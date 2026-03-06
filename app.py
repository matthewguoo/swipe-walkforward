"""
Swipe Walkforward - Main Application
A "Tinder for stocks" backtesting tool
"""
from flask import Flask, render_template, jsonify, request, session
import pandas as pd
import json
import os
from datetime import datetime
from src.data import fetch_stock_data, get_sp500_symbols, scan_universe
from src.trigger import load_trigger, find_triggers, Trigger
from src.simulation import simulate_trade, SwipeDecision, WalkforwardSession, TradeResult

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global state (in production, use Redis or database)
sessions = {}
current_setups = {}


def get_session() -> WalkforwardSession:
    """Get or create walkforward session"""
    session_id = session.get('session_id', 'default')
    if session_id not in sessions:
        sessions[session_id] = WalkforwardSession()
    return sessions[session_id]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/scan', methods=['POST'])
def scan_stocks():
    """Scan for trigger points"""
    data = request.json
    trigger_path = data.get('trigger', 'triggers/oversold_bounce.yaml')
    symbols = data.get('symbols', get_sp500_symbols()[:20])  # Limit for speed
    period = data.get('period', '2y')
    
    trigger = load_trigger(trigger_path)
    
    all_setups = []
    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, period=period)
            triggers = find_triggers(df, trigger)
            
            trigger_indices = df[triggers].index.tolist()
            for idx in trigger_indices:
                all_setups.append({
                    'symbol': symbol,
                    'date': df.iloc[idx]['Date'].isoformat() if hasattr(df.iloc[idx]['Date'], 'isoformat') else str(df.iloc[idx]['Date']),
                    'index': int(idx),
                    'close': float(df.iloc[idx]['Close']),
                    'rsi': float(df.iloc[idx].get('rsi', 0)) if 'rsi' in df.columns else None,
                })
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue
    
    # Store setups
    session_id = session.get('session_id', 'default')
    current_setups[session_id] = all_setups
    
    return jsonify({
        'count': len(all_setups),
        'setups': all_setups
    })


@app.route('/api/setup/<int:setup_idx>')
def get_setup(setup_idx):
    """Get chart data for a specific setup (no future data shown)"""
    session_id = session.get('session_id', 'default')
    setups = current_setups.get(session_id, [])
    
    if setup_idx >= len(setups):
        return jsonify({'error': 'Invalid setup index'}), 404
    
    setup = setups[setup_idx]
    symbol = setup['symbol']
    trigger_idx = setup['index']
    
    # Fetch data
    df = fetch_stock_data(symbol, period='2y')
    
    # Only show data UP TO trigger point (no lookahead)
    df_visible = df.iloc[:trigger_idx + 1].copy()
    
    # Format for candlestick chart
    chart_data = {
        'dates': df_visible['Date'].astype(str).tolist(),
        'open': df_visible['Open'].tolist(),
        'high': df_visible['High'].tolist(),
        'low': df_visible['Low'].tolist(),
        'close': df_visible['Close'].tolist(),
        'volume': df_visible['Volume'].tolist(),
    }
    
    return jsonify({
        'symbol': symbol,
        'trigger_date': setup['date'],
        'trigger_idx': trigger_idx,
        'setup_idx': setup_idx,
        'total_setups': len(setups),
        'chart': chart_data
    })


@app.route('/api/decide', methods=['POST'])
def make_decision():
    """Record a swipe decision and simulate trade"""
    data = request.json
    setup_idx = data['setup_idx']
    decision = data['decision']  # "buy" or "pass"
    
    session_id = session.get('session_id', 'default')
    setups = current_setups.get(session_id, [])
    wf_session = get_session()
    
    if setup_idx >= len(setups):
        return jsonify({'error': 'Invalid setup index'}), 404
    
    setup = setups[setup_idx]
    
    # Create decision record
    swipe = SwipeDecision(
        symbol=setup['symbol'],
        trigger_date=datetime.fromisoformat(setup['date']) if isinstance(setup['date'], str) else setup['date'],
        trigger_index=setup['index'],
        decision=decision
    )
    
    # Simulate trade if bought
    result = None
    if decision == "buy":
        trigger = load_trigger('triggers/oversold_bounce.yaml')
        df = fetch_stock_data(setup['symbol'], period='2y')
        result = simulate_trade(df, setup['index'] + 1, trigger.trade_params)
    
    wf_session.add_decision(swipe, result)
    
    # Get the actual outcome (for reveal)
    df = fetch_stock_data(setup['symbol'], period='2y')
    trigger_idx = setup['index']
    future_data = None
    
    if trigger_idx + 20 < len(df):
        future_df = df.iloc[trigger_idx:trigger_idx + 21]
        future_data = {
            'dates': future_df['Date'].astype(str).tolist(),
            'open': future_df['Open'].tolist(),
            'high': future_df['High'].tolist(),
            'low': future_df['Low'].tolist(),
            'close': future_df['Close'].tolist(),
        }
    
    return jsonify({
        'decision': decision,
        'result': {
            'pnl_pct': result.pnl_pct if result else None,
            'pnl_r': result.pnl_r if result else None,
            'exit_reason': result.exit_reason if result else None,
            'holding_days': result.holding_days if result else None,
        } if result else None,
        'future_chart': future_data,
        'session_stats': wf_session.get_stats()
    })


@app.route('/api/stats')
def get_stats():
    """Get current session statistics"""
    wf_session = get_session()
    return jsonify(wf_session.get_stats())


@app.route('/api/master-chart/<symbol>')
def master_chart(symbol):
    """Get full chart with all trigger points marked"""
    trigger = load_trigger('triggers/oversold_bounce.yaml')
    df = fetch_stock_data(symbol, period='2y')
    triggers = find_triggers(df, trigger)
    
    trigger_indices = df[triggers].index.tolist()
    
    chart_data = {
        'dates': df['Date'].astype(str).tolist(),
        'open': df['Open'].tolist(),
        'high': df['High'].tolist(),
        'low': df['Low'].tolist(),
        'close': df['Close'].tolist(),
        'volume': df['Volume'].tolist(),
        'trigger_indices': trigger_indices,
        'trigger_dates': [df.iloc[i]['Date'].isoformat() if hasattr(df.iloc[i]['Date'], 'isoformat') else str(df.iloc[i]['Date']) for i in trigger_indices]
    }
    
    return jsonify(chart_data)


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the current session"""
    session_id = session.get('session_id', 'default')
    sessions[session_id] = WalkforwardSession()
    return jsonify({'status': 'reset'})


if __name__ == '__main__':
    os.makedirs('data/cache', exist_ok=True)
    app.run(debug=True, port=5000)
