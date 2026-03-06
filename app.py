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
from src.simulation import simulate_trade, SwipeDecision, WalkforwardSession, TradeResult, PortfolioConfig

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
    interval = data.get('interval', '1h')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    trigger = load_trigger(trigger_path)
    
    all_setups = []
    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, period=period, interval=interval, 
                                  start_date=start_date, end_date=end_date)
            triggers = find_triggers(df, trigger)
            
            trigger_indices = df[triggers].index.tolist()
            for idx in trigger_indices:
                row = df.iloc[idx]
                date_val = row['Date'] if 'Date' in df.columns else df.index[idx]
                all_setups.append({
                    'symbol': symbol,
                    'date': date_val.isoformat() if hasattr(date_val, 'isoformat') else str(date_val),
                    'index': int(idx) if isinstance(idx, (int, float)) else df.index.get_loc(idx),
                    'close': float(row['Close']),
                })
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue
    
    # Store setups and interval - sort by date for chronological walkforward
    all_setups.sort(key=lambda x: x['date'])
    
    session_id = session.get('session_id', 'default')
    current_setups[session_id] = {'setups': all_setups, 'interval': interval, 'period': period}
    
    return jsonify({
        'count': len(all_setups),
        'setups': all_setups
    })


@app.route('/api/setup/<int:setup_idx>')
def get_setup(setup_idx):
    """Get chart data for a specific setup (no future data shown)"""
    session_id = session.get('session_id', 'default')
    session_data = current_setups.get(session_id, {'setups': [], 'interval': '1h', 'period': '2y'})
    setups = session_data['setups']
    interval = session_data['interval']
    period = session_data['period']
    
    if setup_idx >= len(setups):
        return jsonify({'error': 'Invalid setup index'}), 404
    
    setup = setups[setup_idx]
    symbol = setup['symbol']
    trigger_idx = setup['index']
    
    # Fetch data
    df = fetch_stock_data(symbol, period=period, interval=interval)
    
    # Only show data UP TO trigger point (no lookahead)
    df_visible = df.iloc[:trigger_idx + 1].copy()
    
    # Format for candlestick chart
    dates = df_visible['Date'].astype(str).tolist() if 'Date' in df_visible.columns else df_visible.index.astype(str).tolist()
    chart_data = {
        'dates': dates,
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
    session_data = current_setups.get(session_id, {'setups': [], 'interval': '1h', 'period': '2y'})
    setups = session_data['setups']
    interval = session_data['interval']
    period = session_data['period']
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
    
    # Get full data for outcome visualization
    df = fetch_stock_data(setup['symbol'], period=period, interval=interval)
    trigger_idx = setup['index']
    
    # Simulate trade if bought
    result = None
    if decision == "buy":
        trigger = load_trigger('triggers/oversold_bounce.yaml')
        result = simulate_trade(df, trigger_idx + 1, trigger.trade_params)
    
    wf_session.add_decision(swipe, result)
    
    # Get the full chart including outcome (show 30 candles after trigger)
    look_ahead = 30
    df_full = df.iloc[:trigger_idx + look_ahead + 1].copy() if trigger_idx + look_ahead < len(df) else df.copy()
    
    dates = df_full['Date'].astype(str).tolist() if 'Date' in df_full.columns else df_full.index.astype(str).tolist()
    full_chart = {
        'dates': dates,
        'open': df_full['Open'].tolist(),
        'high': df_full['High'].tolist(),
        'low': df_full['Low'].tolist(),
        'close': df_full['Close'].tolist(),
        'trigger_idx': trigger_idx,  # To mark the trigger point
    }
    
    # Calculate entry, stop, target prices for visualization
    entry_price = df.iloc[trigger_idx + 1]['Open'] if trigger_idx + 1 < len(df) else None
    trigger = load_trigger('triggers/oversold_bounce.yaml')
    stop_price = entry_price * (1 - trigger.trade_params.stop_loss_pct / 100) if entry_price else None
    target_price = entry_price * (1 + trigger.trade_params.take_profit_pct / 100) if entry_price else None
    
    return jsonify({
        'decision': decision,
        'result': {
            'pnl_pct': result.pnl_pct if result else None,
            'pnl_r': result.pnl_r if result else None,
            'exit_reason': result.exit_reason if result else None,
            'holding_days': result.holding_days if result else None,
        } if result else None,
        'full_chart': full_chart,
        'entry_price': entry_price,
        'stop_price': stop_price,
        'target_price': target_price,
        'session_stats': wf_session.get_stats()
    })


@app.route('/api/stats')
def get_stats():
    """Get current session statistics"""
    wf_session = get_session()
    return jsonify(wf_session.get_stats())


@app.route('/api/portfolio', methods=['POST'])
def set_portfolio():
    """Configure portfolio settings"""
    data = request.json
    wf_session = get_session()
    wf_session.set_portfolio(
        starting_equity=data.get('starting_equity', 10000),
        risk_pct=data.get('risk_per_trade_pct', 2.0),
        max_position_pct=data.get('max_position_pct', 25.0)
    )
    return jsonify({'status': 'ok', 'portfolio': {
        'starting_equity': wf_session.portfolio.starting_equity,
        'risk_per_trade_pct': wf_session.portfolio.risk_per_trade_pct,
        'max_position_pct': wf_session.portfolio.max_position_pct,
    }})


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
