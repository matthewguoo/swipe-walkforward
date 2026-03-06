"""
Trade simulation and P&L tracking
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from .trigger import TradeParams


@dataclass
class TradeResult:
    """Result of a single trade"""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    exit_reason: str  # "stop_loss", "take_profit", "max_hold"
    pnl_pct: float
    pnl_r: float  # P&L in R multiples
    holding_days: int


@dataclass
class SwipeDecision:
    """User's swipe decision on a setup"""
    symbol: str
    trigger_date: datetime
    trigger_index: int
    decision: str  # "buy" or "pass"
    timestamp: datetime = field(default_factory=datetime.now)


def simulate_trade(
    df: pd.DataFrame,
    entry_index: int,
    trade_params: TradeParams
) -> Optional[TradeResult]:
    """
    Simulate a trade from entry point
    
    Args:
        df: Full price data
        entry_index: Index where trade is entered (day after trigger)
        trade_params: Risk/Reward parameters
    
    Returns:
        TradeResult or None if can't simulate
    """
    if entry_index >= len(df) - 1:
        return None
    
    # Entry on open of next day after trigger
    entry_row = df.iloc[entry_index]
    entry_price = entry_row['Open']
    entry_date = entry_row['Date']
    
    stop_price = entry_price * (1 - trade_params.stop_loss_pct / 100)
    target_price = entry_price * (1 + trade_params.take_profit_pct / 100)
    
    # Walk forward through future candles
    for i in range(entry_index, min(entry_index + trade_params.max_hold_days, len(df))):
        row = df.iloc[i]
        
        # Check stop loss (using low)
        if row['Low'] <= stop_price:
            return TradeResult(
                symbol=df.get('symbol', 'UNKNOWN'),
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=row['Date'],
                exit_price=stop_price,
                exit_reason="stop_loss",
                pnl_pct=-trade_params.stop_loss_pct,
                pnl_r=-1.0,
                holding_days=i - entry_index
            )
        
        # Check take profit (using high)
        if row['High'] >= target_price:
            return TradeResult(
                symbol=df.get('symbol', 'UNKNOWN'),
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=row['Date'],
                exit_price=target_price,
                exit_reason="take_profit",
                pnl_pct=trade_params.take_profit_pct,
                pnl_r=trade_params.rr_ratio,
                holding_days=i - entry_index
            )
    
    # Max hold reached, exit at close
    exit_row = df.iloc[min(entry_index + trade_params.max_hold_days - 1, len(df) - 1)]
    exit_price = exit_row['Close']
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    pnl_r = pnl_pct / trade_params.stop_loss_pct
    
    return TradeResult(
        symbol=df.get('symbol', 'UNKNOWN'),
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_row['Date'],
        exit_price=exit_price,
        exit_reason="max_hold",
        pnl_pct=pnl_pct,
        pnl_r=pnl_r,
        holding_days=trade_params.max_hold_days
    )


@dataclass
class WalkforwardSession:
    """Manages a walkforward backtesting session"""
    decisions: List[SwipeDecision] = field(default_factory=list)
    results: List[TradeResult] = field(default_factory=list)
    
    def add_decision(self, decision: SwipeDecision, result: Optional[TradeResult]):
        self.decisions.append(decision)
        if decision.decision == "buy" and result:
            self.results.append(result)
    
    def get_stats(self) -> Dict:
        """Calculate session statistics"""
        if not self.results:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_r': 0,
                'total_r': 0,
            }
        
        wins = [r for r in self.results if r.pnl_r > 0]
        losses = [r for r in self.results if r.pnl_r <= 0]
        
        return {
            'total_decisions': len(self.decisions),
            'buy_decisions': len([d for d in self.decisions if d.decision == 'buy']),
            'pass_decisions': len([d for d in self.decisions if d.decision == 'pass']),
            'total_trades': len(self.results),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.results) * 100 if self.results else 0,
            'avg_r': np.mean([r.pnl_r for r in self.results]),
            'total_r': sum([r.pnl_r for r in self.results]),
            'avg_hold_days': np.mean([r.holding_days for r in self.results]),
            'best_trade_r': max([r.pnl_r for r in self.results]),
            'worst_trade_r': min([r.pnl_r for r in self.results]),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export results to DataFrame for analysis"""
        return pd.DataFrame([
            {
                'symbol': r.symbol,
                'entry_date': r.entry_date,
                'entry_price': r.entry_price,
                'exit_date': r.exit_date,
                'exit_price': r.exit_price,
                'exit_reason': r.exit_reason,
                'pnl_pct': r.pnl_pct,
                'pnl_r': r.pnl_r,
                'holding_days': r.holding_days,
            }
            for r in self.results
        ])
