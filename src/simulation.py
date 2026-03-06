"""
Trade simulation and P&L tracking
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Tuple
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
class OpenPosition:
    """Tracks an open position"""
    symbol: str
    entry_date: datetime
    entry_price: float
    stop_price: float
    target_price: float
    position_size: float  # Dollar amount
    max_hold_candles: int
    candles_held: int = 0


@dataclass
class WalkforwardEvent:
    """An event in the walkforward simulation"""
    event_type: str  # "trigger", "exit", "missed"
    symbol: str
    date: datetime
    details: Dict = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """Portfolio simulation settings"""
    starting_equity: float = 10000.0
    risk_per_trade_pct: float = 2.0  # Risk 2% of equity per trade
    max_position_pct: float = 25.0  # Max 25% of equity in one position


@dataclass
class WalkforwardSession:
    """Manages a walkforward backtesting session"""
    decisions: List[SwipeDecision] = field(default_factory=list)
    results: List[TradeResult] = field(default_factory=list)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    equity_curve: List[float] = field(default_factory=list)
    current_equity: float = 10000.0
    available_cash: float = 10000.0
    open_positions: List[OpenPosition] = field(default_factory=list)
    events: List[WalkforwardEvent] = field(default_factory=list)
    
    def __post_init__(self):
        self.current_equity = self.portfolio.starting_equity
        self.available_cash = self.portfolio.starting_equity
        self.equity_curve = [self.current_equity]
    
    def set_portfolio(self, starting_equity: float, risk_pct: float, max_position_pct: float):
        """Update portfolio settings"""
        self.portfolio = PortfolioConfig(
            starting_equity=starting_equity,
            risk_per_trade_pct=risk_pct,
            max_position_pct=max_position_pct
        )
        self.current_equity = starting_equity
        self.available_cash = starting_equity
        self.equity_curve = [self.current_equity]
        self.open_positions = []
        self.events = []
    
    def can_open_position(self, entry_price: float, stop_pct: float) -> Tuple[bool, float, str]:
        """Check if we can open a position, returns (can_trade, position_size, reason)"""
        risk_amount = self.current_equity * (self.portfolio.risk_per_trade_pct / 100)
        position_size = risk_amount / (stop_pct / 100)  # Size to risk exactly risk_amount
        
        max_position = self.current_equity * (self.portfolio.max_position_pct / 100)
        position_size = min(position_size, max_position)
        
        if position_size > self.available_cash:
            if self.available_cash < risk_amount:
                return False, 0, "insufficient_cash"
            position_size = self.available_cash
        
        return True, position_size, "ok"
    
    def open_position(self, symbol: str, entry_date: datetime, entry_price: float, 
                      stop_price: float, target_price: float, position_size: float, max_hold: int):
        """Open a new position"""
        self.open_positions.append(OpenPosition(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            position_size=position_size,
            max_hold_candles=max_hold
        ))
        self.available_cash -= position_size
        self.events.append(WalkforwardEvent(
            event_type="entry",
            symbol=symbol,
            date=entry_date,
            details={"price": entry_price, "size": position_size}
        ))
    
    def close_position(self, position: OpenPosition, exit_price: float, exit_date: datetime, reason: str):
        """Close an existing position"""
        pnl_pct = (exit_price - position.entry_price) / position.entry_price
        pnl_dollars = position.position_size * pnl_pct
        
        self.available_cash += position.position_size + pnl_dollars
        self.current_equity += pnl_dollars
        self.equity_curve.append(self.current_equity)
        
        self.open_positions.remove(position)
        self.events.append(WalkforwardEvent(
            event_type="exit",
            symbol=position.symbol,
            date=exit_date,
            details={"price": exit_price, "reason": reason, "pnl": pnl_dollars, "pnl_pct": pnl_pct * 100}
        ))
        
        return pnl_dollars, pnl_pct
    
    def add_missed_trigger(self, symbol: str, date: datetime, reason: str):
        """Record a missed trigger due to no cash"""
        self.events.append(WalkforwardEvent(
            event_type="missed",
            symbol=symbol,
            date=date,
            details={"reason": reason}
        ))
    
    def add_decision(self, decision: SwipeDecision, result: Optional[TradeResult]):
        self.decisions.append(decision)
        if decision.decision == "buy" and result:
            self.results.append(result)
            
            # Calculate position size based on risk
            risk_amount = self.current_equity * (self.portfolio.risk_per_trade_pct / 100)
            # Position size = risk_amount / stop_loss_pct
            # PnL = position_size * pnl_pct
            pnl_dollars = risk_amount * result.pnl_r  # pnl_r is in R multiples
            
            self.current_equity += pnl_dollars
            self.equity_curve.append(self.current_equity)
    
    def get_stats(self) -> Dict:
        """Calculate session statistics"""
        if not self.results:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_r': 0,
                'total_r': 0,
                'starting_equity': self.portfolio.starting_equity,
                'current_equity': self.current_equity,
                'available_cash': self.available_cash,
                'total_return_pct': 0,
                'equity_curve': self.equity_curve,
            }
        
        wins = [r for r in self.results if r.pnl_r > 0]
        losses = [r for r in self.results if r.pnl_r <= 0]
        
        total_return_pct = ((self.current_equity - self.portfolio.starting_equity) / self.portfolio.starting_equity) * 100
        
        # Calculate max drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
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
            'starting_equity': self.portfolio.starting_equity,
            'current_equity': self.current_equity,
            'available_cash': self.available_cash,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_dd,
            'equity_curve': self.equity_curve,
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
