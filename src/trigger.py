"""
Trigger definition and detection system
"""
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .indicators import INDICATORS


@dataclass
class TradeParams:
    """Risk/Reward parameters for a trade"""
    stop_loss_pct: float = 5.0  # Stop loss as % below entry
    take_profit_pct: float = 10.0  # Take profit as % above entry
    max_hold_days: int = 20  # Max days to hold before forced exit
    
    @property
    def rr_ratio(self) -> float:
        return self.take_profit_pct / self.stop_loss_pct


@dataclass
class TriggerCondition:
    """Single condition in a trigger"""
    indicator: str
    period: int
    operator: str  # "<", ">", "<=", ">=", "=="
    value: float
    column: str = "Close"  # Which price column to use


@dataclass
class Trigger:
    """Complete trigger definition"""
    name: str
    conditions: List[TriggerCondition]
    logic: str = "AND"  # "AND" or "OR"
    trade_params: TradeParams = None
    
    def __post_init__(self):
        if self.trade_params is None:
            self.trade_params = TradeParams()


def load_trigger(path: str) -> Trigger:
    """Load trigger from YAML file"""
    with open(path) as f:
        config = yaml.safe_load(f)
    
    conditions = []
    for c in config.get('conditions', []):
        conditions.append(TriggerCondition(
            indicator=c['indicator'],
            period=c.get('period', 14),
            operator=c['operator'],
            value=c['value'],
            column=c.get('column', 'Close')
        ))
    
    trade_params = TradeParams()
    if 'trade_params' in config:
        tp = config['trade_params']
        trade_params = TradeParams(
            stop_loss_pct=tp.get('stop_loss_pct', 5.0),
            take_profit_pct=tp.get('take_profit_pct', 10.0),
            max_hold_days=tp.get('max_hold_days', 20)
        )
    
    return Trigger(
        name=config['name'],
        conditions=conditions,
        logic=config.get('logic', 'AND'),
        trade_params=trade_params
    )


def evaluate_condition(df: pd.DataFrame, condition: TriggerCondition) -> pd.Series:
    """Evaluate a single condition, returns boolean Series"""
    indicator_fn = INDICATORS.get(condition.indicator)
    if indicator_fn is None:
        raise ValueError(f"Unknown indicator: {condition.indicator}")
    
    # Get the indicator values
    if condition.indicator == 'volume_spike':
        values = indicator_fn(df['Volume'], period=condition.period)
    else:
        values = indicator_fn(df[condition.column], period=condition.period)
    
    # Apply comparison
    ops = {
        '<': lambda v, t: v < t,
        '>': lambda v, t: v > t,
        '<=': lambda v, t: v <= t,
        '>=': lambda v, t: v >= t,
        '==': lambda v, t: v == t,
    }
    return ops[condition.operator](values, condition.value)


def find_triggers(df: pd.DataFrame, trigger: Trigger) -> pd.Series:
    """Find all trigger points in the data, returns boolean Series"""
    results = []
    for condition in trigger.conditions:
        results.append(evaluate_condition(df, condition))
    
    if trigger.logic == "AND":
        combined = results[0]
        for r in results[1:]:
            combined = combined & r
    else:  # OR
        combined = results[0]
        for r in results[1:]:
            combined = combined | r
    
    return combined
