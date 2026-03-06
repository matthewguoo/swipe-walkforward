# Swipe Walkforward

A "Tinder for stocks" backtesting tool that lets you train and test your trading intuition.

## Concept

1. **Define triggers** using a simple config language (e.g., RSI < 30 AND price_drop > 10%)
2. **Walk forward** through historical data, finding trigger points
3. **Swipe interface** shows chart UP TO the trigger (future hidden)
4. **Decide** buy or pass on each setup
5. **Simulate P&L** based on your decisions
6. **Analyze** your intuition as a dataset

## Trigger Definition

Triggers are defined in YAML:

```yaml
name: oversold_bounce
conditions:
  - indicator: rsi
    period: 14
    operator: "<"
    value: 30
  - indicator: pct_from_high
    period: 20
    operator: ">"
    value: 10
logic: "AND"
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Usage

1. Define your trigger in `triggers/`
2. Run the scanner to find historical trigger points
3. Swipe through setups and make decisions
4. Review your simulated P&L and decision patterns
