"""
Data fetching and management
"""
import yfinance as yf
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import os
import json


CACHE_DIR = "data/cache"


def fetch_stock_data(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol
    
    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        period: Period to fetch if no dates (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, 1wk, 1mo)
        use_cache: Whether to use cached data
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_file = f"{CACHE_DIR}/{symbol}_{period}_{interval}.parquet"
    
    if use_cache and os.path.exists(cache_file):
        # Check if cache is fresh (less than 1 day old for daily, 1 hour for intraday)
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        cache_ttl = timedelta(hours=1) if interval != "1d" else timedelta(days=1)
        if datetime.now() - mtime < cache_ttl:
            return pd.read_parquet(cache_file)
    
    ticker = yf.Ticker(symbol)
    
    # yfinance has limits on intraday data
    # 1m: 7 days, 5m/15m: 60 days, 1h: 730 days
    if start_date and end_date:
        df = ticker.history(start=start_date, end=end_date, interval=interval)
    else:
        df = ticker.history(period=period, interval=interval)
    
    # Clean up column names
    df = df.reset_index()
    df.columns = [c.replace(' ', '_') for c in df.columns]
    
    # Save to cache
    df.to_parquet(cache_file)
    
    return df


def get_sp500_symbols() -> List[str]:
    """Get list of S&P 500 symbols"""
    # Using a static list for reliability
    # In production, could scrape Wikipedia or use an API
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        df = pd.read_csv(url)
        return df['Symbol'].tolist()
    except:
        # Fallback to a small test set
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'INTC']


def scan_universe(
    symbols: List[str],
    trigger_fn,
    period: str = "2y"
) -> pd.DataFrame:
    """
    Scan a universe of stocks for trigger points
    
    Returns DataFrame with columns: symbol, date, trigger_index
    """
    all_triggers = []
    
    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, period=period)
            triggers = trigger_fn(df)
            
            trigger_dates = df[triggers]['Date'].tolist()
            for i, date in enumerate(trigger_dates):
                all_triggers.append({
                    'symbol': symbol,
                    'date': date,
                    'trigger_index': df[triggers].index[i]
                })
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue
    
    return pd.DataFrame(all_triggers)
