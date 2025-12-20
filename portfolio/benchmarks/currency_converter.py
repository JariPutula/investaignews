"""
Currency conversion utilities for benchmark data.
"""

import pandas as pd
from datetime import datetime
from typing import Optional

import streamlit as st


# Yahoo Finance ticker for EUR/USD exchange rate
EXCHANGE_RATE_TICKER = "EURUSD=X"
BASE_CURRENCY = "EUR"


@st.cache_data(ttl=60 * 60)  # Cache for 1 hour
def get_exchange_rate(
    ticker: str = EXCHANGE_RATE_TICKER,
    date: Optional[datetime] = None
) -> Optional[float]:
    """
    Get current or historical EUR/USD exchange rate from Yahoo Finance.
    
    Args:
        ticker: Exchange rate ticker (default: EURUSD=X)
        date: Date for historical rate (None for current rate)
    
    Returns:
        Exchange rate (EUR/USD), or None if fetch fails
    """
    try:
        import yfinance as yf
    except ImportError:
        return None
    
    try:
        if date is not None:
            # Fetch historical rate for specific date
            start_date = date.strftime("%Y-%m-%d")
            end_date = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            # Fetch current rate
            data = yf.download(ticker, period="1d", progress=False)
        
        if data is None or data.empty:
            return None
        
        # Use 'Close' price (or 'Adj Close' if available)
        if 'Adj Close' in data.columns:
            rate = data['Adj Close'].iloc[-1]
        else:
            rate = data['Close'].iloc[-1]
        
        if pd.isna(rate):
            return None
        
        return float(rate)
    except Exception:
        return None


def fetch_exchange_rate_for_date(
    date: datetime,
    ticker: str = EXCHANGE_RATE_TICKER
) -> Optional[float]:
    """
    Fetch EUR/USD exchange rate for a specific date.
    
    Args:
        date: Date to fetch rate for
        ticker: Exchange rate ticker (default: EURUSD=X)
    
    Returns:
        Exchange rate (EUR/USD), or None if fetch fails
    """
    return get_exchange_rate(ticker, date)


def convert_usd_to_eur(
    usd_value: float,
    exchange_rate: Optional[float] = None,
    date: Optional[datetime] = None
) -> Optional[float]:
    """
    Convert USD value to EUR.
    
    Args:
        usd_value: Value in USD
        exchange_rate: Exchange rate (EUR/USD). If None, will fetch current rate.
        date: Date for historical conversion (if exchange_rate is None)
    
    Returns:
        Value in EUR, or None if conversion fails
    """
    if exchange_rate is None:
        exchange_rate = get_exchange_rate(date=date)
        if exchange_rate is None:
            return None
    
    return usd_value * exchange_rate


def convert_benchmark_to_base_currency(
    value: float,
    ticker: str,
    date: Optional[datetime] = None,
    base_currency: str = BASE_CURRENCY
) -> Optional[float]:
    """
    Convert benchmark value to base currency (EUR).
    
    Args:
        value: Benchmark value in its native currency
        ticker: Benchmark ticker symbol
        date: Date for conversion (for historical rates)
        base_currency: Target currency (default: EUR)
    
    Returns:
        Value in base currency, or None if conversion fails
    """
    from portfolio.benchmarks.benchmark_config import get_benchmark_currency
    
    benchmark_currency = get_benchmark_currency(ticker)
    
    # If already in base currency, return as-is
    if benchmark_currency == base_currency:
        return value
    
    # Convert from benchmark currency to base currency
    if benchmark_currency == "USD" and base_currency == "EUR":
        return convert_usd_to_eur(value, date=date)
    elif benchmark_currency == "GBP" and base_currency == "EUR":
        # GBP to EUR conversion (would need GBPUSD=X and EURUSD=X)
        # For now, use approximate rate or fetch GBP/EUR directly
        gbp_eur_rate = get_exchange_rate("GBPEUR=X", date=date)
        if gbp_eur_rate:
            return value * gbp_eur_rate
        return None
    elif benchmark_currency == "JPY" and base_currency == "EUR":
        # JPY to EUR conversion
        jpy_eur_rate = get_exchange_rate("JPYEUR=X", date=date)
        if jpy_eur_rate:
            return value * jpy_eur_rate
        return None
    else:
        # Unsupported conversion
        return None

