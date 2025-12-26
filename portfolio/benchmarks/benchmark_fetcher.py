"""
Fetch benchmark data from Yahoo Finance.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import streamlit as st

from portfolio.benchmarks.benchmark_config import (
    get_benchmark_ticker,
    get_benchmark_currency,
)
from portfolio.benchmarks.currency_converter import (
    convert_benchmark_to_base_currency,
    get_exchange_rate,
)


@st.cache_data(ttl=60 * 60 * 24)  # Cache for 24 hours
def fetch_benchmark_data(
    benchmark_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    period: str = "1y"
) -> Optional[pd.DataFrame]:
    """
    Fetch benchmark data from Yahoo Finance.
    
    Args:
        benchmark_name: Name of benchmark (e.g., "S&P 500")
        start_date: Start date for historical data
        end_date: End date for historical data
        period: Period if dates not specified (default: "1y")
    
    Returns:
        DataFrame with benchmark prices, or None if fetch fails
        Columns: Date, Close, Adj Close (if available)
    """
    try:
        import yfinance as yf
    except ImportError:
        return None
    
    try:
        ticker = get_benchmark_ticker(benchmark_name)
        
        if start_date and end_date:
            data = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )
        else:
            data = yf.download(ticker, period=period, progress=False)
        
        if data is None or data.empty:
            return None
        
        # Handle MultiIndex columns (yfinance sometimes returns these)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten column names
            data.columns = data.columns.get_level_values(-1)
        
        # Use Adj Close if available, otherwise Close
        if 'Adj Close' in data.columns:
            prices = data['Adj Close'].copy()
        elif 'Close' in data.columns:
            prices = data['Close'].copy()
        else:
            # Fallback: use first numeric column
            numeric_cols = data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            if len(numeric_cols) > 0:
                prices = data[numeric_cols[0]].copy()
            else:
                return None
        
        # Ensure prices is a Series (1D)
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]  # Take first column if DataFrame
        
        # Convert to DataFrame with Date index
        # Ensure all values are 1D arrays
        result = pd.DataFrame({
            'price': prices.values.flatten(),  # Ensure 1D
        }, index=prices.index)
        
        result.index.name = 'date'
        
        return result
    except Exception as e:
        print(f"Error fetching benchmark {benchmark_name}: {e}")
        return None


def fetch_benchmark_for_date_range(
    benchmark_name: str,
    dates: List[datetime],
    convert_to_eur: bool = True
) -> pd.DataFrame:
    """
    Fetch benchmark values for specific dates (matching snapshot dates).
    
    Args:
        benchmark_name: Name of benchmark (e.g., "S&P 500")
        dates: List of dates to fetch values for (datetime objects or pandas Timestamp)
        convert_to_eur: Whether to convert to EUR (default: True)
    
    Returns:
        DataFrame with columns: date, price_usd, price_eur (if converted)
    """
    if not dates:
        return pd.DataFrame()
    
    # Ensure dates are datetime objects
    datetime_dates = []
    for date in dates:
        if isinstance(date, pd.Timestamp):
            datetime_dates.append(date.to_pydatetime())
        elif isinstance(date, datetime):
            datetime_dates.append(date)
        else:
            # Try to convert
            try:
                datetime_dates.append(pd.to_datetime(date).to_pydatetime())
            except:
                continue
    
    if not datetime_dates:
        return pd.DataFrame()
    
    # Get date range
    start_date = min(datetime_dates) - timedelta(days=7)  # Add buffer
    end_date = max(datetime_dates) + timedelta(days=1)
    
    # Fetch benchmark data
    benchmark_data = fetch_benchmark_data(
        benchmark_name,
        start_date=start_date,
        end_date=end_date
    )
    
    if benchmark_data is None or benchmark_data.empty:
        return pd.DataFrame()
    
    # Get ticker for currency conversion
    ticker = get_benchmark_ticker(benchmark_name)
    
    # Ensure we have a 'price' column
    if 'price' not in benchmark_data.columns:
        return pd.DataFrame()
    
    # For each snapshot date, find closest benchmark value
    result_data = []
    for snapshot_date in datetime_dates:
        # Find closest date in benchmark data
        # Use asof to get last valid value on or before snapshot_date
        try:
            closest_date = benchmark_data.index.asof(snapshot_date)
            if pd.isna(closest_date):
                # No data before this date, try forward fill
                after_dates = benchmark_data.index[benchmark_data.index >= snapshot_date]
                if len(after_dates) > 0:
                    closest_date = after_dates[0]
                else:
                    continue
        except Exception:
            # Fallback: find closest date manually
            before_dates = benchmark_data.index[benchmark_data.index <= snapshot_date]
            if len(before_dates) == 0:
                continue
            closest_date = before_dates[-1]
        
        try:
            price_usd = benchmark_data.loc[closest_date, 'price']
        except (KeyError, IndexError):
            continue
        
        if pd.isna(price_usd):
            continue
        
        result_row = {
            'date': snapshot_date,
            'price_native': float(price_usd),
            'closest_benchmark_date': closest_date,
        }
        
        # Convert to EUR if requested
        if convert_to_eur:
            price_eur = convert_benchmark_to_base_currency(
                float(price_usd),
                ticker,
                date=snapshot_date
            )
            if price_eur is not None:
                result_row['price_eur'] = price_eur
            # If conversion fails, keep price_native (will be handled in normalization)
        
        result_data.append(result_row)
    
    if not result_data:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(result_data)
    result_df.set_index('date', inplace=True)
    
    return result_df


def normalize_benchmark_data(
    benchmark_df: pd.DataFrame,
    base_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Normalize benchmark data to start at 100 (for comparison charts).
    
    Args:
        benchmark_df: DataFrame with benchmark prices
        base_date: Date to normalize from (default: first date)
    
    Returns:
        DataFrame with normalized values (starting at 100)
    """
    if benchmark_df.empty:
        return benchmark_df
    
    # Use first date as base if not specified
    if base_date is None:
        base_date = benchmark_df.index[0]
    
    # Get base value - prefer price_eur, fallback to price_native
    price_col = None
    base_value = None
    
    if 'price_eur' in benchmark_df.columns:
        # Check if price_eur has valid (non-None, non-NaN) values
        price_eur_values = benchmark_df['price_eur'].dropna()
        if not price_eur_values.empty:
            price_col = 'price_eur'
            # Try to get value at base_date, or use first valid value
            try:
                base_value = benchmark_df.loc[base_date, 'price_eur']
                if pd.isna(base_value):
                    # If base_date value is NaN, use first valid value
                    base_value = price_eur_values.iloc[0]
                    base_date = price_eur_values.index[0]
            except (KeyError, IndexError):
                # base_date not in index, use first valid value
                base_value = price_eur_values.iloc[0]
                base_date = price_eur_values.index[0]
    
    # Fallback to price_native if price_eur not available or all NaN
    if price_col is None and 'price_native' in benchmark_df.columns:
        price_native_values = benchmark_df['price_native'].dropna()
        if not price_native_values.empty:
            price_col = 'price_native'
            # Try to get value at base_date, or use first valid value
            try:
                base_value = benchmark_df.loc[base_date, 'price_native']
                if pd.isna(base_value):
                    # If base_date value is NaN, use first valid value
                    base_value = price_native_values.iloc[0]
                    base_date = price_native_values.index[0]
            except (KeyError, IndexError):
                # base_date not in index, use first valid value
                base_value = price_native_values.iloc[0]
                base_date = price_native_values.index[0]
    
    if price_col is None or pd.isna(base_value) or base_value == 0:
        return benchmark_df
    
    # Normalize to 100
    result = benchmark_df.copy()
    # Only normalize rows with valid price values
    valid_mask = result[price_col].notna()
    result.loc[valid_mask, 'normalized'] = (result.loc[valid_mask, price_col] / base_value) * 100
    
    return result


def fetch_multiple_benchmarks(
    benchmark_names: List[str],
    dates: List[datetime],
    convert_to_eur: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple benchmarks for the same date range.
    
    Args:
        benchmark_names: List of benchmark names
        dates: List of dates to fetch values for
        convert_to_eur: Whether to convert to EUR
    
    Returns:
        Dictionary mapping benchmark names to DataFrames
    """
    results = {}
    
    for benchmark_name in benchmark_names:
        try:
            data = fetch_benchmark_for_date_range(
                benchmark_name,
                dates,
                convert_to_eur=convert_to_eur
            )
            if not data.empty:
                results[benchmark_name] = data
            else:
                # Log empty data but don't fail silently
                import warnings
                warnings.warn(f"Empty data returned for {benchmark_name}")
        except Exception as e:
            # Log error but continue with other benchmarks
            import warnings
            warnings.warn(f"Error fetching {benchmark_name}: {e}")
            continue
    
    return results

