"""
Historical performance calculations and metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from portfolio.historical.snapshot_loader import (
    get_portfolio_value_over_time,
    load_all_snapshots,
)
from portfolio.benchmarks.benchmark_fetcher import (
    fetch_benchmark_for_date_range,
    normalize_benchmark_data,
)


def calculate_period_returns(
    portfolio_timeline: pd.DataFrame,
    periods: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate portfolio returns for different time periods.
    
    Args:
        portfolio_timeline: DataFrame with columns: snapshot_date, total_value_eur
        periods: List of periods to calculate (default: all standard periods)
    
    Returns:
        Dictionary mapping period names to returns (as percentages)
    """
    if portfolio_timeline.empty:
        return {}
    
    if periods is None:
        periods = ["1M", "3M", "6M", "1Y", "YTD", "All-time"]
    
    # Ensure snapshot_date is the index
    if 'snapshot_date' in portfolio_timeline.columns:
        portfolio_timeline = portfolio_timeline.set_index('snapshot_date')
    
    # Get current date and latest portfolio value
    latest_date = portfolio_timeline.index.max()
    latest_value = portfolio_timeline.loc[latest_date, 'total_value_eur']
    
    # Get earliest date
    earliest_date = portfolio_timeline.index.min()
    
    # Calculate returns for each period
    returns = {}
    
    for period in periods:
        if period == "All-time":
            start_date = earliest_date
        elif period == "YTD":
            start_date = datetime(latest_date.year, 1, 1)
        elif period == "1Y":
            start_date = latest_date - timedelta(days=365)
        elif period == "6M":
            start_date = latest_date - timedelta(days=180)
        elif period == "3M":
            start_date = latest_date - timedelta(days=90)
        elif period == "1M":
            start_date = latest_date - timedelta(days=30)
        else:
            continue
        
        # Find closest date >= start_date
        valid_dates = portfolio_timeline.index[portfolio_timeline.index >= start_date]
        if len(valid_dates) == 0:
            returns[period] = None
            continue
        
        start_value = portfolio_timeline.loc[valid_dates[0], 'total_value_eur']
        
        if start_value == 0 or pd.isna(start_value):
            returns[period] = None
            continue
        
        # Calculate return
        period_return = ((latest_value / start_value) - 1) * 100
        returns[period] = period_return
    
    return returns


def calculate_benchmark_period_returns(
    benchmark_data: pd.DataFrame,
    periods: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate benchmark returns for different time periods.
    
    Args:
        benchmark_data: DataFrame with date index and price_eur column
        periods: List of periods to calculate
    
    Returns:
        Dictionary mapping period names to returns (as percentages)
    """
    if benchmark_data.empty or 'price_eur' not in benchmark_data.columns:
        return {}
    
    if periods is None:
        periods = ["1M", "3M", "6M", "1Y", "YTD", "All-time"]
    
    # Get latest date and value
    latest_date = benchmark_data.index.max()
    latest_value = benchmark_data.loc[latest_date, 'price_eur']
    
    # Get earliest date
    earliest_date = benchmark_data.index.min()
    
    returns = {}
    
    for period in periods:
        if period == "All-time":
            start_date = earliest_date
        elif period == "YTD":
            start_date = datetime(latest_date.year, 1, 1)
        elif period == "1Y":
            start_date = latest_date - timedelta(days=365)
        elif period == "6M":
            start_date = latest_date - timedelta(days=180)
        elif period == "3M":
            start_date = latest_date - timedelta(days=90)
        elif period == "1M":
            start_date = latest_date - timedelta(days=30)
        else:
            continue
        
        # Find closest date >= start_date
        valid_dates = benchmark_data.index[benchmark_data.index >= start_date]
        if len(valid_dates) == 0:
            returns[period] = None
            continue
        
        start_value = benchmark_data.loc[valid_dates[0], 'price_eur']
        
        if start_value == 0 or pd.isna(start_value):
            returns[period] = None
            continue
        
        # Calculate return
        period_return = ((latest_value / start_value) - 1) * 100
        returns[period] = period_return
    
    return returns


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate tracking error (standard deviation of excess returns).
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns (aligned by date)
    
    Returns:
        Tracking error (annualized, as percentage)
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return np.nan
    
    # Calculate excess returns
    excess_returns = portfolio_returns - benchmark_returns
    
    # Calculate standard deviation (tracking error)
    tracking_error = excess_returns.std()
    
    # Annualize (assuming daily returns, adjust if needed)
    # For monthly data, multiply by sqrt(12); for daily, sqrt(252)
    # We'll assume monthly snapshots, so multiply by sqrt(12)
    annualized_te = tracking_error * np.sqrt(12) * 100
    
    return annualized_te


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate information ratio (excess return / tracking error).
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
    
    Returns:
        Information ratio
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return np.nan
    
    # Calculate excess returns
    excess_returns = portfolio_returns - benchmark_returns
    avg_excess_return = excess_returns.mean()
    
    # Calculate tracking error
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return np.nan
    
    # Information ratio
    ir = avg_excess_return / tracking_error
    
    return ir


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate beta (sensitivity to benchmark).
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
    
    Returns:
        Beta coefficient
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return np.nan
    
    # Calculate covariance and variance
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return np.nan
    
    beta = covariance / benchmark_variance
    
    return beta


def calculate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate alpha (excess return adjusted for beta).
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Annual risk-free rate (default: 2%)
    
    Returns:
        Alpha (annualized, as percentage)
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return np.nan
    
    # Calculate beta
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    
    if np.isnan(beta):
        return np.nan
    
    # Calculate average returns (monthly)
    avg_portfolio_return = portfolio_returns.mean()
    avg_benchmark_return = benchmark_returns.mean()
    
    # Monthly risk-free rate
    monthly_rf = risk_free_rate / 12
    
    # Alpha = Portfolio Return - (Risk-free + Beta * (Benchmark Return - Risk-free))
    alpha = avg_portfolio_return - (monthly_rf + beta * (avg_benchmark_return - monthly_rf))
    
    # Annualize
    alpha_annualized = alpha * 12 * 100
    
    return alpha_annualized


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 12,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Series of returns
        window: Rolling window size (default: 12 months)
        annualize: Whether to annualize volatility
    
    Returns:
        Series of rolling volatility values
    """
    if len(returns) < window:
        return pd.Series(dtype=float)
    
    rolling_std = returns.rolling(window=window).std()
    
    if annualize:
        # Annualize (assuming monthly data)
        rolling_std = rolling_std * np.sqrt(12) * 100
    else:
        rolling_std = rolling_std * 100
    
    return rolling_std


def calculate_rolling_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    window: int = 12
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        window: Rolling window size
    
    Returns:
        Series of rolling Sharpe ratios
    """
    if len(returns) < window:
        return pd.Series(dtype=float)
    
    # Monthly risk-free rate
    monthly_rf = risk_free_rate / 12
    
    # Calculate excess returns
    excess_returns = returns - monthly_rf
    
    # Rolling mean and std
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    
    # Sharpe ratio (annualized)
    sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
    
    return sharpe


def calculate_maximum_drawdown(portfolio_values: pd.Series) -> Tuple[float, datetime, datetime]:
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values: Series of portfolio values over time
    
    Returns:
        Tuple of (max_drawdown_pct, peak_date, trough_date)
    """
    if portfolio_values.empty:
        return (np.nan, None, None)
    
    # Calculate running maximum (peak)
    running_max = portfolio_values.expanding().max()
    
    # Calculate drawdown
    drawdown = (portfolio_values - running_max) / running_max * 100
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    
    # Find peak before trough
    peak_value = running_max.loc[trough_idx]
    peak_idx = portfolio_values[portfolio_values.index <= trough_idx][
        portfolio_values[portfolio_values.index <= trough_idx] == peak_value
    ].index
    
    if len(peak_idx) > 0:
        peak_date = peak_idx[0]
    else:
        peak_date = trough_idx
    
    return (max_dd, peak_date, trough_idx)


def compare_portfolio_to_benchmark(
    portfolio_timeline: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    benchmark_name: str,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Comprehensive comparison of portfolio vs benchmark.
    
    Args:
        portfolio_timeline: Portfolio value timeline
        benchmark_data: Benchmark price data
        benchmark_name: Name of benchmark
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with comparison metrics
    """
    if portfolio_timeline.empty or benchmark_data.empty:
        return {}
    
    # Ensure indices are datetime
    if 'snapshot_date' in portfolio_timeline.columns:
        portfolio_timeline = portfolio_timeline.set_index('snapshot_date')
    
    if not isinstance(portfolio_timeline.index, pd.DatetimeIndex):
        portfolio_timeline.index = pd.to_datetime(portfolio_timeline.index)
    
    if not isinstance(benchmark_data.index, pd.DatetimeIndex):
        benchmark_data.index = pd.to_datetime(benchmark_data.index)
    
    # Align data by date
    common_dates = portfolio_timeline.index.intersection(benchmark_data.index)
    
    if len(common_dates) < 2:
        return {}
    
    # Get aligned data
    portfolio_values = portfolio_timeline.loc[common_dates, 'total_value_eur']
    benchmark_values = benchmark_data.loc[common_dates, 'price_eur']
    
    # Calculate returns
    portfolio_returns = portfolio_values.pct_change().dropna()
    benchmark_returns = benchmark_values.pct_change().dropna()
    
    # Align returns
    common_return_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_return_dates]
    benchmark_returns = benchmark_returns.loc[common_return_dates]
    
    if len(portfolio_returns) < 2:
        return {}
    
    # Calculate metrics
    comparison = {
        'benchmark_name': benchmark_name,
        'period_returns': {
            'portfolio': calculate_period_returns(portfolio_timeline),
            'benchmark': calculate_benchmark_period_returns(benchmark_data),
        },
        'tracking_error': calculate_tracking_error(portfolio_returns, benchmark_returns),
        'information_ratio': calculate_information_ratio(portfolio_returns, benchmark_returns),
        'beta': calculate_beta(portfolio_returns, benchmark_returns),
        'alpha': calculate_alpha(portfolio_returns, benchmark_returns, risk_free_rate),
        'correlation': portfolio_returns.corr(benchmark_returns),
        'volatility': {
            'portfolio': portfolio_returns.std() * np.sqrt(12) * 100,  # Annualized
            'benchmark': benchmark_returns.std() * np.sqrt(12) * 100,
        },
        'max_drawdown': {
            'portfolio': calculate_maximum_drawdown(portfolio_values),
            'benchmark': calculate_maximum_drawdown(benchmark_values),
        },
    }
    
    return comparison


def get_historical_performance_summary(
    user_name: str,
    benchmark_names: Optional[List[str]] = None,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Get comprehensive historical performance summary.
    
    Args:
        user_name: User name for snapshot loading
        benchmark_names: List of benchmark names to compare against
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with portfolio timeline, benchmarks, and comparison metrics
    """
    # Load portfolio timeline
    portfolio_timeline = get_portfolio_value_over_time(user_name)
    
    if portfolio_timeline.empty:
        return {
            'portfolio_timeline': pd.DataFrame(),
            'benchmarks': {},
            'comparisons': {},
        }
    
    # Ensure snapshot_date is the index (get_portfolio_value_over_time already sets it)
    if 'snapshot_date' in portfolio_timeline.columns:
        portfolio_timeline = portfolio_timeline.set_index('snapshot_date')
    
    # Ensure index is datetime
    if not isinstance(portfolio_timeline.index, pd.DatetimeIndex):
        portfolio_timeline.index = pd.to_datetime(portfolio_timeline.index)
    
    # Get snapshot dates as datetime objects
    datetime_dates = []
    for date in portfolio_timeline.index:
        if isinstance(date, pd.Timestamp):
            datetime_dates.append(date.to_pydatetime())
        elif isinstance(date, datetime):
            datetime_dates.append(date)
        else:
            try:
                datetime_dates.append(pd.to_datetime(date).to_pydatetime())
            except:
                continue
    
    if not datetime_dates:
        return {
            'portfolio_timeline': portfolio_timeline,
            'benchmarks': {},
            'comparisons': {},
        }
    
    # Fetch benchmarks
    if benchmark_names is None:
        from portfolio.benchmarks.benchmark_config import get_recommended_benchmarks
        benchmark_names = get_recommended_benchmarks("global")
    
    from portfolio.benchmarks.benchmark_fetcher import fetch_multiple_benchmarks
    
    benchmarks_data = fetch_multiple_benchmarks(
        benchmark_names,
        datetime_dates,
        convert_to_eur=True
    )
    
    # Calculate comparisons
    comparisons = {}
    for benchmark_name, benchmark_data in benchmarks_data.items():
        comparison = compare_portfolio_to_benchmark(
            portfolio_timeline,
            benchmark_data,
            benchmark_name,
            risk_free_rate
        )
        if comparison:
            comparisons[benchmark_name] = comparison
    
    return {
        'portfolio_timeline': portfolio_timeline,
        'benchmarks': benchmarks_data,
        'comparisons': comparisons,
        'period_returns': calculate_period_returns(portfolio_timeline),
    }

