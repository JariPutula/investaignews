"""
Risk metrics calculations for portfolio analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Parameters:
        returns: Array of portfolio returns
        risk_free_rate: Annual risk-free rate (default 0%)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)
    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (downside risk-adjusted returns).
    
    Parameters:
        returns: Array of portfolio returns
        risk_free_rate: Annual risk-free rate (default 0%)
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)
    return sortino


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown.
    
    Parameters:
        returns: Array of portfolio returns
    
    Returns:
        Maximum drawdown as percentage
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert to pandas Series if needed
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return max_drawdown * 100  # Return as percentage


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Parameters:
        returns: Array of portfolio returns
        confidence: Confidence level (default 95%)
    
    Returns:
        VaR as percentage
    """
    if len(returns) == 0:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence) * 100)
    return abs(var) * 100  # Return as positive percentage


def calculate_portfolio_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio risk metrics.
    
    Parameters:
        df: DataFrame with holdings data containing:
            - market_total_eur: Current market value
            - change_pct: Percentage change
        risk_free_rate: Annual risk-free rate (default 0%)
    
    Returns:
        Dictionary with all risk metrics:
            - portfolio_return: Weighted average return
            - portfolio_volatility: Portfolio volatility
            - sharpe_ratio: Sharpe ratio
            - sortino_ratio: Sortino ratio
            - max_drawdown: Maximum drawdown percentage
            - var_95: Value at Risk at 95% confidence
            - var_99: Value at Risk at 99% confidence
            - beta: Beta (simplified, defaults to 1.0)
            - top_5_concentration: Top 5 holdings concentration %
            - top_10_concentration: Top 10 holdings concentration %
            - herfindahl_index: Herfindahl-Hirschman Index
    """
    # Use change_pct as proxy for returns (weighted by portfolio value)
    total_value = df['market_total_eur'].sum()
    weights = df['market_total_eur'] / total_value
    
    # Portfolio return (weighted average)
    portfolio_return = (weights * df['change_pct']).sum()
    
    # Portfolio volatility (simplified - using weighted standard deviation)
    portfolio_volatility = np.sqrt(np.sum(weights**2 * df['change_pct'].std()**2))
    
    # For metrics that need time series, we'll use the holdings' returns as proxy
    # In a real implementation, you'd use historical portfolio returns
    returns_array = df['change_pct'].values / 100  # Convert to decimal
    
    # Calculate metrics
    sharpe = calculate_sharpe_ratio(returns_array, risk_free_rate, periods_per_year=1)
    sortino = calculate_sortino_ratio(returns_array, risk_free_rate, periods_per_year=1)
    max_dd = calculate_max_drawdown(returns_array)
    var_95 = calculate_var(returns_array, confidence=0.95)
    var_99 = calculate_var(returns_array, confidence=0.99)
    
    # Concentration metrics
    top_5_concentration = df.nlargest(5, 'market_total_eur')['market_total_eur'].sum() / total_value * 100
    top_10_concentration = df.nlargest(10, 'market_total_eur')['market_total_eur'].sum() / total_value * 100
    herfindahl_index = np.sum(weights**2) * 10000  # HHI scaled by 10000
    
    # Beta (simplified - would need market returns in real implementation)
    # Using average change_pct as market proxy
    market_return = df['change_pct'].mean()
    portfolio_return_avg = portfolio_return
    # Simplified beta calculation
    beta = 1.0  # Default, would need historical data for accurate calculation
    
    return {
        'portfolio_return': portfolio_return,
        'portfolio_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'var_95': var_95,
        'var_99': var_99,
        'beta': beta,
        'top_5_concentration': top_5_concentration,
        'top_10_concentration': top_10_concentration,
        'herfindahl_index': herfindahl_index
    }

