"""
Historical portfolio snapshot management and loading.
"""

from portfolio.historical.snapshot_manager import (
    find_snapshot_files,
    parse_snapshot_date,
    get_snapshot_info,
)
from portfolio.historical.snapshot_loader import (
    load_snapshot,
    load_all_snapshots,
    load_latest_snapshot,
    get_portfolio_value_over_time,
)
from portfolio.historical.performance_tracker import (
    calculate_period_returns,
    calculate_benchmark_period_returns,
    calculate_tracking_error,
    calculate_information_ratio,
    calculate_beta,
    calculate_alpha,
    calculate_rolling_volatility,
    calculate_rolling_sharpe_ratio,
    calculate_maximum_drawdown,
    compare_portfolio_to_benchmark,
    get_historical_performance_summary,
)

__all__ = [
    # Snapshot management
    "find_snapshot_files",
    "parse_snapshot_date",
    "get_snapshot_info",
    # Snapshot loading
    "load_snapshot",
    "load_all_snapshots",
    "load_latest_snapshot",
    "get_portfolio_value_over_time",
    # Performance tracking
    "calculate_period_returns",
    "calculate_benchmark_period_returns",
    "calculate_tracking_error",
    "calculate_information_ratio",
    "calculate_beta",
    "calculate_alpha",
    "calculate_rolling_volatility",
    "calculate_rolling_sharpe_ratio",
    "calculate_maximum_drawdown",
    "compare_portfolio_to_benchmark",
    "get_historical_performance_summary",
]

