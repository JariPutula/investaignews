"""
Benchmark data fetching and currency conversion for portfolio comparison.
"""

from portfolio.benchmarks.benchmark_config import (
    DEFAULT_BENCHMARKS,
    get_benchmark_ticker,
    get_benchmark_currency,
    list_available_benchmarks,
    get_recommended_benchmarks,
)
from portfolio.benchmarks.currency_converter import (
    convert_usd_to_eur,
    get_exchange_rate,
    fetch_exchange_rate_for_date,
)
from portfolio.benchmarks.benchmark_fetcher import (
    fetch_benchmark_data,
    fetch_benchmark_for_date_range,
    normalize_benchmark_data,
)

__all__ = [
    # Config
    "DEFAULT_BENCHMARKS",
    "get_benchmark_ticker",
    "get_benchmark_currency",
    "list_available_benchmarks",
    "get_recommended_benchmarks",
    # Currency
    "convert_usd_to_eur",
    "get_exchange_rate",
    "fetch_exchange_rate_for_date",
    # Fetching
    "fetch_benchmark_data",
    "fetch_benchmark_for_date_range",
    "normalize_benchmark_data",
]

