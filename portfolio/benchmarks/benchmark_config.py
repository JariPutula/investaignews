"""
Benchmark index definitions and configuration.
"""

# Default benchmark indices (Yahoo Finance tickers)
DEFAULT_BENCHMARKS = {
    "S&P 500": "^GSPC",
    "EURO STOXX 50": "^STOXX50E",
    "MSCI World": "URTH",  # ETF proxy for MSCI World
    "NASDAQ Composite": "^IXIC",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
}

# Additional benchmarks (can be added to selection)
ADDITIONAL_BENCHMARKS = {
    "Dow Jones": "^DJI",
    "CAC 40": "^FCHI",
    "OMX Helsinki 25": "^OMXH25",
    "MSCI Emerging Markets": "EEM",  # ETF proxy
    "Russell 2000": "^RUT",
}

# Currency information for benchmarks
BENCHMARK_CURRENCIES = {
    "^GSPC": "USD",  # S&P 500
    "^IXIC": "USD",  # NASDAQ
    "^DJI": "USD",  # Dow Jones
    "^GDAXI": "EUR",  # DAX
    "^FCHI": "EUR",  # CAC 40
    "^STOXX50E": "EUR",  # EURO STOXX 50
    "^FTSE": "GBP",  # FTSE 100
    "^N225": "JPY",  # Nikkei 225
    "^OMXH25": "EUR",  # OMX Helsinki
    "URTH": "USD",  # MSCI World ETF
    "EEM": "USD",  # MSCI EM ETF
    "^RUT": "USD",  # Russell 2000
}

# Recommended default benchmarks for different portfolio types
RECOMMENDED_BENCHMARKS = {
    "global": ["S&P 500", "EURO STOXX 50", "MSCI World"],
    "us_focused": ["S&P 500", "NASDAQ Composite", "Dow Jones"],
    "european": ["EURO STOXX 50", "DAX", "FTSE 100"],
    "finnish": ["OMX Helsinki 25", "EURO STOXX 50", "MSCI World"],
}


def get_benchmark_ticker(benchmark_name: str) -> str:
    """
    Get Yahoo Finance ticker for a benchmark name.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "S&P 500")
    
    Returns:
        Yahoo Finance ticker symbol
    
    Raises:
        KeyError: If benchmark name not found
    """
    if benchmark_name in DEFAULT_BENCHMARKS:
        return DEFAULT_BENCHMARKS[benchmark_name]
    elif benchmark_name in ADDITIONAL_BENCHMARKS:
        return ADDITIONAL_BENCHMARKS[benchmark_name]
    else:
        raise KeyError(f"Benchmark '{benchmark_name}' not found")


def get_benchmark_currency(ticker: str) -> str:
    """
    Get currency for a benchmark ticker.
    
    Args:
        ticker: Yahoo Finance ticker symbol
    
    Returns:
        Currency code (e.g., "USD", "EUR")
        Defaults to "USD" if not found
    """
    return BENCHMARK_CURRENCIES.get(ticker, "USD")


def list_available_benchmarks() -> dict:
    """
    List all available benchmarks.
    
    Returns:
        Dictionary with 'default' and 'additional' benchmark lists
    """
    return {
        "default": list(DEFAULT_BENCHMARKS.keys()),
        "additional": list(ADDITIONAL_BENCHMARKS.keys()),
        "all": list(DEFAULT_BENCHMARKS.keys()) + list(ADDITIONAL_BENCHMARKS.keys()),
    }


def get_recommended_benchmarks(portfolio_type: str = "global") -> list:
    """
    Get recommended benchmarks for a portfolio type.
    
    Args:
        portfolio_type: Type of portfolio ("global", "us_focused", "european", "finnish")
    
    Returns:
        List of recommended benchmark names
    """
    return RECOMMENDED_BENCHMARKS.get(portfolio_type, RECOMMENDED_BENCHMARKS["global"])

