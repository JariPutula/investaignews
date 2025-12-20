"""
Test script for benchmark fetching and currency conversion functionality.
"""

import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio.benchmarks.benchmark_config import (
    DEFAULT_BENCHMARKS,
    get_benchmark_ticker,
    get_benchmark_currency,
    list_available_benchmarks,
    get_recommended_benchmarks,
)
from portfolio.benchmarks.currency_converter import (
    get_exchange_rate,
    convert_usd_to_eur,
    convert_benchmark_to_base_currency,
)
from portfolio.benchmarks.benchmark_fetcher import (
    fetch_benchmark_data,
    fetch_benchmark_for_date_range,
    normalize_benchmark_data,
    fetch_multiple_benchmarks,
)
from portfolio.historical.snapshot_manager import find_snapshot_files


def test_benchmark_config():
    """Test benchmark configuration."""
    print("=" * 60)
    print("TEST 1: Benchmark Configuration")
    print("=" * 60)
    
    print(f"Default benchmarks: {list(DEFAULT_BENCHMARKS.keys())}")
    print(f"\nAvailable benchmarks:")
    available = list_available_benchmarks()
    print(f"  Default: {available['default']}")
    print(f"  Additional: {available['additional']}")
    
    # Test getting tickers
    print(f"\nTicker lookups:")
    for name in ["S&P 500", "EURO STOXX 50", "MSCI World"]:
        try:
            ticker = get_benchmark_ticker(name)
            currency = get_benchmark_currency(ticker)
            print(f"  {name}: {ticker} ({currency})")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print(f"\nRecommended benchmarks:")
    for portfolio_type in ["global", "us_focused", "european", "finnish"]:
        recommended = get_recommended_benchmarks(portfolio_type)
        print(f"  {portfolio_type}: {recommended}")
    
    print()


def test_exchange_rate():
    """Test exchange rate fetching."""
    print("=" * 60)
    print("TEST 2: Exchange Rate Fetching")
    print("=" * 60)
    
    try:
        # Current rate
        current_rate = get_exchange_rate()
        if current_rate:
            print(f"✓ Current EUR/USD rate: {current_rate:.4f}")
        else:
            print("✗ Failed to fetch current exchange rate")
        
        # Historical rate (30 days ago)
        historical_date = datetime.now() - timedelta(days=30)
        hist_rate = get_exchange_rate(date=historical_date)
        if hist_rate:
            print(f"✓ Historical EUR/USD rate ({historical_date.strftime('%Y-%m-%d')}): {hist_rate:.4f}")
        else:
            print(f"✗ Failed to fetch historical exchange rate for {historical_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_currency_conversion():
    """Test currency conversion."""
    print("=" * 60)
    print("TEST 3: Currency Conversion")
    print("=" * 60)
    
    try:
        # Test USD to EUR conversion
        usd_value = 1000.0
        eur_value = convert_usd_to_eur(usd_value)
        
        if eur_value:
            print(f"✓ ${usd_value:,.2f} USD = €{eur_value:,.2f} EUR")
        else:
            print("✗ Currency conversion failed")
        
        # Test with explicit exchange rate
        test_rate = 1.10
        eur_value_explicit = convert_usd_to_eur(usd_value, exchange_rate=test_rate)
        print(f"✓ ${usd_value:,.2f} USD @ {test_rate} = €{eur_value_explicit:,.2f} EUR")
        
        # Test benchmark currency conversion
        ticker = "^GSPC"  # S&P 500 (USD)
        benchmark_value = 5000.0
        converted = convert_benchmark_to_base_currency(benchmark_value, ticker)
        if converted:
            print(f"✓ S&P 500 ${benchmark_value:,.2f} = €{converted:,.2f} EUR")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_fetch_benchmark_data():
    """Test fetching benchmark data."""
    print("=" * 60)
    print("TEST 4: Fetch Benchmark Data")
    print("=" * 60)
    
    try:
        # Fetch S&P 500 data
        benchmark_name = "S&P 500"
        print(f"Fetching {benchmark_name} data...")
        
        data = fetch_benchmark_data(benchmark_name, period="1mo")
        
        if data is not None and not data.empty:
            print(f"✓ Fetched {len(data)} data points")
            print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"  Latest price: ${data['price'].iloc[-1]:,.2f}")
            print(f"  Columns: {list(data.columns)}")
        else:
            print(f"✗ Failed to fetch {benchmark_name} data")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_fetch_benchmark_for_dates():
    """Test fetching benchmark for specific dates."""
    print("=" * 60)
    print("TEST 5: Fetch Benchmark for Snapshot Dates")
    print("=" * 60)
    
    try:
        # Get snapshot dates
        snapshots = find_snapshot_files("jari")
        if not snapshots:
            print("✗ No snapshots found to test with")
            return
        
        dates = [date for date, _ in snapshots]
        print(f"Testing with {len(dates)} snapshot dates:")
        for date in dates:
            print(f"  - {date.strftime('%Y-%m-%d')}")
        
        # Fetch S&P 500 for these dates
        benchmark_name = "S&P 500"
        print(f"\nFetching {benchmark_name} for snapshot dates...")
        
        benchmark_data = fetch_benchmark_for_date_range(
            benchmark_name,
            dates,
            convert_to_eur=True
        )
        
        if not benchmark_data.empty:
            print(f"✓ Fetched data for {len(benchmark_data)} dates")
            print("\nBenchmark Data:")
            print(benchmark_data.to_string())
        else:
            print(f"✗ Failed to fetch {benchmark_name} data for snapshot dates")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_normalize_benchmark():
    """Test benchmark normalization."""
    print("=" * 60)
    print("TEST 6: Benchmark Normalization")
    print("=" * 60)
    
    try:
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        prices = [100 + i * 0.5 for i in range(len(dates))]
        
        sample_data = pd.DataFrame({
            'price_eur': prices
        }, index=dates)
        
        normalized = normalize_benchmark_data(sample_data)
        
        if 'normalized' in normalized.columns:
            print("✓ Normalization successful")
            print(f"  First value: {normalized['normalized'].iloc[0]:.2f} (should be 100.00)")
            print(f"  Last value: {normalized['normalized'].iloc[-1]:.2f}")
        else:
            print("✗ Normalization failed")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_fetch_multiple_benchmarks():
    """Test fetching multiple benchmarks."""
    print("=" * 60)
    print("TEST 7: Fetch Multiple Benchmarks")
    print("=" * 60)
    
    try:
        # Get snapshot dates
        snapshots = find_snapshot_files("jari")
        if not snapshots:
            print("✗ No snapshots found to test with")
            return
        
        dates = [date for date, _ in snapshots]
        
        # Fetch multiple benchmarks
        benchmark_names = ["S&P 500", "EURO STOXX 50"]
        print(f"Fetching {len(benchmark_names)} benchmarks for {len(dates)} dates...")
        
        results = fetch_multiple_benchmarks(
            benchmark_names,
            dates,
            convert_to_eur=True
        )
        
        if results:
            print(f"✓ Successfully fetched {len(results)} benchmarks")
            for name, data in results.items():
                print(f"\n  {name}:")
                print(f"    Data points: {len(data)}")
                if not data.empty:
                    print(f"    Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    if 'price_eur' in data.columns:
                        print(f"    Latest price (EUR): €{data['price_eur'].iloc[-1]:,.2f}")
        else:
            print("✗ Failed to fetch benchmarks")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_integration_with_snapshots():
    """Test integration with snapshot system."""
    print("=" * 60)
    print("TEST 8: Integration with Snapshots")
    print("=" * 60)
    
    try:
        from portfolio.historical.snapshot_loader import (
            load_all_snapshots,
            get_portfolio_value_over_time,
        )
        
        # Load portfolio timeline
        portfolio_timeline = get_portfolio_value_over_time("jari")
        
        if portfolio_timeline.empty:
            print("✗ No portfolio timeline data available")
            return
        
        print(f"✓ Portfolio timeline: {len(portfolio_timeline)} data points")
        
        # Get dates
        dates = portfolio_timeline['snapshot_date'].tolist()
        
        # Fetch benchmark for same dates
        benchmark_name = "S&P 500"
        benchmark_data = fetch_benchmark_for_date_range(
            benchmark_name,
            dates,
            convert_to_eur=True
        )
        
        if not benchmark_data.empty:
            print(f"✓ Benchmark data fetched for {len(benchmark_data)} dates")
            
            # Normalize both for comparison
            portfolio_normalized = portfolio_timeline.copy()
            portfolio_normalized['normalized'] = (
                portfolio_normalized['total_value_eur'] / 
                portfolio_timeline['total_value_eur'].iloc[0] * 100
            )
            
            benchmark_normalized = normalize_benchmark_data(benchmark_data)
            
            print("\nComparison (normalized to 100 at start):")
            print("Date       | Portfolio | Benchmark")
            print("-" * 40)
            for date in dates[:5]:  # Show first 5
                if date in portfolio_normalized.index and date in benchmark_normalized.index:
                    port_val = portfolio_normalized.loc[date, 'normalized']
                    bench_val = benchmark_normalized.loc[date, 'normalized']
                    print(f"{date.strftime('%Y-%m-%d')} | {port_val:8.2f} | {bench_val:8.2f}")
        else:
            print("✗ Failed to fetch benchmark data")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


if __name__ == "__main__":
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("BENCHMARK FUNCTIONALITY TESTS")
    print("=" * 60 + "\n")
    
    test_benchmark_config()
    test_exchange_rate()
    test_currency_conversion()
    test_fetch_benchmark_data()
    test_fetch_benchmark_for_dates()
    test_normalize_benchmark()
    test_fetch_multiple_benchmarks()
    test_integration_with_snapshots()
    
    print("=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)

