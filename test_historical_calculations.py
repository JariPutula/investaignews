"""
Test script for historical performance calculations.
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from portfolio.historical.snapshot_loader import get_portfolio_value_over_time
from portfolio.benchmarks.benchmark_fetcher import fetch_benchmark_for_date_range


def test_period_returns():
    """Test period returns calculation."""
    print("=" * 60)
    print("TEST 1: Period Returns Calculation")
    print("=" * 60)
    
    try:
        portfolio_timeline = get_portfolio_value_over_time("jari")
        
        if portfolio_timeline.empty:
            print("✗ No portfolio timeline data available")
            return
        
        print(f"✓ Portfolio timeline: {len(portfolio_timeline)} data points")
        
        # Calculate period returns
        period_returns = calculate_period_returns(portfolio_timeline)
        
        if period_returns:
            print("\nPeriod Returns:")
            for period, return_pct in period_returns.items():
                if return_pct is not None:
                    print(f"  {period:10s}: {return_pct:7.2f}%")
                else:
                    print(f"  {period:10s}: N/A (insufficient data)")
        else:
            print("✗ No period returns calculated")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_benchmark_comparison():
    """Test portfolio vs benchmark comparison."""
    print("=" * 60)
    print("TEST 2: Portfolio vs Benchmark Comparison")
    print("=" * 60)
    
    try:
        # Get portfolio timeline
        portfolio_timeline = get_portfolio_value_over_time("jari")
        
        if portfolio_timeline.empty:
            print("✗ No portfolio timeline data available")
            return
        
        # Get dates
        dates = portfolio_timeline.index.tolist()
        
        # Fetch benchmark
        benchmark_name = "S&P 500"
        print(f"Fetching {benchmark_name} data...")
        
        benchmark_data = fetch_benchmark_for_date_range(
            benchmark_name,
            dates,
            convert_to_eur=True
        )
        
        if benchmark_data.empty:
            print(f"✗ Failed to fetch {benchmark_name} data")
            return
        
        print(f"✓ Fetched benchmark data: {len(benchmark_data)} data points")
        
        # Compare
        comparison = compare_portfolio_to_benchmark(
            portfolio_timeline,
            benchmark_data,
            benchmark_name
        )
        
        if comparison:
            print(f"\nComparison Metrics ({benchmark_name}):")
            print(f"  Beta: {comparison.get('beta', 'N/A'):.3f}")
            print(f"  Alpha: {comparison.get('alpha', 'N/A'):.2f}%")
            print(f"  Tracking Error: {comparison.get('tracking_error', 'N/A'):.2f}%")
            print(f"  Information Ratio: {comparison.get('information_ratio', 'N/A'):.3f}")
            print(f"  Correlation: {comparison.get('correlation', 'N/A'):.3f}")
            
            if 'volatility' in comparison:
                print(f"\n  Volatility (Annualized):")
                print(f"    Portfolio: {comparison['volatility']['portfolio']:.2f}%")
                print(f"    Benchmark: {comparison['volatility']['benchmark']:.2f}%")
            
            if 'max_drawdown' in comparison:
                port_dd = comparison['max_drawdown']['portfolio']
                bench_dd = comparison['max_drawdown']['benchmark']
                print(f"\n  Maximum Drawdown:")
                if not pd.isna(port_dd[0]):
                    print(f"    Portfolio: {port_dd[0]:.2f}%")
                if not pd.isna(bench_dd[0]):
                    print(f"    Benchmark: {bench_dd[0]:.2f}%")
        else:
            print("✗ Comparison calculation failed")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_risk_metrics():
    """Test risk metrics calculations."""
    print("=" * 60)
    print("TEST 3: Risk Metrics Over Time")
    print("=" * 60)
    
    try:
        portfolio_timeline = get_portfolio_value_over_time("jari")
        
        if portfolio_timeline.empty:
            print("✗ No portfolio timeline data available")
            return
        
        # Calculate returns
        portfolio_values = portfolio_timeline['total_value_eur']
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        if len(portfolio_returns) < 12:
            print(f"✗ Insufficient data for rolling metrics (need at least 12 data points, have {len(portfolio_returns)})")
            return
        
        # Calculate rolling volatility
        rolling_vol = calculate_rolling_volatility(portfolio_returns, window=12)
        
        if not rolling_vol.empty:
            print(f"✓ Rolling Volatility (12-month window):")
            print(f"  Latest: {rolling_vol.iloc[-1]:.2f}%")
            print(f"  Average: {rolling_vol.mean():.2f}%")
            print(f"  Min: {rolling_vol.min():.2f}%")
            print(f"  Max: {rolling_vol.max():.2f}%")
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = calculate_rolling_sharpe_ratio(portfolio_returns, window=12)
        
        if not rolling_sharpe.empty:
            print(f"\n✓ Rolling Sharpe Ratio (12-month window):")
            print(f"  Latest: {rolling_sharpe.iloc[-1]:.3f}")
            print(f"  Average: {rolling_sharpe.mean():.3f}")
        
        # Calculate maximum drawdown
        max_dd, peak_date, trough_date = calculate_maximum_drawdown(portfolio_values)
        
        if not pd.isna(max_dd):
            print(f"\n✓ Maximum Drawdown:")
            print(f"  Drawdown: {max_dd:.2f}%")
            if peak_date:
                print(f"  Peak Date: {peak_date.strftime('%Y-%m-%d')}")
            if trough_date:
                print(f"  Trough Date: {trough_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_comprehensive_summary():
    """Test comprehensive performance summary."""
    print("=" * 60)
    print("TEST 4: Comprehensive Performance Summary")
    print("=" * 60)
    
    try:
        summary = get_historical_performance_summary(
            "jari",
            benchmark_names=["S&P 500", "EURO STOXX 50"]
        )
        
        if not summary:
            print("✗ Failed to generate summary")
            return
        
        print("✓ Summary generated")
        
        # Portfolio timeline
        portfolio_timeline = summary.get('portfolio_timeline', pd.DataFrame())
        if not portfolio_timeline.empty:
            print(f"\nPortfolio Timeline: {len(portfolio_timeline)} data points")
            print(f"  Latest value: €{portfolio_timeline['total_value_eur'].iloc[-1]:,.2f}")
        
        # Benchmarks
        benchmarks = summary.get('benchmarks', {})
        print(f"\nBenchmarks: {len(benchmarks)}")
        for name, data in benchmarks.items():
            if not data.empty:
                print(f"  {name}: {len(data)} data points")
        
        # Period returns
        period_returns = summary.get('period_returns', {})
        if period_returns:
            print("\nPeriod Returns:")
            for period, return_pct in period_returns.items():
                if return_pct is not None:
                    print(f"  {period:10s}: {return_pct:7.2f}%")
        
        # Comparisons
        comparisons = summary.get('comparisons', {})
        if comparisons:
            print("\nBenchmark Comparisons:")
            for benchmark_name, comp in comparisons.items():
                print(f"\n  {benchmark_name}:")
                print(f"    Beta: {comp.get('beta', 'N/A'):.3f}")
                print(f"    Alpha: {comp.get('alpha', 'N/A'):.2f}%")
                print(f"    Correlation: {comp.get('correlation', 'N/A'):.3f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_period_comparison_table():
    """Test period comparison table generation."""
    print("=" * 60)
    print("TEST 5: Period Comparison Table")
    print("=" * 60)
    
    try:
        summary = get_historical_performance_summary(
            "jari",
            benchmark_names=["S&P 500"]
        )
        
        comparisons = summary.get('comparisons', {})
        period_returns = summary.get('period_returns', {})
        
        if not comparisons or not period_returns:
            print("✗ Insufficient data for comparison table")
            return
        
        # Get first benchmark comparison
        benchmark_name = list(comparisons.keys())[0]
        comparison = comparisons[benchmark_name]
        
        portfolio_periods = comparison.get('period_returns', {}).get('portfolio', {})
        benchmark_periods = comparison.get('period_returns', {}).get('benchmark', {})
        
        print(f"\nPeriod Returns Comparison (vs {benchmark_name}):")
        print(f"{'Period':<12} {'Portfolio':<12} {'Benchmark':<12} {'Difference':<12}")
        print("-" * 50)
        
        for period in ["1M", "3M", "6M", "1Y", "YTD", "All-time"]:
            port_return = portfolio_periods.get(period)
            bench_return = benchmark_periods.get(period)
            
            if port_return is not None and bench_return is not None:
                diff = port_return - bench_return
                print(f"{period:<12} {port_return:>10.2f}% {bench_return:>10.2f}% {diff:>10.2f}%")
            elif port_return is not None:
                print(f"{period:<12} {port_return:>10.2f}% {'N/A':<12} {'N/A':<12}")
            elif bench_return is not None:
                print(f"{period:<12} {'N/A':<12} {bench_return:>10.2f}% {'N/A':<12}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    print("\n" + "=" * 60)
    print("HISTORICAL PERFORMANCE CALCULATIONS TESTS")
    print("=" * 60 + "\n")
    
    test_period_returns()
    test_benchmark_comparison()
    test_risk_metrics()
    test_comprehensive_summary()
    test_period_comparison_table()
    
    print("=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)

