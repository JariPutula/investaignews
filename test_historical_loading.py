"""
Test script for historical snapshot loading functionality.
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio.historical.snapshot_manager import (
    find_snapshot_files,
    parse_snapshot_date,
    get_snapshot_info,
    get_latest_snapshot_path,
)
from portfolio.historical.snapshot_loader import (
    load_snapshot,
    load_all_snapshots,
    load_latest_snapshot,
    get_portfolio_value_over_time,
)
from portfolio.data_loader import load_data


def test_date_parsing():
    """Test date parsing from filenames."""
    print("=" * 60)
    print("TEST 1: Date Parsing")
    print("=" * 60)
    
    test_cases = [
        ("20112025_assets_jari.csv", "jari", datetime(2025, 11, 20)),
        ("01012024_assets_jari.csv", "jari", datetime(2024, 1, 1)),
        ("31122023_assets_jari.csv", "jari", datetime(2023, 12, 31)),
        ("invalid_file.csv", "jari", None),
        ("20112025_assets_wronguser.csv", "jari", None),
    ]
    
    for filename, user_name, expected_date in test_cases:
        result = parse_snapshot_date(filename, user_name)
        status = "✓" if result == expected_date else "✗"
        print(f"{status} {filename} -> {result} (expected: {expected_date})")
    
    print()


def test_snapshot_discovery():
    """Test snapshot file discovery."""
    print("=" * 60)
    print("TEST 2: Snapshot File Discovery")
    print("=" * 60)
    
    snapshots = find_snapshot_files("jari")
    print(f"Found {len(snapshots)} historical snapshots:")
    for date, filepath in snapshots:
        print(f"  - {date.strftime('%Y-%m-%d')}: {os.path.basename(filepath)}")
    
    print()


def test_latest_snapshot_path():
    """Test latest snapshot path detection."""
    print("=" * 60)
    print("TEST 3: Latest Snapshot Path")
    print("=" * 60)
    
    latest_path = get_latest_snapshot_path("jari")
    if latest_path:
        print(f"✓ Latest snapshot found: {os.path.basename(latest_path)}")
    else:
        print("✗ Latest snapshot not found")
    
    print()


def test_snapshot_info():
    """Test snapshot info gathering."""
    print("=" * 60)
    print("TEST 4: Snapshot Info")
    print("=" * 60)
    
    info = get_snapshot_info("jari")
    print(f"Latest path: {info['latest_path']}")
    print(f"Historical count: {info['historical_count']}")
    print(f"Date range: {info['date_range']}")
    print()


def test_load_single_snapshot():
    """Test loading a single snapshot."""
    print("=" * 60)
    print("TEST 5: Load Single Snapshot")
    print("=" * 60)
    
    try:
        # Try to load latest snapshot
        latest_path = get_latest_snapshot_path("jari")
        if latest_path:
            df = load_snapshot(latest_path, enrich=False)
            print(f"✓ Loaded latest snapshot: {len(df)} holdings")
            print(f"  Total value: €{df['market_total_eur'].sum():,.2f}")
            print(f"  Columns: {list(df.columns)}")
            if 'snapshot_date' in df.columns:
                print(f"  Snapshot date: {df['snapshot_date'].iloc[0]}")
        else:
            print("✗ Latest snapshot not found")
    except Exception as e:
        print(f"✗ Error loading snapshot: {e}")
    
    print()


def test_load_all_snapshots():
    """Test loading all historical snapshots."""
    print("=" * 60)
    print("TEST 6: Load All Snapshots")
    print("=" * 60)
    
    try:
        all_snapshots = load_all_snapshots("jari", enrich=False)
        if not all_snapshots.empty:
            print(f"✓ Loaded {len(all_snapshots)} total records from all snapshots")
            unique_dates = all_snapshots['snapshot_date'].unique()
            print(f"  Unique snapshot dates: {len(unique_dates)}")
            for date in sorted(unique_dates):
                count = len(all_snapshots[all_snapshots['snapshot_date'] == date])
                total_value = all_snapshots[
                    all_snapshots['snapshot_date'] == date
                ]['market_total_eur'].sum()
                print(f"    {date.strftime('%Y-%m-%d')}: {count} holdings, €{total_value:,.2f}")
        else:
            print("✗ No snapshots found")
    except Exception as e:
        print(f"✗ Error loading snapshots: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_load_latest_snapshot():
    """Test loading latest snapshot with fallback."""
    print("=" * 60)
    print("TEST 7: Load Latest Snapshot (with fallback)")
    print("=" * 60)
    
    try:
        df = load_latest_snapshot("jari", enrich=False)
        print(f"✓ Loaded latest snapshot: {len(df)} holdings")
        print(f"  Total value: €{df['market_total_eur'].sum():,.2f}")
        if 'snapshot_date' in df.columns:
            print(f"  Snapshot date: {df['snapshot_date'].iloc[0]}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_portfolio_value_over_time():
    """Test portfolio value calculation over time."""
    print("=" * 60)
    print("TEST 8: Portfolio Value Over Time")
    print("=" * 60)
    
    try:
        timeline = get_portfolio_value_over_time("jari")
        if not timeline.empty:
            print(f"✓ Calculated portfolio timeline: {len(timeline)} data points")
            print("\nPortfolio Value Timeline:")
            print(timeline.to_string(index=False))
        else:
            print("✗ No timeline data available")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_backward_compatibility():
    """Test backward compatibility with existing load_data function."""
    print("=" * 60)
    print("TEST 9: Backward Compatibility (load_data)")
    print("=" * 60)
    
    try:
        # Test loading with new system (should use latest snapshot)
        df = load_data(user_name="jari")
        print(f"✓ load_data() works: {len(df)} holdings")
        print(f"  Total value: €{df['market_total_eur'].sum():,.2f}")
        
        # Test loading with explicit path (old behavior)
        if os.path.exists("holdings_from_op.csv"):
            df_old = load_data(csv_path="holdings_from_op.csv")
            print(f"✓ load_data(csv_path=...) works: {len(df_old)} holdings")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_compare_snapshots():
    """Test comparing two snapshots to see differences."""
    print("=" * 60)
    print("TEST 10: Compare Snapshots")
    print("=" * 60)
    
    try:
        # Load latest and historical snapshot
        latest = load_latest_snapshot("jari", enrich=False)
        snapshots = find_snapshot_files("jari")
        
        if snapshots:
            # Load first historical snapshot
            hist_date, hist_path = snapshots[0]
            historical = load_snapshot(hist_path, snapshot_date=hist_date, enrich=False)
            
            print(f"Comparing:")
            print(f"  Latest: {latest['market_total_eur'].sum():,.2f} EUR")
            print(f"  Historical ({hist_date.strftime('%Y-%m-%d')}): {historical['market_total_eur'].sum():,.2f} EUR")
            
            # Find differences in holdings
            latest_holdings = set(latest['name'].values)
            hist_holdings = set(historical['name'].values)
            
            added = latest_holdings - hist_holdings
            removed = hist_holdings - latest_holdings
            
            if added:
                print(f"\n  Added holdings: {', '.join(added)}")
            if removed:
                print(f"  Removed holdings: {', '.join(removed)}")
            
            # Compare quantities for common holdings
            common = latest_holdings & hist_holdings
            print(f"\n  Common holdings: {len(common)}")
            
            # Show quantity changes for a few holdings
            print("\n  Quantity changes (sample):")
            for name in list(common)[:5]:
                latest_qty = latest[latest['name'] == name]['quantity'].iloc[0]
                hist_qty = historical[historical['name'] == name]['quantity'].iloc[0]
                if latest_qty != hist_qty:
                    print(f"    {name}: {hist_qty} -> {latest_qty}")
        else:
            print("✗ No historical snapshots to compare")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HISTORICAL SNAPSHOT LOADING TESTS")
    print("=" * 60 + "\n")
    
    test_date_parsing()
    test_snapshot_discovery()
    test_latest_snapshot_path()
    test_snapshot_info()
    test_load_single_snapshot()
    test_load_all_snapshots()
    test_load_latest_snapshot()
    test_portfolio_value_over_time()
    test_backward_compatibility()
    test_compare_snapshots()
    
    print("=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)

