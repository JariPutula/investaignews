"""
Helper script to create test snapshot files for historical performance testing.

This script creates multiple snapshot files from your latest snapshot with
slightly modified values to simulate portfolio changes over time.
"""

import pandas as pd
import shutil
import os
from datetime import datetime, timedelta
from config import DATA_DIR, DEFAULT_USER_NAME

def create_test_snapshot(source_file: str, target_file: str, value_change_pct: float = 0.0):
    """
    Create a test snapshot by copying source and modifying values.
    
    Args:
        source_file: Source CSV file to copy
        target_file: Target CSV file name
        value_change_pct: Percentage change to apply to market values (default: 0%)
    """
    # Read source file
    df = pd.read_csv(source_file)
    
    # Apply value change
    if value_change_pct != 0:
        # Modify market prices
        df['market_price_eur'] = df['market_price_eur'] * (1 + value_change_pct / 100)
        df['market_total_eur'] = df['quantity'] * df['market_price_eur']
        df['change_eur'] = df['market_total_eur'] - df['purchase_total_eur']
        df['change_pct'] = (df['change_eur'] / df['purchase_total_eur']) * 100
    
    # Save to target file
    df.to_csv(target_file, index=False)
    print(f"✓ Created {target_file} (change: {value_change_pct:+.2f}%)")


def main():
    """Create test snapshots for the last few months."""
    user_name = DEFAULT_USER_NAME
    source_file = os.path.join(DATA_DIR, f"latest_assets_{user_name}.csv")
    
    # Check if source file exists
    try:
        df = pd.read_csv(source_file)
        print(f"✓ Found source file: {source_file}")
        print(f"  Total value: €{df['market_total_eur'].sum():,.2f}")
    except FileNotFoundError:
        print(f"✗ Source file not found: {source_file}")
        print("  Please ensure latest_assets_jari.csv exists")
        return
    
    # Get today's date
    today = datetime.now()
    
    # Create snapshots for the last 3 months (monthly)
    print("\nCreating monthly snapshots...")
    for months_ago in range(3, 0, -1):
        snapshot_date = today - timedelta(days=30 * months_ago)
        date_str = snapshot_date.strftime("%d%m%Y")
        target_file = os.path.join(DATA_DIR, f"{date_str}_assets_{user_name}.csv")
        
        # Apply a small change to simulate portfolio growth
        # -3 months: -2%, -2 months: -1%, -1 month: +1%
        change_pct = -2.0 + (3 - months_ago) * 1.0
        
        create_test_snapshot(source_file, target_file, value_change_pct=change_pct)
    
    # Create a snapshot from 1 week ago
    print("\nCreating weekly snapshot...")
    week_ago = today - timedelta(days=7)
    date_str = week_ago.strftime("%d%m%Y")
    target_file = os.path.join(DATA_DIR, f"{date_str}_assets_{user_name}.csv")
    create_test_snapshot(source_file, target_file, value_change_pct=0.5)
    
    print("\n" + "=" * 60)
    print("Test snapshots created!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  - {os.path.join(DATA_DIR, week_ago.strftime('%d%m%Y') + '_assets_' + user_name + '.csv')} (1 week ago)")
    for months_ago in range(3, 0, -1):
        snapshot_date = today - timedelta(days=30 * months_ago)
        print(f"  - {os.path.join(DATA_DIR, snapshot_date.strftime('%d%m%Y') + '_assets_' + user_name + '.csv')} ({months_ago} months ago)")
    print(f"\n  - {os.path.join(DATA_DIR, 'latest_assets_' + user_name + '.csv')} (current)")
    print("\nYou now have multiple snapshots for testing historical performance!")


if __name__ == "__main__":
    main()

