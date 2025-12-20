"""
Load portfolio snapshots from CSV files.
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple

from config import DEFAULT_CSV_PATH, DEFAULT_USER_NAME
from portfolio.classification import reset_unclassified_tracking
from portfolio.historical.snapshot_manager import (
    find_snapshot_files,
    get_latest_snapshot_path,
    parse_snapshot_date,
)


def load_snapshot(
    filepath: str,
    snapshot_date: Optional[datetime] = None,
    enrich: bool = True
) -> pd.DataFrame:
    """
    Load a single snapshot from CSV file.
    
    Args:
        filepath: Path to CSV file
        snapshot_date: Date of the snapshot (if None, will try to parse from filename)
        enrich: Whether to add geography/sector classifications
    
    Returns:
        DataFrame with holdings data, including 'snapshot_date' column
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Snapshot file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError(f"Snapshot file is empty: {filepath}")
    
    # Fill NaN values with 0 for calculations
    df = df.fillna(0)
    
    # Parse date from filename if not provided
    if snapshot_date is None:
        filename = os.path.basename(filepath)
        # Try to extract user name from filename or use default
        user_name = DEFAULT_USER_NAME
        snapshot_date = parse_snapshot_date(filename, user_name)
        
        # If still None, use file modification time as fallback
        if snapshot_date is None:
            mtime = os.path.getmtime(filepath)
            snapshot_date = datetime.fromtimestamp(mtime)
    
    # Add snapshot date column
    df['snapshot_date'] = snapshot_date
    
    # Enrich with classifications if requested
    if enrich:
        from portfolio.classification import classify_geography, classify_sector
        reset_unclassified_tracking()
        df['geography'] = df['name'].apply(classify_geography)
        df['sector'] = df['name'].apply(classify_sector)
    
    return df


def load_all_snapshots(
    user_name: Optional[str] = None,
    directory: Optional[str] = None,
    enrich: bool = True
) -> pd.DataFrame:
    """
    Load all historical snapshots for a user.
    
    Args:
        user_name: User name (default: from config)
        directory: Directory to search (default: current directory)
        enrich: Whether to add geography/sector classifications
    
    Returns:
        Combined DataFrame with all snapshots, including 'snapshot_date' column
        Returns empty DataFrame if no snapshots found
    """
    if user_name is None:
        user_name = DEFAULT_USER_NAME
    
    snapshot_files = find_snapshot_files(user_name, directory)
    
    if not snapshot_files:
        return pd.DataFrame()
    
    all_snapshots = []
    
    for date, filepath in snapshot_files:
        try:
            df = load_snapshot(filepath, snapshot_date=date, enrich=enrich)
            all_snapshots.append(df)
        except Exception as e:
            # Log error but continue with other snapshots
            print(f"Warning: Failed to load snapshot {filepath}: {e}")
            continue
    
    if not all_snapshots:
        return pd.DataFrame()
    
    # Combine all snapshots
    combined_df = pd.concat(all_snapshots, ignore_index=True)
    
    return combined_df


def load_latest_snapshot(
    user_name: Optional[str] = None,
    directory: Optional[str] = None,
    enrich: bool = True,
    fallback_to_default: bool = True
) -> pd.DataFrame:
    """
    Load the latest snapshot (latest_assets_{user}.csv).
    
    Falls back to holdings_from_op.csv if latest snapshot not found.
    
    Args:
        user_name: User name (default: from config)
        directory: Directory to search (default: current directory)
        enrich: Whether to add geography/sector classifications
        fallback_to_default: Whether to fall back to DEFAULT_CSV_PATH if latest not found
    
    Returns:
        DataFrame with holdings data
    
    Raises:
        FileNotFoundError: If no snapshot file found and fallback disabled
    """
    if user_name is None:
        user_name = DEFAULT_USER_NAME
    
    if directory is None:
        directory = os.getcwd()
    
    # Try to load latest snapshot
    latest_path = get_latest_snapshot_path(user_name, directory)
    
    if latest_path:
        return load_snapshot(latest_path, enrich=enrich)
    
    # Fallback to default CSV if enabled
    if fallback_to_default:
        default_path = os.path.join(directory, DEFAULT_CSV_PATH)
        if os.path.exists(default_path):
            return load_snapshot(default_path, enrich=enrich)
    
    # No file found
    raise FileNotFoundError(
        f"Latest snapshot not found for user '{user_name}' "
        f"and fallback file '{DEFAULT_CSV_PATH}' not found"
    )


def get_portfolio_value_over_time(
    user_name: Optional[str] = None,
    directory: Optional[str] = None,
    include_latest: bool = True
) -> pd.DataFrame:
    """
    Calculate portfolio total value over time from snapshots.
    
    Args:
        user_name: User name (default: from config)
        directory: Directory to search (default: current directory)
        include_latest: Whether to include latest snapshot (default: True)
    
    Returns:
        DataFrame with columns: snapshot_date, total_value_eur, total_purchase_eur, total_gain_eur, total_gain_pct
    """
    if user_name is None:
        user_name = DEFAULT_USER_NAME
    
    # Load historical snapshots
    all_snapshots = load_all_snapshots(user_name, directory, enrich=False)
    
    # Optionally include latest snapshot
    if include_latest:
        try:
            latest_snapshot = load_latest_snapshot(user_name, directory, enrich=False, fallback_to_default=False)
            if not latest_snapshot.empty:
                # Use current date for latest snapshot if no date is set
                if 'snapshot_date' not in latest_snapshot.columns or latest_snapshot['snapshot_date'].isna().all():
                    from datetime import datetime
                    latest_snapshot['snapshot_date'] = datetime.now()
                
                # Combine with historical snapshots
                if all_snapshots.empty:
                    all_snapshots = latest_snapshot
                else:
                    all_snapshots = pd.concat([all_snapshots, latest_snapshot], ignore_index=True)
        except FileNotFoundError:
            # Latest snapshot not found, continue with historical only
            pass
    
    if all_snapshots.empty:
        return pd.DataFrame()
    
    # Group by snapshot date and calculate totals
    portfolio_timeline = all_snapshots.groupby('snapshot_date').agg({
        'market_total_eur': 'sum',
        'purchase_total_eur': 'sum',
        'change_eur': 'sum',
    }).reset_index()
    
    # Calculate gain percentage
    portfolio_timeline['total_gain_pct'] = (
        portfolio_timeline['change_eur'] / portfolio_timeline['purchase_total_eur'] * 100
    )
    
    # Rename columns for clarity
    portfolio_timeline = portfolio_timeline.rename(columns={
        'market_total_eur': 'total_value_eur',
        'purchase_total_eur': 'total_purchase_eur',
        'change_eur': 'total_gain_eur',
    })
    
    # Sort by date
    portfolio_timeline = portfolio_timeline.sort_values('snapshot_date')
    
    # Set snapshot_date as index for easier date-based operations
    portfolio_timeline = portfolio_timeline.set_index('snapshot_date')
    
    # Ensure index is datetime
    portfolio_timeline.index = pd.to_datetime(portfolio_timeline.index)
    
    return portfolio_timeline

