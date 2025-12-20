"""
Snapshot file management: discovery, parsing, and metadata extraction.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from config import (
    HISTORICAL_SNAPSHOT_PATTERN,
    LATEST_SNAPSHOT_PATTERN,
    SNAPSHOT_DATE_FORMAT,
)


def parse_snapshot_date(filename: str, user_name: str) -> Optional[datetime]:
    """
    Parse date from snapshot filename.
    
    Expected format: {DDMMYYYY}_assets_{user}.csv
    Example: 20112025_assets_jari.csv -> 2025-11-20
    
    Args:
        filename: Name of the snapshot file
        user_name: User name to match in filename
    
    Returns:
        Parsed datetime object, or None if parsing fails
    """
    # Pattern: {DDMMYYYY}_assets_{user}.csv
    pattern = r"(\d{8})_assets_" + re.escape(user_name) + r"\.csv"
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    date_str = match.group(1)  # DDMMYYYY
    
    try:
        # Parse DDMMYYYY format
        date_obj = datetime.strptime(date_str, SNAPSHOT_DATE_FORMAT)
        return date_obj
    except ValueError:
        return None


def find_snapshot_files(
    user_name: str,
    directory: Optional[str] = None
) -> List[Tuple[datetime, str]]:
    """
    Find all historical snapshot files for a given user.
    
    Args:
        user_name: User name (e.g., "jari")
        directory: Directory to search (default: current directory)
    
    Returns:
        List of (date, filepath) tuples, sorted by date (oldest first)
    """
    if directory is None:
        directory = os.getcwd()
    
    snapshot_files = []
    
    # Search for files matching the pattern: {DDMMYYYY}_assets_{user}.csv
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
        
        # Try to parse date from filename
        date_obj = parse_snapshot_date(filename, user_name)
        
        if date_obj is not None:
            filepath = os.path.join(directory, filename)
            snapshot_files.append((date_obj, filepath))
    
    # Sort by date (oldest first)
    snapshot_files.sort(key=lambda x: x[0])
    
    return snapshot_files


def get_latest_snapshot_path(user_name: str, directory: Optional[str] = None) -> Optional[str]:
    """
    Get path to the latest snapshot file.
    
    Args:
        user_name: User name (e.g., "jari")
        directory: Directory to search (default: current directory)
    
    Returns:
        Path to latest snapshot file, or None if not found
    """
    if directory is None:
        directory = os.getcwd()
    
    # Pattern: latest_assets_{user}.csv
    filename = LATEST_SNAPSHOT_PATTERN.format(user=user_name)
    filepath = os.path.join(directory, filename)
    
    if os.path.exists(filepath):
        return filepath
    
    return None


def get_snapshot_info(user_name: str, directory: Optional[str] = None) -> dict:
    """
    Get information about available snapshots.
    
    Args:
        user_name: User name (e.g., "jari")
        directory: Directory to search (default: current directory)
    
    Returns:
        Dictionary with snapshot information:
        {
            'latest_path': str or None,
            'historical_count': int,
            'historical_dates': List[datetime],
            'date_range': (earliest, latest) or None
        }
    """
    if directory is None:
        directory = os.getcwd()
    
    latest_path = get_latest_snapshot_path(user_name, directory)
    historical_snapshots = find_snapshot_files(user_name, directory)
    
    historical_dates = [date for date, _ in historical_snapshots]
    
    date_range = None
    if historical_dates:
        date_range = (min(historical_dates), max(historical_dates))
    
    return {
        'latest_path': latest_path,
        'historical_count': len(historical_snapshots),
        'historical_dates': historical_dates,
        'date_range': date_range,
        'historical_files': historical_snapshots,
    }

