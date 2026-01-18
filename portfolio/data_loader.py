"""
Data loading and enrichment for portfolio holdings.
"""

import os
import pandas as pd
from typing import Optional

import streamlit as st

from config import DEFAULT_CSV_PATH, DEFAULT_USER_NAME, DATA_DIR
from portfolio.classification import (
    classify_geography,
    classify_sector,
    reset_unclassified_tracking,
)


@st.cache_data
def load_data(csv_path: Optional[str] = None, user_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load holdings data from CSV file.
    
    This function maintains backward compatibility:
    - If csv_path is provided, loads that file (old behavior)
    - If csv_path is None, tries to load latest_assets_{user}.csv
    - Falls back to DEFAULT_CSV_PATH if latest snapshot not found
    
    Args:
        csv_path: Path to CSV file. If None, tries latest snapshot then falls back to DEFAULT_CSV_PATH.
        user_name: User name for snapshot loading (default: from config)
    
    Returns:
        DataFrame with holdings data
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV file is empty or invalid
    """
    if csv_path is None:
        # Try to load latest snapshot first
        if user_name is None:
            user_name = DEFAULT_USER_NAME
        
        try:
            from portfolio.historical.snapshot_loader import load_latest_snapshot
            # Load latest snapshot (will fallback to DEFAULT_CSV_PATH if not found)
            df = load_latest_snapshot(user_name=user_name, enrich=False)
            return df
        except (FileNotFoundError, ImportError):
            # Fallback to default CSV path
            csv_path = DEFAULT_CSV_PATH
    
    # Load from specified path (backward compatibility)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")
    
    # Fill NaN values with 0 for calculations
    df = df.fillna(0)
    
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich DataFrame with geography and sector classifications.
    
    This function:
    1. Resets unclassified tracking
    2. Adds 'geography' column
    3. Adds 'sector' column
    
    Args:
        df: DataFrame with holdings data (must have 'name' column)
    
    Returns:
        Enriched DataFrame with 'geography' and 'sector' columns
    """
    # Reset tracking before classification
    reset_unclassified_tracking()
    
    # Create a copy to avoid modifying original
    df_enriched = df.copy()
    
    # Add geography and sector classification
    df_enriched['geography'] = df_enriched['name'].apply(classify_geography)
    df_enriched['sector'] = df_enriched['name'].apply(classify_sector)
    
    return df_enriched


def validate_data(df: pd.DataFrame) -> tuple[bool, Optional[str]]:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, error_message)
        If valid, returns (True, None)
        If invalid, returns (False, error_message)
    """
    required_columns = [
        'name',
        'market_total_eur',
        'purchase_total_eur',
        'change_eur',
        'change_pct'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    return True, None

