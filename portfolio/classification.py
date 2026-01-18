"""
Classification logic for portfolio holdings by geography and sector.
"""

import pandas as pd
from typing import List

from config import GEOGRAPHY_KEYWORDS, SECTOR_KEYWORDS


# Track unclassified items (module-level state)
unclassified_geography: List[str] = []
unclassified_sector: List[str] = []


def reset_unclassified_tracking():
    """Reset the unclassified tracking lists."""
    global unclassified_geography, unclassified_sector
    unclassified_geography.clear()
    unclassified_sector.clear()


def get_unclassified_geography() -> List[str]:
    """Get list of unclassified geography items."""
    return unclassified_geography.copy()


def get_unclassified_sector() -> List[str]:
    """Get list of unclassified sector items."""
    return unclassified_sector.copy()


def classify_geography(name) -> str:
    """
    Classify holdings by geography with improved pattern matching.
    
    Args:
        name: Holding name to classify
    
    Returns:
        Geography classification string ('U.S.', 'Finland', 'Europe/Global', 'Other')
    """
    if pd.isna(name):
        return 'Other'
    
    name_upper = str(name).upper()
    
    # Check if it's an ETF or index fund
    is_etf = any(keyword in name_upper for keyword in ['ETF', 'INDEKSI', 'INDEX', 'FUND'])
    
    # Check Finland holdings first (most specific)
    if any(keyword in name_upper for keyword in GEOGRAPHY_KEYWORDS['Finland']['keywords']):
        return 'Finland'
    if any(suffix in name_upper for suffix in GEOGRAPHY_KEYWORDS['Finland']['company_suffixes']):
        return 'Finland'
    
    # Check Europe/Global keywords (before US, to avoid conflicts)
    if any(keyword in name_upper for keyword in GEOGRAPHY_KEYWORDS['Europe/Global']['keywords']):
        return 'Europe/Global'
    
    # Check European company suffixes (before US, to avoid conflicts)
    if any(suffix in name_upper for suffix in GEOGRAPHY_KEYWORDS['Europe/Global']['company_suffixes']):
        return 'Europe/Global'
    
    # Check US holdings
    if any(keyword in name_upper for keyword in GEOGRAPHY_KEYWORDS['U.S.']['keywords']):
        return 'U.S.'
    if is_etf and any(pattern in name_upper for pattern in GEOGRAPHY_KEYWORDS['U.S.']['etf_patterns']):
        return 'U.S.'
    # US company suffixes (INC, CORP, CO, COMPANY) are strong indicators of US companies
    if any(suffix in name_upper for suffix in GEOGRAPHY_KEYWORDS['U.S.']['company_suffixes']):
        return 'U.S.'
    
    # Check Europe/Global ETFs
    if is_etf:
        if any(pattern in name_upper for pattern in GEOGRAPHY_KEYWORDS['Europe/Global']['etf_patterns']):
            return 'Europe/Global'
        # Default ETFs to Europe/Global if not clearly US
        if not any(pattern in name_upper for pattern in GEOGRAPHY_KEYWORDS['U.S.']['etf_patterns']):
            return 'Europe/Global'
    
    # If unclassified, track it
    if name not in unclassified_geography:
        unclassified_geography.append(name)
    
    return 'Other'


def classify_sector(name) -> str:
    """
    Classify holdings by sector with improved pattern matching.
    
    Args:
        name: Holding name to classify
    
    Returns:
        Sector classification string
    """
    if pd.isna(name):
        return 'Other'
    
    name_upper = str(name).upper()
    is_etf = any(keyword in name_upper for keyword in ['ETF', 'INDEKSI', 'INDEX', 'FUND'])
    
    # Check Broad Market ETFs first (before thematic)
    if is_etf:
        for pattern in SECTOR_KEYWORDS['Broad Market ETF']['etf_patterns']:
            if pattern in name_upper:
                return 'Broad Market ETF'
    
    # Check Fixed Income ETFs (bonds) before thematic
    if is_etf:
        for pattern in SECTOR_KEYWORDS['Fixed Income']['etf_patterns']:
            if pattern in name_upper:
                return 'Fixed Income'
    
    # Check all sectors
    for sector, config in SECTOR_KEYWORDS.items():
        if sector in ['Broad Market ETF', 'Fixed Income', 'Thematic ETF']:
            continue  # Already checked or will check later
        
        # Check keywords
        if any(keyword in name_upper for keyword in config['keywords']):
            return sector
        
        # Check ETF patterns
        if is_etf and 'etf_patterns' in config:
            if any(pattern in name_upper for pattern in config['etf_patterns']):
                return sector
    
    # Check Thematic ETFs
    if is_etf:
        for pattern in SECTOR_KEYWORDS['Thematic ETF']['etf_patterns']:
            if pattern in name_upper:
                return 'Thematic ETF'
        # Default ETF classification
        if 'ETF' in name_upper or 'INDEKSI' in name_upper:
            return 'Thematic ETF'
    
    # If unclassified, track it
    if name not in unclassified_sector:
        unclassified_sector.append(name)
    
    return 'Other'

