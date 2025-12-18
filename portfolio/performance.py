"""
Performance metrics calculations for portfolio analysis.
"""

import pandas as pd
from typing import Dict


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate overall portfolio performance metrics.
    
    Args:
        df: DataFrame with holdings data containing columns:
            - market_total_eur: Current market value
            - purchase_total_eur: Purchase value
            - change_eur: Unrealized gain/loss
    
    Returns:
        Dictionary with performance metrics:
            - total_market_value: Total current market value
            - total_unrealized_gain: Total unrealized gain/loss
            - overall_performance_pct: Overall performance percentage
    """
    total_market_value = df['market_total_eur'].sum()
    total_purchase_value = df['purchase_total_eur'].sum()
    total_unrealized_gain = df['change_eur'].sum()
    
    if total_purchase_value > 0:
        overall_performance_pct = (total_unrealized_gain / total_purchase_value) * 100
    else:
        overall_performance_pct = 0
    
    return {
        'total_market_value': total_market_value,
        'total_unrealized_gain': total_unrealized_gain,
        'overall_performance_pct': overall_performance_pct
    }

