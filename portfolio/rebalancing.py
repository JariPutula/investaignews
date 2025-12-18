"""
Rebalancing calculations for portfolio optimization.
"""

import pandas as pd
from typing import Dict


def calculate_rebalancing(
    df: pd.DataFrame,
    target_allocations: Dict[str, float],
    transaction_cost_pct: float = 0.0,
    tax_rate: float = 0.30
) -> pd.DataFrame:
    """
    Calculate rebalancing recommendations.
    
    Parameters:
        df: DataFrame with holdings data containing:
            - sector: Sector classification
            - market_total_eur: Current market value
            - change_eur: Unrealized gain/loss
            - purchase_total_eur: Purchase value
        target_allocations: Dictionary with target percentages (e.g., {'Technology': 20, 'Healthcare': 15})
        transaction_cost_pct: Transaction cost as percentage (default 0%)
        tax_rate: Capital gains tax rate (default 30% for Finland)
    
    Returns:
        DataFrame with rebalancing recommendations containing:
            - sector: Sector name
            - current_%: Current allocation percentage
            - target_%: Target allocation percentage
            - current_value_eur: Current value in EUR
            - target_value_eur: Target value in EUR
            - difference_eur: Difference to rebalance
            - action: BUY, SELL, or HOLD
            - transaction_cost_eur: Estimated transaction cost
            - tax_implication_eur: Estimated tax on capital gains
            - total_cost_eur: Total cost (transaction + tax)
    """
    total_value = df['market_total_eur'].sum()
    
    # Calculate current allocations by sector
    current_allocations = df.groupby('sector')['market_total_eur'].sum() / total_value * 100
    
    # Create rebalancing DataFrame
    rebalancing_data = []
    
    for sector, target_pct in target_allocations.items():
        current_pct = current_allocations.get(sector, 0)
        current_value = df[df['sector'] == sector]['market_total_eur'].sum()
        target_value = (target_pct / 100) * total_value
        difference = target_value - current_value
        
        # Calculate transaction costs
        transaction_cost = abs(difference) * (transaction_cost_pct / 100)
        
        # Calculate tax implications (only for selling)
        tax_implication = 0
        if difference < 0:  # Selling
            # Estimate capital gains on sold portion
            sector_df = df[df['sector'] == sector].copy()
            if len(sector_df) > 0:
                # Calculate average gain percentage for this sector
                sector_gain_pct = (sector_df['change_eur'].sum() / sector_df['purchase_total_eur'].sum()) * 100 if sector_df['purchase_total_eur'].sum() > 0 else 0
                # Estimate capital gains on the amount being sold
                sold_pct = abs(difference) / current_value if current_value > 0 else 0
                estimated_gain = sector_df['change_eur'].sum() * sold_pct
                tax_implication = max(0, estimated_gain * tax_rate)  # Only tax on gains
        
        rebalancing_data.append({
            'sector': sector,
            'current_%': round(current_pct, 2),
            'target_%': target_pct,
            'current_value_eur': round(current_value, 2),
            'target_value_eur': round(target_value, 2),
            'difference_eur': round(difference, 2),
            'action': 'BUY' if difference > 0 else 'SELL' if difference < 0 else 'HOLD',
            'transaction_cost_eur': round(transaction_cost, 2),
            'tax_implication_eur': round(tax_implication, 2),
            'total_cost_eur': round(transaction_cost + tax_implication, 2)
        })
    
    rebalancing_df = pd.DataFrame(rebalancing_data)
    rebalancing_df = rebalancing_df.sort_values('difference_eur', key=abs, ascending=False)
    
    return rebalancing_df

