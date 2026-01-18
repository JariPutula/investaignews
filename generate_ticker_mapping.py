"""
Helper script to generate ticker_mapping.txt from existing CSV file.

This creates a template mapping file that you can edit manually.
"""

import os
import pandas as pd
from config import DATA_DIR, DEFAULT_USER_NAME


def generate_ticker_mapping_from_csv(
    csv_file: str = None,
    output_file: str = None
) -> None:
    """
    Generate ticker_mapping.txt from existing CSV file.
    
    Args:
        csv_file: Path to CSV file (default: latest_assets_jari.csv)
        output_file: Output mapping file (default: data/ticker_mapping.txt)
    """
    if csv_file is None:
        csv_file = os.path.join(DATA_DIR, f"latest_assets_{DEFAULT_USER_NAME}.csv")
    
    if output_file is None:
        output_file = os.path.join(DATA_DIR, 'ticker_mapping.txt')
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    if 'name' not in df.columns or 'ticker' not in df.columns:
        print("Error: CSV must have 'name' and 'ticker' columns")
        return
    
    # Generate mapping file
    print(f"Generating ticker mapping from: {csv_file}")
    print(f"Output file: {output_file}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Ticker Mapping File\n")
        f.write("# Format: company_name,ticker\n")
        f.write("# One mapping per line\n")
        f.write("# Lines starting with # are comments\n")
        f.write("# Empty lines are ignored\n\n")
        f.write("# Generated from existing CSV file\n")
        f.write("# Edit this file manually to add/update ticker mappings\n\n")
        
        # Sort by name for easier editing
        sorted_df = df.sort_values('name')
        
        for _, row in sorted_df.iterrows():
            name = row['name']
            ticker = row['ticker']
            
            # Skip if ticker is empty or "NONE"
            if pd.isna(ticker) or str(ticker).strip().upper() == 'NONE' or str(ticker).strip() == '':
                f.write(f"# {name},\n")  # Comment out missing tickers
            else:
                f.write(f"{name},{ticker}\n")
    
    # Count statistics
    total = len(df)
    with_ticker = len(df[df['ticker'].notna() & (df['ticker'].astype(str).str.strip() != '') & (df['ticker'].astype(str).str.upper() != 'NONE')])
    missing = total - with_ticker
    
    print(f"✓ Generated {output_file}")
    print(f"  Total holdings: {total}")
    print(f"  With tickers: {with_ticker}")
    print(f"  Missing tickers: {missing}")
    if missing > 0:
        print(f"\n  Note: Holdings with missing tickers are commented out in the mapping file")
        print(f"  Uncomment and fill in the ticker values manually")
    
    print(f"\nYou can now edit {output_file} to:")
    print(f"  - Add missing tickers")
    print(f"  - Update existing tickers")
    print(f"  - Add new holdings before converting HTML files")


if __name__ == "__main__":
    import sys
    
    csv_file = None
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    generate_ticker_mapping_from_csv(csv_file)

