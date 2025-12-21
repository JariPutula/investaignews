"""
Utility to convert OP Bank HTML export to CSV asset file format.

This script:
1. Parses the HTML file from OP Bank (Arvopaperisäilytys - Sijoitukset _ OP.htm)
2. Extracts holdings data from the HTML table
3. Maps Finnish column names to English using nimimap.txt
4. Infers ticker symbols from company names (or uses manual mapping)
5. Creates a CSV file with the same structure as existing asset files
"""

import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import DATA_DIR, DEFAULT_USER_NAME


# Column mapping from Finnish to English (from nimimap.txt)
FINNISH_TO_ENGLISH = {
    'Laji': 'name',  # Will be extracted from link text
    'Omistus kpl': 'quantity',
    'Hankinta-hinta EUR': 'purchase_price_eur',
    'Hankinta-arvo yht. EUR': 'purchase_total_eur',
    'Markkina-hinta EUR': 'market_price_eur',
    'Markkina-arvo yht. EUR': 'market_total_eur',
    'Muutos EUR': 'change_eur',
    'Muutos %': 'change_pct',
}

def infer_ticker_from_name(name: str) -> str:
    """
    Very basic ticker inference - only tries obvious patterns.
    Primary source should be ticker_mapping.txt file.
    
    Args:
        name: Company or ETF name
    
    Returns:
        Ticker symbol, or empty string if cannot be inferred
    """
    if not name:
        return ''
    
    # Try to extract ticker from parentheses: "Company Name (TICKER)"
    ticker_match = re.search(r'\(([A-Z]{1,6})\)', name)
    if ticker_match:
        return ticker_match.group(1)
    
    # Return empty - user should add to ticker_mapping.txt
    return ''


def parse_finnish_number(text: str) -> float:
    """
    Parse Finnish-formatted number (comma as decimal, non-breaking space as thousand separator).
    
    Examples:
        "1 936,65" -> 1936.65
        "66,70" -> 66.70
        "1 269,65" -> 1269.65
        "-1 269,65" -> -1269.65
        "+1 269,65" -> 1269.65
    
    Args:
        text: Number string with Finnish formatting
    
    Returns:
        Float value, or 0.0 if parsing fails
    """
    if not text:
        return 0.0
    
    # Check for negative sign first
    is_negative = text.strip().startswith('-') or text.strip().startswith('−')
    
    # Remove all whitespace (including non-breaking spaces \xa0)
    text = text.replace('\xa0', '').replace(' ', '').replace('\u00A0', '').replace('\u2009', '')
    
    # Remove + sign (but keep - for negative numbers)
    text = text.replace('+', '')
    
    # Replace comma with dot for decimal separator
    text = text.replace(',', '.')
    
    # Remove any remaining non-numeric characters except dot and minus at start
    # Keep minus only at the beginning
    cleaned = re.sub(r'[^\d\.\-]', '', text)
    if is_negative and not cleaned.startswith('-'):
        cleaned = '-' + cleaned.lstrip('-')
    
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def parse_html_table(html_file: str, ticker_mapping: Dict[str, str]) -> List[Dict]:
    """
    Parse HTML file and extract holdings data.
    
    Args:
        html_file: Path to HTML file
    
    Returns:
        List of dictionaries with holdings data
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the main table with class "Taulukko ws2-omistukset"
    # The table structure has:
    # - Header row with class "TaulukkoOtsikkorivi"
    # - Data rows with class "Pieni" cells
    
    holdings = []
    
    # Find all tables with the holdings class
    tables = soup.find_all('table', class_='Taulukko ws2-omistukset')
    
    for table in tables:
        # Find all data rows (skip header row)
        rows = table.find_all('tr')
        
        for row in rows:
            # Skip header row
            if 'TaulukkoOtsikkorivi' in row.get('class', []):
                continue
            
            # Skip summary/total rows
            row_text = row.get_text()
            if 'Yhteensä' in row_text or 'TOTAL' in row_text.upper():
                continue
            
            # Skip rows that are just separators or action buttons
            if 'Osta' in row_text and 'Myy' in row_text and len(row_text.strip()) < 20:
                continue
            
            # Find cells with class "Pieni" - these contain the data
            # The structure has: name (colspan=2), quantity, purchase_price, purchase_total, 
            # market_price, market_total, change_eur, change_pct, actions
            cells = row.find_all('td', class_='Pieni')
            
            # Need at least 8 cells (name colspan=2 counts as 1 in DOM, then 7 data cells, last is actions)
            # But we only need the first 8 for data (skip the last action cell)
            if len(cells) < 8:
                # Try finding all td cells and filter
                all_cells = row.find_all('td')
                # Use first 8 cells that have data (skip action cells)
                data_cells = []
                for cell in all_cells:
                    # Skip action cells (contain "ToimintoLinkki" or just "Osta"/"Myy")
                    if cell.find('div', class_='ToimintoLinkki'):
                        continue
                    if cell.find('table') and ('Osta' in cell.get_text() or 'Myy' in cell.get_text()):
                        continue
                    data_cells.append(cell)
                    if len(data_cells) >= 8:
                        break
                
                if len(data_cells) >= 8:
                    cells = data_cells
                else:
                    continue
            else:
                # Use first 8 cells (skip the last action cell if present)
                cells = cells[:8]
            
            # Debug: print cell structure for first row
            if len(holdings) == 0 and len(cells) >= 8:
                print(f"Debug: First row has {len(cells)} cells")
                for i, cell in enumerate(cells[:8]):
                    text = cell.get_text(strip=True)[:30]
                    colspan = cell.get('colspan', '1')
                    print(f"  Cell {i}: colspan={colspan}, text='{text}'")
            
            try:
                # Extract data from cells
                # The structure: name (colspan=2), quantity, purchase_price, purchase_total,
                # market_price, market_total, change_eur, change_pct, actions
                # When name has colspan="2", it takes up index 0, so data starts at index 1
                
                # Cell 0: Name (may have colspan="2")
                name_cell = cells[0] if cells else None
                if not name_cell:
                    continue
                
                # Check if name cell has colspan
                colspan = int(name_cell.get('colspan', 1))
                
                # Try to find link first
                link = name_cell.find('a', class_='EiAlleviivattu')
                if not link:
                    # Try any link
                    link = name_cell.find('a')
                
                if link:
                    name = link.get_text(strip=True)
                else:
                    name = name_cell.get_text(strip=True)
                
                # Clean up name - remove extra whitespace
                name = ' '.join(name.split())
                
                if not name or len(name) < 2:
                    continue
                
                # Skip if it looks like a button or action text
                if name.upper() in ['OSTA', 'MYY', 'VAIHDA']:
                    continue
                
                # Calculate data cell start index
                # Name cell has colspan="2" but is still only one element in the list
                # So data starts at index 1 (right after name cell)
                data_start_idx = 1
                
                # Need at least 7 data cells after name
                if len(cells) < data_start_idx + 7:
                    continue
                
                # Cell [data_start_idx]: Quantity (has two numbers separated by <br/>)
                # Format: "10\n10" or "10<br/>10" - we want the first number
                quantity_cell = cells[data_start_idx] if len(cells) > data_start_idx else None
                if quantity_cell:
                    # Get text with line breaks preserved
                    quantity_text = quantity_cell.get_text(separator='\n', strip=True)
                    # Split by newline and take first non-empty value
                    lines = [line.strip() for line in quantity_text.split('\n') if line.strip()]
                    if lines:
                        # First line should be the quantity
                        quantity = parse_finnish_number(lines[0])
                    else:
                        quantity = 0.0
                else:
                    quantity = 0.0
                
                # Cell [data_start_idx + 1]: Purchase price
                purchase_price_text = cells[data_start_idx + 1].get_text(strip=True) if len(cells) > data_start_idx + 1 else ''
                purchase_price = parse_finnish_number(purchase_price_text)
                
                # Cell [data_start_idx + 2]: Purchase total
                purchase_total_text = cells[data_start_idx + 2].get_text(strip=True) if len(cells) > data_start_idx + 2 else ''
                purchase_total = parse_finnish_number(purchase_total_text)
                
                # Cell [data_start_idx + 3]: Market price
                market_price_text = cells[data_start_idx + 3].get_text(strip=True) if len(cells) > data_start_idx + 3 else ''
                market_price = parse_finnish_number(market_price_text)
                
                # Cell [data_start_idx + 4]: Market total
                market_total_text = cells[data_start_idx + 4].get_text(strip=True) if len(cells) > data_start_idx + 4 else ''
                market_total = parse_finnish_number(market_total_text)
                
                # Cell [data_start_idx + 5]: Change EUR (may have + or - sign)
                change_text = cells[data_start_idx + 5].get_text(strip=True) if len(cells) > data_start_idx + 5 else ''
                change_eur = parse_finnish_number(change_text)
                
                # Cell [data_start_idx + 6]: Change %
                change_pct_text = cells[data_start_idx + 6].get_text(strip=True) if len(cells) > data_start_idx + 6 else ''
                # Remove % sign before parsing
                change_pct_text = change_pct_text.replace('%', '')
                change_pct = parse_finnish_number(change_pct_text)
                
                # Get ticker from mapping file (primary source)
                # Only use basic inference if not found in mapping
                ticker = ''
                name_upper = name.upper()
                if name_upper in ticker_mapping:
                    ticker = ticker_mapping[name_upper]
                else:
                    # Try very basic inference (e.g., from parentheses)
                    ticker = infer_ticker_from_name(name)
                    # If still empty, it will be reported for manual addition
                
                holding = {
                    'ticker': ticker,
                    'name': name,
                    'quantity': quantity,
                    'purchase_price_eur': purchase_price,
                    'purchase_total_eur': purchase_total,
                    'market_price_eur': market_price,
                    'market_total_eur': market_total,
                    'change_eur': change_eur,
                    'change_pct': change_pct,
                }
                
                holdings.append(holding)
                
            except (ValueError, AttributeError, IndexError) as e:
                # Skip rows that can't be parsed
                print(f"Warning: Skipping row due to parsing error: {e}")
                continue
    
    return holdings


def load_ticker_mapping(mapping_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load manual ticker mappings from a file.
    
    This is the PRIMARY source for ticker symbols.
    Format: name,ticker (one per line)
    Lines starting with # are comments and ignored.
    
    Args:
        mapping_file: Path to mapping file (default: data/ticker_mapping.txt)
    
    Returns:
        Dictionary mapping company names (uppercase) to tickers
    """
    if mapping_file is None:
        mapping_file = os.path.join(DATA_DIR, 'ticker_mapping.txt')
    
    mapping = {}
    
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Split by comma (handle quoted values if needed)
                parts = line.split(',')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    ticker = parts[1].strip()
                    if name and ticker:
                        # Use uppercase for case-insensitive matching
                        mapping[name.upper()] = ticker
                    else:
                        print(f"Warning: Invalid mapping at line {line_num}: {line}")
                else:
                    print(f"Warning: Invalid format at line {line_num}: {line}")
    else:
        print(f"Note: Ticker mapping file not found: {mapping_file}")
        print("  Create this file with format: company_name,ticker")
        print("  Example: ABBVIE INC,ABBV")
    
    return mapping




def convert_html_to_csv(
    html_file: str,
    output_file: Optional[str] = None,
    user_name: Optional[str] = None,
    use_timestamp: bool = True
) -> str:
    """
    Convert HTML file to CSV asset file.
    
    Args:
        html_file: Path to HTML file
        output_file: Output CSV file path (if None, generates name)
        user_name: User name for filename (default: from config)
        use_timestamp: Whether to add timestamp to output filename
    
    Returns:
        Path to created CSV file
    """
    if user_name is None:
        user_name = DEFAULT_USER_NAME
    
    # Load ticker mappings FIRST (primary source)
    print(f"Loading ticker mappings from {os.path.join(DATA_DIR, 'ticker_mapping.txt')}...")
    ticker_mapping = load_ticker_mapping()
    if ticker_mapping:
        print(f"✓ Loaded {len(ticker_mapping)} ticker mappings")
    else:
        print("⚠ No ticker mappings found - all tickers will need to be added manually")
    
    # Parse HTML
    print(f"\nParsing HTML file: {html_file}")
    holdings = parse_html_table(html_file, ticker_mapping)
    
    if not holdings:
        raise ValueError("No holdings found in HTML file")
    
    print(f"✓ Found {len(holdings)} holdings")
    
    # Count holdings with missing tickers
    missing_tickers = sum(1 for h in holdings if not h['ticker'])
    if missing_tickers > 0:
        print(f"\n⚠ Warning: {missing_tickers} holdings have missing tickers")
        print("  Add these to data/ticker_mapping.txt with format: company_name,ticker")
    
    # Create DataFrame
    df = pd.DataFrame(holdings)
    
    # Ensure correct column order
    column_order = [
        'ticker', 'name', 'quantity', 'purchase_price_eur',
        'purchase_total_eur', 'market_price_eur', 'market_total_eur',
        'change_eur', 'change_pct'
    ]
    df = df[column_order]
    
    # Generate output filename if not provided
    if output_file is None:
        if use_timestamp:
            timestamp = datetime.now().strftime("%d%m%Y")
            output_file = os.path.join(DATA_DIR, f"{timestamp}_assets_{user_name}_from_html.csv")
        else:
            output_file = os.path.join(DATA_DIR, f"latest_assets_{user_name}_from_html.csv")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Created CSV file: {output_file}")
    print(f"  Total holdings: {len(df)}")
    print(f"  Total value: €{df['market_total_eur'].sum():,.2f}")
    
    # Show holdings with missing tickers and generate mapping template
    if missing_tickers > 0:
        print("\n" + "=" * 60)
        print("Holdings with missing tickers:")
        print("=" * 60)
        missing_df = df[df['ticker'] == '']
        print("\nAdd these lines to data/ticker_mapping.txt:\n")
        for idx, row in missing_df.iterrows():
            print(f"{row['name']},")
        print("\n" + "=" * 60)
    
    return output_file


def main():
    """Main function to run the converter."""
    import sys
    
    # Default HTML file
    html_file = os.path.join(DATA_DIR, "Arvopaperisäilytys - Sijoitukset _ OP.htm")
    
    # Allow command-line argument for HTML file
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file not found: {html_file}")
        print(f"Expected location: {os.path.join(DATA_DIR, 'Arvopaperisäilytys - Sijoitukset _ OP.htm')}")
        return
    
    try:
        output_file = convert_html_to_csv(html_file, use_timestamp=True)
        print(f"\n✓ Conversion complete!")
        print(f"  Output file: {output_file}")
        print(f"\nNote: Review the file and fill in any missing tickers before using it.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

