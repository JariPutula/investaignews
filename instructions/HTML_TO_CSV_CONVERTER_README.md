# HTML to CSV Converter Utility

## Overview

This utility converts OP Bank HTML export files (`Arvopaperisäilytys - Sijoitukset _ OP.htm`) to the CSV format used by the investment dashboard.

## Features

- ✅ Parses HTML table structure from OP Bank export
- ✅ Maps Finnish column names to English using `nimimap.txt`
- ✅ Automatically infers ticker symbols from company names
- ✅ Supports manual ticker mapping via `ticker_mapping.txt`
- ✅ Creates timestamped output files (doesn't overwrite existing files)
- ✅ Validates data and reports missing tickers

## Usage

### Basic Usage

```bash
python convert_html_to_csv.py
```

This will:
1. Look for `data/Arvopaperisäilytys - Sijoitukset _ OP.htm`
2. Parse the HTML and extract holdings
3. Create a timestamped CSV file: `data/{DDMMYYYY}_assets_jari_from_html.csv`

### Custom HTML File

```bash
python convert_html_to_csv.py path/to/your/file.htm
```

## File Structure

### Input: HTML File
- Location: `data/Arvopaperisäilytys - Sijoitukset _ OP.htm`
- Format: OP Bank HTML export with table structure

### Output: CSV File
- Format: Same as `latest_assets_jari.csv`
- Columns: `ticker`, `name`, `quantity`, `purchase_price_eur`, `purchase_total_eur`, `market_price_eur`, `market_total_eur`, `change_eur`, `change_pct`
- Location: `data/{DDMMYYYY}_assets_jari_from_html.csv` (timestamped to avoid overwriting)

## Ticker Mapping

**The converter relies on `data/ticker_mapping.txt` as the PRIMARY source for ticker symbols.**

### Creating the Mapping File

You have two options:

#### Option 1: Generate from Existing CSV

If you already have a CSV file with tickers, generate the mapping file:

```bash
python generate_ticker_mapping.py
```

Or specify a CSV file:
```bash
python generate_ticker_mapping.py data/latest_assets_jari.csv
```

This will create `data/ticker_mapping.txt` with all your existing mappings.

#### Option 2: Create Manually

Create `data/ticker_mapping.txt` with the following format:

```
# Ticker Mapping File
# Format: company_name,ticker
# One mapping per line
# Lines starting with # are comments

ABBVIE INC,ABBV
ALPHABET INC -CL C,GOOG
AMAZON COM INC - ORD SHS,AMZN
ASTRAZENECA PLC,AZN
BAE SYSTEMS,BBA.L
OP-AMERIKKA INDEKSI A,OPAMERIKKA
SPDR MSCI WORLD SMALL CAP UCITS ETF,ZPRS.F
VANECK MORNINGSTAR GLOBAL WIDE MOAT UCITS ETF - A ACC,GOAT.MI
```

**Important Notes:**
- Company names must match EXACTLY (case-insensitive) as they appear in the HTML
- One mapping per line: `company_name,ticker`
- Comments start with `#`
- Empty lines are ignored
- The converter will show you which holdings need tickers added

## Column Mapping

The converter maps Finnish column names to English:

| Finnish (HTML) | English (CSV) |
|----------------|---------------|
| Laji | name |
| Omistus kpl | quantity |
| Hankinta-hinta EUR | purchase_price_eur |
| Hankinta-arvo yht. EUR | purchase_total_eur |
| Markkina-hinta EUR | market_price_eur |
| Markkina-arvo yht. EUR | market_total_eur |
| Muutos EUR | change_eur |
| Muutos % | change_pct |

**Note**: The `ticker` column is not present in the HTML file and must be inferred or manually mapped.

## Workflow

### Initial Setup (First Time)

1. **Generate Ticker Mapping**: If you have an existing CSV with tickers:
   ```bash
   python generate_ticker_mapping.py
   ```
   This creates `data/ticker_mapping.txt` from your existing CSV.

2. **Edit Mapping File**: Review and update `data/ticker_mapping.txt`:
   - Add any missing tickers
   - Update tickers if needed
   - Add new holdings you expect to see

### Regular Usage

1. **Export from OP Bank**: Download your holdings as HTML from OP Bank website
2. **Save HTML File**: Place it in `data/` folder as `Arvopaperisäilytys - Sijoitukset _ OP.htm`
3. **Run Converter**: Execute `python convert_html_to_csv.py`
4. **Review Output**: Check the generated CSV file
5. **Fix Missing Tickers** (if any):
   - The converter will show which holdings need tickers
   - Add mappings to `data/ticker_mapping.txt`
   - Re-run converter if needed
6. **Verify Data**: Compare with existing `latest_assets_jari.csv` to ensure structure matches
7. **Rename if Correct**: Once verified, you can rename the file to replace `latest_assets_jari.csv` (if desired)

## Example Output

```
Parsing HTML file: data/Arvopaperisäilytys - Sijoitukset _ OP.htm
Found 48 holdings
Loaded 5 manual ticker mappings
Warning: 3 holdings have missing tickers
✓ Created CSV file: data/20122025_assets_jari_from_html.csv
  Total holdings: 48
  Total value: €202,281.20

Holdings with missing tickers:
  - OP-AMERIKKA INDEKSI A
  - SPDR MSCI WORLD SMALL CAP UCITS ETF
  - VANECK MORNINGSTAR GLOBAL WIDE MOAT UCITS ETF

To fix: Add mappings to data/ticker_mapping.txt (format: name,ticker)
```

## Troubleshooting

### No Holdings Found
- Check that the HTML file is from OP Bank and has the correct table structure
- Verify the file encoding is UTF-8

### Missing Tickers
- Review the list of holdings with missing tickers
- Add mappings to `data/ticker_mapping.txt`
- Re-run the converter

### Parsing Errors
- The HTML structure may have changed
- Check the HTML file structure matches the expected format
- Report the issue with a sample of the HTML structure

## Integration with Dashboard

After conversion:
1. Review the generated CSV file
2. Verify all tickers are filled (or acceptable to leave empty)
3. Compare structure with `latest_assets_jari.csv`
4. Once satisfied, you can:
   - Use the timestamped file as a historical snapshot
   - Rename it to `latest_assets_jari.csv` (after backing up the old one)
   - Or keep both files for comparison

## Dependencies

- `pandas`: Data manipulation
- `beautifulsoup4`: HTML parsing
- Standard library: `os`, `re`, `datetime`

Install with:
```bash
pip install beautifulsoup4
```

## Notes

- **Ticker mapping is manual**: The converter relies on `ticker_mapping.txt` - you maintain this file
- The converter creates timestamped files to avoid overwriting existing data
- Missing tickers are reported but don't prevent file creation
- Use `generate_ticker_mapping.py` to create initial mapping from existing CSV
- The converter handles Finnish number formatting (comma as decimal separator, non-breaking spaces as thousand separators)
- Company names in the mapping file must match exactly (case-insensitive) as they appear in the HTML

