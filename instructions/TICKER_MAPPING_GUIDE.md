# Ticker Mapping Guide

## Overview

The HTML to CSV converter uses `data/ticker_mapping.txt` as the **primary source** for ticker symbols. This file is maintained manually by you.

## File Format

```
# Ticker Mapping File
# Format: company_name,ticker
# One mapping per line
# Lines starting with # are comments
# Empty lines are ignored

ABBVIE INC,ABBV
ALPHABET INC -CL C,GOOG
AMAZON COM INC - ORD SHS,AMZN
ASTRAZENECA PLC,AZN
BAE SYSTEMS,BBA.L
```

## Creating the Mapping File

### From Existing CSV

If you already have a CSV file with tickers:

```bash
python generate_ticker_mapping.py
```

This will:
- Read `data/latest_assets_jari.csv`
- Generate `data/ticker_mapping.txt` with all mappings
- Comment out holdings with missing tickers

### Manually

Create `data/ticker_mapping.txt` and add mappings one per line:

```
company_name,ticker
```

## Important Rules

1. **Exact Match**: Company names must match **exactly** (case-insensitive) as they appear in the HTML file
2. **One per Line**: Each mapping is on its own line
3. **No Spaces**: No spaces around the comma: `name,ticker` not `name, ticker`
4. **Comments**: Lines starting with `#` are ignored
5. **Empty Lines**: Empty lines are ignored

## Finding Ticker Symbols

### Stocks
- US stocks: Use Yahoo Finance (e.g., AAPL, MSFT, GOOG)
- European stocks: May include exchange suffix (e.g., `BBA.L` for London, `.HE` for Helsinki)
- Example: `NOKIA OYJ` → `NOKIA.HE`

### ETFs
- ETFs often have specific tickers (e.g., `IWRD.L` for iShares MSCI World)
- Some may use ISIN codes or exchange-specific codes
- Check the ETF provider's website or your broker

### Examples

```
# US Stocks
MICROSOFT CORP - ORD SHS,MSFT
APPLE INC,AAPL

# European Stocks
NOKIA OYJ,NOKIA.HE
SAMPO OYJ A,SAMPO.HE

# ETFs
ISHARES MSCI WORLD UCITS ETF - USD DIS,IWRD.L
SPDR MSCI WORLD SMALL CAP UCITS ETF - USD ACC,ZPRS.F

# Funds (may not have ticker - use NONE or leave empty)
OP-AMERIKKA INDEKSI A,NONE
```

## Updating the Mapping File

1. **Add New Holdings**: When you get new holdings, add them to the mapping file
2. **Update Tickers**: If a ticker changes, update the mapping
3. **Remove Old Holdings**: You can remove or comment out holdings you no longer have

## Troubleshooting

### Ticker Not Found

If the converter shows "missing ticker" for a holding:

1. Check the exact company name in the HTML file
2. Add it to `ticker_mapping.txt` with the correct ticker
3. Re-run the converter

### Wrong Ticker

If a ticker is incorrect:

1. Find the correct ticker (check Yahoo Finance, broker, etc.)
2. Update the mapping in `ticker_mapping.txt`
3. Re-run the converter

### Case Sensitivity

Company names are matched case-insensitively, but the exact spelling (including punctuation) must match.

Example:
- ✅ `ABBVIE INC` matches `abbvie inc`
- ❌ `ABBVIE INC` does NOT match `ABBVIE INCORPORATED`

## Best Practices

1. **Keep it Updated**: Update the mapping file when you add new holdings
2. **Use Comments**: Comment out holdings you're unsure about: `# UNKNOWN COMPANY,`
3. **Backup**: Keep a backup of your mapping file
4. **Version Control**: Consider adding `ticker_mapping.txt` to version control (but be careful with sensitive data)

## Example Workflow

1. Export HTML from OP Bank
2. Run converter: `python convert_html_to_csv.py`
3. Check output for missing tickers
4. Add missing tickers to `ticker_mapping.txt`
5. Re-run converter
6. Verify the CSV file is correct

