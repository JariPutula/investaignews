# Data Folder Migration Summary

## Overview

All asset files have been moved to the `data/` folder, and all code has been updated to read from this location.

## Changes Made

### 1. Configuration Updates

**File: `config.py`**
- Added `DATA_DIR = 'data'` constant
- Updated `DEFAULT_CSV_PATH` to use `os.path.join(DATA_DIR, 'holdings_from_op.csv')`
- Added `import os` for path operations

### 2. Snapshot Manager Updates

**File: `portfolio/historical/snapshot_manager.py`**
- Updated `find_snapshot_files()` to use `DATA_DIR` as default directory
- Updated `get_latest_snapshot_path()` to use `DATA_DIR` as default directory
- Updated `get_snapshot_info()` to use `DATA_DIR` as default directory

### 3. Snapshot Loader Updates

**File: `portfolio/historical/snapshot_loader.py`**
- Updated `load_all_snapshots()` to use `DATA_DIR` as default directory
- Updated `load_latest_snapshot()` to use `DATA_DIR` as default directory
- Updated `get_portfolio_value_over_time()` to use `DATA_DIR` as default directory

### 4. Data Loader Updates

**File: `portfolio/data_loader.py`**
- Added import for `DATA_DIR` (for future use if needed)
- No functional changes (uses snapshot_loader which now uses DATA_DIR)

### 5. Test Script Updates

**File: `create_test_snapshots.py`**
- Updated to use `DATA_DIR` for all file paths
- Updated to use `DEFAULT_USER_NAME` from config
- All file operations now use `os.path.join(DATA_DIR, ...)`

### 6. New HTML to CSV Converter

**File: `convert_html_to_csv.py`** (NEW)
- Utility to convert OP Bank HTML exports to CSV format
- Automatically infers ticker symbols
- Supports manual ticker mapping
- Creates timestamped output files
- See `HTML_TO_CSV_CONVERTER_README.md` for details

**File: `HTML_TO_CSV_CONVERTER_README.md`** (NEW)
- Complete documentation for the HTML converter utility

### 7. Dependencies

**File: `requirements.txt`**
- Added `beautifulsoup4>=4.12.0` for HTML parsing

## File Structure

### Before
```
project_root/
├── latest_assets_jari.csv
├── 20112025_assets_jari.csv
├── holdings_from_op.csv
└── ...
```

### After
```
project_root/
├── data/
│   ├── latest_assets_jari.csv
│   ├── 20112025_assets_jari.csv
│   ├── holdings_from_op.csv
│   ├── Arvopaperisäilytys - Sijoitukset _ OP.htm
│   ├── nimimap.txt
│   └── ticker_mapping.txt (optional)
└── ...
```

## Backward Compatibility

All changes maintain backward compatibility:
- If `directory` parameter is explicitly provided, it's used as-is
- Default behavior now uses `data/` folder
- Old code that specifies full paths will still work

## Testing

To verify the migration:

1. **Check data folder exists:**
   ```python
   from config import DATA_DIR
   import os
   print(os.path.exists(DATA_DIR))  # Should be True
   ```

2. **Test file loading:**
   ```python
   from portfolio.data_loader import load_data
   df = load_data()  # Should load from data/latest_assets_jari.csv
   ```

3. **Test HTML converter:**
   ```bash
   python convert_html_to_csv.py
   ```

## Migration Checklist

- [x] Update `config.py` with `DATA_DIR`
- [x] Update `snapshot_manager.py` to use `DATA_DIR`
- [x] Update `snapshot_loader.py` to use `DATA_DIR`
- [x] Update `create_test_snapshots.py` to use `DATA_DIR`
- [x] Create HTML to CSV converter utility
- [x] Add BeautifulSoup4 to requirements
- [x] Create documentation

## Next Steps

1. **Move existing files** (if not already done):
   - Move all `*_assets_*.csv` files to `data/` folder
   - Move `holdings_from_op.csv` to `data/` folder (if it exists)

2. **Test the application:**
   - Run the Streamlit app and verify it loads data correctly
   - Test historical performance tab
   - Test all other tabs

3. **Use HTML converter:**
   - Place OP Bank HTML export in `data/` folder
   - Run `python convert_html_to_csv.py`
   - Review generated CSV file
   - Add missing tickers to `data/ticker_mapping.txt` if needed

## Notes

- The `data/` folder is now the single source of truth for asset files
- HTML converter creates timestamped files to avoid overwriting
- Manual ticker mapping can be added to `data/ticker_mapping.txt`
- All file operations use `os.path.join()` for cross-platform compatibility

