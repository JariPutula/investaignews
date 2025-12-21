# How to Add New Stocks to Classification

When you add a new stock to your portfolio that isn't automatically classified, you'll see a warning like:
- "Unclassified Sector: BNP PARIBAS"
- "Unclassified Geography: BNP PARIBAS"

## Quick Answer

**Update only one file: `config.py`**

## Step-by-Step Guide

### 1. Identify the Stock

Check the warning message in the dashboard to see:
- Company name (e.g., "BNP PARIBAS")
- What's missing (Sector, Geography, or both)

### 2. Determine Classification

**For Sector:**
- What industry is the company in?
  - Technology (software, hardware, semiconductors)
  - Healthcare (pharmaceuticals, medical devices)
  - Financial Services (banks, insurance, financial companies)
  - Energy/Utilities (oil, gas, renewable energy, utilities)
  - Real Estate (REITs, property companies)
  - Consumer Discretionary (retail, entertainment, automotive)
  - Consumer Staples (food, beverages, household products)
  - Industrials (manufacturing, aerospace, defense)
  - Materials (mining, chemicals, construction materials)
  - Communication Services (telecom, media)
  - Broad Market ETF (diversified index funds)
  - Thematic ETF (sector-specific ETFs)

**For Geography:**
- Where is the company based?
  - U.S. (United States companies)
  - Finland (Finnish companies)
  - Europe/Global (European or global companies)

### 3. Edit `config.py`

Open `config.py` and find the appropriate section:

#### For Sector Classification

Find `SECTOR_KEYWORDS` dictionary and add to the appropriate sector:

```python
'Financial Services': {
    'keywords': ['BANCO', 'SANTANDER', 'SAMPO', 'MANDATUM', 'NORDEA', 'BANK',
                'JPMORGAN', 'GOLDMAN', 'MORGAN STANLEY', 'WELLS FARGO', 'CITI',
                'BNP PARIBAS', 'BNP'],  # <-- Add your company here
    'etf_patterns': ['FINANCIAL', 'BANK', 'INSURANCE']
},
```

**Tips:**
- Add the full company name: `'BNP PARIBAS'`
- Also add a shorter version if useful: `'BNP'`
- Use uppercase (matching is case-insensitive, but convention is uppercase)
- Add to the `'keywords'` list within the appropriate sector

#### For Geography Classification

Find `GEOGRAPHY_KEYWORDS` dictionary and add to the appropriate region:

```python
'Europe/Global': {
    'keywords': ['BAE SYSTEMS', 'BAE', 'IPSEN', 'BNP PARIBAS', 'BNP'],  # <-- Add here
    'company_suffixes': ['PLC', 'SA', 'SPA', 'AS', 'ABP', 'AB', 'AG', 'SE', 'NV'],
    'etf_patterns': ['EMU', 'EUROPE', 'EURO', 'EMERGING', 'WORLD', 'GLOBAL', 'ALLWORLD']
}
```

**Tips:**
- Add the full company name: `'BNP PARIBAS'`
- Also add a shorter version if useful: `'BNP'`
- If the company has a specific suffix (like 'SA' for French companies), it might already be covered in `'company_suffixes'`

### 4. Restart the Application

After editing `config.py`:
1. Save the file
2. Restart your Streamlit application
3. The new stock should now be classified correctly

## Example: BNP PARIBAS

BNP PARIBAS is a French bank, so:

**Sector:** Financial Services
**Geography:** Europe/Global

**Changes in `config.py`:**

```python
# In SECTOR_KEYWORDS:
'Financial Services': {
    'keywords': [..., 'BNP PARIBAS', 'BNP'],  # Added
    ...
}

# In GEOGRAPHY_KEYWORDS:
'Europe/Global': {
    'keywords': [..., 'BNP PARIBAS', 'BNP'],  # Added
    ...
}
```

## Common Classifications

### US Technology Companies
- **Sector:** Technology
- **Geography:** U.S.
- Examples: `'APPLE'`, `'MICROSOFT'`, `'GOOGLE'`

### European Banks
- **Sector:** Financial Services
- **Geography:** Europe/Global
- Examples: `'BNP PARIBAS'`, `'DEUTSCHE BANK'`, `'UBS'`

### Finnish Companies
- **Sector:** (varies by company)
- **Geography:** Finland
- Examples: `'NOKIA'`, `'SAMPO'`, `'FORTUM'`

### ETFs
- **Sector:** Broad Market ETF or Thematic ETF
- **Geography:** Usually Europe/Global (unless US-specific)
- Add to `'etf_patterns'` if it's a pattern, or `'keywords'` for specific ETFs

## Troubleshooting

### Still Not Classified After Update?

1. **Check spelling:** Company name must match exactly (case-insensitive)
2. **Check format:** Make sure you added it to the `'keywords'` list (not `'etf_patterns'` or `'company_suffixes'`)
3. **Restart app:** You must restart Streamlit for changes to take effect
4. **Check both:** Make sure you added it to both Sector AND Geography if both were missing

### Multiple Matches?

If a company matches multiple keywords, the first match wins. Order matters in the dictionaries.

## Files You DON'T Need to Edit

- ❌ `portfolio/classification.py` - Uses keywords from `config.py` automatically
- ❌ `app.py` - Uses classification functions automatically
- ❌ Any other files - Classification is centralized in `config.py`

## Summary

**Only edit: `config.py`**

1. Find the appropriate sector/geography dictionary
2. Add company name to `'keywords'` list
3. Restart the application
4. Done!

