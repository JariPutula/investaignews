# How to Add Historical Snapshots

## Current Situation

You currently have:
- `latest_assets_jari.csv` - Your current/latest portfolio snapshot
- `20112025_assets_jari.csv` - One historical snapshot (November 20, 2025)

For meaningful historical performance analysis, you need **multiple snapshots over time**.

## Option 1: Create Historical Snapshots from Past Data

If you have portfolio data from previous dates, create snapshot files:

### File Naming Format
```
{DDMMYYYY}_assets_{user}.csv
```

Examples:
- `20112025_assets_jari.csv` = November 20, 2025
- `01122025_assets_jari.csv` = December 1, 2025
- `15122025_assets_jari.csv` = December 15, 2025

### Steps to Create a Snapshot

1. **Copy your latest snapshot**:
   ```bash
   cp latest_assets_jari.csv 15122025_assets_jari.csv
   ```

2. **Modify the values** in the new file to reflect the portfolio state on that date:
   - Update `market_price_eur` to reflect prices on that date
   - Update `market_total_eur` accordingly
   - Update `change_eur` and `change_pct` based on the difference from purchase price
   - Keep `purchase_price_eur` and `purchase_total_eur` the same (these are historical)

3. **Save the file** with the date in the filename

## Option 2: Use Latest Snapshot as Current Point

The code has been updated to **automatically include the latest snapshot** in the timeline (with today's date). This means:

- `latest_assets_jari.csv` will be included as the most recent data point
- You still need historical snapshots for meaningful comparisons

## Recommended Snapshot Schedule

For good historical analysis, create snapshots:

- **Monthly**: At the end of each month
- **Quarterly**: At the end of each quarter
- **After major changes**: After significant portfolio changes

### Example Timeline

If today is December 20, 2025, you might have:
- `20112025_assets_jari.csv` - November 20, 2025 (you have this)
- `01122025_assets_jari.csv` - December 1, 2025
- `latest_assets_jari.csv` - December 20, 2025 (current)

## Quick Test: Create a Few Test Snapshots

To test the functionality quickly, you can:

1. **Copy your existing snapshot with different dates**:
   ```bash
   # Copy to create a few test snapshots
   cp latest_assets_jari.csv 01122025_assets_jari.csv
   cp latest_assets_jari.csv 15122025_assets_jari.csv
   ```

2. **Modify the values slightly** in each file to simulate portfolio changes:
   - Change some `market_price_eur` values
   - Recalculate `market_total_eur`, `change_eur`, `change_pct`

3. **Run the tests again** - you should see multiple data points

## Date Format

The date format is **DDMMYYYY** (day, month, year):
- `20112025` = 20th November 2025
- `01122025` = 1st December 2025
- `15122025` = 15th December 2025

## Notes

- The `latest_assets_jari.csv` file is now automatically included in the timeline
- Historical snapshots should reflect the actual portfolio state on those dates
- More snapshots = better historical analysis
- At least 2-3 snapshots are needed for basic comparisons
- 12+ snapshots are ideal for rolling metrics (volatility, Sharpe ratio)

