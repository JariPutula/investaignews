# Refactoring Proposal: Splitting app.py into Modules

## Proposed Module Structure

```
investaig/
├── app.py                          # Main entry point (minimal, ~100 lines)
├── portfolio/                      # Portfolio analysis modules
│   ├── __init__.py
│   ├── classification.py           # Classification logic (geography, sector)
│   ├── data_loader.py              # Data loading and validation
│   ├── performance.py              # Performance metrics
│   ├── rebalancing.py              # Rebalancing calculations
│   ├── risk_metrics.py             # Risk calculations (Sharpe, Sortino, VaR, etc.)
│   └── correlation.py              # Correlation matrix calculations
├── ai/                             # AI-related utilities
│   ├── __init__.py
│   ├── openai_client.py            # OpenAI client initialization
│   └── suggestions.py              # AI suggestions generation
├── ui/                             # Streamlit UI components
│   ├── __init__.py
│   ├── dashboard_tab.py           # Tab 1: Interactive Dashboard
│   ├── strategic_analysis_tab.py   # Tab 2: Strategic Analysis & AI
│   ├── rebalancing_tab.py          # Tab 3: Rebalancing Calculator
│   ├── risk_metrics_tab.py         # Tab 4: Risk Metrics
│   ├── news_sentiments_tab.py      # Tab 5: News and Sentiments
│   └── components.py                # Reusable UI components
└── config.py                       # Configuration constants and settings
```

## Module Breakdown

### 1. `config.py`
**Purpose:** Centralized configuration and constants

**Contains:**
- `GEOGRAPHY_KEYWORDS`
- `SECTOR_KEYWORDS`
- `CORR_DEBUG` (or better: session state management)
- File paths (CSV path)
- Default values (tax rates, risk-free rates, etc.)
- Model names list

**Lines:** ~150

---

### 2. `portfolio/classification.py`
**Purpose:** Classification logic for holdings

**Contains:**
- `classify_geography(name)` function
- `classify_sector(name)` function
- `unclassified_geography` and `unclassified_sector` tracking lists
- Classification helper functions

**Dependencies:** `config.py` (for keywords)

**Lines:** ~200

---

### 3. `portfolio/data_loader.py`
**Purpose:** Data loading and validation

**Contains:**
- `load_data()` function
- `validate_data(df)` function
- `enrich_data(df)` function (adds geography/sector columns)
- CSV path configuration

**Dependencies:** `portfolio.classification`, `config.py`

**Lines:** ~100

---

### 4. `portfolio/performance.py`
**Purpose:** Performance metrics calculations

**Contains:**
- `calculate_performance_metrics(df)` function
- Helper functions for performance calculations

**Dependencies:** None (pure calculations)

**Lines:** ~50

---

### 5. `portfolio/rebalancing.py`
**Purpose:** Rebalancing calculations

**Contains:**
- `calculate_rebalancing(df, target_allocations, ...)` function
- Rebalancing helper functions

**Dependencies:** None (pure calculations)

**Lines:** ~80

---

### 6. `portfolio/risk_metrics.py`
**Purpose:** Risk metric calculations

**Contains:**
- `calculate_sharpe_ratio(returns, ...)`
- `calculate_sortino_ratio(returns, ...)`
- `calculate_max_drawdown(returns)`
- `calculate_var(returns, confidence)`
- `calculate_portfolio_risk_metrics(df, ...)`
- Risk calculation helpers

**Dependencies:** None (pure calculations)

**Lines:** ~150

---

### 7. `portfolio/correlation.py`
**Purpose:** Correlation matrix calculations

**Contains:**
- `calculate_correlation_matrix(df)` function
- `fetch_prices(tickers_list, ...)` cached function
- Ticker mapping logic
- Correlation debug info management

**Dependencies:** `yfinance` (optional), `config.py`

**Lines:** ~150

---

### 8. `ai/openai_client.py`
**Purpose:** OpenAI client management

**Contains:**
- `get_openai_client()` function
- Client initialization logic
- API key management

**Dependencies:** `streamlit`, `openai`

**Lines:** ~50

---

### 9. `ai/suggestions.py`
**Purpose:** AI suggestions generation

**Contains:**
- `build_portfolio_summary(df)` function
- `generate_ai_suggestions(df, user_goals, risk_profile, ...)` function
- Prompt building logic

**Dependencies:** `ai.openai_client`, `portfolio.performance`

**Lines:** ~100

---

### 10. `ui/components.py`
**Purpose:** Reusable UI components

**Contains:**
- `render_classification_warnings(df, unclassified_geo, unclassified_sector)`
- `render_metrics_display(metrics)`
- `render_performance_charts(df)`
- Other reusable UI widgets

**Dependencies:** `streamlit`, `plotly`

**Lines:** ~200

---

### 11. `ui/dashboard_tab.py`
**Purpose:** Tab 1 - Interactive Dashboard UI

**Contains:**
- `render_dashboard_tab(df)` function
- All UI code for the dashboard tab

**Dependencies:** `portfolio.*`, `ui.components`, `streamlit`, `plotly`

**Lines:** ~200

---

### 12. `ui/strategic_analysis_tab.py`
**Purpose:** Tab 2 - Strategic Analysis & AI UI

**Contains:**
- `render_strategic_analysis_tab(df)` function
- AI suggestions UI
- Sector/geography analysis UI

**Dependencies:** `portfolio.*`, `ai.suggestions`, `ui.components`

**Lines:** ~200

---

### 13. `ui/rebalancing_tab.py`
**Purpose:** Tab 3 - Rebalancing Calculator UI

**Contains:**
- `render_rebalancing_tab(df)` function
- Rebalancing input forms
- Results display

**Dependencies:** `portfolio.rebalancing`, `ui.components`

**Lines:** ~250

---

### 14. `ui/risk_metrics_tab.py`
**Purpose:** Tab 4 - Risk Metrics UI

**Contains:**
- `render_risk_metrics_tab(df)` function
- Risk metrics display
- Correlation matrix visualization

**Dependencies:** `portfolio.risk_metrics`, `portfolio.correlation`, `ui.components`

**Lines:** ~300

---

### 15. `ui/news_sentiments_tab.py`
**Purpose:** Tab 5 - News and Sentiments UI

**Contains:**
- `render_news_sentiments_tab()` function
- News Sentinel UI code

**Dependencies:** `news_sentinel.*`, `ui.components`

**Lines:** ~200

---

### 16. `app.py` (Refactored)
**Purpose:** Main entry point - minimal orchestration

**Contains:**
- Streamlit page config
- Data loading and initialization
- Tab creation and routing
- Main app structure

**Dependencies:** All other modules

**Lines:** ~100-150

---

## Migration Strategy

### Phase 1: Extract Pure Functions (Low Risk)
1. Move `portfolio/risk_metrics.py` - pure calculations
2. Move `portfolio/performance.py` - pure calculations
3. Move `portfolio/rebalancing.py` - pure calculations

### Phase 2: Extract Data & Classification (Medium Risk)
4. Move `portfolio/classification.py`
5. Move `portfolio/data_loader.py`
6. Move `config.py`

### Phase 3: Extract AI Utilities (Medium Risk)
7. Move `ai/openai_client.py`
8. Move `ai/suggestions.py`

### Phase 4: Extract Correlation (Medium Risk - has dependencies)
9. Move `portfolio/correlation.py`

### Phase 5: Extract UI Components (Higher Risk - needs testing)
10. Move `ui/components.py`
11. Move `ui/dashboard_tab.py`
12. Move `ui/strategic_analysis_tab.py`
13. Move `ui/rebalancing_tab.py`
14. Move `ui/risk_metrics_tab.py`
15. Move `ui/news_sentiments_tab.py`

### Phase 6: Refactor Main App (Final)
16. Refactor `app.py` to use all modules

---

## Benefits

1. **Maintainability:** Each module has a single responsibility
2. **Testability:** Pure functions can be easily unit tested
3. **Reusability:** Functions can be imported and used elsewhere
4. **Readability:** Smaller files are easier to understand
5. **Collaboration:** Multiple developers can work on different modules
6. **Performance:** Easier to optimize specific modules

---

## Potential Challenges

1. **Circular Dependencies:** Need to be careful with import order
2. **Streamlit Caching:** `@st.cache_data` decorators need to stay in the right place
3. **Global State:** `CORR_DEBUG` and unclassified lists need proper management
4. **Testing:** Need to ensure all imports work correctly after refactoring

---

## Next Steps

1. Review this proposal
2. Start with Phase 1 (lowest risk)
3. Test after each phase
4. Gradually migrate remaining code

