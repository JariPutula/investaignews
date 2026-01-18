import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import os
import time

# Import portfolio calculation modules
from portfolio import (
    calculate_performance_metrics,
    calculate_rebalancing,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_portfolio_risk_metrics,
    load_data,
    enrich_data,
    get_unclassified_geography,
    get_unclassified_sector,
    reset_unclassified_tracking,
)

# Import historical performance modules
from portfolio.historical import (
    get_portfolio_value_over_time,
    get_historical_performance_summary,
    calculate_period_returns,
)
from portfolio.benchmarks import (
    DEFAULT_BENCHMARKS,
    list_available_benchmarks,
    get_recommended_benchmarks,
)
from config import DEFAULT_USER_NAME

# Debug container for correlation diagnostics
CORR_DEBUG = {}

# Page configuration
st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Classification configuration and functions moved to:
# - config.py (GEOGRAPHY_KEYWORDS, SECTOR_KEYWORDS)
# - portfolio/classification.py (classify_geography, classify_sector)
# - portfolio/data_loader.py (load_data, enrich_data)
# Imported at top of file

# Performance, rebalancing, and risk metrics functions moved to portfolio/ module
# Imported at top of file

def calculate_correlation_matrix(df):
    """
    Calculate correlation matrix for holdings
    
    Note: This is a simplified version using change_pct as proxy for returns.
    For accurate correlations, historical price data would be needed.
    
    Parameters:
    - df: DataFrame with holdings data
    
    Returns:
    - Correlation matrix DataFrame
    """
    # Try to compute Pearson correlations from historical returns when tickers exist.
    # Fall back to a simple sector/geography proxy for missing tickers or failures.
    holdings = df[['name', 'change_pct', 'sector', 'geography']].copy()
    # Optional overrides for mapping holding name -> yfinance ticker
    TICKER_OVERRIDES = {
        'ABBVIE': 'ABBV',
        'ALPHABET': 'GOOG',
        'AMAZON': 'AMZN',
        'MICROSOFT': 'MSFT',
        'QUALCOMM': 'QCOM',
        'PFIZER': 'PFE',
        'MERCK': 'MRK',
        'NOKIA': 'NOK',
        'TESLA': 'TSLA',
        'WALT DISNEY': 'DIS',
    }

    def infer_ticker(name):
        # Use explicit ticker column if present
        if 'ticker' in df.columns:
            row = df[df['name'] == name]
            if not row.empty:
                val = row.iloc[0].get('ticker')
                if pd.notna(val):
                    s = str(val).strip()
                    if s == '' or s.upper() in ['NONE', 'N/A', 'NA', 'UNKNOWN']:
                        return None
                    return s

        name_up = str(name).upper()
        for key, val in TICKER_OVERRIDES.items():
            if key in name_up:
                return val

        return None

    name_to_ticker = {name: infer_ticker(name) for name in holdings['name']}
    tickers = [t for t in set(name_to_ticker.values()) if t]

    @st.cache_data(ttl=60 * 60 * 24)
    def fetch_prices(tickers_list, period='1y', interval='1d'):
        try:
            import yfinance as yf
        except Exception:
            return pd.DataFrame()

        success = []
        failed = []
        series_list = []

        for t in tickers_list:
            try:
                # Suppress stderr temporarily to avoid HTTP 404 error messages from yfinance
                import sys
                import io
                from contextlib import redirect_stderr
                
                # Capture stderr to suppress HTTP error messages (404, etc.)
                stderr_capture = io.StringIO()
                with redirect_stderr(stderr_capture):
                    data = yf.download(t, period=period, interval=interval, progress=False, threads=False, auto_adjust=True)
                
                if data is None or data.empty:
                    failed.append(t)
                    continue

                # Prefer 'Adj Close'
                if 'Adj Close' in data:
                    s = data['Adj Close']
                else:
                    # Fallback to first numeric column
                    s = data.iloc[:, 0]

                # Ensure Series and name it by ticker
                if isinstance(s, pd.Series):
                    s = s.rename(t)
                    series_list.append(s)
                    success.append(t)
                else:
                    failed.append(t)
            except Exception as e:
                # Track failed tickers (will be displayed to user)
                failed.append(t)

        if len(series_list) == 0:
            CORR_DEBUG['fetch_success'] = success
            CORR_DEBUG['fetch_failed'] = failed
            return pd.DataFrame()

        prices = pd.concat(series_list, axis=1)
        CORR_DEBUG['fetch_success'] = success
        CORR_DEBUG['fetch_failed'] = failed
        return prices

    corr_df = None
    if len(tickers) >= 2:
        prices = fetch_prices(tickers, period='1y', interval='1d')
        if not prices.empty:
            returns = prices.pct_change().dropna(how='all')
            try:
                # Filter out tickers with insufficient return observations
                min_returns = 50
                counts = returns.count()
                good_tickers = counts[counts >= min_returns].index.tolist()
                excluded = [c for c in returns.columns if c not in good_tickers]
                CORR_DEBUG['returns_count'] = counts.to_dict()
                CORR_DEBUG['excluded_due_to_insufficient_data'] = excluded

                if len(good_tickers) >= 2:
                    returns_filtered = returns[good_tickers]
                    corr_df = returns_filtered.corr()
                else:
                    corr_df = None
            except Exception:
                corr_df = None

    # Store debug info in module-level container for display in the UI
    try:
        CORR_DEBUG['detected_tickers'] = tickers
        CORR_DEBUG['tickers_with_price_data'] = list(corr_df.columns) if corr_df is not None else []
    except Exception:
        CORR_DEBUG['detected_tickers'] = []
        CORR_DEBUG['tickers_with_price_data'] = []

    # Include ALL holdings in correlation matrix
    # Use real ticker-based correlations when available, fallback to proxy for others
    valid_holdings_indices = [i for i, name in enumerate(holdings['name']) if name_to_ticker.get(name) is not None]
    CORR_DEBUG['excluded_unmapped_holdings'] = [holdings.iloc[i]['name'] for i in range(len(holdings)) if i not in valid_holdings_indices]
    
    # Use all holdings for the correlation matrix (not just those with tickers)
    all_holdings = holdings.reset_index(drop=True)

    correlation_data = []
    for i, row1 in all_holdings.iterrows():
        row_correlations = []
        for j, row2 in all_holdings.iterrows():
            if i == j:
                corr = 1.0
            else:
                name1 = row1['name']
                name2 = row2['name']
                t1 = name_to_ticker.get(name1)
                t2 = name_to_ticker.get(name2)

                used_real = False
                if corr_df is not None and t1 is not None and t2 is not None and t1 in corr_df.columns and t2 in corr_df.columns:
                    try:
                        corr = float(corr_df.loc[t1, t2])
                        if pd.isna(corr):
                            raise ValueError
                        used_real = True
                    except Exception:
                        corr = None

                if not used_real:
                    # Fallback: use sector/geography proxy for all holdings without real correlation data
                    sector_match = 1.0 if row1['sector'] == row2['sector'] else 0.3
                    geo_match = 1.0 if row1['geography'] == row2['geography'] else 0.5
                    corr = (sector_match + geo_match) / 2

            try:
                corr = max(-1.0, min(1.0, float(corr)))
            except Exception:
                corr = 0.0

            row_correlations.append(corr)
        correlation_data.append(row_correlations)
    
    corr_matrix = pd.DataFrame(
        correlation_data,
        index=all_holdings['name'],
        columns=all_holdings['name']
    )
    
    return corr_matrix

# Portfolio risk metrics function moved to portfolio/risk_metrics.py
# Imported at top of file

# =========================
# AI Suggestions Utilities
# =========================
def build_portfolio_summary(df):
    """Create a compact textual summary of the portfolio for prompting an LLM."""
    total_value = df['market_total_eur'].sum()
    geo = (df.groupby('geography')['market_total_eur'].sum() / total_value * 100).round(1)
    sector = (df.groupby('sector')['market_total_eur'].sum() / total_value * 100).round(1).sort_values(ascending=False)
    top_holdings = df.nlargest(8, 'market_total_eur')[['name', 'market_total_eur', 'change_pct']]

    lines = []
    lines.append(f"Total portfolio value: €{total_value:,.2f}")
    lines.append("Geographic allocation (%): " + ", ".join([f"{k}: {v:.1f}%" for k, v in geo.items()]))
    lines.append("Sector allocation (%): " + ", ".join([f"{k}: {v:.1f}%" for k, v in sector.items()]))
    lines.append("Top holdings:")
    for _, row in top_holdings.iterrows():
        lines.append(f"- {row['name']}: €{row['market_total_eur']:,.0f}, Change: {row['change_pct']:.1f}%")
    return "\n".join(lines)

def get_openai_client():
    """Initialize OpenAI client from environment or Streamlit secrets; return None if unavailable."""
    # Prefer env first so UI-entered key works without requiring secrets.toml
    api_key = os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)  # st.secrets may not exist/configured
        except Exception:
            api_key = None
    try:
        # Prefer new SDK if available
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        return client, api_key
    except Exception:
        try:
            # Fallback to legacy
            import openai  # type: ignore
            openai.api_key = api_key
            return openai, api_key
        except Exception:
            return None, api_key

def generate_ai_suggestions(df, user_goals, risk_profile, model_name="gpt-4o-mini"):
    """Generate dynamic AI suggestions using OpenAI based on portfolio context."""
    client, api_key = get_openai_client()
    if not api_key:
        return None, "Missing API key. Provide OPENAI_API_KEY via Secrets or Environment."
    if client is None:
        return None, "OpenAI SDK not installed. Please install 'openai' package."

    portfolio_summary = build_portfolio_summary(df)

    system_prompt = (
        "You are an expert investment analyst. Provide specific, actionable, and risk-aware portfolio suggestions. "
        "Focus on diversification, concentration risk, sector/geography balance, and pragmatic next steps. "
        "Avoid giving tax or legal advice beyond general considerations. Currency is EUR."
    )

    user_prompt = f"""
Portfolio Summary:
{portfolio_summary}

Investor context:
- Goals: {user_goals or "Not specified"}
- Risk profile: {risk_profile}

Requirements:
- Provide 4-8 concise suggestions grouped by themes (Diversification, Risk, Rebalancing, Opportunities).
- Include rationale and concrete actions (what to buy/sell/hold, approximate % weights or € amounts).
- Consider current concentrations and recent performance.
- Keep it tailored to the provided portfolio; avoid generic lists.
"""

    try:
        # New SDK style
        from openai import OpenAI  # type: ignore
        if isinstance(client, OpenAI):
            resp = client.chat.completions.create(
                model=model_name,
                temperature=1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content
            return content, None
        else:
            # Legacy SDK
            resp = client.ChatCompletion.create(
                model=model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp["choices"][0]["message"]["content"]
            return content, None
    except Exception as e:
        return None, f"OpenAI error: {e}"

# Load and enrich data
df = load_data()
df = enrich_data(df)  # Adds geography and sector columns

# Main app
st.title("📊 Portfolio Analytics Dashboard")

# Display warnings for unclassified items
unclassified_geo = get_unclassified_geography()
unclassified_sect = get_unclassified_sector()
if unclassified_geo or unclassified_sect:
    with st.expander("⚠️ Classification Warnings - Action Required", expanded=True):
        st.warning("**Some holdings could not be automatically classified. Please review and update the classification rules.**")
        
        if unclassified_geo:
            st.write("**Unclassified Geography:**")
            unclassified_geo_df = df[df['name'].isin(unclassified_geo)][['name', 'geography', 'market_total_eur']].copy()
            st.dataframe(unclassified_geo_df, width='stretch', hide_index=True)
            st.info("💡 **Tip:** Add keywords for these holdings to `GEOGRAPHY_KEYWORDS` in `config.py` to improve classification.")
        
        if unclassified_sect:
            st.write("**Unclassified Sector:**")
            unclassified_sector_df = df[df['name'].isin(unclassified_sect)][['name', 'sector', 'market_total_eur']].copy()
            st.dataframe(unclassified_sector_df, width='stretch', hide_index=True)
            st.info("💡 **Tip:** Add keywords for these holdings to `SECTOR_KEYWORDS` in `config.py` to improve classification.")
        
        st.markdown("""
        **How to fix:**
        1. Review the unclassified holdings above
        2. Open `config.py` and locate the `GEOGRAPHY_KEYWORDS` or `SECTOR_KEYWORDS` dictionaries
        3. Add the missing company names or keywords to the appropriate category
        4. Restart the application
        """)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Interactive Dashboard",
    "🤖 Strategic Analysis & AI Suggestions",
    "⚖️ Rebalancing Calculator",
    "🎲 Risk Metrics",
    "📰 News and Sentiments",
    "📊 Historical Performance"
])

with tab1:
    st.header("Portfolio Overview")
    
    # Calculate metrics
    metrics = calculate_performance_metrics(df)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Market Value",
            value=f"€{metrics['total_market_value']:,.2f}"
        )
    
    with col2:
        gain_color = "normal" if metrics['total_unrealized_gain'] >= 0 else "inverse"
        st.metric(
            label="Total Unrealized Gain",
            value=f"€{metrics['total_unrealized_gain']:,.2f}",
            delta=f"{metrics['overall_performance_pct']:.2f}%"
        )
    
    with col3:
        performance_color = "normal" if metrics['overall_performance_pct'] >= 0 else "inverse"
        st.metric(
            label="Overall Performance",
            value=f"{metrics['overall_performance_pct']:.2f}%"
        )
    
    st.divider()
    
    # Concentration Charts
    st.header("Concentration Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Geographic Breakdown")
        
        # Calculate geographic distribution
        geo_dist = df.groupby('geography')['market_total_eur'].sum().reset_index()
        geo_dist['percentage'] = (geo_dist['market_total_eur'] / geo_dist['market_total_eur'].sum()) * 100
        
        # Create pie chart
        fig_geo = px.pie(
            geo_dist,
            values='market_total_eur',
            names='geography',
            title='Portfolio by Geography',
            color_discrete_map={
                'U.S.': '#1f77b4',
                'Finland': '#2ca02c',
                'Europe/Global': '#ff7f0e',
                'Other': '#d62728'
            }
        )
        fig_geo.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_geo, width='stretch')
        
        # Display percentages
        st.write("**Breakdown:**")
        for _, row in geo_dist.iterrows():
            st.write(f"- {row['geography']}: {row['percentage']:.2f}% (€{row['market_total_eur']:,.2f})")
    
    with col2:
        st.subheader("Top 10 Holdings")
        
        # Get top 10 holdings by market value
        top_10 = df.nlargest(10, 'market_total_eur')[['name', 'market_total_eur']].copy()
        top_10 = top_10.sort_values('market_total_eur', ascending=True)
        
        # Create horizontal bar chart
        fig_top10 = px.bar(
            top_10,
            x='market_total_eur',
            y='name',
            orientation='h',
            title='Top 10 Holdings by Market Value',
            labels={'market_total_eur': 'Market Value (€)', 'name': 'Holding'},
            color='market_total_eur',
            color_continuous_scale='Blues'
        )
        fig_top10.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_top10, width='stretch')
    
    st.divider()
    
    # Performance Waterfall
    st.header("Performance Analysis")
    
    # Get top 5 gainers and bottom 5 losers
    top_gainers = df.nlargest(5, 'change_eur')[['name', 'change_eur', 'change_pct']].copy()
    bottom_losers = df.nsmallest(5, 'change_eur')[['name', 'change_eur', 'change_pct']].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Gainers")
        
        fig_gainers = px.bar(
            top_gainers,
            x='change_eur',
            y='name',
            orientation='h',
            title='Top 5 Gainers by Change (€)',
            labels={'change_eur': 'Gain (€)', 'name': 'Holding'},
            color='change_eur',
            color_continuous_scale='Greens'
        )
        fig_gainers.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_gainers, width='stretch')
        
        # Display table
        st.write("**Details:**")
        display_gainers = top_gainers.copy()
        display_gainers.columns = ['Holding', 'Gain (€)', 'Gain (%)']
        display_gainers['Gain (€)'] = display_gainers['Gain (€)'].apply(lambda x: f"€{x:,.2f}")
        display_gainers['Gain (%)'] = display_gainers['Gain (%)'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_gainers, width='stretch', hide_index=True)
    
    with col2:
        st.subheader("Bottom 5 Losers")
        
        fig_losers = px.bar(
            bottom_losers,
            x='change_eur',
            y='name',
            orientation='h',
            title='Bottom 5 Losers by Change (€)',
            labels={'change_eur': 'Loss (€)', 'name': 'Holding'},
            color='change_eur',
            color_continuous_scale='Reds'
        )
        fig_losers.update_layout(
            yaxis={'categoryorder': 'total descending'},
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_losers, width='stretch')
        
        # Display table
        st.write("**Details:**")
        display_losers = bottom_losers.copy()
        display_losers.columns = ['Holding', 'Loss (€)', 'Loss (%)']
        display_losers['Loss (€)'] = display_losers['Loss (€)'].apply(lambda x: f"€{x:,.2f}")
        display_losers['Loss (%)'] = display_losers['Loss (%)'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_losers, width='stretch', hide_index=True)

with tab2:
    st.header("Strategic Analysis & AI-Augmented Suggestions")
    
    # Calculate current portfolio metrics
    metrics = calculate_performance_metrics(df)
    geo_dist = df.groupby('geography')['market_total_eur'].sum()
    total_value = metrics['total_market_value']
    
    st.subheader("Current Portfolio Composition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Geographic Distribution:**")
        for geo, value in geo_dist.items():
            pct = (value / total_value) * 100
            st.write(f"- {geo}: {pct:.2f}%")
    
    with col2:
        # Calculate concentration risk
        top_5_concentration = df.nlargest(5, 'market_total_eur')['market_total_eur'].sum() / total_value * 100
        st.write("**Concentration Metrics:**")
        st.write(f"- Top 5 Holdings: {top_5_concentration:.2f}% of portfolio")
        st.write(f"- Number of Holdings: {len(df)}")
    
    st.divider()
    
    # Sector Analysis
    st.subheader("Sector Analysis")
    
    sector_dist = df.groupby('sector')['market_total_eur'].sum().reset_index()
    sector_dist['percentage'] = (sector_dist['market_total_eur'] / sector_dist['market_total_eur'].sum()) * 100
    sector_dist = sector_dist.sort_values('market_total_eur', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector distribution chart
        fig_sector = px.bar(
            sector_dist,
            x='sector',
            y='percentage',
            title='Portfolio by Sector (%)',
            labels={'sector': 'Sector', 'percentage': 'Percentage (%)'},
            color='percentage',
            color_continuous_scale='Viridis'
        )
        fig_sector.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_sector, width='stretch')
    
    with col2:
        st.write("**Sector Breakdown:**")
        for _, row in sector_dist.iterrows():
            st.write(f"- **{row['sector']}**: {row['percentage']:.2f}% (€{row['market_total_eur']:,.2f})")
    
    st.divider()
    
    
    # ------------------------------
    # AI-Augmented Dynamic Suggestions
    # ------------------------------
    st.divider()
    st.subheader("🧠 AI-Generated Strategic Suggestions")

    with st.expander("Configure AI and Generate Suggestions", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            user_goals = st.text_area(
                "Your investment goals (optional)",
                placeholder="e.g., Long-term growth, moderate volatility, dividend income, buy a house in 7 years...",
                help="Provide any context you want the AI to consider"
            )
        with col2:
            risk_profile = st.selectbox(
                "Risk profile",
                options=["Conservative", "Moderate", "Aggressive"],
                index=1
            )

        colk1, colk2 = st.columns([2, 1])
        with colk1:
            api_key_input = st.text_input(
                "OpenAI API Key (not stored; used for this session)",
                type="password",
                placeholder="sk-...",
                help="Alternatively, set OPENAI_API_KEY in Streamlit Secrets or environment."
            )
        with colk2:
            model_name = st.selectbox(
                "Model",
                options=["gpt-5", "gpt-5-mini", "gpt-4.1-mini"],
                index=0
            )

        # Streaming option: simulate progressively rendering the AI output
        streaming_output = st.checkbox(
            "Stream AI output (simulate typing)",
            value=False,
            help="Display AI suggestions gradually as they arrive (simulated)."
            )

        # If user provides a key here, prefer it for this session
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

        if st.button("Generate AI Suggestions", type="primary"):
            # Progress UI elements (avoid using global spinner which grays out the whole app)
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: prepare context
            status_text.info("Preparing portfolio context for the AI...")
            progress_bar.progress(20)

            # Step 2: call the AI (this may take a while)
            status_text.info("Calling OpenAI and waiting for the response...")
            progress_bar.progress(50)
            
            ai_text, ai_err = generate_ai_suggestions(df, user_goals, risk_profile, model_name=model_name)

            # Step 3: render results
            progress_bar.progress(100)
            if ai_err:
                status_text.error("OpenAI returned an error.")
                st.error(ai_err)
            else:
                # Optionally stream the AI output to make the UI feel more responsive
                if streaming_output and ai_text:
                    status_text.info("Streaming AI output...")
                    container = st.empty()
                    # Render in chunks to simulate streaming
                    chunk_size = 120
                    for i in range(0, len(ai_text), chunk_size):
                        chunk = ai_text[: i + chunk_size]
                        container.markdown(chunk)
                        time.sleep(0.04)
                    status_text.success("AI suggestions generated successfully.")
                else:
                    status_text.success("AI suggestions generated successfully.")
                    st.markdown(ai_text)

            # Keep final status visible but remove the progress bar to avoid clutter
            progress_bar.empty()

with tab3:
    st.header("⚖️ Portfolio Rebalancing Calculator")
    
    st.markdown("""
    This tool helps you calculate exactly how much to buy or sell to reach your target portfolio allocation.
    Set your target allocations below, and we'll calculate the required transactions.
    """)
    
    # Get current sector distribution
    current_sector_dist = df.groupby('sector')['market_total_eur'].sum()
    total_value = df['market_total_eur'].sum()
    current_sector_pct = (current_sector_dist / total_value * 100).round(2)
    
    # Display current allocations
    st.subheader("Current Portfolio Allocation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Sector Allocation:**")
        for sector, pct in current_sector_pct.items():
            st.write(f"- {sector}: {pct:.2f}% (€{current_sector_dist[sector]:,.2f})")
    
    with col2:
        # Show current allocation chart
        fig_current = px.pie(
            values=current_sector_dist.values,
            names=current_sector_dist.index,
            title='Current Allocation',
            hole=0.3
        )
        st.plotly_chart(fig_current, width='stretch')
    
    st.divider()
    
    # Target allocation inputs
    st.subheader("Set Target Allocations")
    
    st.info("💡 **Tip:** Target allocations should sum to 100%. The calculator will normalize if needed.")
    
    # Get unique sectors
    unique_sectors = sorted(df['sector'].unique())
    
    # Create input columns for target allocations
    target_allocations = {}
    
    # Use columns for better layout
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, sector in enumerate(unique_sectors):
        col_idx = idx % num_cols
        with cols[col_idx]:
            current_pct = current_sector_pct.get(sector, 0)
            target_pct = st.number_input(
                f"{sector}",
                min_value=0.0,
                max_value=100.0,
                value=float(current_pct),
                step=0.5,
                help=f"Current: {current_pct:.2f}%"
            )
            target_allocations[sector] = target_pct
    
    # Settings
    st.subheader("Rebalancing Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_cost_pct = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            help="Percentage cost per transaction (e.g., 0.1% for broker fees)"
        )
    
    with col2:
        tax_rate = st.number_input(
            "Capital Gains Tax Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=30.0,
            step=1.0,
            help="Tax rate on capital gains (Finland default: 30%)"
        ) / 100
    
    # Calculate rebalancing
    if st.button("Calculate Rebalancing", type="primary"):
        # Normalize target allocations to sum to 100%
        total_target = sum(target_allocations.values())
        if total_target > 0:
            target_allocations_normalized = {k: (v / total_target * 100) for k, v in target_allocations.items()}
        else:
            target_allocations_normalized = target_allocations
        
        # Calculate rebalancing recommendations
        rebalancing_df = calculate_rebalancing(
            df, 
            target_allocations_normalized, 
            transaction_cost_pct=transaction_cost_pct,
            tax_rate=tax_rate
        )
        
        st.divider()
        st.subheader("📊 Rebalancing Recommendations")
        
        # Summary metrics
        total_buy = rebalancing_df[rebalancing_df['action'] == 'BUY']['difference_eur'].sum()
        total_sell = abs(rebalancing_df[rebalancing_df['action'] == 'SELL']['difference_eur'].sum())
        total_transaction_cost = rebalancing_df['transaction_cost_eur'].sum()
        total_tax = rebalancing_df['tax_implication_eur'].sum()
        total_cost = rebalancing_df['total_cost_eur'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total to Buy", f"€{total_buy:,.2f}")
        with col2:
            st.metric("Total to Sell", f"€{total_sell:,.2f}")
        with col3:
            st.metric("Transaction Costs", f"€{total_transaction_cost:,.2f}")
        with col4:
            st.metric("Tax Implications", f"€{total_tax:,.2f}")
        
        st.info(f"💰 **Total Rebalancing Cost:** €{total_cost:,.2f}")
        
        # Display rebalancing table
        st.write("**Detailed Rebalancing Plan:**")
        
        # Format the dataframe for display
        display_df = rebalancing_df.copy()
        display_df.columns = ['Sector', 'Current %', 'Target %', 'Current Value (€)', 
                             'Target Value (€)', 'Difference (€)', 'Action', 
                             'Transaction Cost (€)', 'Tax (€)', 'Total Cost (€)']
        
        # Format currency columns
        for col in ['Current Value (€)', 'Target Value (€)', 'Difference (€)', 
                   'Transaction Cost (€)', 'Tax (€)', 'Total Cost (€)']:
            display_df[col] = display_df[col].apply(lambda x: f"€{x:,.2f}")
        
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        # Visual comparison
        st.subheader("Current vs Target Allocation")
        
        comparison_data = []
        for sector in unique_sectors:
            current = current_sector_pct.get(sector, 0)
            target = target_allocations_normalized.get(sector, 0)
            comparison_data.append({
                'Sector': sector,
                'Current %': current,
                'Target %': target
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Current %',
            x=comparison_df['Sector'],
            y=comparison_df['Current %'],
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Target %',
            x=comparison_df['Sector'],
            y=comparison_df['Target %'],
            marker_color='lightgreen'
        ))
        
        fig_comparison.update_layout(
            title='Current vs Target Allocation Comparison',
            xaxis_title='Sector',
            yaxis_title='Allocation (%)',
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_comparison, width='stretch')
        
        # Step-by-step instructions
        st.subheader("📋 Step-by-Step Rebalancing Instructions")
        
        buy_actions = rebalancing_df[rebalancing_df['action'] == 'BUY'].sort_values('difference_eur', ascending=False)
        sell_actions = rebalancing_df[rebalancing_df['action'] == 'SELL'].sort_values('difference_eur')
        
        if len(sell_actions) > 0:
            st.write("**Step 1: Sell Holdings**")
            for idx, row in sell_actions.iterrows():
                st.write(f"1. Sell **€{abs(row['difference_eur']):,.2f}** worth of **{row['sector']}** holdings")
                st.write(f"   - Current: {row['current_%']:.2f}% → Target: {row['target_%']:.2f}%")
                st.write(f"   - Estimated tax: €{row['tax_implication_eur']:,.2f}")
        
        if len(buy_actions) > 0:
            st.write("**Step 2: Buy Holdings**")
            for idx, row in buy_actions.iterrows():
                st.write(f"1. Buy **€{row['difference_eur']:,.2f}** worth of **{row['sector']}** holdings")
                st.write(f"   - Current: {row['current_%']:.2f}% → Target: {row['target_%']:.2f}%")
        
        # Export option
        st.divider()
        st.subheader("💾 Export Rebalancing Plan")
        
        csv = rebalancing_df.to_csv(index=False)
        st.download_button(
            label="Download Rebalancing Plan as CSV",
            data=csv,
            file_name="rebalancing_plan.csv",
            mime="text/csv"
        )
        
        st.info("💡 **Tip:** Review the rebalancing plan carefully. Consider executing in phases to minimize market impact and transaction costs.")

with tab4:
    st.header("🎲 Enhanced Risk Metrics")
    
    st.markdown("""
    This section provides comprehensive risk analysis for your portfolio, including risk-adjusted returns,
    downside risk measures, and concentration metrics.
    """)
    
    # Settings
    st.subheader("Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Annual risk-free rate (e.g., government bond yield)"
        ) / 100
    
    with col2:
        st.info("💡 **Note:** Some metrics (Beta, accurate correlations) require historical price data. Current calculations use available data as proxies.")
    
    # Calculate risk metrics
    risk_metrics = calculate_portfolio_risk_metrics(df, risk_free_rate=risk_free_rate)
    
    st.divider()
    
    # Display key risk metrics
    st.subheader("📊 Key Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sharpe = risk_metrics['sharpe_ratio']
        sharpe_color = "normal" if sharpe > 1 else "off" if sharpe > 0 else "inverse"
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            help="Risk-adjusted return. >1 is good, >2 is excellent"
        )
        if sharpe > 2:
            st.success("Excellent risk-adjusted returns")
        elif sharpe > 1:
            st.info("Good risk-adjusted returns")
        elif sharpe > 0:
            st.warning("Moderate risk-adjusted returns")
        else:
            st.error("Poor risk-adjusted returns")
    
    with col2:
        sortino = risk_metrics['sortino_ratio']
        sortino_color = "normal" if sortino > 1 else "off" if sortino > 0 else "inverse"
        st.metric(
            "Sortino Ratio",
            f"{sortino:.2f}",
            help="Downside risk-adjusted return. Focuses on negative volatility only"
        )
        if sortino > 2:
            st.success("Excellent downside protection")
        elif sortino > 1:
            st.info("Good downside protection")
        else:
            st.warning("Limited downside protection")
    
    with col3:
        max_dd = risk_metrics['max_drawdown']
        st.metric(
            "Maximum Drawdown",
            f"{max_dd:.2f}%",
            help="Largest peak-to-trough decline"
        )
        if abs(max_dd) < 10:
            st.success("Low drawdown risk")
        elif abs(max_dd) < 20:
            st.info("Moderate drawdown risk")
        else:
            st.warning("High drawdown risk")
    
    with col4:
        beta = risk_metrics['beta']
        st.metric(
            "Beta",
            f"{beta:.2f}",
            help="Portfolio sensitivity vs market (1.0 = market average). Requires historical data for accuracy"
        )
        if beta < 0.8:
            st.info("Lower volatility than market")
        elif beta > 1.2:
            st.warning("Higher volatility than market")
        else:
            st.success("Similar volatility to market")
    
    st.divider()
    
    # Value at Risk
    st.subheader("💸 Value at Risk (VaR)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_95 = risk_metrics['var_95']
        st.metric(
            "VaR (95% Confidence)",
            f"{var_95:.2f}%",
            help="Maximum expected loss with 95% confidence"
        )
        st.info(f"With 95% confidence, your portfolio is not expected to lose more than {var_95:.2f}% in a given period.")
    
    with col2:
        var_99 = risk_metrics['var_99']
        st.metric(
            "VaR (99% Confidence)",
            f"{var_99:.2f}%",
            help="Maximum expected loss with 99% confidence"
        )
        st.info(f"With 99% confidence, your portfolio is not expected to lose more than {var_99:.2f}% in a given period.")
    
    # Calculate portfolio value at risk in euros
    total_value = df['market_total_eur'].sum()
    var_95_eur = total_value * (var_95 / 100)
    var_99_eur = total_value * (var_99 / 100)
    
    st.write("**Portfolio Value at Risk:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- 95% VaR: **€{var_95_eur:,.2f}**")
    with col2:
        st.write(f"- 99% VaR: **€{var_99_eur:,.2f}**")
    
    st.divider()
    
    # Portfolio Statistics
    st.subheader("📈 Portfolio Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Portfolio Return",
            f"{risk_metrics['portfolio_return']:.2f}%",
            help="Weighted average return of portfolio holdings"
        )
    
    with col2:
        st.metric(
            "Portfolio Volatility",
            f"{risk_metrics['portfolio_volatility']:.2f}%",
            help="Standard deviation of portfolio returns"
        )
    
    with col3:
        # Calculate risk-return ratio
        risk_return_ratio = risk_metrics['portfolio_return'] / risk_metrics['portfolio_volatility'] if risk_metrics['portfolio_volatility'] > 0 else 0
        st.metric(
            "Risk-Return Ratio",
            f"{risk_return_ratio:.2f}",
            help="Return per unit of risk"
        )
    
    st.divider()
    
    # Concentration Risk
    st.subheader("🎯 Concentration Risk Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_5 = risk_metrics['top_5_concentration']
        st.metric(
            "Top 5 Holdings Concentration",
            f"{top_5:.2f}%",
            help="Percentage of portfolio in top 5 holdings"
        )
        if top_5 > 50:
            st.warning("High concentration - consider diversifying")
        elif top_5 > 30:
            st.info("Moderate concentration")
        else:
            st.success("Well-diversified")
    
    with col2:
        top_10 = risk_metrics['top_10_concentration']
        st.metric(
            "Top 10 Holdings Concentration",
            f"{top_10:.2f}%",
            help="Percentage of portfolio in top 10 holdings"
        )
    
    with col3:
        hhi = risk_metrics['herfindahl_index']
        st.metric(
            "Herfindahl-Hirschman Index (HHI)",
            f"{hhi:.0f}",
            help="Concentration measure: <1500 = competitive, 1500-2500 = moderate, >2500 = concentrated"
        )
        if hhi < 1500:
            st.success("Well-diversified portfolio")
        elif hhi < 2500:
            st.info("Moderately concentrated")
        else:
            st.warning("Highly concentrated portfolio")
    
    # Display top holdings for context
    st.write("**Top 10 Holdings by Value:**")
    top_holdings = df.nlargest(10, 'market_total_eur')[['name', 'market_total_eur', 'sector', 'geography']].copy()
    top_holdings['percentage'] = (top_holdings['market_total_eur'] / total_value * 100).round(2)
    top_holdings.columns = ['Holding', 'Value (€)', 'Sector', 'Geography', 'Portfolio %']
    top_holdings['Value (€)'] = top_holdings['Value (€)'].apply(lambda x: f"€{x:,.2f}")
    st.dataframe(top_holdings, width='stretch', hide_index=True)
    
    st.divider()
    
    # Correlation Matrix
    st.subheader("🔗 Correlation Analysis")
    
    st.info("💡 **Note:** This correlation matrix is a simplified proxy based on sector and geography similarity. For accurate correlations, historical price data would be needed.")
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(df)
    
    # Display failed tickers prominently if any
    try:
        dbg = CORR_DEBUG
        failed_tickers = dbg.get('fetch_failed', [])
        if failed_tickers:
            st.warning(
                f"⚠️ **Tickers not found on Yahoo Finance:** {', '.join(failed_tickers)}\n\n"
                f"These tickers were excluded from the correlation matrix calculation. "
                f"This may be due to incorrect ticker symbols, delisted stocks, or data unavailability."
            )
    except Exception:
        pass
    
    # Show debug diagnostics about ticker detection / price fetch
    try:
        dbg = CORR_DEBUG
        with st.expander("Correlation debug info", expanded=False):
            for key, val in dbg.items():
                if isinstance(val, list) and len(val) > 20:
                    st.write(f"**{key}** ({len(val)} items): {val[:5]} ... {val[-5:]}")
                elif isinstance(val, dict):
                    st.write(f"**{key}**: {len(val)} entries")
                    with st.expander(f"  Show {key}"):
                        for k, v in val.items():
                            st.write(f"  {k}: {v}")
                else:
                    st.write(f"**{key}**: {val}")
    except Exception as e:
        st.write(f"Debug error: {e}")
    
    # Create heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0.5,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig_corr.update_layout(
        title='Portfolio Correlation Matrix (Proxy)',
        xaxis_title='Holdings',
        yaxis_title='Holdings',
        height=600,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_corr, width='stretch')
    
    # Correlation insights
    st.write("**Correlation Insights:**")
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > 0.7:
                high_corr_pairs.append({
                    'Holding 1': corr_matrix.columns[i],
                    'Holding 2': corr_matrix.columns[j],
                    'Correlation': f"{corr_value:.2f}"
                })
    
    if high_corr_pairs:
        st.warning("**High Correlation Pairs (>0.7):**")
        high_corr_df = pd.DataFrame(high_corr_pairs)
        st.dataframe(high_corr_df, width='stretch', hide_index=True)
        st.info("💡 High correlation means these holdings tend to move together, reducing diversification benefits.")
    else:
        st.success("✅ No extremely high correlations detected. Your portfolio shows good diversification.")
    
    st.divider()
    
    # Risk Summary
    st.subheader("📋 Risk Summary & Recommendations")
    
    # Generate risk assessment
    risk_level = "Low"
    risk_issues = []
    risk_strengths = []
    
    if abs(max_dd) > 20:
        risk_level = "High"
        risk_issues.append("High maximum drawdown risk")
    elif abs(max_dd) > 10:
        risk_level = "Medium"
        risk_issues.append("Moderate drawdown risk")
    else:
        risk_strengths.append("Low drawdown risk")
    
    if top_5 > 50:
        risk_level = "High" if risk_level == "Low" else risk_level
        risk_issues.append("High concentration in top 5 holdings")
    elif top_5 < 30:
        risk_strengths.append("Good diversification")
    
    if sharpe < 0:
        risk_level = "High" if risk_level == "Low" else risk_level
        risk_issues.append("Negative risk-adjusted returns")
    elif sharpe > 1:
        risk_strengths.append("Good risk-adjusted returns")
    
    if hhi > 2500:
        risk_level = "High" if risk_level == "Low" else risk_level
        risk_issues.append("High concentration (HHI > 2500)")
    
    # Display risk level
    if risk_level == "Low":
        st.success(f"**Overall Risk Level: {risk_level}** ✅")
    elif risk_level == "Medium":
        st.info(f"**Overall Risk Level: {risk_level}** ⚠️")
    else:
        st.error(f"**Overall Risk Level: {risk_level}** 🚨")
    
    if risk_strengths:
        st.write("**Portfolio Strengths:**")
        for strength in risk_strengths:
            st.success(f"✅ {strength}")
    
    if risk_issues:
        st.write("**Areas of Concern:**")
        for issue in risk_issues:
            st.warning(f"⚠️ {issue}")
    
    # Recommendations
    if risk_issues:
        st.write("**Recommendations:**")
        if "High concentration" in str(risk_issues):
            st.info("💡 Consider diversifying by reducing positions in top holdings and spreading across more positions or ETFs.")
        if "High maximum drawdown" in str(risk_issues):
            st.info("💡 Consider adding more defensive assets (bonds, consumer staples) to reduce drawdown risk.")
        if "Negative risk-adjusted returns" in str(risk_issues):
            st.info("💡 Review underperforming holdings and consider rebalancing to better-performing sectors.")
    
    st.divider()
    
    # Export risk report
    st.subheader("💾 Export Risk Report")
    
    risk_report_data = {
        'Metric': [
            'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown (%)', 'Beta',
            'VaR 95% (%)', 'VaR 99% (%)', 'Portfolio Return (%)', 'Portfolio Volatility (%)',
            'Top 5 Concentration (%)', 'Top 10 Concentration (%)', 'Herfindahl Index',
            'Risk-Free Rate (%)', 'Total Portfolio Value (€)'
        ],
        'Value': [
            f"{risk_metrics['sharpe_ratio']:.2f}",
            f"{risk_metrics['sortino_ratio']:.2f}",
            f"{risk_metrics['max_drawdown']:.2f}",
            f"{risk_metrics['beta']:.2f}",
            f"{risk_metrics['var_95']:.2f}",
            f"{risk_metrics['var_99']:.2f}",
            f"{risk_metrics['portfolio_return']:.2f}",
            f"{risk_metrics['portfolio_volatility']:.2f}",
            f"{risk_metrics['top_5_concentration']:.2f}",
            f"{risk_metrics['top_10_concentration']:.2f}",
            f"{risk_metrics['herfindahl_index']:.0f}",
            f"{risk_free_rate * 100:.2f}",
            f"€{total_value:,.2f}"
        ]
    }
    
    risk_report_df = pd.DataFrame(risk_report_data)
    csv_report = risk_report_df.to_csv(index=False)
    
    st.download_button(
        label="Download Risk Report as CSV",
        data=csv_report,
        file_name="risk_report.csv",
        mime="text/csv"
    )

with tab5:
    st.header("📰 News and Sentiments")
    
    # Disclaimer
    st.warning(
        "⚠️ **Disclaimer:** This feature summarizes public news and is for informational/educational purposes only. "
        "It is **not** investment advice."
    )
    
    # Import News Sentinel components
    try:
        from news_sentinel.agent import NewsSentinelAgent
        from news_sentinel.utils import parse_tickers, format_date_for_display
        from news_sentinel.config import get_openai_api_key
    except ImportError as e:
        st.error(f"Failed to import News Sentinel module: {e}")
        st.stop()
    
    st.divider()
    
    # Input section
    st.subheader("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker_input = st.text_area(
            "Tickers (comma or space separated)",
            placeholder="AAPL, MSFT, SPY",
            help="Enter one or more stock tickers separated by commas or spaces"
        )
    
    with col2:
        backend = st.selectbox(
            "Search Backend",
            options=["Tavily", "DuckDuckGo"],
            index=0,
            help="Choose the web search backend for fetching news"
        )
    
    # Optional controls
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            days_lookback = st.number_input(
                "Days Lookback",
                min_value=1,
                max_value=30,
                value=7,
                help="Number of days to look back for news"
            )
        
        with col2:
            max_articles_per_ticker = st.number_input(
                "Max Articles per Ticker",
                min_value=5,
                max_value=20,
                value=10,
                help="Maximum number of articles to fetch per ticker"
            )
    
    # Logging toggle
    show_logs = st.checkbox(
        "Show detailed logs",
        value=False,
        help="Enable to see detailed logging of agent operations"
    )
    
    # API Key input (optional, will use env/secrets if not provided)
    with st.expander("OpenAI Configuration", expanded=False):
        api_key_input = st.text_input(
            "OpenAI API Key (optional - will use environment/secrets if not provided)",
            type="password",
            placeholder="sk-...",
            help="If not provided, will use OPENAI_API_KEY from environment or Streamlit secrets"
        )
        
        if api_key_input:
            import os
            os.environ["OPENAI_API_KEY"] = api_key_input
    
    st.divider()
    
    # Analyze button
    if st.button("🔍 Analyze News", type="primary"):
        # Parse tickers
        tickers = parse_tickers(ticker_input)
        
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
            st.stop()
        
        # Validate API key
        api_key = get_openai_api_key()
        if not api_key:
            st.error(
                "OpenAI API key not found. Please provide it in the configuration section above, "
                "or set OPENAI_API_KEY environment variable or in Streamlit secrets."
            )
            st.stop()
        
        # Initialize agent
        try:
            agent = NewsSentinelAgent(openai_api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize News Sentinel Agent: {e}")
            st.stop()
        
        # Set up logging if enabled
        if show_logs:
            # Clear log buffer before starting
            from news_sentinel.logger import clear_log_buffer, setup_logger
            import logging
            clear_log_buffer()
            
            # Set up logger (will capture to buffer)
            setup_logger(
                name="news_sentinel",
                level=logging.INFO,
                streamlit_container=None,  # We'll display from buffer after
                enable_console=True
            )
        
        # Process with spinner
        with st.spinner(f"Fetching and analyzing news for {', '.join(tickers)}..."):
            try:
                # Use cached function for fetching news
                @st.cache_data(
                    ttl=600,  # Cache for 10 minutes
                    show_spinner=False
                )
                def cached_analyze(
                    ticker_list: tuple,
                    backend_name: str,
                    days: int,
                    max_articles: int,
                    api_key: str,
                ):
                    """Cached analysis function."""
                    # Set up logger (will capture to buffer)
                    from news_sentinel.logger import setup_logger
                    import logging
                    setup_logger(
                        name="news_sentinel",
                        level=logging.INFO,
                        streamlit_container=None,  # Buffer only
                        enable_console=True
                    )
                    
                    agent = NewsSentinelAgent(openai_api_key=api_key)
                    return agent.analyze(
                        tickers=list(ticker_list),
                        backend_name=backend_name.lower(),
                        days_lookback=days,
                        max_articles_per_ticker=max_articles,
                    )
                
                # Call cached function
                articles, analysis_result = cached_analyze(
                    ticker_list=tuple(tickers),
                    backend_name=backend.lower(),
                    days=days_lookback,
                    max_articles=max_articles_per_ticker,
                    api_key=api_key,
                )
                
                # Display logs if enabled
                if show_logs:
                    from news_sentinel.logger import get_logs
                    logs = get_logs()
                    
                    if logs:
                        with st.expander("📋 Agent Logs", expanded=True):
                            st.info(f"Captured {len(logs)} log entries:")
                            for log_entry in logs:
                                level = log_entry['level']
                                msg = log_entry['message']
                                
                                if level >= logging.ERROR:
                                    st.error(f"🔴 {msg}")
                                elif level >= logging.WARNING:
                                    st.warning(f"⚠️ {msg}")
                                elif level >= logging.INFO:
                                    st.info(f"ℹ️ {msg}")
                                else:  # DEBUG
                                    st.text(f"🔍 {msg}")
                    else:
                        with st.expander("📋 Agent Logs", expanded=False):
                            st.info("No logs captured. This may be due to caching. Try clearing the cache or running again.")
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)
                st.stop()
        
        # Display results
        st.success(f"✅ Analysis complete! Found {len(articles)} articles.")
        
        st.divider()
        
        # News Sentinel Overview
        st.subheader("📊 News Sentinel Overview")
        st.info(analysis_result.overall_summary)
        
        # Overall sentiment score
        overall_score = analysis_result.overall_sentiment_score
        sentiment_color = (
            "🟢" if overall_score > 0.2
            else "🔴" if overall_score < -0.2
            else "🟡"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Overall Sentiment Score",
                f"{overall_score:.2f}",
                help="Range: -1.0 (very negative) to 1.0 (very positive)"
            )
        with col2:
            st.write(f"**Sentiment:** {sentiment_color} {analysis_result.overall_sentiment_score:.2f}")
        
        st.divider()
        
        # Per-ticker analysis
        st.subheader("📈 Per-Ticker Analysis")
        
        # Create tabs for each ticker
        ticker_tabs = st.tabs([f"**{ts.ticker}**" for ts in analysis_result.ticker_sentiments])
        
        for idx, (ticker_sentiment, ticker_tab) in enumerate(
            zip(analysis_result.ticker_sentiments, ticker_tabs)
        ):
            with ticker_tab:
                # Sentiment badge
                sentiment_label = ticker_sentiment.sentiment_label
                sentiment_score = ticker_sentiment.sentiment_score
                
                if sentiment_label == "positive":
                    badge_color = "🟢"
                    st.success(f"**Sentiment:** {badge_color} {sentiment_label.upper()} (Score: {sentiment_score:.2f})")
                elif sentiment_label == "negative":
                    badge_color = "🔴"
                    st.error(f"**Sentiment:** {badge_color} {sentiment_label.upper()} (Score: {sentiment_score:.2f})")
                else:
                    badge_color = "🟡"
                    st.info(f"**Sentiment:** {badge_color} {sentiment_label.upper()} (Score: {sentiment_score:.2f})")
                
                # Summary
                st.write("**Summary:**")
                st.write(ticker_sentiment.summary)
                
                # Key themes
                if ticker_sentiment.key_themes:
                    st.write("**Key Themes:**")
                    themes_str = ", ".join(ticker_sentiment.key_themes)
                    st.write(f"_{themes_str}_")
                
                # Impactful headlines
                st.write("**Impactful Headlines:**")
                
                for headline in ticker_sentiment.impactful_headlines:
                    with st.expander(f"📰 {headline.title}", expanded=False):
                        st.write(f"**Why it matters:** {headline.why_it_matters}")
                        st.write(f"**Source:** {headline.source}")
                        st.write(f"**Date:** {headline.date or 'Date not available'}")
                        st.markdown(f"**Link:** [{headline.url}]({headline.url})")
        
        st.divider()
        
        # Charts
        st.subheader("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment by ticker bar chart
            ticker_data = {
                "Ticker": [ts.ticker for ts in analysis_result.ticker_sentiments],
                "Sentiment Score": [ts.sentiment_score for ts in analysis_result.ticker_sentiments],
            }
            sentiment_df = pd.DataFrame(ticker_data)
            
            fig_sentiment = px.bar(
                sentiment_df,
                x="Ticker",
                y="Sentiment Score",
                title="Sentiment Score by Ticker",
                color="Sentiment Score",
                color_continuous_scale=["red", "yellow", "green"],
                color_continuous_midpoint=0,
            )
            fig_sentiment.update_layout(height=400)
            st.plotly_chart(fig_sentiment, width='stretch')
        
        with col2:
            # Article counts by sentiment
            article_counts = analysis_result.article_counts
            counts_data = {
                "Sentiment": list(article_counts.keys()),
                "Count": list(article_counts.values()),
            }
            counts_df = pd.DataFrame(counts_data)
            
            fig_counts = px.bar(
                counts_df,
                x="Sentiment",
                y="Count",
                title="Article Count by Sentiment",
                color="Sentiment",
                color_discrete_map={
                    "positive": "green",
                    "neutral": "yellow",
                    "negative": "red",
                },
            )
            fig_counts.update_layout(height=400)
            st.plotly_chart(fig_counts, width='stretch')
        
        st.divider()
        
        # Raw articles count
        st.write(f"**Total articles analyzed:** {len(articles)}")
        st.write(f"**Backend used:** {backend}")

with tab6:
    st.header("📊 Historical Performance")
    
    st.markdown("""
    This section shows the development of your portfolio over time, comparing it against 
    benchmark indices. Historical snapshots are loaded from dated CSV files (e.g., `20112025_assets_jari.csv`).
    """)
    
    # User selection
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input(
            "User/Portfolio Name",
            value=DEFAULT_USER_NAME,
            help="Name used in snapshot filenames (e.g., 'jari' for latest_assets_jari.csv)"
        )
    
    with col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Annual risk-free rate for risk-adjusted metrics"
        ) / 100
    
    st.divider()
    
    # Load historical performance summary
    try:
        with st.spinner("Loading historical data and fetching benchmarks..."):
            # Get available benchmarks
            available_benchmarks = list_available_benchmarks()
            recommended = get_recommended_benchmarks("global")
            
            # Benchmark selection
            st.subheader("Benchmark Selection")
            selected_benchmarks = st.multiselect(
                "Select benchmarks to compare against",
                options=available_benchmarks['all'],
                default=recommended[:2] if len(recommended) >= 2 else recommended,
                help="Select one or more benchmarks to compare your portfolio performance"
            )
            
            if not selected_benchmarks:
                st.warning("Please select at least one benchmark for comparison.")
                st.stop()
            
            # Get performance summary
            summary = get_historical_performance_summary(
                user_name=user_name,
                benchmark_names=selected_benchmarks,
                risk_free_rate=risk_free_rate
            )
            
            portfolio_timeline = summary.get('portfolio_timeline', pd.DataFrame())
            benchmarks_data = summary.get('benchmarks', {})
            comparisons = summary.get('comparisons', {})
            period_returns = summary.get('period_returns', {})
            
            # Debug: Show which benchmarks were successfully fetched
            if benchmarks_data:
                st.success(f"✓ Successfully fetched {len(benchmarks_data)} benchmark(s): {', '.join(benchmarks_data.keys())}")
            if len(selected_benchmarks) > len(benchmarks_data):
                missing = set(selected_benchmarks) - set(benchmarks_data.keys())
                if missing:
                    st.warning(f"⚠️ Failed to fetch {len(missing)} benchmark(s): {', '.join(missing)}")
            
            if portfolio_timeline.empty:
                st.warning(
                    f"⚠️ **No historical snapshots found for user '{user_name}'.**\n\n"
                    "To use this feature:\n"
                    "1. Create snapshot files named: `{DDMMYYYY}_assets_{user}.csv`\n"
                    "2. Example: `20112025_assets_jari.csv` for November 20, 2025\n"
                    "3. The `latest_assets_{user}.csv` file is automatically included\n\n"
                    "See `HOW_TO_ADD_SNAPSHOTS.md` for more details."
                )
                st.stop()
            
            # Display snapshot info
            num_snapshots = len(portfolio_timeline)
            date_range = f"{portfolio_timeline.index.min().strftime('%Y-%m-%d')} to {portfolio_timeline.index.max().strftime('%Y-%m-%d')}"
            
            st.info(f"📅 **Data Range:** {date_range} ({num_snapshots} snapshot{'s' if num_snapshots != 1 else ''})")
            
            if num_snapshots < 2:
                st.warning("⚠️ **Limited data:** You need at least 2 snapshots for meaningful historical analysis. Consider adding more snapshot files.")
            
            st.divider()
            
            # Portfolio Value Over Time
            st.subheader("💰 Portfolio Value Over Time")
            
            if not portfolio_timeline.empty:
                # Create value chart
                fig_value = go.Figure()
                
                # Portfolio value line
                fig_value.add_trace(go.Scatter(
                    x=portfolio_timeline.index,
                    y=portfolio_timeline['total_value_eur'],
                    mode='lines+markers',
                    name='Portfolio Value',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                fig_value.update_layout(
                    title='Portfolio Total Value Over Time',
                    xaxis_title='Date',
                    yaxis_title='Value (EUR)',
                    height=500,
                    hovermode='x unified',
                    yaxis=dict(tickformat='€,.0f')
                )
                
                st.plotly_chart(fig_value, width='stretch')
                
                # Display current metrics
                latest_value = portfolio_timeline['total_value_eur'].iloc[-1]
                earliest_value = portfolio_timeline['total_value_eur'].iloc[0]
                total_change = latest_value - earliest_value
                total_change_pct = ((latest_value / earliest_value) - 1) * 100 if earliest_value > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Value", f"€{latest_value:,.2f}")
                with col2:
                    st.metric("Total Change", f"€{total_change:,.2f}", f"{total_change_pct:+.2f}%")
                with col3:
                    st.metric("Number of Snapshots", num_snapshots)
            
            st.divider()
            
            # Benchmark Comparison (Normalized)
            if benchmarks_data:
                st.subheader("📈 Portfolio vs Benchmarks (Normalized)")
                
                from portfolio.benchmarks.benchmark_fetcher import normalize_benchmark_data
                
                # Normalize portfolio to 100 at start
                portfolio_normalized = portfolio_timeline.copy()
                if len(portfolio_normalized) > 0:
                    base_value = portfolio_normalized['total_value_eur'].iloc[0]
                    portfolio_normalized['normalized'] = (portfolio_normalized['total_value_eur'] / base_value) * 100
                
                # Create comparison chart
                fig_comparison = go.Figure()
                
                # Portfolio line
                if len(portfolio_normalized) > 0:
                    fig_comparison.add_trace(go.Scatter(
                        x=portfolio_normalized.index,
                        y=portfolio_normalized['normalized'],
                        mode='lines+markers',
                        name='Portfolio',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                
                # Benchmark lines
                colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                for idx, (benchmark_name, benchmark_df) in enumerate(benchmarks_data.items()):
                    # Check if benchmark has valid data (either price_eur or price_native)
                    if benchmark_df.empty:
                        st.warning(f"⚠️ {benchmark_name}: No data available")
                        continue
                    
                    # Check for price column (prefer price_eur, fallback to price_native)
                    if 'price_eur' not in benchmark_df.columns and 'price_native' not in benchmark_df.columns:
                        st.warning(f"⚠️ {benchmark_name}: Missing price column")
                        continue
                    
                    # Normalize benchmark data
                    normalized_benchmark = normalize_benchmark_data(benchmark_df)
                    if normalized_benchmark.empty or 'normalized' not in normalized_benchmark.columns:
                        st.warning(f"⚠️ {benchmark_name}: Failed to normalize data")
                        continue
                    
                    # Add to chart
                    color = colors[idx % len(colors)]
                    fig_comparison.add_trace(go.Scatter(
                        x=normalized_benchmark.index,
                        y=normalized_benchmark['normalized'],
                        mode='lines+markers',
                        name=benchmark_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=6)
                    ))
                
                fig_comparison.update_layout(
                    title='Portfolio vs Benchmarks (Normalized to 100 at Start)',
                    xaxis_title='Date',
                    yaxis_title='Normalized Value (Base = 100)',
                    height=500,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig_comparison, width='stretch')
                
                st.info("💡 **Note:** All values are normalized to 100 at the start date for easy comparison.")
            
            st.divider()
            
            # Period Returns Table
            if period_returns:
                st.subheader("📊 Period Returns")
                
                # Create period returns table
                periods = ["1M", "3M", "6M", "1Y", "YTD", "All-time"]
                returns_data = []
                
                for period in periods:
                    return_pct = period_returns.get(period)
                    if return_pct is not None:
                        returns_data.append({
                            'Period': period,
                            'Return (%)': f"{return_pct:.2f}%"
                        })
                
                if returns_data:
                    returns_df = pd.DataFrame(returns_data)
                    st.dataframe(returns_df, width='stretch', hide_index=True)
                else:
                    st.info("Period returns require more historical data points.")
            
            # Benchmark Comparison Metrics
            if comparisons:
                st.divider()
                st.subheader("🔍 Benchmark Comparison Metrics")
                
                # Create comparison metrics table
                comparison_data = []
                for benchmark_name, comp in comparisons.items():
                    comparison_data.append({
                        'Benchmark': benchmark_name,
                        'Beta': f"{comp.get('beta', np.nan):.3f}" if not pd.isna(comp.get('beta')) else "N/A",
                        'Alpha (%)': f"{comp.get('alpha', np.nan):.2f}" if not pd.isna(comp.get('alpha')) else "N/A",
                        'Tracking Error (%)': f"{comp.get('tracking_error', np.nan):.2f}" if not pd.isna(comp.get('tracking_error')) else "N/A",
                        'Information Ratio': f"{comp.get('information_ratio', np.nan):.3f}" if not pd.isna(comp.get('information_ratio')) else "N/A",
                        'Correlation': f"{comp.get('correlation', np.nan):.3f}" if not pd.isna(comp.get('correlation')) else "N/A",
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch', hide_index=True)
                    
                    # Explanation
                    with st.expander("📖 Metric Explanations", expanded=False):
                        st.markdown("""
                        - **Beta**: Portfolio sensitivity to benchmark (1.0 = moves with market, >1.0 = more volatile, <1.0 = less volatile)
                        - **Alpha**: Risk-adjusted excess return (positive = outperformance, negative = underperformance)
                        - **Tracking Error**: Volatility of excess returns (lower = more consistent relative performance)
                        - **Information Ratio**: Alpha / Tracking Error (higher = better risk-adjusted outperformance)
                        - **Correlation**: How closely portfolio moves with benchmark (1.0 = perfect correlation, 0 = no correlation)
                        """)
            
            # Period Comparison Table (Portfolio vs Benchmarks)
            if comparisons and period_returns:
                st.divider()
                st.subheader("📋 Period Returns Comparison")
                
                # Get first benchmark for comparison
                if len(comparisons) > 0:
                    benchmark_name = list(comparisons.keys())[0]
                    comparison = comparisons[benchmark_name]
                    
                    portfolio_periods = comparison.get('period_returns', {}).get('portfolio', {})
                    benchmark_periods = comparison.get('period_returns', {}).get('benchmark', {})
                    
                    if portfolio_periods and benchmark_periods:
                        comparison_table_data = []
                        for period in ["1M", "3M", "6M", "1Y", "YTD", "All-time"]:
                            port_return = portfolio_periods.get(period)
                            bench_return = benchmark_periods.get(period)
                            
                            if port_return is not None and bench_return is not None:
                                diff = port_return - bench_return
                                comparison_table_data.append({
                                    'Period': period,
                                    'Portfolio (%)': f"{port_return:.2f}%",
                                    f'{benchmark_name} (%)': f"{bench_return:.2f}%",
                                    'Difference (%)': f"{diff:+.2f}%"
                                })
                        
                        if comparison_table_data:
                            comparison_table_df = pd.DataFrame(comparison_table_data)
                            st.dataframe(comparison_table_df, width='stretch', hide_index=True)
            
            # Risk Metrics Over Time (if enough data)
            if not portfolio_timeline.empty and len(portfolio_timeline) >= 12:
                st.divider()
                st.subheader("📉 Risk Metrics Over Time")
                
                from portfolio.historical.performance_tracker import (
                    calculate_rolling_volatility,
                    calculate_rolling_sharpe_ratio,
                )
                
                # Calculate returns
                portfolio_values = portfolio_timeline['total_value_eur']
                portfolio_returns = portfolio_values.pct_change().dropna()
                
                if len(portfolio_returns) >= 12:
                    # Rolling volatility
                    rolling_vol = calculate_rolling_volatility(portfolio_returns, window=12)
                    
                    if not rolling_vol.empty:
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Scatter(
                            x=rolling_vol.index,
                            y=rolling_vol.values,
                            mode='lines',
                            name='12-Month Rolling Volatility',
                            line=dict(color='#d62728', width=2)
                        ))
                        fig_vol.update_layout(
                            title='Rolling Volatility (12-Month Window, Annualized)',
                            xaxis_title='Date',
                            yaxis_title='Volatility (%)',
                            height=400
                        )
                        st.plotly_chart(fig_vol, width='stretch')
                    
                    # Rolling Sharpe ratio
                    rolling_sharpe = calculate_rolling_sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate, window=12)
                    
                    if not rolling_sharpe.empty:
                        fig_sharpe = go.Figure()
                        fig_sharpe.add_trace(go.Scatter(
                            x=rolling_sharpe.index,
                            y=rolling_sharpe.values,
                            mode='lines',
                            name='12-Month Rolling Sharpe Ratio',
                            line=dict(color='#2ca02c', width=2)
                        ))
                        fig_sharpe.update_layout(
                            title='Rolling Sharpe Ratio (12-Month Window)',
                            xaxis_title='Date',
                            yaxis_title='Sharpe Ratio',
                            height=400
                        )
                        st.plotly_chart(fig_sharpe, width='stretch')
            elif not portfolio_timeline.empty:
                st.info("💡 **Note:** Risk metrics over time require at least 12 data points. You currently have {len(portfolio_timeline)} snapshot(s).")
            
            # Allocation Evolution (if enough snapshots)
            if not portfolio_timeline.empty and num_snapshots >= 2:
                st.divider()
                st.subheader("🌍 Allocation Evolution")
                
                from portfolio.historical.snapshot_loader import load_all_snapshots, load_latest_snapshot
                from config import DATA_DIR
                
                try:
                    # Load all historical snapshots
                    all_snapshots = load_all_snapshots(user_name, directory=DATA_DIR, enrich=True)
                    
                    # Also include the latest snapshot with current date
                    try:
                        latest_snapshot = load_latest_snapshot(user_name, directory=DATA_DIR, enrich=True, fallback_to_default=False)
                        if not latest_snapshot.empty:
                            # Use current date for latest snapshot if no date is set
                            if 'snapshot_date' not in latest_snapshot.columns or latest_snapshot['snapshot_date'].isna().all():
                                from datetime import datetime
                                latest_snapshot['snapshot_date'] = datetime.now()
                            
                            # Combine with historical snapshots
                            if all_snapshots.empty:
                                all_snapshots = latest_snapshot
                            else:
                                all_snapshots = pd.concat([all_snapshots, latest_snapshot], ignore_index=True)
                    except FileNotFoundError:
                        # Latest snapshot not found, continue with historical only
                        pass
                    
                    # Debug: Show unique dates to help diagnose issues
                    if not all_snapshots.empty and 'snapshot_date' in all_snapshots.columns:
                        unique_dates = sorted(all_snapshots['snapshot_date'].unique())
                        st.caption(f"📅 Found {len(unique_dates)} unique snapshot dates")
                    
                    if not all_snapshots.empty and 'geography' in all_snapshots.columns:
                        # Group by date and geography
                        geo_evolution = all_snapshots.groupby(['snapshot_date', 'geography'])['market_total_eur'].sum().reset_index()
                        geo_evolution_pivot = geo_evolution.pivot(index='snapshot_date', columns='geography', values='market_total_eur').fillna(0)
                        
                        # Calculate percentages
                        geo_evolution_pct = geo_evolution_pivot.div(geo_evolution_pivot.sum(axis=1), axis=0) * 100
                        
                        # Stacked area chart
                        fig_geo = go.Figure()
                        for geo in geo_evolution_pct.columns:
                            fig_geo.add_trace(go.Scatter(
                                x=geo_evolution_pct.index,
                                y=geo_evolution_pct[geo],
                                mode='lines',
                                name=geo,
                                stackgroup='one',
                                fill='tonexty' if geo != geo_evolution_pct.columns[0] else 'tozeroy'
                            ))
                        
                        fig_geo.update_layout(
                            title='Geographic Allocation Over Time',
                            xaxis_title='Date',
                            yaxis_title='Allocation (%)',
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_geo, width='stretch')
                    
                    if not all_snapshots.empty and 'sector' in all_snapshots.columns:
                        # Sector evolution
                        sector_evolution = all_snapshots.groupby(['snapshot_date', 'sector'])['market_total_eur'].sum().reset_index()
                        sector_evolution_pivot = sector_evolution.pivot(index='snapshot_date', columns='sector', values='market_total_eur').fillna(0)
                        sector_evolution_pct = sector_evolution_pivot.div(sector_evolution_pivot.sum(axis=1), axis=0) * 100
                        
                        # Show top 5 sectors
                        top_sectors = sector_evolution_pct.iloc[-1].nlargest(5).index
                        sector_evolution_pct_top = sector_evolution_pct[top_sectors]
                        
                        fig_sector = go.Figure()
                        for sector in sector_evolution_pct_top.columns:
                            fig_sector.add_trace(go.Scatter(
                                x=sector_evolution_pct_top.index,
                                y=sector_evolution_pct_top[sector],
                                mode='lines',
                                name=sector,
                                stackgroup='one',
                                fill='tonexty' if sector != sector_evolution_pct_top.columns[0] else 'tozeroy'
                            ))
                        
                        fig_sector.update_layout(
                            title='Sector Allocation Over Time (Top 5 Sectors)',
                            xaxis_title='Date',
                            yaxis_title='Allocation (%)',
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_sector, width='stretch')
                except Exception as e:
                    st.warning(f"Could not load allocation evolution data: {e}")
            
            st.divider()
            
            # Data Info
            with st.expander("ℹ️ About Historical Performance", expanded=False):
                st.markdown("""
                **How it works:**
                - Historical snapshots are loaded from files named `{DDMMYYYY}_assets_{user}.csv`
                - The `latest_assets_{user}.csv` file is automatically included as the most recent data point
                - Benchmark data is fetched from Yahoo Finance and converted to EUR for comparison
                - All metrics are calculated based on the available snapshot dates
                
                **Requirements:**
                - At least 2 snapshots for basic comparisons
                - 12+ snapshots recommended for rolling metrics (volatility, Sharpe ratio)
                - Benchmark data requires internet connection and may take a few seconds to fetch
                
                **File Naming:**
                - Historical: `20112025_assets_jari.csv` (November 20, 2025)
                - Latest: `latest_assets_jari.csv`
                - Date format: DDMMYYYY (day, month, year)
                """)
    
    except Exception as e:
        st.error(f"Error loading historical performance data: {e}")
        import traceback
        with st.expander("Error Details", expanded=False):
            st.code(traceback.format_exc())

