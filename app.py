"""
LLM-Powered Stock Market Analyzer
Streamlit Dashboard – main entry point.

Run with:
    streamlit run app.py
"""

import json
import logging
import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

import config
from modules.analytics  import get_earnings_calendar, get_sector_heatmap, get_peer_comparison, get_insider_activity
from modules.macro      import get_market_overview, get_relative_strength, get_yield_curve
from modules.risk       import get_portfolio_risk_metrics, get_correlation_matrix, get_rebalancing_suggestions
from modules.sec_filing import get_recent_filings, analyze_filing, summarize_earnings
from modules import alerts as alert_mgr

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MarketMind – LLM Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Global */
  .main { background-color: #0e1117; }
  .stApp { background-color: #0e1117; }

  /* Metric cards */
  div[data-testid="metric-container"] {
      background: #1a1f2e;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 16px;
  }

  /* Sentiment badges */
  .badge {
      display: inline-block;
      padding: 4px 14px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 14px;
  }
  .bullish  { background: #1a4731; color: #4ade80; }
  .bearish  { background: #4a1a1a; color: #f87171; }
  .neutral  { background: #2a2a1a; color: #facc15; }
  .mixed    { background: #1a2a4a; color: #60a5fa; }

  /* Chat bubbles */
  .user-bubble {
      background: #1e3a5f;
      border-radius: 12px 12px 0 12px;
      padding: 10px 16px;
      margin: 4px 0;
      max-width: 80%;
      float: right;
      clear: both;
  }
  .assistant-bubble {
      background: #1a1f2e;
      border: 1px solid #2d3748;
      border-radius: 12px 12px 12px 0;
      padding: 10px 16px;
      margin: 4px 0;
      max-width: 90%;
      float: left;
      clear: both;
  }
  .clearfix::after { content: ""; display: table; clear: both; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] {
      font-size: 15px;
      font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "etrade_authenticated":  False,
        "etrade_account_id":     "",
        "etrade_accounts":       [],
        "portfolio":             [],
        "chat_history":          [],
        "agent":                 None,
        "oauth_url":             "",
        "selected_ticker":       "GOOGL",
        "price_alerts":          [],
        "alerts_last_checked":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Agent initialisation (per-session, NOT cached – caching None would break it)
# ─────────────────────────────────────────────────────────────────────────────

def _init_agent() -> tuple:
    """
    Attempt to create a StockAgent.
    Returns (agent, error_message).  One of them will be None.
    Never uses @st.cache_resource so a first-run failure doesn't get cached.
    """
    try:
        from modules.llm_agent import StockAgent
        return StockAgent(), None
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_currency(val: float, decimals: int = 2) -> str:
    if val >= 1_000_000_000:
        return f"${val/1_000_000_000:.2f}B"
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    return f"${val:,.{decimals}f}"


def _sentiment_badge(label: str) -> str:
    css = {"BULLISH": "bullish", "BEARISH": "bearish", "NEUTRAL": "neutral", "MIXED": "mixed"}
    cls = css.get(label.upper(), "neutral")
    return f'<span class="badge {cls}">{label}</span>'


def _delta_color(val: float) -> str:
    return "normal" if val >= 0 else "inverse"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – E*Trade OAuth
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/ETrade_logo.svg/320px-ETrade_logo.svg.png",
                 width=140)
        st.markdown("### E*Trade Portfolio")

        if not config.ETRADE_CONSUMER_KEY or not config.ETRADE_CONSUMER_SECRET:
            st.warning("E*Trade keys not configured.\nAdd them to your `.env` file.")
            # Manual portfolio fallback
            st.markdown("---")
            st.markdown("**Manual Ticker Entry**")
            manual = st.text_area(
                "Enter tickers (comma-separated)",
                value=st.session_state.get("manual_tickers", "META,NVDA,GOOGL"),
                key="manual_tickers_input",
            )
            if st.button("Load Manual Portfolio"):
                tickers = [t.strip().upper() for t in manual.split(",") if t.strip()]
                st.session_state["portfolio"] = [
                    {"symbol": t, "quantity": 0, "market_value": 0,
                     "cost_basis": 0, "gain_loss": 0, "gain_loss_pct": 0,
                     "current_price": 0, "position_type": "MANUAL"}
                    for t in tickers
                ]
                st.success(f"Loaded {len(tickers)} tickers.")
            return

        # ── OAuth flow ─────────────────────────────────────────────────────────
        if not st.session_state["etrade_authenticated"]:
            col1, col2 = st.columns(2)
            with col1:
                env_label = "🔴 Live" if config.ETRADE_ENV == "live" else "🟡 Sandbox"
                st.caption(f"Mode: **{env_label}**")

            if st.button("Connect to E*Trade", use_container_width=True):
                try:
                    from modules.etrade_client import get_request_token
                    url = get_request_token()
                    st.session_state["oauth_url"] = url
                except Exception as exc:
                    st.error(f"OAuth init failed: {exc}")

            if st.session_state["oauth_url"]:
                st.info(
                    "1. Click the link below to authorise MarketMind\n"
                    "2. Log in to E*Trade\n"
                    "3. Copy the verification code and paste it below"
                )
                st.markdown(f"[Open E*Trade Authorisation]({st.session_state['oauth_url']})")

                verifier = st.text_input("Verification code", key="oauth_verifier")
                if st.button("Submit Verifier") and verifier:
                    from modules.etrade_client import get_access_token, list_accounts
                    if get_access_token(verifier.strip()):
                        st.session_state["etrade_authenticated"] = True
                        accounts = list_accounts()
                        st.session_state["etrade_accounts"] = accounts
                        if accounts:
                            st.session_state["etrade_account_id"] = accounts[0].get("accountIdKey", "")
                        st.success("Connected to E*Trade!")
                        st.rerun()
                    else:
                        st.error("Invalid verifier code. Please try again.")
        else:
            st.success("Connected to E*Trade")
            if st.button("Disconnect"):
                from modules.etrade_client import logout as etrade_logout
                etrade_logout()
                st.session_state["etrade_authenticated"] = False
                st.session_state["etrade_account_id"]    = ""
                st.session_state["etrade_accounts"]      = []
                st.session_state["oauth_url"]            = ""
                st.session_state["portfolio"]            = []
                st.rerun()

            # Account selector
            accounts = st.session_state["etrade_accounts"]
            if accounts:
                acct_labels = {
                    a.get("accountIdKey", ""): f"{a.get('accountDesc', '')} ({a.get('accountId', '')})"
                    for a in accounts
                }
                selected = st.selectbox(
                    "Select account",
                    options=list(acct_labels.keys()),
                    format_func=lambda k: acct_labels[k],
                )
                st.session_state["etrade_account_id"] = selected

            if st.button("Refresh Portfolio", use_container_width=True):
                _load_portfolio()

        # ── Quick search ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Quick Ticker Lookup**")
        ticker_input = st.text_input(
            "Enter ticker symbol",
            value=st.session_state["selected_ticker"],
            key="ticker_search",
        ).upper().strip()
        if ticker_input:
            st.session_state["selected_ticker"] = ticker_input

        # ── Price Alerts ───────────────────────────────────────────────────────
        st.markdown("---")
        _render_alerts_sidebar()


def _render_alerts_sidebar():
    """Render the price-alert section inside the sidebar."""
    with st.expander("🔔 Price Alerts", expanded=False):
        # Check active alerts on every render (throttled to once per run)
        alerts = st.session_state.get("price_alerts", [])
        if alerts:
            # Gather current prices for all alert tickers
            alert_tickers = list({a["symbol"] for a in alert_mgr.get_active_alerts(alerts)})
            cur_prices: dict[str, float] = {}
            for sym in alert_tickers:
                try:
                    from modules.stock_data import get_current_price
                    cur_prices[sym] = get_current_price(sym)["price"]
                except Exception:
                    pass

            if cur_prices:
                updated, fired = alert_mgr.check_alerts(alerts, cur_prices)
                st.session_state["price_alerts"] = updated
                for a in fired:
                    cond = "above" if a["condition"] == "above" else "below"
                    st.success(
                        f"🔔 **{a['symbol']}** hit ${a['triggered_price']:,.2f} "
                        f"({cond} ${a['target']:,.2f})"
                    )

        # Add new alert form
        st.markdown("**Add Alert**")
        c1, c2 = st.columns(2)
        with c1:
            alert_sym   = st.text_input("Ticker", key="alert_sym", placeholder="AAPL").upper().strip()
            alert_cond  = st.selectbox("Condition", ["above", "below"], key="alert_cond")
        with c2:
            alert_price = st.number_input("Price ($)", min_value=0.01, value=100.0, key="alert_price")
            alert_note  = st.text_input("Note (optional)", key="alert_note")

        if st.button("Add Alert", key="add_alert_btn"):
            if alert_sym and alert_price > 0:
                st.session_state["price_alerts"] = alert_mgr.add_alert(
                    st.session_state["price_alerts"],
                    alert_sym, alert_price, alert_cond, alert_note,
                )
                st.success(f"Alert set: {alert_sym} {alert_cond} ${alert_price:,.2f}")
                st.rerun()

        # Active alerts list
        active = alert_mgr.get_active_alerts(st.session_state.get("price_alerts", []))
        if active:
            st.markdown("**Active Alerts**")
            for a in active:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"{a['symbol']} {a['condition']} ${a['target']:,.2f}" + (f" — {a['note']}" if a['note'] else ""))
                with col2:
                    if st.button("✕", key=f"rm_alert_{a['id']}"):
                        st.session_state["price_alerts"] = alert_mgr.remove_alert(
                            st.session_state["price_alerts"], a["id"]
                        )
                        st.rerun()

        # Triggered alerts
        triggered = alert_mgr.get_triggered_alerts(st.session_state.get("price_alerts", []))
        if triggered:
            st.markdown("**Triggered**")
            for a in triggered[:5]:
                st.caption(f"✅ {a['symbol']} hit ${a['triggered_price']:,.2f} at {a['triggered_at']}")


def _load_portfolio():
    """Fetch portfolio from E*Trade and store in session state."""
    acct_id = st.session_state.get("etrade_account_id", "")
    if not acct_id:
        st.warning("No account selected.")
        return
    with st.spinner("Loading portfolio..."):
        try:
            from modules.etrade_client import get_portfolio
            positions = get_portfolio(acct_id)
            st.session_state["portfolio"] = positions
            st.success(f"Loaded {len(positions)} positions.")
        except Exception as exc:
            st.error(f"Could not load portfolio: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Portfolio Overview
# ─────────────────────────────────────────────────────────────────────────────

def _render_portfolio_tab():
    st.header("Portfolio Overview")

    portfolio = st.session_state.get("portfolio", [])

    # ── Fetch account balance if authenticated ─────────────────────────────────
    if st.session_state.get("etrade_authenticated") and st.session_state.get("etrade_account_id"):
        try:
            from modules.etrade_client import get_account_balance
            bal = get_account_balance(st.session_state["etrade_account_id"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Account Value",    _fmt_currency(bal["net_account_value"]))
            c2.metric("Cash Available",       _fmt_currency(bal["cash_available"]))
            c3.metric("Margin Buying Power",  _fmt_currency(bal["margin_buying_power"]))
            c4.metric("Account Type",         bal.get("account_type", "N/A"))
        except Exception as exc:
            st.warning(f"Could not load balance: {exc}")

    if not portfolio:
        st.info("No portfolio loaded. Use the sidebar to connect E*Trade or enter tickers manually.")
        return

    df = pd.DataFrame(portfolio)

    # ── Portfolio summary metrics ──────────────────────────────────────────────
    if "market_value" in df.columns and df["market_value"].sum() > 0:
        total_value    = df["market_value"].sum()
        total_cost     = df["cost_basis"].sum()
        total_gain     = df["gain_loss"].sum()
        total_gain_pct = (total_gain / total_cost * 100) if total_cost else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Portfolio Value", _fmt_currency(total_value))
        c2.metric("Total Cost Basis",      _fmt_currency(total_cost))
        c3.metric("Unrealised Gain/Loss",  _fmt_currency(total_gain),
                  f"{total_gain_pct:+.2f}%",
                  delta_color="normal" if total_gain >= 0 else "inverse")
        c4.metric("Positions", len(df))

    st.markdown("---")

    # ── Position table ─────────────────────────────────────────────────────────
    st.subheader("Positions")

    display_cols = [c for c in
        ["symbol", "quantity", "current_price", "market_value",
         "cost_basis", "gain_loss", "gain_loss_pct", "position_type"]
        if c in df.columns]

    styled_df = df[display_cols].copy()
    if "gain_loss_pct" in styled_df.columns:
        styled_df["gain_loss_pct"] = styled_df["gain_loss_pct"].apply(lambda x: f"{x:+.2f}%")
    if "current_price" in styled_df.columns:
        styled_df["current_price"] = styled_df["current_price"].apply(
            lambda x: f"${x:,.2f}" if x else "—")
    if "market_value" in styled_df.columns:
        styled_df["market_value"] = styled_df["market_value"].apply(
            lambda x: _fmt_currency(x) if x else "—")
    if "cost_basis" in styled_df.columns:
        styled_df["cost_basis"] = styled_df["cost_basis"].apply(
            lambda x: _fmt_currency(x) if x else "—")
    if "gain_loss" in styled_df.columns:
        styled_df["gain_loss"] = styled_df["gain_loss"].apply(
            lambda x: f"${x:+,.2f}" if x else "—")

    styled_df.columns = [c.replace("_", " ").title() for c in styled_df.columns]
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Allocation pie ─────────────────────────────────────────────────────────
    if "market_value" in df.columns and df["market_value"].sum() > 0:
        st.subheader("Allocation")
        fig = px.pie(
            df[df["market_value"] > 0],
            values="market_value",
            names="symbol",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white",
            legend=dict(font=dict(color="white")),
            showlegend=True,
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Gain/Loss bar ──────────────────────────────────────────────────────────
    if "gain_loss" in df.columns:
        df_gl = df[df["gain_loss"] != 0].copy()
        if not df_gl.empty:
            st.subheader("Unrealised Gain / Loss by Position")
            df_gl["color"] = df_gl["gain_loss"].apply(lambda x: "#4ade80" if x >= 0 else "#f87171")
            fig2 = go.Figure(go.Bar(
                x=df_gl["symbol"],
                y=df_gl["gain_loss"],
                marker_color=df_gl["color"].tolist(),
                text=df_gl["gain_loss"].apply(lambda x: f"${x:+,.0f}"),
                textposition="outside",
            ))
            fig2.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1f2e",
                font_color="white",
                xaxis=dict(color="white"),
                yaxis=dict(color="white", gridcolor="#2d3748"),
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Portfolio Risk Metrics ──────────────────────────────────────────────────
    _render_portfolio_risk_section(portfolio)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – Stock Dashboard (ratios + fair value)
# ─────────────────────────────────────────────────────────────────────────────

def _render_stock_dashboard_tab():
    st.header("Stock Dashboard")

    sym = st.text_input(
        "Ticker Symbol",
        value=st.session_state["selected_ticker"],
        key="dashboard_ticker",
    ).upper().strip()

    if not sym:
        return

    st.session_state["selected_ticker"] = sym

    col_left, col_right = st.columns([3, 1])
    with col_right:
        period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

    # ── Fetch data ─────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for {sym}..."):
        try:
            from modules.stock_data  import get_current_price, get_key_ratios, get_technical_indicators, get_price_history
            from modules.valuation   import get_fair_value
            from modules.sentiment   import get_sentiment

            price_info = get_current_price(sym)
            ratios     = get_key_ratios(sym)
            tech       = get_technical_indicators(sym)
            fv         = get_fair_value(sym)
            sent       = get_sentiment(sym, price_info.get("name", sym))
            hist_df    = get_price_history(sym, period=period)

        except Exception as exc:
            st.error(f"Error fetching data for {sym}: {exc}")
            return

    # ── Header ─────────────────────────────────────────────────────────────────
    name   = price_info.get("name", sym)
    sector = price_info.get("sector", "N/A")
    price  = price_info.get("price", 0)
    chg    = price_info.get("change", 0)
    chg_p  = price_info.get("change_pct", 0)
    mktcap = price_info.get("market_cap", 0)

    st.markdown(f"## {name} ({sym})")
    st.caption(f"{sector} · {price_info.get('industry', '')} · {price_info.get('currency', 'USD')}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price",         f"${price:,.2f}", f"{chg:+.2f} ({chg_p:+.2f}%)",
              delta_color="normal" if chg >= 0 else "inverse")
    m2.metric("Market Cap",    _fmt_currency(mktcap))
    m3.metric("52W High",      f"${price_info.get('52w_high', 0):,.2f}")
    m4.metric("52W Low",       f"${price_info.get('52w_low', 0):,.2f}")
    m5.metric("Avg Volume",    f"{price_info.get('avg_volume', 0):,}")

    st.markdown("---")

    # ── Price chart with indicators ────────────────────────────────────────────
    if not hist_df.empty:
        st.subheader(f"Price & Volume ({period})")
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.03,
        )

        close = hist_df["Close"]
        sma20  = close.rolling(20).mean()
        sma50  = close.rolling(50).mean()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"], high=hist_df["High"],
            low=hist_df["Low"],   close=hist_df["Close"],
            name="OHLC",
            increasing_line_color="#4ade80",
            decreasing_line_color="#f87171",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=hist_df.index, y=sma20, name="SMA 20",
                                  line=dict(color="#60a5fa", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=sma50, name="SMA 50",
                                  line=dict(color="#f59e0b", width=1.2)), row=1, col=1)

        # Volume bars
        colors = ["#4ade80" if c >= o else "#f87171"
                  for c, o in zip(hist_df["Close"], hist_df["Open"])]
        fig.add_trace(go.Bar(x=hist_df.index, y=hist_df["Volume"],
                             name="Volume", marker_color=colors, opacity=0.6), row=2, col=1)

        fig.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
            font_color="white", height=500,
            xaxis_rangeslider_visible=False,
            legend=dict(font=dict(color="white")),
            xaxis2=dict(color="white", gridcolor="#2d3748"),
            yaxis=dict(color="white",  gridcolor="#2d3748"),
            yaxis2=dict(color="white", gridcolor="#2d3748"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Key Ratios ─────────────────────────────────────────────────────────────
    st.subheader("Key Financial Ratios")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("**Valuation**")
        ratio_df = pd.DataFrame({
            "Ratio": ["P/E (TTM)", "P/E (Forward)", "P/B", "P/S", "EV/EBITDA", "EV/Revenue", "PEG"],
            "Value": [
                f"{ratios['pe_trailing']:.1f}x"   if ratios['pe_trailing']  else "N/A",
                f"{ratios['pe_forward']:.1f}x"    if ratios['pe_forward']   else "N/A",
                f"{ratios['price_to_book']:.2f}x" if ratios['price_to_book'] else "N/A",
                f"{ratios['price_to_sales']:.2f}x"if ratios['price_to_sales'] else "N/A",
                f"{ratios['ev_to_ebitda']:.1f}x"  if ratios['ev_to_ebitda'] else "N/A",
                f"{ratios['ev_to_revenue']:.1f}x" if ratios['ev_to_revenue'] else "N/A",
                f"{ratios['peg_ratio']:.2f}"       if ratios['peg_ratio']   else "N/A",
            ],
        })
        st.dataframe(ratio_df, hide_index=True, use_container_width=True)

    with r2:
        st.markdown("**Profitability**")
        prof_df = pd.DataFrame({
            "Metric": ["Gross Margin", "Oper. Margin", "Net Margin", "EBITDA Margin", "ROE", "ROA"],
            "Value": [
                f"{ratios['gross_margin_pct']:.1f}%",
                f"{ratios['oper_margin_pct']:.1f}%",
                f"{ratios['net_margin_pct']:.1f}%",
                f"{ratios['ebitda_margin_pct']:.1f}%",
                f"{ratios['roe_pct']:.1f}%",
                f"{ratios['roa_pct']:.1f}%",
            ],
        })
        st.dataframe(prof_df, hide_index=True, use_container_width=True)

    with r3:
        st.markdown("**Growth & Health**")
        health_df = pd.DataFrame({
            "Metric": ["Rev Growth (YoY)", "Earn Growth (YoY)", "Current Ratio",
                       "Quick Ratio", "D/E Ratio", "Beta", "Div Yield"],
            "Value": [
                f"{ratios['revenue_growth_pct']:+.1f}%",
                f"{ratios['earnings_growth_pct']:+.1f}%",
                f"{ratios['current_ratio']:.2f}",
                f"{ratios['quick_ratio']:.2f}",
                f"{ratios['debt_to_equity']:.2f}",
                f"{ratios['beta']:.2f}",
                f"{ratios['dividend_yield_pct']:.2f}%",
            ],
        })
        st.dataframe(health_df, hide_index=True, use_container_width=True)

    # ── Technical Indicators ───────────────────────────────────────────────────
    st.subheader("Technical Indicators")
    t1, t2, t3, t4 = st.columns(4)

    rsi = tech.get("rsi_14", 0)
    rsi_color = "#f87171" if rsi > 70 else ("#4ade80" if rsi < 30 else "white")
    t1.metric("RSI (14)", f"{rsi:.1f}", help="Overbought > 70 | Oversold < 30")
    t2.metric("MACD",     f"{tech.get('macd', 0):.4f}",
              f"Signal: {tech.get('macd_signal', 0):.4f}")
    t3.metric("SMA 50",   f"${tech.get('sma_50', 0):,.2f}",
              "Above" if tech.get("above_sma_50") else "Below",
              delta_color="normal" if tech.get("above_sma_50") else "inverse")
    t4.metric("SMA 200",  f"${tech.get('sma_200', 0):,.2f}",
              "Above" if tech.get("above_sma_200") else "Below",
              delta_color="normal" if tech.get("above_sma_200") else "inverse")

    # Bollinger Band gauge
    bb_upper = tech.get("bollinger_upper", 0)
    bb_lower = tech.get("bollinger_lower", 0)
    if bb_upper and bb_lower:
        bb_range   = bb_upper - bb_lower
        bb_pos_pct = ((price - bb_lower) / bb_range * 100) if bb_range else 50
        col_bb1, col_bb2 = st.columns([1, 3])
        col_bb1.metric("Bollinger %B", f"{bb_pos_pct:.1f}%",
                       help="<0% below lower band, >100% above upper band")
        col_bb2.metric("BB Range",
                       f"${bb_lower:,.2f} – ${bb_upper:,.2f}")

    # ── Fair Value Section ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Fair Value Estimates")

    fv_cols = st.columns(5)
    fv_items = [
        ("Graham Number",    fv.get("graham_number", 0)),
        ("EPV",              fv.get("epv", 0)),
        ("DCF (10yr)",       fv.get("dcf_per_share", 0)),
        ("Analyst Mean",     fv.get("analyst_mean", 0)),
        ("Blended Target",   fv.get("blended_fair_value", 0)),
    ]
    for col, (label, val) in zip(fv_cols, fv_items):
        if val:
            diff = ((val - price) / price * 100) if price else 0
            col.metric(label, f"${val:,.2f}", f"{diff:+.1f}% vs current",
                       delta_color="normal" if diff >= 0 else "inverse")
        else:
            col.metric(label, "N/A")

    # Verdict banner
    verdict = fv.get("verdict", "N/A")
    upside  = fv.get("upside_pct", 0)
    vcolor  = {"UNDERVALUED": "#1a4731", "OVERVALUED": "#4a1a1a", "FAIRLY VALUED": "#2a2a1a"}
    vtcolor = {"UNDERVALUED": "#4ade80", "OVERVALUED": "#f87171", "FAIRLY VALUED": "#facc15"}
    v_bg = vcolor.get(verdict, "#1a1f2e")
    v_tc = vtcolor.get(verdict, "white")
    ana_rec = fv.get("analyst_recommendation", "N/A")
    num_ana = fv.get("num_analysts", 0)

    st.markdown(f"""
    <div style="background:{v_bg};border-radius:10px;padding:18px;margin-top:10px;text-align:center;">
      <span style="font-size:22px;font-weight:800;color:{v_tc}">{verdict}</span><br>
      <span style="color:#d1d5db">Blended upside: <b style="color:{v_tc}">{upside:+.1f}%</b>
      · Analyst consensus: <b>{ana_rec}</b> ({num_ana} analysts)</span>
    </div>
    """, unsafe_allow_html=True)

    # ── News Sentiment ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("News Sentiment")

    overall = sent.get("overall_sentiment", "NEUTRAL")
    score   = sent.get("sentiment_score", 0)
    summary = sent.get("summary", "")

    s1, s2 = st.columns([1, 4])
    with s1:
        st.markdown(f"**Overall**<br>{_sentiment_badge(overall)}", unsafe_allow_html=True)
        st.markdown(f"**Score:** {score:+.2f}")
        st.markdown(f"**Articles:** {sent.get('article_count', 0)}")
    with s2:
        st.info(summary)

    bull_pts = sent.get("bull_points", [])
    bear_pts = sent.get("bear_points", [])
    themes   = sent.get("key_themes", [])

    if bull_pts or bear_pts:
        bc, ec = st.columns(2)
        with bc:
            st.markdown("**Bull Points**")
            for p in bull_pts:
                st.markdown(f"✅ {p}")
        with ec:
            st.markdown("**Bear Points**")
            for p in bear_pts:
                st.markdown(f"⚠️ {p}")

    if themes:
        st.markdown("**Key Themes:** " + " · ".join([f"`{t}`" for t in themes]))

    # Recent headlines
    articles = sent.get("articles", [])
    if articles:
        with st.expander("Recent News Headlines", expanded=False):
            for art in articles[:10]:
                title  = art.get("title", "")
                source = art.get("source", "")
                date   = art.get("published_at", "")[:10]
                url    = art.get("url", "#")
                st.markdown(f"- **[{title}]({url})** — *{source}, {date}*")

    # ── Relative Strength vs S&P 500 ───────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Relative Strength vs S&P 500")
    with st.spinner("Loading relative strength data..."):
        try:
            rs_df = get_relative_strength(sym, period=period)
            if not rs_df.empty and sym in rs_df.columns and "SPY" in rs_df.columns:
                import plotly.graph_objects as _go
                fig_rs = _go.Figure()
                fig_rs.add_trace(_go.Scatter(
                    x=rs_df.index, y=rs_df[sym],
                    name=sym, line=dict(color="#60a5fa", width=2),
                ))
                fig_rs.add_trace(_go.Scatter(
                    x=rs_df.index, y=rs_df["SPY"],
                    name="S&P 500 (SPY)", line=dict(color="#9ca3af", width=1.5, dash="dash"),
                ))
                fig_rs.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                    font_color="white", height=300,
                    legend=dict(font=dict(color="white")),
                    yaxis=dict(color="white", gridcolor="#2d3748", title="Normalised (base=100)"),
                    xaxis=dict(color="white"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_rs, use_container_width=True)

                if "Ratio" in rs_df.columns:
                    last_ratio = float(rs_df["Ratio"].dropna().iloc[-1])
                    direction  = "outperforming" if last_ratio > 100 else "underperforming"
                    color      = "#4ade80" if last_ratio > 100 else "#f87171"
                    st.markdown(
                        f"<span style='color:{color}'>**{sym}** is currently "
                        f"**{direction}** the S&P 500 — ratio: {last_ratio:.1f}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Relative strength data unavailable.")
        except Exception as exc:
            st.warning(f"Could not load relative strength: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Market Rundown
# ─────────────────────────────────────────────────────────────────────────────

def _render_market_rundown_tab():
    st.header("Market Rundown")

    portfolio = st.session_state.get("portfolio", [])
    symbols_from_portfolio = [p["symbol"] for p in portfolio if p.get("symbol")]

    if symbols_from_portfolio:
        st.info(f"Generating rundown for portfolio: {', '.join(symbols_from_portfolio)}")
        run_symbols = symbols_from_portfolio
    else:
        default_syms = "MSFT,NVDA,GOOGL,META"
        user_syms = st.text_input(
            "Enter tickers for rundown (comma-separated)",
            value=default_syms,
        )
        run_symbols = [s.strip().upper() for s in user_syms.split(",") if s.strip()]

    if not run_symbols:
        return

    if st.button("Generate Market Rundown", type="primary", use_container_width=True):
        with st.spinner("Analysing portfolio stocks — this may take a moment..."):
            from modules.llm_agent import get_market_rundown
            rundown = get_market_rundown(run_symbols)

        st.markdown("---")
        st.markdown(rundown)

    # ── Quick comparison table ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Quick Comparison")

    if st.button("Load Comparison Data"):
        rows = []
        prog = st.progress(0)
        from modules.stock_data import get_current_price, get_key_ratios
        from modules.valuation  import get_fair_value

        for i, sym in enumerate(run_symbols):
            prog.progress((i + 1) / len(run_symbols))
            try:
                p  = get_current_price(sym)
                r  = get_key_ratios(sym)
                fv = get_fair_value(sym)
                rows.append({
                    "Symbol":          sym,
                    "Price":           f"${p['price']:,.2f}",
                    "Change %":        f"{p['change_pct']:+.2f}%",
                    "Mkt Cap":         _fmt_currency(p.get("market_cap", 0)),
                    "P/E (TTM)":       f"{r['pe_trailing']:.1f}" if r['pe_trailing'] else "N/A",
                    "Net Margin %":    f"{r['net_margin_pct']:.1f}%",
                    "Rev Growth %":    f"{r['revenue_growth_pct']:+.1f}%",
                    "Fair Value":      f"${fv['blended_fair_value']:,.2f}" if fv['blended_fair_value'] else "N/A",
                    "Upside %":        f"{fv['upside_pct']:+.1f}%" if fv['blended_fair_value'] else "N/A",
                    "Verdict":         fv.get("verdict", "N/A"),
                })
            except Exception as exc:
                rows.append({"Symbol": sym, "Error": str(exc)})

        prog.empty()
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 – AI Chat
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat_tab():
    st.header("AI Stock Analyst Chat")
    st.caption("Ask anything about stocks — prices, fundamentals, sentiment, fair value, strategy.")

    if not config.ANTHROPIC_API_KEY:
        st.error(
            "**ANTHROPIC_API_KEY is not set.**\n\n"
            "Add it to your `.env` file:\n```\nANTHROPIC_API_KEY=sk-ant-...\n```"
        )
        return

    # Lazy-init agent into session state (re-tried every run until it succeeds)
    if st.session_state["agent"] is None:
        agent, err = _init_agent()
        if err:
            st.error(
                f"**Failed to initialise the AI agent.**\n\n"
                f"```\n{err}\n```\n\n"
                "Make sure all packages are installed: `pip install -r requirements.txt`"
            )
            return
        st.session_state["agent"] = agent

    agent = st.session_state["agent"]

    # ── Chat history display ───────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="📈"):
                    st.markdown(msg["content"])

    # ── Input ──────────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "E.g. 'What's the fair value of NVDA?' or 'Compare Apple and Microsoft margins'"
    )

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="📈"):
            with st.spinner("Analysing..."):
                try:
                    response = agent.chat(user_input)
                except Exception as exc:
                    response = f"Error: {exc}"

            st.markdown(response)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

    # ── Controls ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = []
            if agent:
                agent.reset()
            st.rerun()

    # ── Suggested questions ────────────────────────────────────────────────────
    with st.expander("Suggested questions"):
        suggestions = [
            "What is the fair value of GOOGL and is it overvalued?",
            "Analyse the sentiment for TSLA based on recent news",
            "Compare MSFT and GOOGL fundamentals",
            "What are the key risks for NVDA right now?",
            "Explain the MACD signal for META",
            "Which of my portfolio stocks has the best upside?",
            "What does the RSI say about AMZN right now?",
        ]
        for q in suggestions:
            if st.button(q, key=f"sug_{q[:20]}"):
                # Trigger via rerun trick
                st.session_state["chat_history"].append({"role": "user", "content": q})
                if agent:
                    with st.spinner("Analysing..."):
                        resp = agent.chat(q)
                    st.session_state["chat_history"].append({"role": "assistant", "content": resp})
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Risk section (embedded in Portfolio tab)
# ─────────────────────────────────────────────────────────────────────────────

def _render_portfolio_risk_section(portfolio: list):
    """Risk metrics + rebalancing — shown at the bottom of the Portfolio tab."""
    if not portfolio or not any(float(p.get("market_value", 0) or 0) > 0 for p in portfolio):
        return

    st.markdown("---")
    st.subheader("Portfolio Risk Analytics")

    period = st.selectbox("Analysis period", ["6mo", "1y", "2y"], index=1, key="risk_period")

    if st.button("Compute Risk Metrics", key="compute_risk"):
        with st.spinner("Calculating risk metrics..."):
            metrics = get_portfolio_risk_metrics(portfolio, period=period)

        if "error" in metrics:
            st.warning(metrics["error"])
        else:
            # ── Portfolio-level KPIs ────────────────────────────────────────
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Ann. Return",    f"{metrics['ann_return_pct']:+.1f}%")
            r2.metric("Ann. Volatility", f"{metrics['ann_vol_pct']:.1f}%")
            r3.metric("Sharpe Ratio",   f"{metrics['sharpe_ratio']:.2f}",
                      help="Risk-adjusted return vs 5% risk-free rate")
            r4.metric("Max Drawdown",   f"{metrics['max_drawdown_pct']:.1f}%",
                      delta_color="inverse")
            r5.metric("Portfolio Beta", f"{metrics['beta']:.2f}" if metrics.get("beta") else "N/A",
                      help="Sensitivity to S&P 500 moves")

            st.caption(f"1-day 95% Value at Risk (parametric): **{metrics['var_95_pct']:.2f}%** of portfolio value")

            # ── Per-stock breakdown ─────────────────────────────────────────
            st.markdown("**Per-Stock Risk Breakdown**")
            sm_df = pd.DataFrame(metrics["stock_metrics"])
            sm_df.columns = [c.replace("_", " ").title() for c in sm_df.columns]
            st.dataframe(sm_df, use_container_width=True, hide_index=True)

            # ── Correlation matrix ──────────────────────────────────────────
            symbols = [p["symbol"] for p in portfolio if float(p.get("market_value", 0) or 0) > 0]
            if len(symbols) >= 2:
                st.markdown("**Return Correlation Matrix**")
                with st.spinner("Loading correlation data..."):
                    corr = get_correlation_matrix(symbols, period=period)
                if not corr.empty:
                    import plotly.express as _px
                    fig_corr = _px.imshow(
                        corr,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        text_auto=True,
                    )
                    fig_corr.update_layout(
                        paper_bgcolor="#0e1117",
                        plot_bgcolor="#1a1f2e",
                        font_color="white",
                        height=400,
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Rebalancing ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Rebalancing Suggestions")
    st.caption("Enter target allocations (%) for each position. Totals should sum to 100.")

    symbols  = [p["symbol"] for p in portfolio]
    targets: dict[str, float] = {}
    n_cols   = min(4, len(symbols))
    cols     = st.columns(n_cols)
    eq_weight = round(100 / len(symbols), 1) if symbols else 0

    for i, sym in enumerate(symbols):
        with cols[i % n_cols]:
            targets[sym] = st.number_input(
                sym, min_value=0.0, max_value=100.0,
                value=eq_weight, step=1.0,
                key=f"tgt_{sym}",
            )

    total_target = sum(targets.values())
    st.caption(f"Total target allocation: **{total_target:.1f}%** (should be 100%)")

    if st.button("Generate Rebalancing Plan", key="rebal_btn"):
        plan = get_rebalancing_suggestions(portfolio, targets)
        if plan.empty:
            st.info("No rebalancing needed or no portfolio data.")
        else:
            def _color_drift(val):
                try:
                    v = float(val)
                    if abs(v) < 1:    return "color: #9ca3af"
                    return "color: #f87171" if v > 0 else "color: #4ade80"
                except Exception:
                    return ""
            styled = plan.style.applymap(_color_drift, subset=["Drift"])
            st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 – Analytics (Earnings Calendar, Sector Heatmap, Peers, Insider)
# ─────────────────────────────────────────────────────────────────────────────

def _render_analytics_tab():
    st.header("Market Analytics")

    # ── Economic Indicators ────────────────────────────────────────────────────
    st.subheader("Economic Indicators")
    with st.spinner("Loading macro data..."):
        macro = get_market_overview()

    if macro:
        cols = st.columns(min(5, len(macro)))
        for i, item in enumerate(macro[:10]):
            with cols[i % 5]:
                delta_color = "normal" if item["change_pct"] >= 0 else "inverse"
                st.metric(
                    item["label"],
                    item["fmt_value"],
                    f"{item['change_pct']:+.2f}%",
                    delta_color=delta_color,
                )

    # ── Yield Curve ─────────────────────────────────────────────────────────────
    with st.expander("US Treasury Yield Curve", expanded=False):
        with st.spinner("Loading yield curve..."):
            yc = get_yield_curve()
        if not yc.empty:
            import plotly.express as _px
            fig_yc = _px.line(yc, x="Maturity", y="Yield (%)", markers=True)
            fig_yc.update_traces(line_color="#60a5fa", marker_color="#60a5fa")
            fig_yc.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
                font_color="white", height=250,
                xaxis=dict(color="white"), yaxis=dict(color="white", gridcolor="#2d3748"),
            )
            st.plotly_chart(fig_yc, use_container_width=True)

    st.markdown("---")

    # ── Sector Heatmap ──────────────────────────────────────────────────────────
    st.subheader("Sector Performance Heatmap")
    heat_period = st.radio(
        "Period", ["1d", "1w", "1mo", "ytd"],
        horizontal=True, index=2, key="heat_period",
    )

    with st.spinner("Loading sector data..."):
        sector_df = get_sector_heatmap(heat_period)

    if not sector_df.empty:
        import plotly.express as _px
        # Bar chart
        sector_df_sorted = sector_df.sort_values("change_pct", ascending=True)
        colors = ["#4ade80" if v >= 0 else "#f87171" for v in sector_df_sorted["change_pct"]]
        fig_sec = go.Figure(go.Bar(
            x=sector_df_sorted["change_pct"],
            y=sector_df_sorted["sector"],
            orientation="h",
            marker_color=colors,
            text=sector_df_sorted["change_pct"].apply(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        ))
        fig_sec.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
            font_color="white", height=380,
            xaxis=dict(color="white", gridcolor="#2d3748", title="Change (%)"),
            yaxis=dict(color="white"),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

        # Table with YTD
        disp = sector_df[["sector", "etf", "price", "change_pct", "ytd_pct"]].copy()
        disp.columns = ["Sector", "ETF", "Price", f"Change % ({heat_period})", "YTD %"]
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Earnings Calendar ───────────────────────────────────────────────────────
    st.subheader("Earnings Calendar")

    portfolio   = st.session_state.get("portfolio", [])
    port_syms   = [p["symbol"] for p in portfolio if p.get("symbol")]
    default_sym = ", ".join(port_syms) if port_syms else "AAPL, MSFT, GOOGL, NVDA, META"
    earn_input  = st.text_input(
        "Tickers (comma-separated)", value=default_sym, key="earn_tickers",
    )
    earn_syms   = [s.strip().upper() for s in earn_input.split(",") if s.strip()]

    if earn_syms and st.button("Load Earnings Calendar", key="load_earnings"):
        with st.spinner("Fetching earnings dates..."):
            cal = get_earnings_calendar(earn_syms)

        if cal:
            cal_rows = []
            for c in cal:
                eps_est = f"${c['eps_estimate']:.2f}" if c.get("eps_estimate") is not None else "N/A"
                rev_est = f"${c['revenue_estimate']/1e9:.2f}B" if c.get("revenue_estimate") else "N/A"
                surp    = f"{c['last_surprise_pct']:+.1f}%" if c.get("last_surprise_pct") is not None else "N/A"
                cal_rows.append({
                    "Symbol":          c["symbol"],
                    "Name":            c.get("name", c["symbol"]),
                    "Earnings Date":   c.get("earnings_date", "N/A"),
                    "EPS Estimate":    eps_est,
                    "Rev Estimate":    rev_est,
                    "Last Surprise":   surp,
                    "Trailing EPS":    f"${c['trailing_eps']:.2f}" if c.get("trailing_eps") else "N/A",
                    "Sector":          c.get("sector", "N/A"),
                })
            st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Peer Comparison ─────────────────────────────────────────────────────────
    st.subheader("Peer Comparison")
    peer_sym = st.text_input(
        "Stock to compare", value=st.session_state["selected_ticker"],
        key="peer_ticker",
    ).upper().strip()

    if peer_sym and st.button("Compare with Peers", key="peer_btn"):
        with st.spinner(f"Fetching peer data for {peer_sym}..."):
            peer_df = get_peer_comparison(peer_sym)

        if not peer_df.empty:
            # Highlight the target ticker row
            def _highlight_target(row):
                if row["Ticker"] == peer_sym:
                    return ["background-color: #1e3a5f"] * len(row)
                return [""] * len(row)
            styled_peers = peer_df.style.apply(_highlight_target, axis=1)
            st.dataframe(styled_peers, use_container_width=True, hide_index=True)
        else:
            st.info(f"No peer data found for {peer_sym}.")

    st.markdown("---")

    # ── Insider & Institutional Activity ────────────────────────────────────────
    st.subheader("Insider & Institutional Activity")
    insider_sym = st.text_input(
        "Ticker", value=st.session_state["selected_ticker"], key="insider_ticker",
    ).upper().strip()

    if insider_sym and st.button("Load Insider Data", key="insider_btn"):
        with st.spinner(f"Loading insider data for {insider_sym}..."):
            activity = get_insider_activity(insider_sym)

        # Major holders summary
        mh = activity.get("major_holders", {})
        if mh:
            st.markdown("**Ownership Breakdown**")
            mh_cols = st.columns(min(4, len(mh)))
            for i, (label, val) in enumerate(list(mh.items())[:4]):
                mh_cols[i].metric(label, str(val))

        # Insider transactions
        ins_df = activity.get("insider_transactions", pd.DataFrame())
        if not ins_df.empty:
            st.markdown("**Recent Insider Transactions**")
            st.dataframe(ins_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No insider transaction data available.")

        # Institutional holders
        inst_df = activity.get("institutional_holders", pd.DataFrame())
        if not inst_df.empty:
            st.markdown("**Top Institutional Holders**")
            st.dataframe(inst_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No institutional holder data available.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 – AI Research (SEC Filing Analyzer + Earnings Summarizer)
# ─────────────────────────────────────────────────────────────────────────────

def _render_research_tab():
    st.header("AI Research")
    st.caption("Powered by SEC EDGAR (free, no API key) + Claude AI analysis.")

    if not config.ANTHROPIC_API_KEY:
        st.error("**ANTHROPIC_API_KEY** is not set. Add it to your secrets to use AI Research.")
        return

    research_sym = st.text_input(
        "Ticker Symbol",
        value=st.session_state["selected_ticker"],
        key="research_ticker",
    ).upper().strip()

    if not research_sym:
        return

    tab_earn, tab_annual, tab_quarterly = st.tabs([
        "📋 Earnings Release (8-K)",
        "📄 Annual Report (10-K)",
        "📊 Quarterly Report (10-Q)",
    ])

    with tab_earn:
        st.markdown(f"Summarise the most recent **earnings release (8-K)** for **{research_sym}**.")

        # Show filing list
        with st.expander("Available 8-K Filings", expanded=False):
            with st.spinner("Fetching SEC filing list..."):
                filings_8k = get_recent_filings(research_sym, form_types=["8-K"], max_count=5)
            if filings_8k:
                for f in filings_8k:
                    st.markdown(f"- **{f['form']}** filed {f['date']} — {f['desc']}")
            else:
                st.info("No 8-K filings found. Check the ticker symbol.")

        if st.button("Summarise Earnings Release", key="earn_sum_btn", type="primary"):
            with st.spinner("Fetching filing from SEC EDGAR and analysing with Claude..."):
                result = summarize_earnings(research_sym)
            st.markdown(result)

    with tab_annual:
        st.markdown(f"Analyse the most recent **annual report (10-K)** for **{research_sym}**.")

        with st.expander("Available 10-K Filings", expanded=False):
            with st.spinner("Fetching SEC filing list..."):
                filings_10k = get_recent_filings(research_sym, form_types=["10-K"], max_count=3)
            if filings_10k:
                for f in filings_10k:
                    st.markdown(f"- **{f['form']}** filed {f['date']} — {f['desc']}")
            else:
                st.info("No 10-K filings found.")

        if st.button("Analyse Annual Report", key="ann_btn", type="primary"):
            with st.spinner("Fetching 10-K from SEC EDGAR and analysing with Claude..."):
                result = analyze_filing(research_sym, "10-K")
            st.markdown(result)

    with tab_quarterly:
        st.markdown(f"Analyse the most recent **quarterly report (10-Q)** for **{research_sym}**.")

        with st.expander("Available 10-Q Filings", expanded=False):
            with st.spinner("Fetching SEC filing list..."):
                filings_10q = get_recent_filings(research_sym, form_types=["10-Q"], max_count=4)
            if filings_10q:
                for f in filings_10q:
                    st.markdown(f"- **{f['form']}** filed {f['date']} — {f['desc']}")
            else:
                st.info("No 10-Q filings found.")

        if st.button("Analyse Quarterly Report", key="qtr_btn", type="primary"):
            with st.spinner("Fetching 10-Q from SEC EDGAR and analysing with Claude..."):
                result = analyze_filing(research_sym, "10-Q")
            st.markdown(result)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _render_sidebar()

    st.markdown("""
    <div style="text-align:center;padding:4px 0 20px 0;">
      <span style="font-size:2rem;font-weight:900;color:#60a5fa;">📈 MarketMind</span>
      <span style="color:#9ca3af;margin-left:12px;font-size:1rem;">LLM-Powered Stock Market Analyzer</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💼 Portfolio",
        "📊 Stock Dashboard",
        "📰 Market Rundown",
        "🌐 Analytics",
        "🔬 AI Research",
        "🤖 AI Analyst Chat",
    ])

    with tab1:
        _render_portfolio_tab()

    with tab2:
        _render_stock_dashboard_tab()

    with tab3:
        _render_market_rundown_tab()

    with tab4:
        _render_analytics_tab()

    with tab5:
        _render_research_tab()

    with tab6:
        _render_chat_tab()


if __name__ == "__main__":
    main()
