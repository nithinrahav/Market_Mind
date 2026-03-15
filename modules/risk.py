"""
Portfolio risk analytics: beta, Sharpe ratio, max drawdown, correlation matrix,
Value-at-Risk, and rebalancing suggestions.
All calculations use numpy/pandas — no additional dependencies required.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache

import config

log = logging.getLogger(__name__)
_cache: TTLCache = TTLCache(maxsize=64, ttl=config.CACHE_TTL)

RISK_FREE_RATE = 0.05   # ~5% annual; update as needed
TRADING_DAYS   = 252


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_close(raw: Any, ticker: str) -> pd.Series:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                return raw[ticker]["Close"].dropna()
        else:
            return raw["Close"].dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def _download_returns(symbols: list[str], benchmark: str, period: str) -> pd.DataFrame:
    """Download adjusted closes and return daily pct-change returns."""
    all_tickers = list(dict.fromkeys(symbols + [benchmark]))
    try:
        raw = yf.download(
            all_tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        close = pd.DataFrame({
            t: _extract_close(raw, t)
            for t in all_tickers
        })
        close = close.dropna(how="all")
        return close.pct_change().dropna()
    except Exception as exc:
        log.error("Returns download failed: %s", exc)
        return pd.DataFrame()


def _beta(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    aligned = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0
    cov     = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    bvar    = float(np.var(aligned.iloc[:, 1]))
    return float(cov[0, 1] / bvar) if bvar > 0 else 1.0


def _sharpe(daily_ret: pd.Series) -> float:
    if daily_ret.empty or daily_ret.std() == 0:
        return 0.0
    ann_ret = float(daily_ret.mean() * TRADING_DAYS)
    ann_vol = float(daily_ret.std() * np.sqrt(TRADING_DAYS))
    return round((ann_ret - RISK_FREE_RATE) / ann_vol, 2) if ann_vol > 0 else 0.0


def _max_drawdown(daily_ret: pd.Series) -> float:
    cum      = (1 + daily_ret).cumprod()
    roll_max = cum.cummax()
    return round(float((cum / roll_max - 1).min()) * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio_risk_metrics(
    portfolio: list[dict],
    period: str = "1y",
    benchmark: str = "SPY",
) -> dict:
    """
    Compute portfolio-level and per-stock risk metrics.

    portfolio: list of dicts with 'symbol' and 'market_value'.
    Returns dict with ann_return_pct, ann_vol_pct, sharpe_ratio,
    max_drawdown_pct, beta, var_95_pct, and per-stock breakdown.
    """
    positions = [p for p in portfolio if float(p.get("market_value", 0) or 0) > 0]
    if len(positions) < 2:
        return {"error": "Need at least 2 positions with market value to compute risk metrics."}

    symbols   = [p["symbol"] for p in positions]
    values    = [float(p["market_value"]) for p in positions]
    total_val = sum(values)
    weights   = np.array([v / total_val for v in values])

    key = f"risk_{'_'.join(sorted(symbols))}_{period}"
    if key in _cache:
        return _cache[key]

    returns = _download_returns(symbols, benchmark, period)
    if returns.empty:
        return {"error": "Could not download price history."}

    bench_ret = returns.get(benchmark, pd.Series(dtype=float))
    avail     = [s for s in symbols if s in returns.columns]
    if len(avail) < 2:
        return {"error": "Insufficient price history for risk calculations."}

    stock_rets = returns[avail]
    w_aligned  = np.array([weights[symbols.index(s)] for s in avail])
    w_aligned  = w_aligned / w_aligned.sum()

    port_ret = pd.Series(
        stock_rets.values @ w_aligned,
        index=stock_rets.index,
    )

    ann_ret   = round(float(port_ret.mean() * TRADING_DAYS) * 100, 2)
    ann_vol   = round(float(port_ret.std() * np.sqrt(TRADING_DAYS)) * 100, 2)
    sharpe    = _sharpe(port_ret)
    max_dd    = _max_drawdown(port_ret)
    port_beta = round(_beta(port_ret, bench_ret), 2) if not bench_ret.empty else None

    # 1-day 95% Value at Risk (parametric, normal distribution)
    var_95 = round(float(port_ret.mean() - 1.645 * port_ret.std()) * 100, 2)

    stock_metrics = []
    for sym in avail:
        s_ret = stock_rets[sym]
        stock_metrics.append({
            "symbol":     sym,
            "weight_pct": round(float(w_aligned[avail.index(sym)]) * 100, 1),
            "ann_return": round(float(s_ret.mean() * TRADING_DAYS) * 100, 1),
            "ann_vol":    round(float(s_ret.std() * np.sqrt(TRADING_DAYS)) * 100, 1),
            "sharpe":     _sharpe(s_ret),
            "max_dd":     _max_drawdown(s_ret),
            "beta":       round(_beta(s_ret, bench_ret), 2) if not bench_ret.empty else None,
        })

    result = {
        "period":          period,
        "benchmark":       benchmark,
        "ann_return_pct":  ann_ret,
        "ann_vol_pct":     ann_vol,
        "sharpe_ratio":    sharpe,
        "max_drawdown_pct": max_dd,
        "beta":            port_beta,
        "var_95_pct":      var_95,
        "stock_metrics":   stock_metrics,
    }
    _cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def get_correlation_matrix(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """Return pairwise return correlation matrix for the given symbols."""
    key = f"corr_{'_'.join(sorted(symbols))}_{period}"
    if key in _cache:
        return _cache[key]

    returns = _download_returns(symbols, "SPY", period)
    avail   = [s for s in symbols if s in returns.columns]
    if len(avail) < 2:
        return pd.DataFrame()

    df = returns[avail].corr().round(2)
    _cache[key] = df
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rebalancing Suggestions
# ─────────────────────────────────────────────────────────────────────────────

def get_rebalancing_suggestions(
    portfolio: list[dict],
    target_allocations: dict[str, float],
) -> pd.DataFrame:
    """
    Compare current portfolio allocation vs target and suggest trades.

    portfolio: list of dicts with 'symbol' and 'market_value'.
    target_allocations: {symbol: target_pct} — values should sum to 100.
    Returns DataFrame sorted by abs(drift), largest first.
    """
    total_value = sum(float(p.get("market_value", 0) or 0) for p in portfolio)
    if total_value == 0:
        return pd.DataFrame()

    rows = []
    for pos in portfolio:
        sym     = pos["symbol"]
        mv      = float(pos.get("market_value", 0) or 0)
        cur_pct = mv / total_value * 100
        tgt_pct = float(target_allocations.get(sym, 0))
        drift   = cur_pct - tgt_pct
        trade   = abs(drift / 100) * total_value

        if abs(drift) < 1.0:
            action = "Hold"
        elif drift > 0:
            action = f"Reduce  (−${trade:,.0f})"
        else:
            action = f"Add  (+${trade:,.0f})"

        rows.append({
            "Symbol":    sym,
            "Value":     f"${mv:,.0f}",
            "Current %": round(cur_pct, 1),
            "Target %":  round(tgt_pct, 1),
            "Drift":     round(drift, 1),
            "Action":    action,
        })

    return pd.DataFrame(rows).sort_values("Drift", key=abs, ascending=False)
