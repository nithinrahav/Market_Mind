"""
Macro module: Economic Indicators, Relative Strength vs S&P 500, Yield Curve.
All data sourced from yfinance proxy tickers — no API key required.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache

import config

log = logging.getLogger(__name__)
_cache: TTLCache = TTLCache(maxsize=128, ttl=config.CACHE_TTL)

# ── Macro proxy tickers ────────────────────────────────────────────────────────
MACRO_PROXIES: dict[str, str] = {
    "S&P 500":         "^GSPC",
    "NASDAQ 100":      "^NDX",
    "Dow Jones":       "^DJI",
    "Russell 2000":    "^RUT",
    "VIX":             "^VIX",
    "10Y Treasury":    "^TNX",
    "2Y Treasury":     "^IRX",
    "Gold":            "GC=F",
    "Crude Oil (WTI)": "CL=F",
    "USD Index":       "DX-Y.NYB",
}

YIELD_CURVE_TICKERS: dict[str, str] = {
    "3M":  "^IRX",
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else default
    except (TypeError, ValueError):
        return default


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


# ─────────────────────────────────────────────────────────────────────────────
# Market Overview
# ─────────────────────────────────────────────────────────────────────────────

def get_market_overview() -> list[dict]:
    """
    Current price + daily change for all MACRO_PROXIES.
    Returns list of dicts for display cards.
    """
    key = "macro_overview"
    if key in _cache:
        return _cache[key]

    results: list[dict] = []
    tickers = list(MACRO_PROXIES.values())

    try:
        raw = yf.download(
            tickers,
            period="5d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        for label, ticker in MACRO_PROXIES.items():
            try:
                prices = _extract_close(raw, ticker)
                if len(prices) < 2:
                    continue

                current = float(prices.iloc[-1])
                prev    = float(prices.iloc[-2])
                chg     = current - prev
                chg_pct = chg / prev * 100 if prev else 0.0

                if ticker in ("^TNX", "^IRX", "^FVX", "^TYX"):
                    fmt_val = f"{current:.2f}%"
                elif ticker in ("^VIX", "DX-Y.NYB"):
                    fmt_val = f"{current:.2f}"
                else:
                    fmt_val = f"{current:,.2f}"

                results.append({
                    "label":      label,
                    "ticker":     ticker,
                    "value":      current,
                    "fmt_value":  fmt_val,
                    "change":     chg,
                    "change_pct": round(chg_pct, 2),
                })
            except Exception as exc:
                log.warning("Macro indicator %s failed: %s", ticker, exc)

    except Exception as exc:
        log.error("Market overview download failed: %s", exc)

    if results:
        _cache[key] = results
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Yield Curve
# ─────────────────────────────────────────────────────────────────────────────

def get_yield_curve() -> pd.DataFrame:
    """
    Return US Treasury yield curve data.
    DataFrame columns: Maturity, Yield (%)
    """
    key = "yield_curve"
    if key in _cache:
        return _cache[key]

    rows = []
    tickers = list(YIELD_CURVE_TICKERS.values())

    try:
        raw = yf.download(
            tickers,
            period="5d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        for maturity, ticker in YIELD_CURVE_TICKERS.items():
            try:
                prices = _extract_close(raw, ticker)
                if not prices.empty:
                    rows.append({
                        "Maturity":  maturity,
                        "Yield (%)": round(float(prices.iloc[-1]), 3),
                    })
            except Exception:
                pass
    except Exception as exc:
        log.error("Yield curve download failed: %s", exc)

    order = {"3M": 0, "5Y": 1, "10Y": 2, "30Y": 3}
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Maturity", key=lambda x: x.map(order))
        _cache[key] = df
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Relative Strength vs S&P 500
# ─────────────────────────────────────────────────────────────────────────────

def get_relative_strength(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Compute normalised price (base=100) for symbol vs SPY.
    Returns DataFrame with columns: {symbol}, SPY, Ratio.
    Ratio = symbol_norm / SPY_norm * 100 (rising = outperforming).
    """
    key = f"rel_strength_{symbol}_{period}"
    if key in _cache:
        return _cache[key]

    sym_upper = symbol.upper()
    tickers   = [sym_upper, "SPY"]

    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        close = pd.DataFrame()
        for t in tickers:
            s = _extract_close(raw, t)
            if not s.empty:
                close[t] = s

        if close.empty or len(close) < 5:
            return pd.DataFrame()

        # Normalise to 100 at start
        result = pd.DataFrame(index=close.index)
        for t in tickers:
            if t in close.columns:
                s = close[t].dropna()
                result[t] = (s / s.iloc[0]) * 100

        if sym_upper in result.columns and "SPY" in result.columns:
            result["Ratio"] = result[sym_upper] / result["SPY"] * 100

        _cache[key] = result
        return result

    except Exception as exc:
        log.error("Relative strength failed for %s: %s", symbol, exc)
        return pd.DataFrame()
