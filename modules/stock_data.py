"""
Stock data module – fetches prices, fundamentals and key ratios via yfinance.
All results are cached with cachetools to avoid hammering Yahoo Finance.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache, cached
from tenacity import retry, stop_after_attempt, wait_exponential

import config

log = logging.getLogger(__name__)

# ── In-memory cache (keyed by function + args) ────────────────────────────────
_cache: TTLCache = TTLCache(maxsize=256, ttl=config.CACHE_TTL)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.upper())


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Price history
# ─────────────────────────────────────────────────────────────────────────────

def get_price_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    period: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
    interval: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
    """
    key = f"price_hist_{symbol}_{period}_{interval}"
    if key in _cache:
        return _cache[key]

    tk = _ticker(symbol)
    df = tk.history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    _cache[key] = df
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Current snapshot
# ─────────────────────────────────────────────────────────────────────────────

def get_current_price(symbol: str) -> dict:
    """
    Returns dict with current price info:
      price, change, change_pct, volume, market_cap, day_high, day_low,
      52w_high, 52w_low, avg_volume
    """
    key = f"cur_price_{symbol}"
    if key in _cache:
        return _cache[key]

    tk = _ticker(symbol)
    info = tk.info

    price      = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    prev_close = _safe_float(info.get("previousClose") or info.get("regularMarketPreviousClose"))
    change     = round(price - prev_close, 4)
    change_pct = round((change / prev_close * 100) if prev_close else 0, 2)

    result = {
        "symbol":      symbol.upper(),
        "price":       price,
        "prev_close":  prev_close,
        "change":      change,
        "change_pct":  change_pct,
        "volume":      int(info.get("volume") or info.get("regularMarketVolume") or 0),
        "avg_volume":  int(info.get("averageVolume") or 0),
        "day_high":    _safe_float(info.get("dayHigh") or info.get("regularMarketDayHigh")),
        "day_low":     _safe_float(info.get("dayLow") or info.get("regularMarketDayLow")),
        "52w_high":    _safe_float(info.get("fiftyTwoWeekHigh")),
        "52w_low":     _safe_float(info.get("fiftyTwoWeekLow")),
        "market_cap":  info.get("marketCap", 0),
        "name":        info.get("longName") or info.get("shortName", symbol),
        "sector":      info.get("sector", "N/A"),
        "industry":    info.get("industry", "N/A"),
        "currency":    info.get("currency", "USD"),
    }
    _cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Key financial ratios
# ─────────────────────────────────────────────────────────────────────────────

def get_key_ratios(symbol: str) -> dict:
    """
    Returns a dict of the most important fundamental ratios.
    """
    key = f"ratios_{symbol}"
    if key in _cache:
        return _cache[key]

    tk   = _ticker(symbol)
    info = tk.info

    # ── Valuation ──────────────────────────────────────────────────────────────
    pe_trailing  = _safe_float(info.get("trailingPE"))
    pe_forward   = _safe_float(info.get("forwardPE"))
    pb           = _safe_float(info.get("priceToBook"))
    ps           = _safe_float(info.get("priceToSalesTrailing12Months"))
    ev_ebitda    = _safe_float(info.get("enterpriseToEbitda"))
    ev_revenue   = _safe_float(info.get("enterpriseToRevenue"))
    peg          = _safe_float(info.get("pegRatio"))

    # ── Profitability ──────────────────────────────────────────────────────────
    gross_margin  = _safe_float(info.get("grossMargins"))
    oper_margin   = _safe_float(info.get("operatingMargins"))
    net_margin    = _safe_float(info.get("profitMargins"))
    roe           = _safe_float(info.get("returnOnEquity"))
    roa           = _safe_float(info.get("returnOnAssets"))
    ebitda_margin = _safe_float(info.get("ebitdaMargins"))

    # ── Growth ─────────────────────────────────────────────────────────────────
    rev_growth    = _safe_float(info.get("revenueGrowth"))
    earn_growth   = _safe_float(info.get("earningsGrowth"))
    earn_quarterly= _safe_float(info.get("earningsQuarterlyGrowth"))

    # ── Financial health ───────────────────────────────────────────────────────
    current_ratio = _safe_float(info.get("currentRatio"))
    quick_ratio   = _safe_float(info.get("quickRatio"))
    debt_equity   = _safe_float(info.get("debtToEquity"))
    interest_cov  = _safe_float(info.get("interestCoverage"))  # may be absent
    free_cashflow = info.get("freeCashflow", 0) or 0

    # ── Per share ──────────────────────────────────────────────────────────────
    eps_ttm       = _safe_float(info.get("trailingEps"))
    eps_forward   = _safe_float(info.get("forwardEps"))
    bvps          = _safe_float(info.get("bookValue"))
    dividend_yield= _safe_float(info.get("dividendYield"))
    payout_ratio  = _safe_float(info.get("payoutRatio"))

    # ── Beta / volatility ──────────────────────────────────────────────────────
    beta          = _safe_float(info.get("beta"))

    result = {
        # Valuation
        "pe_trailing":    pe_trailing,
        "pe_forward":     pe_forward,
        "price_to_book":  pb,
        "price_to_sales": ps,
        "ev_to_ebitda":   ev_ebitda,
        "ev_to_revenue":  ev_revenue,
        "peg_ratio":      peg,

        # Profitability (as %)
        "gross_margin_pct":   round(gross_margin * 100, 2),
        "oper_margin_pct":    round(oper_margin * 100, 2),
        "net_margin_pct":     round(net_margin * 100, 2),
        "ebitda_margin_pct":  round(ebitda_margin * 100, 2),
        "roe_pct":            round(roe * 100, 2),
        "roa_pct":            round(roa * 100, 2),

        # Growth (as %)
        "revenue_growth_pct":  round(rev_growth * 100, 2),
        "earnings_growth_pct": round(earn_growth * 100, 2),
        "earnings_qoq_pct":    round(earn_quarterly * 100, 2),

        # Financial health
        "current_ratio":  current_ratio,
        "quick_ratio":    quick_ratio,
        "debt_to_equity": debt_equity,
        "free_cashflow":  free_cashflow,

        # Per share
        "eps_ttm":        eps_ttm,
        "eps_forward":    eps_forward,
        "book_value_per_share": bvps,
        "dividend_yield_pct":  round(dividend_yield * 100, 2),
        "payout_ratio_pct":    round(payout_ratio * 100, 2),

        # Risk
        "beta": beta,
    }
    _cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Income / cashflow summary
# ─────────────────────────────────────────────────────────────────────────────

def get_financials_summary(symbol: str) -> dict:
    """
    Returns last 4 quarters / years of key line items as DataFrames inside a dict.
    Keys: income_annual, income_quarterly, cashflow_annual, balance_annual
    """
    key = f"financials_{symbol}"
    if key in _cache:
        return _cache[key]

    tk = _ticker(symbol)

    try:
        result = {
            "income_annual":     tk.financials,
            "income_quarterly":  tk.quarterly_financials,
            "cashflow_annual":   tk.cashflow,
            "balance_annual":    tk.balance_sheet,
        }
    except Exception as exc:
        log.warning("Could not fetch financials for %s: %s", symbol, exc)
        result = {}

    _cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Analyst targets
# ─────────────────────────────────────────────────────────────────────────────

def get_analyst_targets(symbol: str) -> dict:
    """Returns analyst price targets and recommendation summary."""
    key = f"analyst_{symbol}"
    if key in _cache:
        return _cache[key]

    tk   = _ticker(symbol)
    info = tk.info

    result = {
        "target_low":    _safe_float(info.get("targetLowPrice")),
        "target_mean":   _safe_float(info.get("targetMeanPrice")),
        "target_median": _safe_float(info.get("targetMedianPrice")),
        "target_high":   _safe_float(info.get("targetHighPrice")),
        "recommendation": info.get("recommendationKey", "N/A").upper(),
        "num_analysts":  int(info.get("numberOfAnalystOpinions") or 0),
    }

    _cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators (computed from price history)
# ─────────────────────────────────────────────────────────────────────────────

def get_technical_indicators(symbol: str) -> dict:
    """
    Returns common technical indicators computed from daily price history.
    Keys: sma_20, sma_50, sma_200, rsi_14, macd, macd_signal,
          bollinger_upper, bollinger_lower, atr_14
    """
    key = f"technical_{symbol}"
    if key in _cache:
        return _cache[key]

    df = get_price_history(symbol, period="1y", interval="1d")
    if df.empty:
        return {}

    close = df["Close"]

    # Moving averages
    sma_20  = close.rolling(20).mean().iloc[-1]
    sma_50  = close.rolling(50).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1]

    # RSI (14-period)
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi.iloc[-1])

    # MACD
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20-day, 2σ)
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ATR (14-period)
    high = df["High"]
    low  = df["Low"]
    tr   = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1])

    last_close = float(close.iloc[-1])

    result = {
        "current_price":   last_close,
        "sma_20":          round(float(sma_20), 4),
        "sma_50":          round(float(sma_50), 4),
        "sma_200":         round(float(sma_200), 4),
        "rsi_14":          round(rsi_14, 2),
        "macd":            round(float(macd_line.iloc[-1]), 4),
        "macd_signal":     round(float(macd_signal.iloc[-1]), 4),
        "macd_histogram":  round(float(macd_line.iloc[-1] - macd_signal.iloc[-1]), 4),
        "bollinger_upper": round(float(bb_upper.iloc[-1]), 4),
        "bollinger_lower": round(float(bb_lower.iloc[-1]), 4),
        "atr_14":          round(atr_14, 4),
        # Signal interpretations
        "above_sma_20":    last_close > float(sma_20),
        "above_sma_50":    last_close > float(sma_50),
        "above_sma_200":   last_close > float(sma_200),
    }
    _cache[key] = result
    return result
