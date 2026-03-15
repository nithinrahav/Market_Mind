"""
Analytics module: Earnings Calendar, Sector Heatmap, Peer Comparison,
and Insider & Institutional Activity.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

import config

log = logging.getLogger(__name__)
_cache: TTLCache = TTLCache(maxsize=256, ttl=config.CACHE_TTL)

# ── Sector ETFs ────────────────────────────────────────────────────────────────
SECTOR_ETFS: dict[str, str] = {
    "Technology":       "XLK",
    "Financials":       "XLF",
    "Health Care":      "XLV",
    "Consumer Disc.":   "XLY",
    "Industrials":      "XLI",
    "Communication":    "XLC",
    "Consumer Staples": "XLP",
    "Energy":           "XLE",
    "Utilities":        "XLU",
    "Real Estate":      "XLRE",
    "Materials":        "XLB",
}

# ── Curated peer groups (industry keyword → tickers) ─────────────────────────
PEER_MAP: dict[str, list[str]] = {
    "semiconductor":           ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU", "AMAT"],
    "internet content":        ["GOOGL", "META", "SNAP", "PINS", "RDDT"],
    "software—application":    ["MSFT", "CRM", "ADBE", "NOW", "ORCL", "WDAY", "INTU"],
    "software—infrastructure": ["MSFT", "AMZN", "GOOGL", "SNOW", "DDOG", "MDB"],
    "consumer electronics":    ["AAPL", "DELL", "HPQ", "SONY"],
    "e-commerce":              ["AMZN", "SHOP", "EBAY", "ETSY"],
    "electric vehicle":        ["TSLA", "RIVN", "LCID", "NIO", "LI", "XPEV"],
    "streaming":               ["NFLX", "DIS", "PARA", "WBD"],
    "investment banking":      ["JPM", "GS", "MS", "BAC", "C", "WFC"],
    "drug manufacturers":      ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY"],
    "biotechnology":           ["GILD", "BIIB", "REGN", "VRTX", "MRNA", "AMGN"],
    "automotive":              ["F", "GM", "TSLA", "TM", "STLA"],
    "aerospace":               ["BA", "LMT", "RTX", "NOC", "GD"],
    "specialty retail":        ["WMT", "TGT", "COST", "HD", "LOW"],
    "oil & gas":               ["XOM", "CVX", "COP", "SLB", "EOG"],
    "insurance":               ["BRK-B", "MET", "PRU", "AIG", "PGR"],
    "telecom":                 ["T", "VZ", "TMUS"],
    "airlines":                ["DAL", "UAL", "AAL", "LUV", "ALK"],
    "payment":                 ["V", "MA", "PYPL", "SQ", "AXP"],
    "medical devices":         ["MDT", "ABT", "BSX", "SYK", "ZBH"],
    "banking":                 ["JPM", "BAC", "WFC", "C", "USB", "PNC"],
    "reit":                    ["PLD", "AMT", "EQIX", "CCI", "PSA"],
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.upper())


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else default
    except (TypeError, ValueError):
        return default


def _find_peers(symbol: str, industry: str, sector: str) -> list[str]:
    """Match ticker to a peer group by longest keyword match."""
    text = (industry + " " + sector).lower()
    best: list[str] = []
    for keyword, peers in PEER_MAP.items():
        if keyword in text:
            candidates = [p for p in peers if p.upper() != symbol.upper()]
            if len(candidates) > len(best):
                best = candidates
    return best[:6]


def _extract_close(raw: Any, ticker: str) -> pd.Series:
    """Safely extract Close prices from a yfinance download result."""
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
# Earnings Calendar
# ─────────────────────────────────────────────────────────────────────────────

def get_earnings_calendar(symbols: list[str]) -> list[dict]:
    """
    Return upcoming earnings info for a list of tickers.
    Gracefully handles yfinance API inconsistencies across versions.
    """
    results = []
    for sym in symbols:
        try:
            tk   = _ticker(sym)
            info = tk.info or {}
            cal  = getattr(tk, "calendar", None)

            earnings_date    = None
            eps_estimate     = None
            revenue_estimate = None

            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                earnings_date    = (ed[0] if isinstance(ed, list) and ed else ed)
                eps_estimate     = cal.get("EPS Estimate")
                revenue_estimate = cal.get("Revenue Estimate")
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                try:
                    earnings_date = cal.columns[0]
                    if "EPS Estimate" in cal.index:
                        eps_estimate = cal.loc["EPS Estimate", cal.columns[0]]
                except Exception:
                    pass

            if hasattr(earnings_date, "strftime"):
                earnings_date = earnings_date.strftime("%Y-%m-%d")
            elif earnings_date is not None:
                earnings_date = str(earnings_date)[:10]

            # Last earnings surprise from earnings_dates
            last_surprise = None
            try:
                ed_df = tk.earnings_dates
                if ed_df is not None and not ed_df.empty and "Surprise(%)" in ed_df.columns:
                    last_surprise = float(ed_df["Surprise(%)"].dropna().iloc[0])
            except Exception:
                pass

            results.append({
                "symbol":           sym.upper(),
                "name":             info.get("shortName", sym),
                "earnings_date":    earnings_date or "N/A",
                "eps_estimate":     _safe_float(eps_estimate) if eps_estimate is not None else None,
                "revenue_estimate": _safe_float(revenue_estimate) if revenue_estimate is not None else None,
                "trailing_eps":     _safe_float(info.get("trailingEps")),
                "last_surprise_pct": last_surprise,
                "sector":           info.get("sector", "N/A"),
            })
        except Exception as exc:
            log.warning("Earnings calendar failed for %s: %s", sym, exc)
            results.append({"symbol": sym.upper(), "name": sym, "earnings_date": "N/A"})

    def _sort_key(r: dict) -> str:
        d = r.get("earnings_date", "N/A")
        return d if d and d != "N/A" else "9999-99-99"

    return sorted(results, key=_sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# Sector Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def get_sector_heatmap(period: str = "1mo") -> pd.DataFrame:
    """
    Fetch performance of all 11 S&P 500 sector ETFs.
    period: "1d" | "1w" | "1mo" | "ytd"
    Returns DataFrame: sector, etf, price, change_pct, ytd_pct
    """
    key = f"sector_heatmap_{period}"
    if key in _cache:
        return _cache[key]

    tickers_list = list(SECTOR_ETFS.values())
    rows = []

    try:
        raw = yf.download(
            tickers_list,
            period="ytd",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        for sector, etf in SECTOR_ETFS.items():
            try:
                prices = _extract_close(raw, etf)
                if len(prices) < 2:
                    continue

                latest  = float(prices.iloc[-1])
                ytd_pct = float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100)

                if period == "1d":
                    chg_pct = float((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100)
                elif period == "1w":
                    idx = max(0, len(prices) - 6)
                    chg_pct = float((prices.iloc[-1] - prices.iloc[idx]) / prices.iloc[idx] * 100)
                elif period == "1mo":
                    idx = max(0, len(prices) - 22)
                    chg_pct = float((prices.iloc[-1] - prices.iloc[idx]) / prices.iloc[idx] * 100)
                else:
                    chg_pct = ytd_pct

                rows.append({
                    "sector":     sector,
                    "etf":        etf,
                    "price":      round(latest, 2),
                    "change_pct": round(chg_pct, 2),
                    "ytd_pct":    round(ytd_pct, 2),
                })
            except Exception as exc:
                log.warning("Sector ETF %s failed: %s", etf, exc)

    except Exception as exc:
        log.error("Sector heatmap download failed: %s", exc)

    df = pd.DataFrame(rows)
    if not df.empty:
        _cache[key] = df
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Peer Comparison
# ─────────────────────────────────────────────────────────────────────────────

def get_peer_comparison(symbol: str) -> pd.DataFrame:
    """
    Fetch key ratios for the symbol and its sector peers.
    Returns a DataFrame with one row per ticker.
    """
    key = f"peers_{symbol}"
    if key in _cache:
        return _cache[key]

    try:
        info     = _ticker(symbol).info or {}
        industry = info.get("industry", "")
        sector   = info.get("sector", "")
        peers    = _find_peers(symbol, industry, sector)
    except Exception:
        peers = []

    rows = []
    for sym in [symbol.upper()] + peers:
        try:
            info = _ticker(sym).info or {}
            rows.append({
                "Ticker":       sym,
                "Name":         (info.get("shortName", sym) or sym)[:22],
                "Price":        _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
                "Mkt Cap ($B)": round(_safe_float(info.get("marketCap")) / 1e9, 1),
                "P/E (TTM)":    _safe_float(info.get("trailingPE")) or None,
                "P/E (Fwd)":    _safe_float(info.get("forwardPE")) or None,
                "EV/EBITDA":    _safe_float(info.get("enterpriseToEbitda")) or None,
                "Rev Growth %": round(_safe_float(info.get("revenueGrowth")) * 100, 1),
                "Net Margin %": round(_safe_float(info.get("profitMargins")) * 100, 1),
                "ROE %":        round(_safe_float(info.get("returnOnEquity")) * 100, 1),
                "Beta":         _safe_float(info.get("beta")) or None,
                "Div Yield %":  round(_safe_float(info.get("dividendYield")) * 100, 2),
            })
        except Exception as exc:
            log.warning("Peer data failed for %s: %s", sym, exc)

    df = pd.DataFrame(rows)
    if not df.empty:
        _cache[key] = df
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Insider & Institutional Activity
# ─────────────────────────────────────────────────────────────────────────────

def get_insider_activity(symbol: str) -> dict:
    """
    Returns insider transactions, institutional holders, and major holder percentages.
    """
    key = f"insider_{symbol}"
    if key in _cache:
        return _cache[key]

    result: dict = {
        "insider_transactions":  pd.DataFrame(),
        "institutional_holders": pd.DataFrame(),
        "major_holders":         {},
    }

    try:
        tk = _ticker(symbol)

        # Insider transactions
        try:
            ins_df = tk.insider_transactions
            if ins_df is not None and not ins_df.empty:
                ins_df = ins_df.reset_index(drop=True)
                # Normalise column names across yfinance versions
                rename = {}
                for col in ins_df.columns:
                    cl = col.lower()
                    if "date" in cl:                    rename[col] = "Date"
                    elif "name" in cl or "insider" in cl: rename[col] = "Insider"
                    elif "title" in cl or "relation" in cl: rename[col] = "Title"
                    elif "transaction" in cl:           rename[col] = "Transaction"
                    elif "shares" in cl or "securities" in cl: rename[col] = "Shares"
                    elif "value" in cl:                 rename[col] = "Value ($)"
                ins_df = ins_df.rename(columns=rename)
                keep = [c for c in ["Date", "Insider", "Title", "Transaction", "Shares", "Value ($)"] if c in ins_df.columns]
                result["insider_transactions"] = ins_df[keep].head(15)
        except Exception as exc:
            log.debug("Insider transactions unavailable for %s: %s", symbol, exc)

        # Institutional holders
        try:
            inst_df = tk.institutional_holders
            if inst_df is not None and not inst_df.empty:
                result["institutional_holders"] = inst_df.head(10).reset_index(drop=True)
        except Exception as exc:
            log.debug("Institutional holders unavailable for %s: %s", symbol, exc)

        # Major holders summary (% breakdown)
        try:
            mh = tk.major_holders
            if mh is not None and not mh.empty:
                mh_dict: dict = {}
                for _, row in mh.iterrows():
                    try:
                        mh_dict[str(row.iloc[1])] = row.iloc[0]
                    except Exception:
                        pass
                result["major_holders"] = mh_dict
        except Exception as exc:
            log.debug("Major holders unavailable for %s: %s", symbol, exc)

    except Exception as exc:
        log.warning("Insider activity failed for %s: %s", symbol, exc)

    _cache[key] = result
    return result
