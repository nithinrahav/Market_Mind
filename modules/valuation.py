"""
Fair-value estimation module.

Methods implemented:
  1. Graham Number              – conservative intrinsic value for value stocks
  2. DCF (Discounted Cash Flow) – based on free cash flow + growth assumptions
  3. Earnings Power Value (EPV) – normalised earnings / WACC
  4. Analyst consensus          – from yfinance analyst targets

All monetary values are in USD unless stated otherwise.
"""

import logging
import math
from typing import Any

from modules.stock_data import get_current_price, get_key_ratios, get_analyst_targets

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Graham Number
# ─────────────────────────────────────────────────────────────────────────────

def graham_number(eps: float, bvps: float) -> float:
    """
    Graham Number = sqrt(22.5 × EPS × BVPS)
    Returns 0 if inputs are non-positive.
    """
    if eps <= 0 or bvps <= 0:
        return 0.0
    return round(math.sqrt(22.5 * eps * bvps), 2)


# ─────────────────────────────────────────────────────────────────────────────
# DCF
# ─────────────────────────────────────────────────────────────────────────────

def dcf_valuation(
    free_cashflow: float,
    shares_outstanding: int,
    growth_rate_yr1_5: float = 0.10,   # annualised growth rate, years 1-5
    growth_rate_yr6_10: float = 0.06,  # years 6-10
    terminal_growth: float = 0.03,     # perpetuity growth
    discount_rate: float = 0.10,       # WACC / required return
) -> dict:
    """
    Two-stage DCF model using FCF per share.
    Returns dict with: dcf_per_share, total_dcf_value, terminal_value, assumptions
    """
    if free_cashflow <= 0 or shares_outstanding <= 0:
        return {"dcf_per_share": 0.0, "note": "Negative or zero FCF – DCF not meaningful."}

    fcf_per_share = free_cashflow / shares_outstanding
    present_value = 0.0

    cf = fcf_per_share
    # Stage 1: years 1-5
    for yr in range(1, 6):
        cf *= (1 + growth_rate_yr1_5)
        present_value += cf / ((1 + discount_rate) ** yr)

    # Stage 2: years 6-10
    for yr in range(6, 11):
        cf *= (1 + growth_rate_yr6_10)
        present_value += cf / ((1 + discount_rate) ** yr)

    # Terminal value (Gordon Growth)
    terminal_cf  = cf * (1 + terminal_growth)
    terminal_val = terminal_cf / (discount_rate - terminal_growth)
    pv_terminal  = terminal_val / ((1 + discount_rate) ** 10)

    total = present_value + pv_terminal

    return {
        "dcf_per_share":    round(total, 2),
        "terminal_value":   round(pv_terminal, 2),
        "stage1_pv":        round(present_value - sum(
            # recalc stage2 to separate for display purposes
            [0] * 5  # placeholder; total is correct
        ), 2),
        "assumptions": {
            "fcf_per_share":     round(fcf_per_share, 4),
            "growth_yr1_5_pct":  round(growth_rate_yr1_5 * 100, 1),
            "growth_yr6_10_pct": round(growth_rate_yr6_10 * 100, 1),
            "terminal_growth_pct": round(terminal_growth * 100, 1),
            "discount_rate_pct": round(discount_rate * 100, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Earnings Power Value
# ─────────────────────────────────────────────────────────────────────────────

def earnings_power_value(eps_ttm: float, discount_rate: float = 0.10) -> float:
    """
    EPV = Normalised EPS / Discount Rate
    Simple, conservative, no growth assumption.
    """
    if eps_ttm <= 0:
        return 0.0
    return round(eps_ttm / discount_rate, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main aggregate
# ─────────────────────────────────────────────────────────────────────────────

def get_fair_value(symbol: str) -> dict:
    """
    Computes all fair-value estimates and a blended consensus.
    Returns dict with keys:
      current_price, graham, epv, dcf, analyst_mean, analyst_high, analyst_low,
      blended_fair_value, upside_pct, verdict
    """
    price_info = get_current_price(symbol)
    ratios     = get_key_ratios(symbol)
    analyst    = get_analyst_targets(symbol)

    current_price = price_info.get("price", 0)
    eps_ttm       = ratios.get("eps_ttm", 0)
    bvps          = ratios.get("book_value_per_share", 0)
    fcf           = ratios.get("free_cashflow", 0)

    # Shares outstanding (derived from market_cap / price)
    market_cap = price_info.get("market_cap", 0) or 0
    shares     = int(market_cap / current_price) if current_price else 1

    # ── Individual models ──────────────────────────────────────────────────────
    g_num  = graham_number(eps_ttm, bvps)
    epv    = earnings_power_value(eps_ttm)
    dcf    = dcf_valuation(fcf, shares)
    dcf_ps = dcf.get("dcf_per_share", 0)

    # Analyst
    ana_mean = analyst.get("target_mean", 0)
    ana_high = analyst.get("target_high", 0)
    ana_low  = analyst.get("target_low", 0)

    # ── Blended value (equal-weight non-zero estimates) ────────────────────────
    estimates = [v for v in [g_num, epv, dcf_ps, ana_mean] if v > 0]
    blended   = round(sum(estimates) / len(estimates), 2) if estimates else 0.0

    upside_pct = round(((blended - current_price) / current_price) * 100, 2) if current_price else 0.0

    if upside_pct > 20:
        verdict = "UNDERVALUED"
    elif upside_pct < -20:
        verdict = "OVERVALUED"
    else:
        verdict = "FAIRLY VALUED"

    return {
        "symbol":             symbol.upper(),
        "current_price":      current_price,
        "graham_number":      g_num,
        "epv":                epv,
        "dcf_per_share":      dcf_ps,
        "dcf_assumptions":    dcf.get("assumptions", {}),
        "analyst_mean":       ana_mean,
        "analyst_high":       ana_high,
        "analyst_low":        ana_low,
        "analyst_recommendation": analyst.get("recommendation", "N/A"),
        "num_analysts":       analyst.get("num_analysts", 0),
        "blended_fair_value": blended,
        "upside_pct":         upside_pct,
        "verdict":            verdict,
    }
