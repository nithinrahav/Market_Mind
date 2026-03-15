"""
SEC EDGAR integration for filing analysis and earnings summarization.
Uses the free SEC EDGAR API — no API key required.
SEC requires a User-Agent header; rate limit is ~10 req/s (we stay well under).
"""
import logging
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache

import anthropic
import config

log = logging.getLogger(__name__)
_cache: TTLCache = TTLCache(maxsize=32, ttl=3600)  # 1-hour cache for filings

_EDGAR_BASE = "https://data.sec.gov"
_ARCHIVES   = "https://www.sec.gov/Archives/edgar/data"
_HEADERS    = {
    "User-Agent": "MarketMind/1.0 research@marketmind.app",
    "Accept":     "application/json",
}

# Module-level CIK mapping (loaded once per process)
_cik_map: dict[str, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# SEC HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _sec_get(url: str, accept_html: bool = False) -> requests.Response:
    """Throttled GET to SEC EDGAR (~5 req/s to stay under the 10 req/s limit)."""
    time.sleep(0.2)
    headers = dict(_HEADERS)
    if accept_html:
        headers["Accept"] = "text/html,application/xhtml+xml"
    resp = requests.get(url, headers=headers, timeout=25)
    resp.raise_for_status()
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# CIK lookup
# ─────────────────────────────────────────────────────────────────────────────

def _load_cik_map() -> None:
    """Load SEC company→CIK mapping (fetched once per process, ~3 MB)."""
    global _cik_map
    if _cik_map:
        return
    try:
        resp = _sec_get("https://www.sec.gov/files/company_tickers.json")
        data = resp.json()
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper().strip()
            cik    = str(entry.get("cik_str", "")).zfill(10)
            if ticker:
                _cik_map[ticker] = cik
        log.info("CIK map loaded: %d companies", len(_cik_map))
    except Exception as exc:
        log.warning("CIK map load failed: %s", exc)


def get_cik(symbol: str) -> str | None:
    """Return the zero-padded 10-digit CIK for a ticker, or None."""
    _load_cik_map()
    return _cik_map.get(symbol.upper().strip())


# ─────────────────────────────────────────────────────────────────────────────
# Filing list
# ─────────────────────────────────────────────────────────────────────────────

def get_recent_filings(
    symbol: str,
    form_types: list[str] | None = None,
    max_count: int = 5,
) -> list[dict]:
    """
    Return recent SEC filings for a symbol.
    form_types: e.g. ["10-K", "10-Q", "8-K"]. Defaults to all three.
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    key = f"filings_{symbol}_{'_'.join(sorted(form_types))}"
    if key in _cache:
        return _cache[key]

    cik = get_cik(symbol)
    if not cik:
        log.warning("No CIK found for %s", symbol)
        return []

    try:
        resp = _sec_get(f"{_EDGAR_BASE}/submissions/CIK{cik}.json")
        data = resp.json()
    except Exception as exc:
        log.error("Submissions fetch failed for %s: %s", symbol, exc)
        return []

    recent  = data.get("filings", {}).get("recent", {})
    forms   = recent.get("form",                  [])
    dates   = recent.get("filingDate",            [])
    accnums = recent.get("accessionNumber",       [])
    docs    = recent.get("primaryDocument",       [])
    descs   = recent.get("primaryDocDescription", [])

    results = []
    cik_int = int(cik)
    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        if len(results) >= max_count:
            break

        accn_clean = accnums[i].replace("-", "") if i < len(accnums) else ""
        doc_name   = docs[i]  if i < len(docs)   else ""
        desc       = descs[i] if i < len(descs)  else form

        filing_url = (
            f"{_ARCHIVES}/{cik_int}/{accn_clean}/{doc_name}"
            if accn_clean and doc_name else ""
        )

        results.append({
            "form": form,
            "date": dates[i] if i < len(dates) else "",
            "url":  filing_url,
            "desc": desc,
            "accn": accnums[i] if i < len(accnums) else "",
        })

    _cache[key] = results
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Filing text extraction
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_filing_text(url: str, max_chars: int = 40_000) -> str:
    """Download an SEC filing and return clean plain text (truncated)."""
    if not url:
        return ""
    try:
        resp         = _sec_get(url, accept_html=True)
        content_type = resp.headers.get("Content-Type", "")

        if "html" in content_type or url.lower().endswith((".htm", ".html")):
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = resp.text

        # Clean excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # For annual/quarterly filings try to find MD&A section
        mda_match = re.search(
            r"(item\s*7[\.\s]+management.{0,40}discussion|results\s+of\s+operations)",
            text, re.IGNORECASE,
        )
        if mda_match:
            text = text[mda_match.start():]

        return text[:max_chars]

    except Exception as exc:
        log.error("Filing text fetch failed for %s: %s", url, exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Claude analysis
# ─────────────────────────────────────────────────────────────────────────────

def _call_claude(prompt: str, max_tokens: int = 1500) -> str:
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp   = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def analyze_filing(symbol: str, form_type: str = "10-K") -> str:
    """
    Fetch the most recent filing of form_type and return a Claude analysis.
    Returns markdown-formatted string.
    """
    if not config.ANTHROPIC_API_KEY:
        return "**Error:** ANTHROPIC_API_KEY is not configured."

    filings = get_recent_filings(symbol, form_types=[form_type], max_count=1)
    if not filings:
        return f"No recent **{form_type}** filings found for **{symbol.upper()}** on SEC EDGAR."

    filing      = filings[0]
    filing_text = _fetch_filing_text(filing["url"])

    if not filing_text.strip():
        return f"Could not retrieve filing text for **{symbol.upper()}**."

    prompt = f"""You are a senior equity research analyst.
Analyze this {form_type} SEC filing for **{symbol.upper()}** (filed {filing['date']}).

FILING EXCERPT (MD&A section or first 40,000 characters):
{filing_text}

Produce a structured analysis in markdown:

## {symbol.upper()} – {form_type} Analysis ({filing['date']})

### Business & Revenue Overview
- Core business and key revenue drivers
- Notable strategic changes or initiatives

### Financial Highlights
- Revenue and earnings trajectory with specific numbers
- Key margin and profitability trends
- Balance sheet and liquidity position

### Top 5 Risk Factors
List the most material risks mentioned in the filing.

### Management Outlook
- Key forward-looking statements or guidance

### Investment Takeaways
- **Bull case** (2-3 points)
- **Bear case** (2-3 points)

Be concise, factual, and cite specific numbers from the filing where present."""

    try:
        return _call_claude(prompt, max_tokens=2000)
    except Exception as exc:
        log.error("Claude filing analysis failed: %s", exc)
        return f"Analysis failed: {exc}"


def summarize_earnings(symbol: str) -> str:
    """
    Fetch the most recent 8-K earnings release and summarise with Claude.
    Returns markdown-formatted string.
    """
    if not config.ANTHROPIC_API_KEY:
        return "**Error:** ANTHROPIC_API_KEY is not configured."

    filings = get_recent_filings(symbol, form_types=["8-K"], max_count=5)
    if not filings:
        return f"No recent 8-K filings found for **{symbol.upper()}**."

    # Prefer earnings-related 8-K
    target = filings[0]
    for f in filings:
        desc_lower = (f.get("desc") or "").lower()
        if any(kw in desc_lower for kw in ["earnings", "results", "revenue", "quarter", "financial"]):
            target = f
            break

    filing_text = _fetch_filing_text(target["url"], max_chars=20_000)
    if not filing_text.strip():
        return f"Could not retrieve 8-K text for **{symbol.upper()}**."

    prompt = f"""You are a financial analyst reviewing an 8-K filing for **{symbol.upper()}** (filed {target['date']}).

FILING TEXT:
{filing_text}

Summarize in markdown:

## {symbol.upper()} Earnings Summary ({target['date']})

### Key Results
- Revenue, EPS, and key metrics (actual vs expected where mentioned)
- Year-over-year and quarter-over-quarter comparisons

### Highlights
- 2-3 standout positives from the quarter

### Concerns
- Any misses, one-time items, or areas of concern

### Guidance & Outlook
- Management's forward guidance (if provided)

### Analyst Takeaway
- One paragraph synthesis: what this means for the stock outlook

Be concise and focus on investment-relevant information."""

    try:
        return _call_claude(prompt, max_tokens=1500)
    except Exception as exc:
        log.error("Earnings summary failed: %s", exc)
        return f"Summary failed: {exc}"
