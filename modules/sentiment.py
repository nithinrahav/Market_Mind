"""
Sentiment analysis module.

Sources (in priority order, depending on which keys are configured):
  1. NewsAPI  – structured news search by ticker + company name
  2. Yahoo Finance RSS – free, no API key required

Each article is scored with a quick lexicon pass first; if Claude is
available those scores are enriched by an LLM summary + overall verdict.
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any
import urllib.request

import anthropic
from cachetools import TTLCache
import config

log = logging.getLogger(__name__)

_cache: TTLCache = TTLCache(maxsize=128, ttl=config.CACHE_TTL)

# ── Simple lexicon for fast, free scoring ─────────────────────────────────────
_POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "record", "profit",
    "growth", "grow", "innovation", "buyback", "dividend", "outperform",
    "upgrade", "strong", "robust", "bullish", "soar", "soars", "exceed",
    "exceeds", "guidance", "raised", "raise", "top", "upside", "momentum",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "plunge", "plunges", "decline", "declines", "loss",
    "losses", "downgrade", "bearish", "concern", "concerns", "risk", "risks",
    "lawsuit", "investigation", "sec", "fraud", "slump", "slumps", "warn",
    "warning", "cut", "cuts", "layoff", "layoffs", "recession", "debt",
    "default", "bankruptcy", "weak", "disappointing",
}


def _lexicon_score(text: str) -> float:
    """Return score in [-1, 1]; positive = bullish."""
    words  = set(re.findall(r"\b\w+\b", text.lower()))
    pos    = len(words & _POSITIVE_WORDS)
    neg    = len(words & _NEGATIVE_WORDS)
    total  = pos + neg
    return (pos - neg) / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# News fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_newsapi(symbol: str, company_name: str, days: int = 7) -> list[dict]:
    """Fetch articles from NewsAPI (requires NEWS_API_KEY)."""
    if not config.NEWS_API_KEY:
        return []

    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=config.NEWS_API_KEY)

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        query     = f'"{symbol}" OR "{company_name}"'

        response  = client.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=20,
        )
        articles = response.get("articles", [])
        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "url":         a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "source":      a.get("source", {}).get("name", ""),
            }
            for a in articles
        ]
    except Exception as exc:
        log.warning("NewsAPI fetch failed for %s: %s", symbol, exc)
        return []


def _fetch_yahoo_rss(symbol: str) -> list[dict]:
    """Fetch articles from Yahoo Finance RSS (free, no key needed)."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    try:
        import xml.etree.ElementTree as ET
        req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_bytes = resp.read()
        root  = ET.fromstring(xml_bytes)
        items = root.findall(".//item")
        articles = []
        for item in items[:20]:
            title = (item.findtext("title") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            link  = (item.findtext("link") or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            articles.append({
                "title":        title,
                "description":  desc,
                "url":          link,
                "published_at": pub,
                "source":       "Yahoo Finance",
            })
        return articles
    except Exception as exc:
        log.warning("Yahoo RSS fetch failed for %s: %s", symbol, exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# LLM-powered analysis
# ─────────────────────────────────────────────────────────────────────────────

def _llm_sentiment_analysis(symbol: str, articles: list[dict]) -> dict:
    """
    Send headline + description snippets to Claude and get back a structured
    sentiment verdict and summary.
    """
    if not config.ANTHROPIC_API_KEY or not articles:
        return {}

    snippet_lines = []
    for i, a in enumerate(articles[:15], 1):
        title = a.get("title", "")
        desc  = (a.get("description") or "")[:200]
        src   = a.get("source", "")
        date  = a.get("published_at", "")[:10]
        snippet_lines.append(f"{i}. [{src} | {date}] {title} — {desc}")

    snippets = "\n".join(snippet_lines)

    prompt = f"""You are a financial analyst. Analyse the following recent news headlines and descriptions for stock ticker **{symbol}**.

NEWS:
{snippets}

Respond ONLY in this exact JSON format (no markdown fences):
{{
  "overall_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL" | "MIXED",
  "sentiment_score": <float -1.0 to 1.0>,
  "key_themes": ["theme1", "theme2", "theme3"],
  "bull_points": ["point1", "point2"],
  "bear_points": ["point1", "point2"],
  "summary": "<2-3 sentence human-readable summary of the market sentiment for {symbol}>"
}}"""

    try:
        client   = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        raw = response.content[0].text.strip()
        # Strip possible ```json ... ``` wrapping
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        log.warning("LLM sentiment analysis failed for %s: %s", symbol, exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_sentiment(symbol: str, company_name: str = "") -> dict:
    """
    Main entry point. Returns a dict with:
      articles, overall_sentiment, sentiment_score, sentiment_label,
      key_themes, bull_points, bear_points, summary
    """
    cache_key = f"sentiment_{symbol}"
    if cache_key in _cache:
        return _cache[cache_key]

    company_name = company_name or symbol

    # Fetch from available sources
    articles = _fetch_newsapi(symbol, company_name)
    if not articles:
        articles = _fetch_yahoo_rss(symbol)

    # Lexicon baseline
    scores = []
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}"
        scores.append(_lexicon_score(text))

    avg_score = sum(scores) / len(scores) if scores else 0.0

    if avg_score > 0.15:
        lex_label = "BULLISH"
    elif avg_score < -0.15:
        lex_label = "BEARISH"
    else:
        lex_label = "NEUTRAL"

    # Enrich with LLM
    llm_result = _llm_sentiment_analysis(symbol, articles)

    result: dict = {
        "symbol":            symbol.upper(),
        "articles":          articles,
        "article_count":     len(articles),
        "overall_sentiment": llm_result.get("overall_sentiment", lex_label),
        "sentiment_score":   llm_result.get("sentiment_score", round(avg_score, 3)),
        "key_themes":        llm_result.get("key_themes", []),
        "bull_points":       llm_result.get("bull_points", []),
        "bear_points":       llm_result.get("bear_points", []),
        "summary":           llm_result.get("summary", "Sentiment based on recent news headlines."),
        "source":            "NewsAPI + Claude" if llm_result else "Lexicon (no LLM key)",
    }

    _cache[cache_key] = result
    return result
