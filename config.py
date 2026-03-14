"""
Central configuration – reads from st.secrets (Streamlit Cloud) with
fallback to .env / environment variables for local development.
"""
import os
import logging
from dotenv import load_dotenv
import streamlit as st
import anthropic

load_dotenv()  # no-op in production; loads .env locally


def _secret(key: str, default: str = "") -> str:
    """Return value from st.secrets if available, else os.environ."""
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        return os.getenv(key, default)


# ── Anthropic ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = _secret("ANTHROPIC_API_KEY")
CLAUDE_MODEL: str = "claude-opus-4-6"

# ── E*Trade ────────────────────────────────────────────────────────────────────
ETRADE_CONSUMER_KEY: str = _secret("ETRADE_CONSUMER_KEY")
ETRADE_CONSUMER_SECRET: str = _secret("ETRADE_CONSUMER_SECRET")
ETRADE_ENV: str = _secret("ETRADE_ENV", "sandbox")          # "sandbox" | "live"

ETRADE_BASE_URL = (
    "https://api.etrade.com"
    if ETRADE_ENV == "live"
    else "https://apisb.etrade.com"
)
ETRADE_REQUEST_TOKEN_URL = f"{ETRADE_BASE_URL}/oauth/request_token"
ETRADE_ACCESS_TOKEN_URL  = f"{ETRADE_BASE_URL}/oauth/access_token"
ETRADE_AUTHORIZE_URL     = (
    "https://us.etrade.com/e/t/etws/authorize"
    if ETRADE_ENV == "live"
    else "https://us.etrade.com/e/t/etws/authorize"
)

# ── News ───────────────────────────────────────────────────────────────────────
NEWS_API_KEY: str = _secret("NEWS_API_KEY")

# ── Alpha Vantage ──────────────────────────────────────────────────────────────
ALPHA_VANTAGE_KEY: str = _secret("ALPHA_VANTAGE_KEY")

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_TTL: int = int(_secret("CACHE_TTL_SECONDS", "300"))

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _secret("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
