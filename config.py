"""
Central configuration – reads from .env / environment variables.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = "claude-opus-4-6"

# ── E*Trade ────────────────────────────────────────────────────────────────────
ETRADE_CONSUMER_KEY: str = os.getenv("ETRADE_CONSUMER_KEY", "")
ETRADE_CONSUMER_SECRET: str = os.getenv("ETRADE_CONSUMER_SECRET", "")
ETRADE_ENV: str = os.getenv("ETRADE_ENV", "sandbox")          # "sandbox" | "live"

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
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# ── Alpha Vantage ──────────────────────────────────────────────────────────────
ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_TTL: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
