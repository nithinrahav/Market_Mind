"""
E*Trade OAuth 1.0a client.

Flow:
  1. Call `get_request_token()` → returns an authorisation URL.
  2. User visits URL, logs in, and gets a verification code.
  3. Call `get_access_token(verifier)` → stores session tokens.
  4. Use `get_portfolio()` / `get_account_balance()` freely.

Tokens are cached in Streamlit session_state so the OAuth dance only
happens once per browser session.
"""

import logging
from typing import Any
import requests
from requests_oauthlib import OAuth1Session

import config

log = logging.getLogger(__name__)

# ── Internal state ────────────────────────────────────────────────────────────
# Three separate OAuth1Sessions are used – one per OAuth step.
# Reusing a single session across steps causes signature_invalid errors because
# E*Trade's sandbox is strict: the callback_uri bleeds into later requests and
# the signing key must change between steps.
_oauth_session: OAuth1Session | None = None   # final authenticated session
_authenticated: bool = False                   # set True only after access token obtained
_account_id_key: str = ""
_resource_owner_key: str = ""                  # oauth_token from request-token step
_resource_owner_secret: str = ""               # oauth_token_secret from request-token step


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Request token
# ─────────────────────────────────────────────────────────────────────────────

def get_request_token() -> str:
    """
    Fetches a request token from E*Trade and returns the authorisation URL
    the user must visit to get their verifier code.

    E*Trade requires a clean session with NO callback_uri for this step –
    passing oauth_callback causes signature_invalid on their sandbox server.
    The authorisation URL is built manually using E*Trade's custom format.
    """
    global _resource_owner_key, _resource_owner_secret, _authenticated

    # Reset any stale state from a previous attempt
    _authenticated = False
    _resource_owner_key = ""
    _resource_owner_secret = ""

    # Step-1 session: consumer credentials + oauth_callback=oob.
    # E*Trade sandbox requires oauth_callback to be present in the request
    # token call (returns 400 parameter_absent if omitted).  "oob" (out-of-band)
    # tells E*Trade the user will copy the verifier code manually.
    step1_session = OAuth1Session(
        config.ETRADE_CONSUMER_KEY,
        client_secret=config.ETRADE_CONSUMER_SECRET,
        callback_uri="oob",
    )

    try:
        fetch_response = step1_session.fetch_request_token(
            config.ETRADE_REQUEST_TOKEN_URL
        )
    except Exception as exc:
        log.error("Failed to fetch E*Trade request token: %s", exc)
        raise

    _resource_owner_key    = fetch_response.get("oauth_token", "")
    _resource_owner_secret = fetch_response.get("oauth_token_secret", "")

    # E*Trade uses a non-standard authorisation URL format
    auth_url = (
        f"{config.ETRADE_AUTHORIZE_URL}"
        f"?key={config.ETRADE_CONSUMER_KEY}&token={_resource_owner_key}"
    )
    log.info("E*Trade authorisation URL generated.")
    return auth_url


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Access token
# ─────────────────────────────────────────────────────────────────────────────

def get_access_token(verifier: str) -> bool:
    """
    Exchange the verifier code for an access token.
    Returns True on success, False on failure.

    A brand-new OAuth1Session is created for this step, seeded with the
    request-token credentials and the user's verifier.  After a successful
    exchange, a third session (access-token credentials only) is stored in
    _oauth_session for all subsequent API calls.
    """
    global _oauth_session, _authenticated, _resource_owner_key, _resource_owner_secret

    if not _resource_owner_key or not _resource_owner_secret:
        log.error("Call get_request_token() before get_access_token().")
        return False

    try:
        # Step-2 session: consumer + request-token credentials + verifier
        step2_session = OAuth1Session(
            config.ETRADE_CONSUMER_KEY,
            client_secret=config.ETRADE_CONSUMER_SECRET,
            resource_owner_key=_resource_owner_key,
            resource_owner_secret=_resource_owner_secret,
            verifier=verifier,
        )
        access_response = step2_session.fetch_access_token(
            config.ETRADE_ACCESS_TOKEN_URL
        )

        access_token        = access_response.get("oauth_token", "")
        access_token_secret = access_response.get("oauth_token_secret", "")

        # Step-3 session: consumer + access-token credentials (no verifier)
        _oauth_session = OAuth1Session(
            config.ETRADE_CONSUMER_KEY,
            client_secret=config.ETRADE_CONSUMER_SECRET,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret,
        )
        _authenticated = True
        log.info("E*Trade access token obtained successfully.")
        return True

    except Exception as exc:
        _authenticated = False
        log.error("Failed to obtain E*Trade access token: %s", exc)
        return False


def is_authenticated() -> bool:
    """
    Return True only after fetch_access_token has succeeded.
    NOTE: OAuth1Session always has a .token *property* on the class, so
    hasattr() is always True – we use a dedicated flag instead.
    """
    return _oauth_session is not None and _authenticated


def logout() -> None:
    """Reset all module-level auth state (call on Disconnect)."""
    global _oauth_session, _authenticated, _resource_owner_key, _resource_owner_secret
    _oauth_session         = None
    _authenticated         = False
    _resource_owner_key    = ""
    _resource_owner_secret = ""
    log.info("E*Trade session cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> dict:
    """Authenticated GET against E*Trade base URL."""
    if not is_authenticated():
        raise RuntimeError("Not authenticated with E*Trade. Complete OAuth flow first.")

    url = f"{config.ETRADE_BASE_URL}{endpoint}"
    resp = _oauth_session.get(url, params=params or {}, headers={"Accept": "application/json"})  # type: ignore[union-attr]
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Account helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_accounts() -> list[dict]:
    """Return all brokerage accounts associated with the credentials."""
    data = _get("/v1/accounts/list")
    accounts = (
        data.get("AccountListResponse", {})
            .get("Accounts", {})
            .get("Account", [])
    )
    if isinstance(accounts, dict):         # single account → wrap in list
        accounts = [accounts]
    return accounts


def _safe_int(value) -> int:
    """Convert value to int; return 0 if it is non-numeric (e.g. 'NO_PDT')."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def get_account_balance(account_id_key: str) -> dict:
    """
    Return a simplified balance dict for the given account.
    Keys: net_account_value, cash_available, margin_buying_power
    """
    data = _get(f"/v1/accounts/{account_id_key}/balance", params={"instType": "BROKERAGE"})
    balance_resp = data.get("BalanceResponse", {})

    computed = balance_resp.get("Computed", {})
    return {
        "net_account_value":    float(computed.get("RealTimeValues", {}).get("totalAccountValue", 0) or 0),
        "cash_available":       float(computed.get("cashAvailableForWithdrawal", 0) or 0),
        "margin_buying_power":  float(computed.get("marginBuyingPower", 0) or 0),
        "account_description":  balance_resp.get("accountDescription", ""),
        "account_type":         balance_resp.get("accountType", ""),
        # dayTraderStatus is numeric when PDT rules apply, or "NO_PDT" when they don't
        "day_trades_remaining": _safe_int(balance_resp.get("dayTraderStatus", 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────────────────────

def get_portfolio(account_id_key: str) -> list[dict]:
    """
    Return a list of ALL positions in the portfolio, across all pages.

    E*Trade caps results at 50 positions per page.  When more pages exist the
    response includes AccountPortfolio[].nextPageNo (a string integer).  We
    keep requesting the next page until that field is absent.
    """
    positions: list[dict] = []
    page: int = 1

    while True:
        data = _get(
            f"/v1/accounts/{account_id_key}/portfolio",
            params={"view": "COMPLETE", "pageNumber": page},
        )

        portfolio_resp     = data.get("PortfolioResponse", {})
        account_portfolios = portfolio_resp.get("AccountPortfolio", [])
        if isinstance(account_portfolios, dict):
            account_portfolios = [account_portfolios]

        next_page: int | None = None

        for acct in account_portfolios:
            raw_positions = acct.get("Position", [])
            if isinstance(raw_positions, dict):
                raw_positions = [raw_positions]

            for pos in raw_positions:
                product = pos.get("Product", {})
                symbol  = product.get("symbol", "N/A")
                qty     = float(pos.get("quantity", 0) or 0)

                quick         = pos.get("Quick", {})
                cur_price     = float(quick.get("lastTrade", 0) or 0)
                mkt_value     = float(pos.get("marketValue", 0) or 0)
                cost          = float(pos.get("costBasis", 0) or 0)
                gain_loss     = float(pos.get("totalGain", 0) or 0)
                gain_loss_pct = (gain_loss / cost * 100) if cost else 0.0

                positions.append({
                    "symbol":        symbol,
                    "quantity":      qty,
                    "current_price": cur_price,
                    "market_value":  mkt_value,
                    "cost_basis":    cost,
                    "gain_loss":     gain_loss,
                    "gain_loss_pct": round(gain_loss_pct, 2),
                    "position_type": pos.get("positionType", "LONG"),
                })

            # nextPageNo is a string like "2"; absent on the last page
            next_page_str = acct.get("nextPageNo")
            if next_page_str:
                next_page = int(next_page_str)

        if next_page is None:
            break
        page = next_page
        log.debug("Fetching portfolio page %d", page)

    log.info("Portfolio loaded: %d positions across pages.", len(positions))
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Quotes
# ─────────────────────────────────────────────────────────────────────────────

def get_quote(symbols: list[str]) -> list[dict]:
    """
    Fetch real-time quotes for a list of ticker symbols via E*Trade.
    Returns list of dicts with keys: symbol, last_price, change, change_pct, volume.
    """
    symbols_str = ",".join(symbols)
    data = _get(f"/v1/market/quote/{symbols_str}")

    quote_data = data.get("QuoteResponse", {}).get("QuoteData", [])
    if isinstance(quote_data, dict):
        quote_data = [quote_data]

    results = []
    for qd in quote_data:
        product  = qd.get("Product", {})
        all_q    = qd.get("All", {})
        results.append({
            "symbol":     product.get("symbol", ""),
            "last_price": float(all_q.get("lastTrade", 0) or 0),
            "change":     float(all_q.get("change", 0) or 0),
            "change_pct": float(all_q.get("pctChange", 0) or 0),
            "volume":     int(all_q.get("totalVolume", 0) or 0),
            "bid":        float(all_q.get("bid", 0) or 0),
            "ask":        float(all_q.get("ask", 0) or 0),
            "high":       float(all_q.get("high", 0) or 0),
            "low":        float(all_q.get("low", 0) or 0),
        })
    return results
