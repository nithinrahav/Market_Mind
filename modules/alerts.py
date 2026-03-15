"""
Price alert management.
Alerts are stored in st.session_state — they persist for the browser session
but reset on page refresh. Use check_alerts() on each app run to evaluate them.
"""
import logging
from datetime import datetime

log = logging.getLogger(__name__)


def add_alert(
    alerts: list[dict],
    symbol: str,
    target_price: float,
    condition: str,
    note: str = "",
) -> list[dict]:
    """
    Return a new alerts list with the alert appended.
    condition: "above" | "below"
    """
    new_alert = {
        "id":              len(alerts),
        "symbol":          symbol.upper().strip(),
        "target":          round(target_price, 4),
        "condition":       condition,
        "note":            note,
        "created":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        "triggered":       False,
        "triggered_at":    None,
        "triggered_price": None,
    }
    log.info("Alert added: %s %s $%.4f", symbol, condition, target_price)
    return alerts + [new_alert]


def remove_alert(alerts: list[dict], alert_id: int) -> list[dict]:
    """Return list with the alert matching alert_id removed."""
    return [a for a in alerts if a["id"] != alert_id]


def check_alerts(
    alerts: list[dict],
    current_prices: dict[str, float],
) -> tuple[list[dict], list[dict]]:
    """
    Evaluate all untriggered alerts against current_prices.
    Returns (updated_alerts, newly_triggered).
    Dicts are copied — does not mutate inputs.
    """
    updated: list[dict]          = []
    newly_triggered: list[dict]  = []

    for alert in alerts:
        if alert["triggered"]:
            updated.append(alert)
            continue

        price = current_prices.get(alert["symbol"])
        if price is None:
            updated.append(alert)
            continue

        fired = (
            (alert["condition"] == "above" and price >= alert["target"]) or
            (alert["condition"] == "below" and price <= alert["target"])
        )

        if fired:
            alert = {
                **alert,
                "triggered":       True,
                "triggered_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                "triggered_price": price,
            }
            newly_triggered.append(alert)

        updated.append(alert)

    return updated, newly_triggered


def get_active_alerts(alerts: list[dict]) -> list[dict]:
    """Return only untriggered alerts."""
    return [a for a in alerts if not a["triggered"]]


def get_triggered_alerts(alerts: list[dict]) -> list[dict]:
    """Return triggered alerts sorted newest-first."""
    return sorted(
        [a for a in alerts if a["triggered"]],
        key=lambda a: a.get("triggered_at", ""),
        reverse=True,
    )
