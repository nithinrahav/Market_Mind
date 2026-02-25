"""General-purpose helper utilities shared across modules."""

import functools
import logging
import time
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timeit(func: F) -> F:
    """Decorator that logs execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0     = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.debug("%s completed in %.3fs", func.__qualname__, elapsed)
        return result
    return wrapper  # type: ignore[return-value]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns *default* instead of raising ZeroDivisionError."""
    return numerator / denominator if denominator else default


def pct_change(old: float, new: float) -> float:
    """Percentage change from old to new, safe against zero division."""
    return safe_divide(new - old, old) * 100


def fmt_large_number(val: float | int) -> str:
    """
    Format large numbers with B/M/K suffix.
    E.g. 2_500_000_000 → "2.50B"
    """
    val = float(val)
    if abs(val) >= 1e12:
        return f"{val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"{val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"{val/1e6:.2f}M"
    if abs(val) >= 1e3:
        return f"{val/1e3:.1f}K"
    return f"{val:.2f}"


def validate_ticker(symbol: str) -> str:
    """
    Basic ticker validation: uppercase, strip whitespace, allow letters + dots + hyphens.
    Raises ValueError for obviously invalid symbols.
    """
    import re
    symbol = symbol.strip().upper()
    if not re.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", symbol):
        raise ValueError(
            f"'{symbol}' does not look like a valid ticker symbol. "
            "Expected 1-10 characters (letters, digits, . or -)."
        )
    return symbol
