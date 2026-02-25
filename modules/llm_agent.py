"""
LLM Agent powered by Claude with tool-use.

Available tools that Claude can call during a conversation:
  • get_stock_price        – current price snapshot
  • get_key_ratios         – fundamental ratios
  • get_technical_indicators – SMA, RSI, MACD, Bollinger
  • get_sentiment          – news sentiment analysis
  • get_fair_value         – multi-model intrinsic value
  • get_price_history      – OHLCV DataFrame (summarised)
  • get_analyst_targets    – consensus targets

The agent maintains conversation history so users can ask follow-up
questions naturally.
"""

import json
import logging
from typing import Any

import anthropic

import config
from modules.stock_data import (
    get_current_price,
    get_key_ratios,
    get_technical_indicators,
    get_price_history,
    get_analyst_targets,
)
from modules.sentiment  import get_sentiment
from modules.valuation  import get_fair_value

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions (schema for Claude)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "get_stock_price",
        "description": (
            "Get the current price snapshot for a stock ticker. "
            "Returns price, daily change, market cap, 52-week high/low, volume, sector, industry."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_key_ratios",
        "description": (
            "Get fundamental financial ratios for a stock: P/E, P/B, EV/EBITDA, "
            "margins, ROE, ROA, debt/equity, beta, dividend yield, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": (
            "Get technical analysis indicators for a stock: SMA-20/50/200, "
            "RSI-14, MACD, Bollinger Bands, ATR. Useful for trend and momentum analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_sentiment",
        "description": (
            "Analyse recent news sentiment for a stock. "
            "Returns overall sentiment (BULLISH/BEARISH/NEUTRAL/MIXED), "
            "key themes, bull & bear points, and a human-readable summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol":       {"type": "string",  "description": "Stock ticker symbol"},
                "company_name": {"type": "string",  "description": "Full company name for better news search (optional)"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_fair_value",
        "description": (
            "Estimate the fair / intrinsic value of a stock using multiple models: "
            "Graham Number, DCF, Earnings Power Value, and analyst consensus targets. "
            "Also returns a verdict: UNDERVALUED / FAIRLY VALUED / OVERVALUED."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_analyst_targets",
        "description": (
            "Get Wall Street analyst price targets (low / mean / median / high) "
            "and the consensus recommendation (BUY / HOLD / SELL)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_price_history_summary",
        "description": (
            "Get a statistical summary of price history for a ticker "
            "(open/close/high/low/volume stats over a period). "
            "Useful for understanding price ranges and trends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string",  "description": "Stock ticker symbol"},
                "period": {
                    "type": "string",
                    "description": "Period: 1mo, 3mo, 6mo, 1y, 2y, 5y",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                },
            },
            "required": ["symbol"],
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Tool executor
# ─────────────────────────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict) -> str:
    """Call the appropriate Python function and return a JSON string result."""
    try:
        if name == "get_stock_price":
            result = get_current_price(inputs["symbol"])

        elif name == "get_key_ratios":
            result = get_key_ratios(inputs["symbol"])

        elif name == "get_technical_indicators":
            result = get_technical_indicators(inputs["symbol"])

        elif name == "get_sentiment":
            result = get_sentiment(
                inputs["symbol"],
                inputs.get("company_name", ""),
            )
            # Remove raw articles list to keep context window small
            result = {k: v for k, v in result.items() if k != "articles"}

        elif name == "get_fair_value":
            result = get_fair_value(inputs["symbol"])

        elif name == "get_analyst_targets":
            result = get_analyst_targets(inputs["symbol"])

        elif name == "get_price_history_summary":
            symbol = inputs["symbol"]
            period = inputs.get("period", "1y")
            df = get_price_history(symbol, period=period)
            if df.empty:
                result = {"error": "No price history available."}
            else:
                close = df["Close"]
                result = {
                    "symbol":        symbol.upper(),
                    "period":        period,
                    "start_date":    str(df.index[0].date()),
                    "end_date":      str(df.index[-1].date()),
                    "start_price":   round(float(close.iloc[0]), 4),
                    "end_price":     round(float(close.iloc[-1]), 4),
                    "min_price":     round(float(close.min()), 4),
                    "max_price":     round(float(close.max()), 4),
                    "mean_price":    round(float(close.mean()), 4),
                    "total_return_pct": round(
                        (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) * 100, 2
                    ),
                    "annualised_volatility_pct": round(
                        float(close.pct_change().std() * (252 ** 0.5) * 100), 2
                    ),
                    "trading_days": len(df),
                }
        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as exc:
        log.error("Tool %s failed: %s", name, exc)
        result = {"error": str(exc)}

    return json.dumps(result, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Agent class
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are **MarketMind**, an expert AI stock market analyst assistant.
You have access to real-time financial data, technical indicators, news sentiment analysis,
and fair-value estimation tools.

Your role:
- Answer questions about stocks clearly and concisely in plain English
- Use available tools to fetch up-to-date data before answering
- Provide balanced, fact-based analysis — highlight both risks and opportunities
- Explain financial jargon in simple terms when needed
- Always cite data sources and note any limitations

You do NOT provide personalised investment advice. Always remind users to consult a
financial advisor before making investment decisions.

When asked about a stock, proactively fetch relevant data (price, ratios, sentiment,
fair value) to give a comprehensive answer."""


class StockAgent:
    """
    Stateful conversational agent. Maintains message history across turns.
    Usage:
        agent = StockAgent()
        response = agent.chat("What do you think about NVDA?")
        response = agent.chat("What's its fair value?")  # context preserved
    """

    def __init__(self):
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in your .env file.")
        self._client   = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._history: list[dict] = []

    def reset(self):
        """Clear conversation history."""
        self._history = []

    def chat(self, user_message: str) -> str:
        """
        Send a message and return the agent's text response.
        Handles multi-round tool-use internally.
        """
        self._history.append({"role": "user", "content": user_message})

        max_iterations = 10  # guard against infinite tool-call loops
        for _ in range(max_iterations):
            response = self._client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self._history,
            )

            # ── If Claude wants to use tools ──────────────────────────────────
            if response.stop_reason == "tool_use":
                # Serialize SDK content blocks → plain dicts so the next API
                # call can reliably round-trip them through the SDK validator.
                content_dicts = []
                for block in response.content:
                    if hasattr(block, "model_dump"):
                        content_dicts.append(block.model_dump())
                    else:
                        content_dicts.append(block)

                self._history.append({
                    "role":    "assistant",
                    "content": content_dicts,
                })

                # Execute each tool and build tool_result messages
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        log.info("Tool call: %s(%s)", block.name, block.input)
                        output = _execute_tool(block.name, block.input)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     output,
                        })

                self._history.append({"role": "user", "content": tool_results})
                # Continue loop so Claude can process tool results

            # ── Final text response ───────────────────────────────────────────
            else:
                text_parts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ]
                final_text = "\n".join(text_parts).strip()
                self._history.append({
                    "role":    "assistant",
                    "content": final_text,
                })
                return final_text

        return "The agent reached the maximum number of reasoning steps. Please try a more specific question."


# ─────────────────────────────────────────────────────────────────────────────
# Market rundown helper (used by the dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def get_market_rundown(symbols: list[str]) -> str:
    """
    Generate a comprehensive market rundown for a list of tickers.
    Returns a formatted markdown string.
    """
    if not config.ANTHROPIC_API_KEY:
        return "⚠️ ANTHROPIC_API_KEY not set. Cannot generate market rundown."

    summaries = []
    for sym in symbols:
        try:
            price   = get_current_price(sym)
            ratios  = get_key_ratios(sym)
            tech    = get_technical_indicators(sym)
            fv      = get_fair_value(sym)
            sent    = get_sentiment(sym, price.get("name", sym))
            sent_clean = {k: v for k, v in sent.items() if k != "articles"}

            summaries.append({
                "symbol":  sym,
                "price":   price,
                "ratios":  ratios,
                "tech":    tech,
                "valuation": fv,
                "sentiment": sent_clean,
            })
        except Exception as exc:
            log.warning("Rundown data fetch failed for %s: %s", sym, exc)
            summaries.append({"symbol": sym, "error": str(exc)})

    data_json = json.dumps(summaries, default=str, indent=2)

    prompt = f"""You are a senior market analyst. Based on the following data for portfolio stocks,
write a concise but comprehensive **Daily Market Rundown** in markdown format.

For each stock include:
- Current price, daily change, and momentum (trend vs SMAs)
- Key fundamental snapshot (P/E, margin, growth)
- Sentiment overview (BULLISH / BEARISH / NEUTRAL)
- Fair-value assessment and upside/downside
- 1-2 sentence investment thesis or key risk to watch

End with a brief **Portfolio Outlook** paragraph (3-5 sentences).

DATA:
{data_json}

Write in a professional yet accessible tone. Use markdown headers, bullet points."""

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp   = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as exc:
        log.error("Market rundown generation failed: %s", exc)
        return f"Error generating market rundown: {exc}"
