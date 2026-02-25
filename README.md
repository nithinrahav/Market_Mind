# MarketMind – LLM-Powered Stock Market Analyzer

A Streamlit dashboard that combines **E\*Trade portfolio data**, **real-time financial analytics**, **news sentiment analysis**, and **Claude AI** to give you a complete picture of your investments.

---

## Features

| Feature | Description |
|---|---|
| **Portfolio Value** | OAuth-connected E\*Trade integration – live positions, balances, P&L |
| **Stock Dashboard** | Key ratios, candlestick chart, technical indicators (RSI, MACD, Bollinger) |
| **Fair Value** | Graham Number, DCF, Earnings Power Value, Analyst consensus – blended target |
| **News Sentiment** | Multi-source news analysis (BULLISH / BEARISH / NEUTRAL / MIXED) powered by Claude |
| **Market Rundown** | AI-generated daily brief for all portfolio stocks |
| **AI Chat** | Conversational stock analyst powered by Claude with tool-use (real data) |

---

## Quick Start

### 1. Clone and install dependencies

```bash
cd LLM_Powered_Stock_Market_Analyzer
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your keys
```

Required keys:
- `ANTHROPIC_API_KEY` – from https://console.anthropic.com
- `ETRADE_CONSUMER_KEY` + `ETRADE_CONSUMER_SECRET` – from https://developer.etrade.com *(optional – manual ticker entry is available)*
- `NEWS_API_KEY` – from https://newsapi.org *(optional – Yahoo Finance RSS is used as fallback)*

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Project Structure

```
LLM_Powered_Stock_Market_Analyzer/
├── app.py                   # Streamlit dashboard (main entry point)
├── config.py                # Environment variable loader
├── requirements.txt
├── .env.example
├── modules/
│   ├── etrade_client.py     # E*Trade OAuth + portfolio/quote API
│   ├── stock_data.py        # yfinance: prices, ratios, technicals
│   ├── sentiment.py         # NewsAPI + Yahoo RSS + Claude sentiment
│   ├── valuation.py         # Graham, DCF, EPV fair-value models
│   └── llm_agent.py         # Claude tool-use agent + market rundown
└── utils/
    └── helpers.py           # Shared utilities
```

---

## E\*Trade OAuth Flow

1. Click **Connect to E\*Trade** in the sidebar
2. Click the authorisation link that appears
3. Log in and authorise MarketMind
4. Copy the verification code from E\*Trade
5. Paste it in the **Verification code** field and click **Submit Verifier**

> Set `ETRADE_ENV=sandbox` in `.env` to use the sandbox (paper trading) environment first.

---

## AI Chat – Example Questions

- *"What is the fair value of NVDA and is it overvalued?"*
- *"Analyse the recent news sentiment for TSLA"*
- *"Compare MSFT and GOOGL on fundamentals"*
- *"What does the RSI say about AAPL right now?"*
- *"Which of my portfolio stocks has the best upside potential?"*

---

## Disclaimer

MarketMind is for **informational and educational purposes only**.
It does **not** provide personalised investment advice.
Always consult a qualified financial advisor before making investment decisions.
