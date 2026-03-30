"""
research_agent.py
-----------------
Responsible ONLY for fetching raw data from yfinance.
No analysis. No interpretation. Just clean structured JSON.

Fixes applied vs original:
  - Module-level safe() utility (was nested inside fetch_stock_data)
  - TTL-based caching to avoid repeated yfinance calls (15-min TTL)
  - Retry logic with exponential backoff on yfinance failures
  - Timezone-aware UTC timestamps
  - Vectorized pandas operations replacing slow iterrows()
  - Validation covers more than just price (sector, history length)
  - News timestamp normalized to ISO date string
  - Bare except replaced with logged warning
  - Ticker count guardrail (max 5 per run)
  - textwrap.shorten() replaces hard [:500] slice
"""

import logging
import textwrap
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

from agents.utils import with_retry

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("research_agent")

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_TICKERS        = 5        # Hard limit per pipeline run
HISTORY_DAYS       = 180      # 6-month price history window
NEWS_LIMIT         = 5        # Max headlines per stock
DESCRIPTION_LIMIT  = 500      # Max chars for company description
CACHE_TTL_SECONDS  = 900      # 15-minute cache TTL
MAX_FETCH_RETRIES  = 3        # Retry attempts for yfinance calls
RETRY_BACKOFF_BASE = 2        # Exponential backoff base (seconds)


# ── In-Memory TTL Cache ────────────────────────────────────────────────────────

class TTLCache:
    """
    Simple in-memory cache with per-entry TTL expiry.
    Thread-safety is not guaranteed — acceptable for single-threaded Gradio use.
    """
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self._store: dict = {}
        self._ttl   = ttl_seconds

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if datetime.now(timezone.utc) > expiry:
            del self._store[key]
            logger.debug("Cache expired for key: %s", key)
            return None
        logger.info("Cache hit for ticker: %s", key)
        return value

    def set(self, key: str, value) -> None:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
        self._store[key] = (value, expiry)
        logger.debug("Cached ticker: %s (expires in %ds)", key, self._ttl)

    def clear(self) -> None:
        self._store.clear()


_cache = TTLCache()


# ── Retry Decorator (imported from agents.utils) ──────────────────────────────
# with_retry is imported from agents.utils — single shared implementation.
# Used here with exponential=True for yfinance API calls (1s, 2s, 4s backoff).


# ── Module-Level Safe Value Extractor ─────────────────────────────────────────

def _safe(info: dict, key: str, default=None):
    """
    Safely extract a value from a yfinance info dict.
    Returns default for None, "N/A", inf, -inf.
    Rounds floats to 4 decimal places.
    """
    val = info.get(key, default)
    if val in (None, "N/A", float("inf"), float("-inf")):
        return default
    try:
        if isinstance(val, float):
            return round(val, 4)
    except Exception:
        pass
    return val


# ── Ticker Formatting ──────────────────────────────────────────────────────────

def format_ticker(ticker: str) -> str:
    """
    Normalize a raw ticker string for yfinance.

    Rules:
      - Strip whitespace, uppercase
      - Append .NS for plain NSE tickers (no existing suffix)
      - Preserve .BO for BSE tickers

    Examples:
      "tcs"       → "TCS.NS"
      "TCS.NS"    → "TCS.NS"
      "TCS.BO"    → "TCS.BO"
      "  infy  "  → "INFY.NS"
    """
    ticker = ticker.strip().upper()
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker += ".NS"
    return ticker


# ── News Parser ────────────────────────────────────────────────────────────────

def _parse_news(raw_news: list) -> list:
    """
    Parse yfinance news list into a normalized list of {headline, date} dicts.
    Handles both old and new yfinance news response formats.
    Returns empty list on any failure — news is non-critical.
    """
    news = []
    for item in raw_news[:NEWS_LIMIT]:
        try:
            content = item.get("content", {})
            title   = content.get("title") or item.get("title", "No title")

            # Normalize publication date to YYYY-MM-DD string
            pub = content.get("pubDate") or item.get("providerPublishTime", "")
            if isinstance(pub, (int, float)):
                pub = datetime.fromtimestamp(pub, tz=timezone.utc).strftime("%Y-%m-%d")
            elif isinstance(pub, str) and "T" in pub:
                pub = pub[:10]   # "2026-03-23T10:00:00Z" → "2026-03-23"

            news.append({"headline": str(title), "date": pub or ""})
        except Exception as exc:
            logger.warning("Skipping malformed news item: %s", exc)
            continue
    return news


# ── Price History Parser ───────────────────────────────────────────────────────

def _parse_price_history(hist_df) -> list:
    """
    Convert yfinance OHLCV DataFrame to a list of dicts.
    Uses vectorized pandas operations instead of iterrows() for performance.

    Handles:
      - MultiIndex columns from newer yfinance versions (>= 0.2.40)
      - Both timezone-aware and timezone-naive DatetimeIndex
    """
    # Flatten MultiIndex columns (newer yfinance returns ('Close', 'TCS.NS'))
    if isinstance(hist_df.columns, pd.MultiIndex):
        hist_df = hist_df.copy()
        hist_df.columns = hist_df.columns.get_level_values(0)

    df = hist_df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Safely strip timezone — tz_localize(None) crashes on naive index
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df["date"] = df.index.strftime("%Y-%m-%d")
    df = df.round({"Open": 2, "High": 2, "Low": 2, "Close": 2})
    df["Volume"] = df["Volume"].astype(int)
    return df[["date", "Open", "High", "Low", "Close", "Volume"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    ).to_dict("records")


# ── Single Stock Fetcher ───────────────────────────────────────────────────────

@with_retry(max_attempts=MAX_FETCH_RETRIES, wait_seconds=RETRY_BACKOFF_BASE, exponential=True)
def _fetch_from_yfinance(formatted_ticker: str) -> tuple:
    """
    Raw yfinance fetcher — wrapped with retry decorator.
    Returns (info_dict, history_dataframe, raw_news_list).
    Raises ValueError on invalid/unavailable ticker.
    """
    stock = yf.Ticker(formatted_ticker)

    # ── Company Info ──
    info = stock.info or {}
    if not info.get("regularMarketPrice") and not info.get("currentPrice"):
        raise ValueError(
            f"Ticker '{formatted_ticker}' not found or no market data available. "
            "Please verify the ticker symbol and try again."
        )

    # ── Price History ──
    end_date   = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    hist_df = stock.history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )
    if hist_df.empty:
        raise ValueError(
            f"No historical price data found for '{formatted_ticker}'. "
            "Please verify the ticker symbol."
        )

    # ── News (non-critical) ──
    try:
        raw_news = stock.news or []
    except Exception as exc:
        logger.warning("[%s] News fetch failed (non-critical): %s", formatted_ticker, exc)
        raw_news = []

    return info, hist_df, raw_news


def fetch_stock_data(ticker: str) -> dict:
    """
    Fetch and structure all data for a single ticker.

    Flow:
      1. Check TTL cache — return cached result if fresh
      2. Format ticker (e.g. TCS → TCS.NS)
      3. Fetch from yfinance with retry + exponential backoff
      4. Parse and structure into a clean dict
      5. Cache result for CACHE_TTL_SECONDS

    Raises:
        ValueError  — invalid ticker or no data available
        RuntimeError — unexpected fetch failure after all retries
    """
    formatted = format_ticker(ticker)

    # ── Cache Check ──
    cached = _cache.get(formatted)
    if cached is not None:
        return cached

    logger.info("Fetching data for %s (%s)...", ticker, formatted)

    try:
        info, hist_df, raw_news = _fetch_from_yfinance(formatted)
    except ValueError:
        raise   # Re-raise ticker-not-found errors as-is
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch data for '{ticker}' after {MAX_FETCH_RETRIES} attempts: {exc}"
        ) from exc

    # ── Validate History Length ──
    if len(hist_df) < 20:
        raise ValueError(
            f"Insufficient price history for '{ticker}' — "
            f"need at least 20 trading days, got {len(hist_df)}. "
            "This may be a newly listed stock."
        )

    # ── Parse Sub-components ──
    price_history = _parse_price_history(hist_df)
    news          = _parse_news(raw_news)

    # ── Assemble Structured Output ──
    result = {
        "ticker":           ticker.upper(),
        "formatted_ticker": formatted,
        "company_name":     _safe(info, "longName", _safe(info, "shortName", ticker)),
        "sector":           _safe(info, "sector",   "N/A"),
        "industry":         _safe(info, "industry", "N/A"),
        "description":      textwrap.shorten(
                                info.get("longBusinessSummary") or "No description available.",
                                width=DESCRIPTION_LIMIT,
                                placeholder="...",
                            ),

        # ── Price ──
        "current_price":   _safe(info, "currentPrice",    _safe(info, "regularMarketPrice")),
        "previous_close":  _safe(info, "previousClose"),
        "52_week_high":    _safe(info, "fiftyTwoWeekHigh"),
        "52_week_low":     _safe(info, "fiftyTwoWeekLow"),
        "average_volume":  _safe(info, "averageVolume"),
        "beta":            _safe(info, "beta"),

        # ── Fundamentals ──
        "market_cap":      _safe(info, "marketCap"),
        "pe_ratio":        _safe(info, "trailingPE"),
        "forward_pe":      _safe(info, "forwardPE"),
        "pb_ratio":        _safe(info, "priceToBook"),
        "eps":             _safe(info, "trailingEps"),
        "book_value":      _safe(info, "bookValue"),
        "dividend_yield":  _safe(info, "dividendYield"),
        "revenue":         _safe(info, "totalRevenue"),
        "net_income":      _safe(info, "netIncomeToCommon"),
        "profit_margin":   _safe(info, "profitMargins"),
        "roe":             _safe(info, "returnOnEquity"),
        "debt_to_equity":  _safe(info, "debtToEquity"),

        # ── History & News ──
        "price_history":   price_history,
        "recent_news":     news,
    }

    _cache.set(formatted, result)
    logger.info(
        "✅ %s — %s | ₹%s | %d history rows | %d news items",
        ticker,
        result["company_name"],
        result["current_price"],
        len(price_history),
        len(news),
    )
    return result


# ── Research Agent Entry Point ─────────────────────────────────────────────────

def run_research_agent(tickers: list) -> dict:
    """
    Run research for a list of tickers.

    Guardrails:
      - Raises ValueError if tickers list is empty
      - Raises ValueError if more than MAX_TICKERS tickers provided
      - Raises ValueError immediately if ANY ticker fails (fail-fast)

    Args:
        tickers: List of cleaned ticker strings (output of parse_tickers)

    Returns:
        {
            "stocks":    { "TCS": {...}, "INFY": {...} },
            "timestamp": "2026-03-23T07:30:00+00:00"   ← UTC ISO 8601
        }
    """
    if not tickers:
        raise ValueError("No tickers provided.")

    if len(tickers) > MAX_TICKERS:
        raise ValueError(
            f"Too many tickers: {len(tickers)} provided, maximum is {MAX_TICKERS}. "
            "Please reduce your selection and try again."
        )

    results = {}

    for ticker in tickers:
        # Any failure here raises ValueError or RuntimeError → pipeline stops immediately
        data = fetch_stock_data(ticker)
        results[ticker.upper()] = data

    return {
        "stocks":    results,
        "timestamp": datetime.now(timezone.utc).isoformat(),   # UTC, timezone-aware
    }
