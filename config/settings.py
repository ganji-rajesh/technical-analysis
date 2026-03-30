"""
settings.py — Centralised configuration constants for the TA Report Generator.

All magic numbers, model identifiers, and tuneable defaults are declared here so
that every other module imports from a single source of truth.  Nothing in this
file should depend on runtime state (no I/O, no API calls at import time).
"""

from __future__ import annotations

# ============================================================================
# Gemini / LLM Configuration
# ============================================================================

GEMINI_MODEL_NAME: str = "gemini-2.0-flash"
"""Inference model used for report generation (Section 4.3 of plan.md)."""

GEMINI_MAX_RETRIES: int = 3
"""Maximum retry attempts on transient Gemini API errors (rate-limit, timeout)."""

GEMINI_RETRY_BACKOFF_BASE: float = 2.0
"""Base multiplier for exponential back-off between retries (seconds)."""

GEMINI_TEMPERATURE: float = 0.4
"""
Sampling temperature for generation.  A moderately low value balances
factual rigour with natural language fluency in financial reports.
"""

GEMINI_MAX_OUTPUT_TOKENS: int = 4096
"""Upper bound on generated token count to prevent runaway responses."""

# ============================================================================
# Data Fetching — yfinance
# ============================================================================

DEFAULT_PERIOD: str = "2y"
"""
Default historical look-back period.  Two years of weekly data yields ~104
candles — sufficient for Dow Theory primary-trend identification.
"""

DEFAULT_INTERVAL: str = "1wk"
"""Weekly granularity for swing-trading focus."""

MIN_DATA_POINTS: int = 52
"""
Minimum number of weekly candles required for a meaningful analysis.
Below this threshold the pipeline raises a ``ValueError``.
"""

# ============================================================================
# Data Fetching — configurable period / interval choices (for the UI)
# ============================================================================

PERIOD_CHOICES: list[str] = ["6mo", "1y", "2y", "3y", "5y"]
"""
Ordered list of yfinance period strings shown in the left-panel dropdown.
The default (``DEFAULT_PERIOD``) must appear in this list.
"""

PERIOD_LABELS: dict[str, str] = {
    "6mo": "6 Months",
    "1y":  "1 Year",
    "2y":  "2 Years  (default)",
    "3y":  "3 Years",
    "5y":  "5 Years",
}
"""Human-readable labels shown next to each period option."""

INTERVAL_CHOICES: list[str] = ["1d", "1wk", "1mo"]
"""
Ordered list of yfinance interval strings shown in the left-panel dropdown.
The default (``DEFAULT_INTERVAL``) must appear in this list.
"""

INTERVAL_LABELS: dict[str, str] = {
    "1d":  "Daily",
    "1wk": "Weekly  (default)",
    "1mo": "Monthly",
}
"""Human-readable labels shown next to each interval option."""

INTERVAL_MIN_POINTS: dict[str, int] = {
    "1d":  20,    # ~1 month of trading days
    "1wk": 52,    # 1 year of weeks
    "1mo": 12,    # 1 year of months
}
"""
Per-interval minimum row count for a meaningful analysis.
Overrides the global ``MIN_DATA_POINTS`` inside the pipeline.
"""

YFINANCE_RETRY_DELAY_SECONDS: float = 5.0
"""Delay before a single retry on yfinance network failure."""

# ============================================================================
# Technical Indicators — pandas-ta
# ============================================================================

EMA_PERIODS: tuple[int, ...] = (20, 50, 100)
"""EMA look-back windows for short / medium / long-term trend detection."""

RSI_PERIOD: int = 14
"""Standard RSI look-back period."""

RSI_OVERSOLD: int = 30
"""RSI level at or below which a security is considered oversold."""

RSI_OVERBOUGHT: int = 70
"""RSI level at or above which a security is considered overbought."""

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
"""Standard 12-26-9 MACD parameters."""

BBANDS_LENGTH: int = 20
BBANDS_STD_DEV: float = 2.0
"""Bollinger Bands: 20-period SMA ± 2 standard deviations."""

ATR_PERIOD: int = 14
"""Average True Range look-back for volatility / stop-loss sizing."""

ADX_PERIOD: int = 14
"""ADX look-back period for trend-strength measurement."""

ADX_TRENDING_THRESHOLD: int = 25
"""ADX above this value indicates a trending market."""

ADX_RANGING_THRESHOLD: int = 20
"""ADX below this value indicates a range-bound market."""

VOLUME_SMA_PERIOD: int = 10
"""Simple moving average period for volume comparison."""

# ============================================================================
# Fibonacci Retracement
# ============================================================================

FIBONACCI_LEVELS: tuple[float, ...] = (0.236, 0.382, 0.500, 0.618, 0.786)
"""Standard Fibonacci retracement ratios."""

# ============================================================================
# Candlestick Patterns — TA-Lib
# ============================================================================

CANDLESTICK_PATTERNS: dict[str, str] = {
    "Hammer":              "CDLHAMMER",
    "Inverted Hammer":     "CDLINVERTEDHAMMER",
    "Shooting Star":       "CDLSHOOTINGSTAR",
    "Engulfing":           "CDLENGULFING",
    "Morning Star":        "CDLMORNINGSTAR",
    "Evening Star":        "CDLEVENINGSTAR",
    "Doji":                "CDLDOJI",
    "Marubozu":            "CDLMARUBOZU",
    "Hanging Man":         "CDLHANGINGMAN",
    "Harami":              "CDLHARAMI",
    "Piercing":            "CDLPIERCING",
    "Dark Cloud Cover":    "CDLDARKCLOUDCOVER",
    "Spinning Top":        "CDLSPINNINGTOP",
    "Three White Soldiers": "CDL3WHITESOLDIERS",
}
"""
Mapping of human-readable pattern names → TA-Lib function identifiers.
Keep alphabetically ordered by display name for deterministic iteration.
"""

PATTERN_LOOKBACK_CANDLES: int = 5
"""Number of recent candles inspected when scoring the Grand Checklist (item #1)."""

# ============================================================================
# Grand TA Checklist (Section 6 of plan.md)
# ============================================================================

CHECKLIST_SR_PROXIMITY_PCT: float = 0.02
"""
Criterion #2 — price is "near" S&R when within this percentage of an
identified level (2 % by default).
"""

CHECKLIST_MIN_RRR: float = 1.5
"""
Criterion #6 — minimum acceptable Risk-Reward Ratio for a trade signal.
"""

# ============================================================================
# Input Validation (Security — Section 9 of plan.md)
# ============================================================================

TICKER_REGEX: str = r"^[A-Z0-9.\^=\-]{1,20}$"
"""
Regex pattern for valid ticker symbols.  Permits uppercase alphanumerics,
dots (``RELIANCE.NS``), carets (``^NSEI``), equals, and hyphens.
Max 20 characters to guard against injection.
"""

API_KEY_PREFIX: str = "AIza"
"""Expected prefix of a Google / Gemini API key for quick format validation."""

API_KEY_MIN_LENGTH: int = 30
"""Minimum character length for a plausible Gemini API key."""

# ============================================================================
# UI / Gradio Defaults
# ============================================================================

REPORT_TAIL_ROWS: int = 20
"""Number of most-recent weekly rows sent to the LLM for context (Section 4.3)."""

GRADIO_THEME: str = "Soft"
"""Gradio built-in theme name used in ``gr.Blocks``."""

# ============================================================================
# Chart Builder Defaults
# ============================================================================

CHART_HEIGHT: int = 480
"""Default pixel height for Plotly chart figures."""

CHART_TEMPLATE: str = "plotly_dark"
"""Plotly template for consistent dark-themed aesthetics across all charts."""
