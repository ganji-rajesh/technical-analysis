"""
analysis_agent.py
-----------------
Responsible for interpreting raw research data.

Two layers:
  1. Python layer  → computes statistical metrics (MA, RSI, Volatility, etc.)
  2. LLM layer     → Google Gemini API interprets metrics per stock
                     + cross-stock comparison

All LLM client, retry logic, and token management imported from shared
agents/utils.py — single source of truth, no duplication.
"""

import json
import logging
import math

import numpy as np

from agents.utils import call_llm, get_api_key

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("analysis_agent")


# ── Configuration ──────────────────────────────────────────────────────────────

# RSI configuration
RSI_PERIOD = 14

# Minimum data points required for analysis
MIN_HISTORY_POINTS = 20


# ── Helper Utilities ───────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float. Returns default on NaN, inf, or error."""
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _sanitize_metrics(metrics: dict) -> dict:
    """
    Replace any NaN or inf values in a metrics dict with None.
    Prevents LLM prompts from receiving invalid float strings.
    """
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
            logger.warning("Metric '%s' was NaN/inf — replaced with None", k)
        else:
            clean[k] = v
    return clean


# ── Python Stats Layer ─────────────────────────────────────────────────────────

def compute_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float:
    """
    Compute RSI using Wilder's Smoothed Moving Average (SMMA) — industry standard.

    This is the correct method used by Bloomberg, TradingView, and all
    professional platforms. The original code used a simple average (np.mean)
    which produces different values from every financial data source.

    Wilder's method:
      - Seed: simple average of first `period` gains/losses
      - Subsequent: avg = (prev_avg * (period-1) + current) / period

    Returns 50.0 if insufficient data (neutral, non-misleading default).
    """
    if len(closes) < period + 1:
        logger.warning(
            "RSI: only %d data points available, need %d. Returning neutral 50.0",
            len(closes), period + 1
        )
        return 50.0

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average of first `period` values (Wilder's initialisation)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Apply Wilder's SMMA for remaining values
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_metrics(stock_data: dict) -> dict:
    """
    Compute all technical and statistical metrics from 6-month price history.

    Fixes vs original:
      - MA-50 returns None (not a false average) when < 50 data points
      - Daily returns volatility guarded against zero-price division
      - All output values sanitized for NaN/inf before returning

    Returns:
        Dict of 10 computed metrics, all floats or None.

    Raises:
        ValueError — insufficient price history
    """
    history = stock_data.get("price_history", [])
    ticker  = stock_data.get("ticker", "UNKNOWN")

    if not history:
        raise ValueError(f"No price history for '{ticker}'. Cannot compute metrics.")

    closes = [_safe_float(h["close"]) for h in history if h.get("close")]

    if len(closes) < MIN_HISTORY_POINTS:
        raise ValueError(
            f"Insufficient price history for '{ticker}': "
            f"need {MIN_HISTORY_POINTS} data points, got {len(closes)}."
        )

    closes_arr    = np.array(closes, dtype=np.float64)
    current_price = _safe_float(stock_data.get("current_price", closes[-1]))

    # ── Moving Averages ──
    ma20 = float(np.mean(closes_arr[-20:]))

    # MA-50: return None if < 50 points — do NOT fabricate a false value
    if len(closes_arr) >= 50:
        ma50 = float(np.mean(closes_arr[-50:]))
    else:
        ma50 = None
        logger.warning(
            "[%s] Only %d data points — MA-50 set to None (insufficient data)",
            ticker, len(closes_arr)
        )

    # ── Price vs MAs ──
    price_vs_ma20 = ((current_price - ma20) / ma20 * 100) if ma20 else 0.0
    price_vs_ma50 = ((current_price - ma50) / ma50 * 100) if ma50 else None

    # ── 20-day Momentum ──
    momentum_20d = (
        ((closes[-1] - closes[-20]) / closes[-20] * 100)
        if closes[-20] and closes[-20] != 0 else 0.0
    )

    # ── Annualized Volatility ──
    # Guard: filter out zero prices before computing returns to avoid inf/NaN
    valid_closes = closes_arr[closes_arr > 0]
    if len(valid_closes) > 1:
        daily_returns = np.diff(valid_closes) / valid_closes[:-1]
        volatility = float(np.std(daily_returns) * np.sqrt(252) * 100)
    else:
        volatility = 0.0
        logger.warning("[%s] Could not compute volatility — all prices zero or one point", ticker)

    # ── 6-Month Return ──
    six_month_return = (
        ((closes[-1] - closes[0]) / closes[0] * 100)
        if closes[0] and closes[0] != 0 else 0.0
    )

    # ── RSI (Wilder's SMMA) ──
    rsi = compute_rsi(closes_arr, period=RSI_PERIOD)

    # ── Distance from 52W High/Low ──
    week52_high   = _safe_float(stock_data.get("52_week_high", 0))
    week52_low    = _safe_float(stock_data.get("52_week_low",  0))
    pct_from_high = ((current_price - week52_high) / week52_high * 100) if week52_high else None
    pct_from_low  = ((current_price - week52_low)  / week52_low  * 100) if week52_low  else None

    raw_metrics = {
        "ma_20":                     round(ma20,          2),
        "ma_50":                     round(ma50, 2) if ma50 is not None else None,
        "price_vs_ma20_pct":         round(price_vs_ma20, 2),
        "price_vs_ma50_pct":         round(price_vs_ma50, 2) if price_vs_ma50 is not None else None,
        "momentum_20d_pct":          round(momentum_20d,  2),
        "volatility_annualized_pct": round(volatility,    2),
        "six_month_return_pct":      round(six_month_return, 2),
        "rsi_14":                    rsi,
        "pct_from_52w_high":         round(pct_from_high, 2) if pct_from_high is not None else None,
        "pct_from_52w_low":          round(pct_from_low,  2) if pct_from_low  is not None else None,
    }

    # Final NaN/inf sanitization pass before metrics leave this function
    return _sanitize_metrics(raw_metrics)


# ── LLM Layer ──────────────────────────────────────────────────────────────────

SYSTEM_ANALYST = (
    "You are a professional equity research analyst specializing in Indian stock markets. "
    "Be precise, data-driven, and concise. Always reference specific numbers in your analysis."
)


def analyze_single_stock(stock_data: dict, metrics: dict) -> str:
    """
    Ask LLM to interpret computed metrics for a single stock.

    Passes sanitized metrics (no NaN/inf) and full fundamentals.
    Returns validated non-empty string.
    """
    ticker = stock_data.get("ticker", "")
    name   = stock_data.get("company_name", ticker)
    news   = "\n".join(
        [f"- {n['headline']}" for n in stock_data.get("recent_news", [])]
    ) or "No recent news available."

    fundamentals = {
        key: stock_data.get(key)
        for key in [
            "current_price", "pe_ratio", "pb_ratio", "eps",
            "roe", "debt_to_equity", "profit_margin", "market_cap",
            "revenue", "net_income", "beta", "dividend_yield",
            "52_week_high", "52_week_low", "sector",
        ]
    }

    prompt = f"""Analyze the following data for {name} ({ticker}):

FUNDAMENTAL DATA:
{json.dumps(fundamentals, indent=2)}

TECHNICAL METRICS:
{json.dumps(metrics, indent=2)}

RECENT NEWS:
{news}

Provide a concise analysis covering:
1. Overall outlook (2-3 sentences)
2. Technical signals — what MA, RSI, and momentum indicate
3. Fundamental strengths and weaknesses
4. Key risks
5. Sentiment label: Bullish / Neutral / Bearish

Be specific with numbers. Keep it under 300 words."""

    logger.info("Running LLM analysis for %s...", ticker)
    result = call_llm(prompt, system=SYSTEM_ANALYST, max_tokens=600)
    logger.info("✅ LLM analysis complete for %s (%d chars)", ticker, len(result))
    return result


def compare_stocks(individual_analyses: list) -> str:
    """
    Ask LLM to compare multiple stocks.
    Only called when 2+ stocks are present.

    Fix: max_tokens scaled by stock count to prevent truncation on 4-5 stocks.
    """
    # Scale token budget with stock count to prevent truncation
    max_tokens = min(300 + (len(individual_analyses) * 150), 900)

    summary = [
        {
            "ticker":  a["ticker"],
            "company": a["company_name"],
            "metrics": a["metrics"],
        }
        for a in individual_analyses
    ]

    prompt = f"""Compare the following Indian stocks based on their metrics:

{json.dumps(summary, indent=2)}

Provide:
1. Which stock shows the strongest momentum and why
2. Relative valuation — which is cheaper or more expensive and why
3. Risk comparison — which carries more risk and why
4. Final ranking from best to worst investment prospect with justification

Be concise and data-driven. Keep it under {max_tokens // 2} words."""

    system = (
        "You are a CFA-certified equity analyst at a top Indian investment bank. "
        "Give balanced, objective comparisons backed by specific data points."
    )

    logger.info("Running cross-stock comparison for %d stocks...", len(individual_analyses))
    result = call_llm(prompt, system=system, max_tokens=max_tokens)
    logger.info("✅ Cross-stock comparison complete (%d chars)", len(result))
    return result


# ── Analysis Agent Entry Point ─────────────────────────────────────────────────

def run_analysis_agent(research_data: dict) -> dict:
    """
    Run the full analysis pipeline for all stocks in research_data.

    Steps:
      1. Validate HF_TOKEN is set (fails fast before any computation)
      2. Compute Python metrics for each stock
      3. Sanitize metrics (remove NaN/inf)
      4. Send to LLM for individual interpretation
      5. If 2+ stocks → cross-stock comparison

    Args:
        research_data: Output dict from run_research_agent()

    Returns:
        {
            "individual_analyses": [
                {
                    "ticker":       str,
                    "company_name": str,
                    "metrics":      dict,   ← sanitized, no NaN
                    "llm_analysis": str,    ← validated non-empty
                },
                ...
            ],
            "comparison_insights": str | None
        }

    Raises:
        EnvironmentError — GEMINI_API_KEY not set
        ValueError       — bad/missing research data
        RuntimeError     — LLM failure after all retries
    """
    # Fail fast — validate token before doing any computation
    get_api_key()

    stocks = research_data.get("stocks", {})
    if not stocks:
        raise ValueError("No stock data found in research output.")

    individual_analyses = []

    for ticker, stock_data in stocks.items():
        logger.info("Computing metrics for %s...", ticker)
        metrics = compute_metrics(stock_data)

        llm_analysis = analyze_single_stock(stock_data, metrics)

        individual_analyses.append({
            "ticker":       ticker,
            "company_name": stock_data.get("company_name", ticker),
            "metrics":      metrics,
            "llm_analysis": llm_analysis,
        })
        logger.info("✅ %s analysis complete.", ticker)

    # Cross-stock comparison (only if 2+ stocks)
    comparison_insights = None
    if len(individual_analyses) >= 2:
        comparison_insights = compare_stocks(individual_analyses)
        logger.info("✅ Comparison complete.")

    return {
        "individual_analyses": individual_analyses,
        "comparison_insights": comparison_insights,
    }
