"""
main_pipeline.py — Orchestrator layer for the TA Report Generator.

Responsibility (plan.md §2, §5):
    Connects the Data, Computation, Chart, and Inference layers into a single
    sequential pipeline invoked by the Gradio UI.

    User Input → validate → fetch → compute → (charts + LLM report) → UI

Design decisions
----------------
* **Single public entry point** (``run_pipeline``) returning a structured
  ``PipelineResult`` — the UI layer never touches domain internals directly.
* **No threading** — the pipeline is deliberately synchronous.  The single
  Gemini call produces a cohesive report (plan.md §10).
* **Progress callback** — accepts an optional callable so Gradio's
  ``gr.Progress`` can display stage-level feedback without coupling the
  orchestrator to the UI framework.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pandas as pd
import plotly.graph_objects as go

from config.settings import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    GEMINI_MODEL_NAME,
    INTERVAL_MIN_POINTS,
    MIN_DATA_POINTS,
    REPORT_TAIL_ROWS,
)
from core.data_fetch import fetch_weekly_data
from core.ta_compute import (
    compute_all_indicators,
    compute_grand_checklist,
    format_checklist_details,
    format_pattern_summary,
)
from core.llm_inference import generate_full_report
from utils.chart_builder import build_all_charts
from utils.validators import sanitise_ticker, validate_api_key

logger = logging.getLogger(__name__)


# ============================================================================
# Public data structures
# ============================================================================

@dataclass(frozen=True)
class PipelineResult:
    """Immutable container for all pipeline outputs consumed by the UI.

    Attributes
    ----------
    report_md:
        Markdown-formatted TA report from Gemini.
    charts:
        Dict of chart name → ``go.Figure`` (keys: price, volume, rsi, macd).
    patterns_df:
        DataFrame of detected candlestick patterns (last 5 weeks).
    indicators_df:
        DataFrame of indicator values (last ``REPORT_TAIL_ROWS`` weeks).
    sr_levels_df:
        DataFrame listing support & resistance price levels.
    fib_levels_df:
        DataFrame listing Fibonacci retracement levels.
    checklist:
        Grand TA Checklist result dict (score, max_score, details).
    ticker:
        The sanitised ticker symbol.
    elapsed_seconds:
        Total wall-clock time for the pipeline run.
    """

    report_md: str
    charts: dict[str, go.Figure]
    ohlcv_df: pd.DataFrame
    patterns_df: pd.DataFrame
    indicators_df: pd.DataFrame
    sr_levels_df: pd.DataFrame
    fib_levels_df: pd.DataFrame
    checklist: dict[str, Any]
    ticker: str
    elapsed_seconds: float


# Type alias for the optional progress callback.
ProgressCallback = Callable[[float, str], None] | None


# ============================================================================
# Public API
# ============================================================================

def run_pipeline(
    ticker: str,
    api_key: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    progress_cb: ProgressCallback = None,
) -> PipelineResult:
    """Execute the full analysis pipeline end-to-end.

    Parameters
    ----------
    ticker:
        Raw ticker string from the UI (will be validated and sanitised).
    api_key:
        Raw Gemini API key from the UI (will be format-validated).
    period:
        yfinance look-back period string (e.g. ``"2y"``, ``"1y"``, ``"6mo"``).
        Defaults to the value of ``DEFAULT_PERIOD`` in ``config/settings.py``.
    interval:
        yfinance candle interval (e.g. ``"1wk"``, ``"1d"``, ``"1mo"``).
        Defaults to the value of ``DEFAULT_INTERVAL`` in ``config/settings.py``.
    progress_cb:
        Optional ``(fraction, description)`` callback for UI progress bars.
        Fraction is a float in [0, 1].

    Returns
    -------
    PipelineResult
        All data the UI needs to populate both tabs.

    Raises
    ------
    ValueError
        Invalid ticker or API key format.
    ConnectionError
        yfinance network failure after retry.
    RuntimeError
        Gemini API failure after retries.
    """
    t_start = time.perf_counter()

    def _progress(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac, msg)
        logger.info("[%.0f%%] %s", frac * 100, msg)

    # ── Stage 1: Validate inputs ───────────────────────────────────────────
    _progress(0.0, "Validating inputs…")
    clean_ticker = sanitise_ticker(ticker)

    key_result = validate_api_key(api_key)
    if not key_result.is_valid:
        raise ValueError(key_result.message)

    # ── Stage 2: Fetch OHLCV data ──────────────────────────────────────────
    _progress(0.10, f"Fetching {interval} data ({period}) for {clean_ticker}…")
    # Determine the per-interval minimum row threshold.
    min_rows = INTERVAL_MIN_POINTS.get(interval, MIN_DATA_POINTS)
    df = fetch_weekly_data(clean_ticker, period=period, interval=interval)

    # ── Stage 3: Compute indicators ────────────────────────────────────────
    _progress(0.30, "Computing technical indicators…")
    computed = compute_all_indicators(df)

    enriched_df: pd.DataFrame = computed["indicators"]
    patterns: dict = computed["patterns"]
    fib_levels: dict[str, float] = computed["fibonacci"]
    sr_levels: list[float] = computed["sr_levels"]

    # ── Stage 4: Build charts ──────────────────────────────────────────────
    _progress(0.50, "Building interactive charts…")
    charts = build_all_charts(
        df=enriched_df,
        ticker=clean_ticker,
        sr_levels=sr_levels,
        fib_levels=fib_levels,
    )

    # ── Stage 5: Generate LLM report ──────────────────────────────────────
    _progress(0.60, f"Generating report via {GEMINI_MODEL_NAME}…")
    report_md = generate_full_report(computed, api_key.strip(), clean_ticker)

    # ── Stage 6: Prepare display DataFrames ────────────────────────────────
    _progress(0.90, "Preparing data tables…")
    ohlcv_df     = _build_ohlcv_dataframe(df)
    patterns_df  = _build_patterns_dataframe(patterns)
    indicators_df = _build_indicators_dataframe(enriched_df)
    sr_levels_df = _build_sr_dataframe(sr_levels)
    fib_levels_df = _build_fib_dataframe(fib_levels)

    # ── Stage 7: Score checklist (already computed inside llm_inference,
    #             but we recompute here to surface it in the UI) ─────────────
    checklist = compute_grand_checklist(enriched_df, patterns, sr_levels)

    elapsed = time.perf_counter() - t_start
    _progress(1.0, f"Done — completed in {elapsed:.1f}s.")

    return PipelineResult(
        report_md=report_md,
        charts=charts,
        ohlcv_df=ohlcv_df,
        patterns_df=patterns_df,
        indicators_df=indicators_df,
        sr_levels_df=sr_levels_df,
        fib_levels_df=fib_levels_df,
        checklist=checklist,
        ticker=clean_ticker,
        elapsed_seconds=elapsed,
    )


# ============================================================================
# DataFrame formatters (for Gradio gr.Dataframe components)
# ============================================================================

def _build_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and format the raw OHLCV columns for display in Section 1."""
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    display = df[ohlcv_cols].copy()

    if hasattr(display.index, "strftime"):
        display.index = display.index.strftime("%Y-%m-%d")
    display = display.reset_index()
    display.rename(columns={display.columns[0]: "Date"}, inplace=True)

    # Round price columns to 2 dp; Volume stays as integer.
    for col in ["Open", "High", "Low", "Close"]:
        if col in display.columns:
            display[col] = display[col].round(2)
    if "Volume" in display.columns:
        display["Volume"] = display["Volume"].astype("Int64")

    return display


def _build_patterns_dataframe(patterns: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a display DataFrame of recent candlestick pattern detections.

    Returns a DataFrame with columns: ``Date``, ``Pattern``, ``Signal``.
    Only non-zero (detected) signals from the last 5 candles are included.
    """
    if not patterns:
        return pd.DataFrame(columns=["Date", "Pattern", "Signal"])

    rows: list[dict[str, Any]] = []
    for name, series in patterns.items():
        recent = series.tail(5)
        nonzero = recent[recent != 0]
        for date, signal in nonzero.items():
            rows.append({
                "Date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "Pattern": name,
                "Signal": "Bullish" if signal > 0 else "Bearish",
            })

    if not rows:
        return pd.DataFrame(columns=["Date", "Pattern", "Signal"])

    return pd.DataFrame(rows).sort_values("Date", ascending=False).reset_index(drop=True)


def _build_indicators_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the last N rows of indicator data for display in Tab 2."""
    display = df.tail(REPORT_TAIL_ROWS).copy()

    # Convert index to a column for Gradio display.
    if hasattr(display.index, "strftime"):
        display.index = display.index.strftime("%Y-%m-%d")
    display = display.reset_index()

    # Round numeric columns for readability.
    numeric_cols = display.select_dtypes(include="number").columns
    display[numeric_cols] = display[numeric_cols].round(2)

    return display


def _build_sr_dataframe(sr_levels: list[float]) -> pd.DataFrame:
    """Build a simple display DataFrame for S&R levels."""
    if not sr_levels:
        return pd.DataFrame(columns=["#", "Price Level"])

    return pd.DataFrame({
        "#": range(1, len(sr_levels) + 1),
        "Price Level": [f"{lv:,.2f}" for lv in sr_levels],
    })


def _build_fib_dataframe(fib_levels: dict[str, float]) -> pd.DataFrame:
    """Build a display DataFrame for Fibonacci retracement levels."""
    if not fib_levels:
        return pd.DataFrame(columns=["Level", "Price"])

    return pd.DataFrame({
        "Level": list(fib_levels.keys()),
        "Price": [f"{p:,.2f}" for p in fib_levels.values()],
    })
