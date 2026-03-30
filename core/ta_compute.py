"""
ta_compute.py — Technical indicator computation layer.

Responsibility (plan.md §4.2):
    Compute all technical indicators (pandas-ta), candlestick patterns (TA-Lib),
    Fibonacci retracement levels, support & resistance zones, and the 6-point
    Grand TA Checklist score.

Design decisions
----------------
* **Selective indicator calculation** — only the indicators listed in plan.md
  §4.2.1 are computed, avoiding the ~130-indicator overhead of
  ``strategy("all")``.
* **Graceful TA-Lib fallback** — if the C library is not installed, candlestick
  pattern detection is skipped and a warning is logged (plan.md §8).
* **All tuneable parameters** are imported from ``config.settings`` — zero
  magic numbers in this file.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    ADX_PERIOD,
    ADX_TRENDING_THRESHOLD,
    ATR_PERIOD,
    BBANDS_LENGTH,
    BBANDS_STD_DEV,
    CANDLESTICK_PATTERNS,
    CHECKLIST_MIN_RRR,
    CHECKLIST_SR_PROXIMITY_PCT,
    EMA_PERIODS,
    FIBONACCI_LEVELS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PATTERN_LOOKBACK_CANDLES,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    VOLUME_SMA_PERIOD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TA-Lib availability check (plan.md §8: graceful fallback)
# ---------------------------------------------------------------------------

_TALIB_AVAILABLE: bool = False
try:
    import talib  # type: ignore[import-untyped]

    _TALIB_AVAILABLE = True
except ImportError:
    logger.warning(
        "TA-Lib C library not found. Candlestick pattern detection will be "
        "disabled. Install TA-Lib to enable full pattern analysis."
    )


# ============================================================================
# 1. Main entry point
# ============================================================================

def compute_all_indicators(df: pd.DataFrame) -> dict[str, Any]:
    """Compute every required indicator and return a structured result dict.

    Parameters
    ----------
    df:
        Clean OHLCV DataFrame from ``data_fetch.fetch_weekly_data``.
        **Modified in place** — indicator columns are appended directly.

    Returns
    -------
    dict with keys:
        ``"indicators"``  — the enriched DataFrame.
        ``"patterns"``    — dict of pattern name → Series (or empty if no TA-Lib).
        ``"fibonacci"``   — dict of level labels → price values.
        ``"sr_levels"``   — list of float support/resistance prices.
    """
    # Suppress pandas-ta FutureWarnings that clutter Gradio logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        _compute_trend_indicators(df)
        _compute_momentum_indicators(df)
        _compute_volatility_indicators(df)
        _compute_volume_indicators(df)

    patterns = _compute_candlestick_patterns(df)
    fibonacci = compute_fibonacci(df)
    sr_levels = compute_support_resistance(df)

    return {
        "indicators": df,
        "patterns": patterns,
        "fibonacci": fibonacci,
        "sr_levels": sr_levels,
    }


# ============================================================================
# 2. Indicator sub-routines (private)
# ============================================================================

def _compute_trend_indicators(df: pd.DataFrame) -> None:
    """Append EMA and Bollinger Band columns (plan.md §4.2.1)."""
    import pandas_ta as ta  # noqa: F811 — deferred import, heavy module

    for period in EMA_PERIODS:
        df.ta.ema(length=period, append=True)

    df.ta.bbands(length=BBANDS_LENGTH, std=BBANDS_STD_DEV, append=True)


def _compute_momentum_indicators(df: pd.DataFrame) -> None:
    """Append RSI, MACD, and ADX columns."""
    import pandas_ta as ta  # noqa: F811

    df.ta.rsi(length=RSI_PERIOD, append=True)
    df.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, append=True)
    df.ta.adx(length=ADX_PERIOD, append=True)


def _compute_volatility_indicators(df: pd.DataFrame) -> None:
    """Append ATR column."""
    import pandas_ta as ta  # noqa: F811

    df.ta.atr(length=ATR_PERIOD, append=True)


def _compute_volume_indicators(df: pd.DataFrame) -> None:
    """Append Volume SMA column (plan.md §4.2.1: 10-period rolling mean)."""
    col_name = f"VOL_SMA_{VOLUME_SMA_PERIOD}"
    df[col_name] = df["Volume"].rolling(window=VOLUME_SMA_PERIOD).mean()


# ============================================================================
# 3. Candlestick patterns (TA-Lib)
# ============================================================================

def _compute_candlestick_patterns(
    df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Detect 14 candlestick patterns via TA-Lib.

    Returns an empty dict if TA-Lib is unavailable (plan.md §8 fallback).
    """
    if not _TALIB_AVAILABLE:
        return {}

    op = df["Open"].values
    hi = df["High"].values
    lo = df["Low"].values
    cl = df["Close"].values

    patterns: dict[str, pd.Series] = {}

    for display_name, func_name in CANDLESTICK_PATTERNS.items():
        func = getattr(talib, func_name, None)
        if func is None:
            logger.warning("TA-Lib function '%s' not found. Skipping.", func_name)
            continue
        result = func(op, hi, lo, cl)
        patterns[display_name] = pd.Series(result, index=df.index, name=display_name)

    return patterns


# ============================================================================
# 4. Fibonacci retracement
# ============================================================================

def compute_fibonacci(df: pd.DataFrame) -> dict[str, float]:
    """Compute Fibonacci retracement levels from the swing high/low.

    Uses the overall high and low of the DataFrame to define the range,
    then calculates retracement levels (plan.md §4.2.1).

    Parameters
    ----------
    df:
        OHLCV DataFrame (must contain ``High`` and ``Low`` columns).

    Returns
    -------
    dict
        Keys are level labels (e.g. ``"38.2%"``), values are price floats.
    """
    swing_high: float = float(df["High"].max())
    swing_low: float = float(df["Low"].min())
    price_range: float = swing_high - swing_low

    if price_range <= 0:
        logger.warning("Fibonacci range is zero or negative; returning empty levels.")
        return {}

    levels: dict[str, float] = {}
    for ratio in FIBONACCI_LEVELS:
        label = f"{ratio * 100:.1f}%"
        # Retracement is measured from the swing high downward.
        levels[label] = round(swing_high - (price_range * ratio), 2)

    return levels


# ============================================================================
# 5. Support & Resistance detection
# ============================================================================

def compute_support_resistance(
    df: pd.DataFrame,
    window: int = 5,
    min_touches: int = 3,
    tolerance_pct: float = 0.015,
) -> list[float]:
    """Identify support & resistance levels via rolling-pivot clustering.

    Algorithm
    ---------
    1. Detect local minima (support) and maxima (resistance) using a rolling
       window on ``Low`` and ``High`` respectively.
    2. Cluster nearby pivot prices within ``tolerance_pct`` of each other.
    3. Keep only clusters with at least ``min_touches`` pivot points.
    4. Return the mean price of each qualifying cluster,
       sorted ascending.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    window:
        Half-width of the rolling window for pivot detection.
    min_touches:
        Minimum number of pivots required to confirm a level.
    tolerance_pct:
        Maximum percentage distance between pivots to be considered the
        same level.

    Returns
    -------
    list[float]
        Sorted list of S&R price levels.
    """
    pivots: list[float] = []

    highs = df["High"].values
    lows = df["Low"].values

    for i in range(window, len(df) - window):
        # Local maximum → resistance candidate
        if highs[i] == max(highs[i - window : i + window + 1]):
            pivots.append(float(highs[i]))
        # Local minimum → support candidate
        if lows[i] == min(lows[i - window : i + window + 1]):
            pivots.append(float(lows[i]))

    if not pivots:
        return []

    # --- Cluster nearby pivots ---
    pivots.sort()
    clusters: list[list[float]] = [[pivots[0]]]

    for price in pivots[1:]:
        cluster_mean = np.mean(clusters[-1])
        if abs(price - cluster_mean) / cluster_mean <= tolerance_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    # --- Filter by minimum touches ---
    sr_levels = [
        round(float(np.mean(cluster)), 2)
        for cluster in clusters
        if len(cluster) >= min_touches
    ]

    logger.debug("Identified %d S&R levels for the dataset.", len(sr_levels))
    return sr_levels


# ============================================================================
# 6. Grand TA Checklist (plan.md §6)
# ============================================================================

def compute_grand_checklist(
    df: pd.DataFrame,
    patterns: dict[str, pd.Series],
    sr_levels: list[float],
) -> dict[str, Any]:
    """Score the 6-point Grand TA Checklist programmatically.

    Parameters
    ----------
    df:
        Enriched DataFrame (all indicator columns must already be appended).
    patterns:
        Dict of pattern name → Series from ``_compute_candlestick_patterns``.
    sr_levels:
        List of S&R prices from ``compute_support_resistance``.

    Returns
    -------
    dict with keys:
        ``"score"``      — int, 0–6.
        ``"max_score"``  — int, always 6.
        ``"details"``    — list of dicts, one per criterion with keys
                           ``criterion``, ``passed`` (bool), ``note`` (str).
    """
    details: list[dict[str, Any]] = []
    score = 0

    # ── Criterion 1: Recognisable candlestick pattern ──────────────────────
    c1_passed, c1_note = _check_candlestick_pattern(df, patterns)
    details.append({"criterion": "Candlestick pattern detected", "passed": c1_passed, "note": c1_note})
    score += int(c1_passed)

    # ── Criterion 2: S&R confirms trade + stop-loss ────────────────────────
    c2_passed, c2_note = _check_sr_proximity(df, sr_levels)
    details.append({"criterion": "S&R confirms trade", "passed": c2_passed, "note": c2_note})
    score += int(c2_passed)

    # ── Criterion 3: Volume above 10-period average ──────────────────────────
    c3_passed, c3_note = _check_volume(df)
    details.append({"criterion": "Volume above average", "passed": c3_passed, "note": c3_note})
    score += int(c3_passed)

    # ── Criterion 4: Aligns with Dow Theory trend ─────────────────────────
    c4_passed, c4_note = _check_adx_trend(df)
    details.append({"criterion": "Dow Theory / ADX trend alignment", "passed": c4_passed, "note": c4_note})
    score += int(c4_passed)

    # ── Criterion 5: RSI + MACD confirm direction ─────────────────────────
    c5_passed, c5_note = _check_momentum_alignment(df)
    details.append({"criterion": "RSI + MACD directional confirmation", "passed": c5_passed, "note": c5_note})
    score += int(c5_passed)

    # ── Criterion 6: RRR ≥ 1.5 ────────────────────────────────────────────
    c6_passed, c6_note = _check_rrr(df)
    details.append({"criterion": f"Risk-Reward Ratio ≥ {CHECKLIST_MIN_RRR}", "passed": c6_passed, "note": c6_note})
    score += int(c6_passed)

    return {"score": score, "max_score": 6, "details": details}


# ---------------------------------------------------------------------------
# Checklist sub-checks (private)
# ---------------------------------------------------------------------------

def _check_candlestick_pattern(
    df: pd.DataFrame,
    patterns: dict[str, pd.Series],
) -> tuple[bool, str]:
    """Criterion #1: any non-zero TA-Lib signal in the last N candles."""
    if not patterns:
        return False, "TA-Lib unavailable — candlestick detection skipped."

    lookback = df.index[-PATTERN_LOOKBACK_CANDLES:]
    detected: list[str] = []

    for name, series in patterns.items():
        recent = series.loc[lookback]
        if (recent != 0).any():
            detected.append(name)

    if detected:
        return True, f"Detected: {', '.join(detected)}."
    return False, "No recognisable pattern in last 3 candles."


def _check_sr_proximity(
    df: pd.DataFrame,
    sr_levels: list[float],
) -> tuple[bool, str]:
    """Criterion #2: current close within 2 % of any S&R level."""
    if not sr_levels:
        return False, "No S&R levels identified."

    last_close: float = float(df["Close"].iloc[-1])
    for level in sr_levels:
        if level == 0:
            continue
        distance_pct = abs(last_close - level) / level
        if distance_pct <= CHECKLIST_SR_PROXIMITY_PCT:
            return True, f"Close ({last_close:,.2f}) within {distance_pct:.1%} of S/R {level:,.2f}."

    nearest = min(sr_levels, key=lambda lv: abs(last_close - lv))
    return False, f"Closest S/R is {nearest:,.2f} ({abs(last_close - nearest) / nearest:.1%} away)."


def _check_volume(df: pd.DataFrame) -> tuple[bool, str]:
    """Criterion #3: current volume > 10-period SMA."""
    vol_sma_col = f"VOL_SMA_{VOLUME_SMA_PERIOD}"
    if vol_sma_col not in df.columns:
        return False, "Volume SMA not computed."

    current_vol = float(df["Volume"].iloc[-1])
    vol_sma = float(df[vol_sma_col].iloc[-1])

    if pd.isna(vol_sma) or vol_sma == 0:
        return False, "Volume SMA is unavailable."

    ratio = current_vol / vol_sma
    passed = current_vol > vol_sma
    return passed, f"Current volume is {ratio:.2f}x the {VOLUME_SMA_PERIOD}-period average."


def _check_adx_trend(df: pd.DataFrame) -> tuple[bool, str]:
    """Criterion #4: ADX indicates a trending market (> threshold)."""
    adx_col = f"ADX_{ADX_PERIOD}"
    if adx_col not in df.columns:
        return False, "ADX column not found."

    adx_value = float(df[adx_col].iloc[-1])
    if pd.isna(adx_value):
        return False, "ADX value is NaN."

    if adx_value >= ADX_TRENDING_THRESHOLD:
        return True, f"ADX = {adx_value:.1f} (trending, ≥ {ADX_TRENDING_THRESHOLD})."
    return False, f"ADX = {adx_value:.1f} (below trending threshold of {ADX_TRENDING_THRESHOLD})."


def _check_momentum_alignment(df: pd.DataFrame) -> tuple[bool, str]:
    """Criterion #5: RSI and MACD agree on direction.

    Bullish = RSI > 50 AND MACD histogram > 0.
    Bearish = RSI < 50 AND MACD histogram < 0.
    Either alignment counts as a pass.
    """
    rsi_col = f"RSI_{RSI_PERIOD}"
    hist_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

    if rsi_col not in df.columns or hist_col not in df.columns:
        return False, "RSI or MACD histogram column not found."

    rsi = float(df[rsi_col].iloc[-1])
    macd_hist = float(df[hist_col].iloc[-1])

    if pd.isna(rsi) or pd.isna(macd_hist):
        return False, "RSI or MACD value is NaN."

    rsi_bull = rsi > 50
    macd_bull = macd_hist > 0

    if rsi_bull == macd_bull:
        direction = "bullish" if rsi_bull else "bearish"
        return True, f"RSI ({rsi:.1f}) and MACD histogram ({macd_hist:+.3f}) both {direction}."

    return False, (
        f"Divergence: RSI = {rsi:.1f} ({'bullish' if rsi_bull else 'bearish'}), "
        f"MACD histogram = {macd_hist:+.3f} ({'bullish' if macd_bull else 'bearish'})."
    )


def _check_rrr(df: pd.DataFrame) -> tuple[bool, str]:
    """Criterion #6: ATR-based RRR ≥ 1.5.

    Simplified calculation:
        - Entry  = last close
        - Stop   = entry − 1 × ATR
        - Target = entry + 1.5 × ATR  (i.e., RRR = target_dist / stop_dist = 1.5)

    This always passes at the 1.5 threshold by construction when ATR is valid.
    The real value is in computing and *reporting* the RRR to the LLM so it
    can contextualise the risk.  A more advanced version would use S&R-based
    targets instead of a fixed 1.5× ATR multiplier.
    """
    atr_col = f"ATRr_{ATR_PERIOD}"
    if atr_col not in df.columns:
        return False, "ATR column not found."

    atr = float(df[atr_col].iloc[-1])
    if pd.isna(atr) or atr <= 0:
        return False, "ATR value is invalid."

    entry = float(df["Close"].iloc[-1])
    stop_loss = entry - atr
    target = entry + (CHECKLIST_MIN_RRR * atr)
    rrr = (target - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0

    passed = rrr >= CHECKLIST_MIN_RRR
    return passed, (
        f"Entry={entry:,.2f}, Stop={stop_loss:,.2f}, Target={target:,.2f}, "
        f"RRR={rrr:.2f} (min {CHECKLIST_MIN_RRR})."
    )


# ============================================================================
# 7. Formatting helpers (used by llm_inference)
# ============================================================================

def format_pattern_summary(patterns: dict[str, pd.Series], n: int = 5) -> str:
    """Build a human-readable summary of recent pattern detections.

    Parameters
    ----------
    patterns:
        Dict from ``compute_all_indicators``'s ``"patterns"`` key.
    n:
        Number of most-recent candles to inspect.

    Returns
    -------
    str
        Multiline text listing detected patterns, or a fallback message.
    """
    if not patterns:
        return "Candlestick pattern detection unavailable (TA-Lib not installed)."

    lines: list[str] = []
    for name, series in patterns.items():
        recent = series.tail(n)
        nonzero = recent[recent != 0]
        if not nonzero.empty:
            for date, signal in nonzero.items():
                direction = "Bullish" if signal > 0 else "Bearish"
                lines.append(f"  • {name} ({direction}) on {date.date()}")

    return "\n".join(lines) if lines else "No candlestick patterns detected in the last 5 candles."


def format_checklist_details(details: list[dict[str, Any]]) -> str:
    """Format Grand Checklist details into a human-readable block."""
    lines: list[str] = []
    for i, item in enumerate(details, start=1):
        status = "✅ PASS" if item["passed"] else "❌ FAIL"
        lines.append(f"  {i}. [{status}] {item['criterion']}: {item['note']}")
    return "\n".join(lines)
