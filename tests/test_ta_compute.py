"""
test_ta_compute.py — Unit tests for ``core.ta_compute``.

Testing strategy
----------------
* Tests target the **public functions** and the **Grand Checklist scorer**.
* A synthetic 104-row weekly OHLCV DataFrame is used as the canonical input.
* pandas-ta is a real dependency (fast enough for unit tests); TA-Lib is
  optionally tested if the C library is installed.
* Each checklist criterion is tested in isolation with hand-crafted
  DataFrame slices so pass/fail outcomes are deterministic.

Run::

    pytest tests/test_ta_compute.py -v
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    ADX_PERIOD,
    ATR_PERIOD,
    CHECKLIST_MIN_RRR,
    CHECKLIST_SR_PROXIMITY_PCT,
    EMA_PERIODS,
    FIBONACCI_LEVELS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    VOLUME_SMA_PERIOD,
)
from core.ta_compute import (
    _TALIB_AVAILABLE,
    compute_all_indicators,
    compute_fibonacci,
    compute_grand_checklist,
    compute_support_resistance,
    format_checklist_details,
    format_pattern_summary,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    """Build a deterministic 104-row weekly OHLCV DataFrame."""
    rng = np.random.default_rng(seed=42)
    n = 104
    dates = pd.date_range(
        end=datetime(2026, 3, 28), periods=n, freq="W-FRI", name="Date",
    )
    close = 100.0 + np.cumsum(rng.normal(0.2, 2, size=n))
    open_ = close + rng.normal(0, 0.5, size=n)
    high = np.maximum(open_, close) + rng.uniform(0.5, 3, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 3, size=n)
    volume = rng.integers(1_000_000, 10_000_000, size=n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture()
def enriched_result(ohlcv_df) -> dict[str, Any]:
    """Run ``compute_all_indicators`` once and cache the result."""
    return compute_all_indicators(ohlcv_df.copy())


# ============================================================================
# 1. compute_all_indicators — structure & column presence
# ============================================================================

class TestComputeAllIndicators:
    """Verify that the enriched dict contains the expected keys and columns."""

    def test_returns_dict_with_required_keys(self, enriched_result):
        for key in ("indicators", "patterns", "fibonacci", "sr_levels"):
            assert key in enriched_result, f"Missing key: {key}"

    def test_indicators_is_dataframe(self, enriched_result):
        assert isinstance(enriched_result["indicators"], pd.DataFrame)

    def test_ema_columns_present(self, enriched_result):
        df = enriched_result["indicators"]
        for period in EMA_PERIODS:
            assert f"EMA_{period}" in df.columns

    def test_rsi_column_present(self, enriched_result):
        df = enriched_result["indicators"]
        assert f"RSI_{RSI_PERIOD}" in df.columns

    def test_macd_columns_present(self, enriched_result):
        df = enriched_result["indicators"]
        prefix = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
        hist_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
        signal_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
        assert prefix in df.columns
        assert hist_col in df.columns
        assert signal_col in df.columns

    def test_bollinger_columns_present(self, enriched_result):
        df = enriched_result["indicators"]
        assert "BBL_20_2.0" in df.columns
        assert "BBM_20_2.0" in df.columns
        assert "BBU_20_2.0" in df.columns

    def test_atr_column_present(self, enriched_result):
        df = enriched_result["indicators"]
        assert f"ATRr_{ATR_PERIOD}" in df.columns

    def test_adx_column_present(self, enriched_result):
        df = enriched_result["indicators"]
        assert f"ADX_{ADX_PERIOD}" in df.columns

    def test_volume_sma_column_present(self, enriched_result):
        df = enriched_result["indicators"]
        assert f"VOL_SMA_{VOLUME_SMA_PERIOD}" in df.columns

    def test_patterns_is_dict(self, enriched_result):
        assert isinstance(enriched_result["patterns"], dict)

    @pytest.mark.skipif(not _TALIB_AVAILABLE, reason="TA-Lib not installed")
    def test_patterns_contain_expected_names_when_talib_available(self, enriched_result):
        patterns = enriched_result["patterns"]
        assert len(patterns) >= 10  # At least most of the 14 patterns
        for name, series in patterns.items():
            assert isinstance(series, pd.Series)

    def test_patterns_empty_when_talib_unavailable(self, ohlcv_df):
        """If TA-Lib is not installed, patterns should be an empty dict."""
        if _TALIB_AVAILABLE:
            pytest.skip("TA-Lib IS installed — cannot test fallback path.")
        result = compute_all_indicators(ohlcv_df.copy())
        assert result["patterns"] == {}


# ============================================================================
# 2. compute_fibonacci
# ============================================================================

class TestComputeFibonacci:
    """Verify Fibonacci retracement level calculations."""

    def test_returns_all_standard_levels(self, ohlcv_df):
        levels = compute_fibonacci(ohlcv_df)
        assert len(levels) == len(FIBONACCI_LEVELS)

    def test_level_labels_match_expected_format(self, ohlcv_df):
        levels = compute_fibonacci(ohlcv_df)
        expected_labels = {f"{r * 100:.1f}%" for r in FIBONACCI_LEVELS}
        assert set(levels.keys()) == expected_labels

    def test_levels_are_between_low_and_high(self, ohlcv_df):
        swing_high = float(ohlcv_df["High"].max())
        swing_low = float(ohlcv_df["Low"].min())
        levels = compute_fibonacci(ohlcv_df)
        for price in levels.values():
            assert swing_low <= price <= swing_high

    def test_levels_descend_as_ratio_increases(self, ohlcv_df):
        levels = compute_fibonacci(ohlcv_df)
        prices = list(levels.values())
        # Higher ratio → lower price (retracement from high).
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1]

    def test_zero_range_returns_empty(self):
        """If high == low the range is zero → empty dict."""
        flat_df = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [100.0] * 10,
            "Low": [100.0] * 10,
            "Close": [100.0] * 10,
            "Volume": [1_000_000.0] * 10,
        })
        assert compute_fibonacci(flat_df) == {}

    def test_specific_calculation(self):
        """Verify exact values for a known high/low pair."""
        df = pd.DataFrame({
            "High": [200.0, 150.0],
            "Low": [100.0, 120.0],
            "Open": [100.0, 120.0],
            "Close": [150.0, 140.0],
            "Volume": [1e6, 1e6],
        })
        levels = compute_fibonacci(df)
        # swing_high = 200, swing_low = 100, range = 100
        assert levels["50.0%"] == 150.0  # 200 - (100 * 0.5)
        assert levels["23.6%"] == 176.4  # 200 - (100 * 0.236)


# ============================================================================
# 3. compute_support_resistance
# ============================================================================

class TestComputeSupportResistance:
    """Verify S&R pivot-clustering logic."""

    def test_returns_list_of_floats(self, ohlcv_df):
        sr = compute_support_resistance(ohlcv_df)
        assert isinstance(sr, list)
        for level in sr:
            assert isinstance(level, float)

    def test_levels_are_sorted_ascending(self, ohlcv_df):
        sr = compute_support_resistance(ohlcv_df)
        assert sr == sorted(sr)

    def test_short_dataframe_returns_empty(self):
        """DataFrame shorter than 2*window should yield no pivots."""
        short = pd.DataFrame({
            "High": [100.0] * 5,
            "Low": [90.0] * 5,
        })
        assert compute_support_resistance(short, window=5) == []

    def test_flat_price_clusters_into_single_level(self):
        """A perfectly flat series should produce at most one S&R level."""
        n = 50
        df = pd.DataFrame({
            "High": [100.0] * n,
            "Low": [100.0] * n,
        })
        sr = compute_support_resistance(df, window=3, min_touches=2)
        assert len(sr) <= 1

    def test_min_touches_filter_works(self, ohlcv_df):
        """Raising min_touches should reduce or maintain the number of levels."""
        sr_loose = compute_support_resistance(ohlcv_df, min_touches=2)
        sr_strict = compute_support_resistance(ohlcv_df, min_touches=5)
        assert len(sr_strict) <= len(sr_loose)


# ============================================================================
# 4. Grand TA Checklist
# ============================================================================

class TestComputeGrandChecklist:
    """Verify the 6-point checklist scorer."""

    def test_returns_expected_structure(self, enriched_result):
        df = enriched_result["indicators"]
        patterns = enriched_result["patterns"]
        sr = enriched_result["sr_levels"]

        checklist = compute_grand_checklist(df, patterns, sr)

        assert "score" in checklist
        assert "max_score" in checklist
        assert "details" in checklist
        assert checklist["max_score"] == 6
        assert 0 <= checklist["score"] <= 6

    def test_details_has_six_criteria(self, enriched_result):
        df = enriched_result["indicators"]
        checklist = compute_grand_checklist(
            df, enriched_result["patterns"], enriched_result["sr_levels"],
        )
        assert len(checklist["details"]) == 6

    def test_each_detail_has_required_keys(self, enriched_result):
        df = enriched_result["indicators"]
        checklist = compute_grand_checklist(
            df, enriched_result["patterns"], enriched_result["sr_levels"],
        )
        for item in checklist["details"]:
            assert "criterion" in item
            assert "passed" in item
            assert "note" in item
            assert isinstance(item["passed"], bool)

    def test_score_equals_sum_of_passed(self, enriched_result):
        df = enriched_result["indicators"]
        checklist = compute_grand_checklist(
            df, enriched_result["patterns"], enriched_result["sr_levels"],
        )
        expected_score = sum(1 for d in checklist["details"] if d["passed"])
        assert checklist["score"] == expected_score

    def test_no_patterns_criterion_1_fails(self, enriched_result):
        """Without patterns, criterion #1 should always fail."""
        df = enriched_result["indicators"]
        checklist = compute_grand_checklist(df, {}, enriched_result["sr_levels"])
        assert checklist["details"][0]["passed"] is False

    def test_no_sr_levels_criterion_2_fails(self, enriched_result):
        """Without S&R levels, criterion #2 should always fail."""
        df = enriched_result["indicators"]
        checklist = compute_grand_checklist(
            df, enriched_result["patterns"], [],
        )
        assert checklist["details"][1]["passed"] is False

    def test_criterion_3_volume_check_with_high_volume(self):
        """Force volume above SMA → criterion #3 should pass."""
        n = 20
        dates = pd.date_range("2025-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            "Close": [100.0] * n,
            "Volume": [1_000_000.0] * (n - 1) + [5_000_000.0],  # last is 5x avg
            f"VOL_SMA_{VOLUME_SMA_PERIOD}": [1_000_000.0] * n,
        }, index=dates)

        from core.ta_compute import _check_volume
        passed, note = _check_volume(df)
        assert passed is True

    def test_criterion_3_volume_check_with_low_volume(self):
        """Force volume below SMA → criterion #3 should fail."""
        n = 20
        dates = pd.date_range("2025-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            "Close": [100.0] * n,
            "Volume": [1_000_000.0] * (n - 1) + [500_000.0],  # last is 0.5x avg
            f"VOL_SMA_{VOLUME_SMA_PERIOD}": [1_000_000.0] * n,
        }, index=dates)

        from core.ta_compute import _check_volume
        passed, note = _check_volume(df)
        assert passed is False

    def test_criterion_4_adx_trending(self):
        """ADX above threshold → criterion #4 should pass."""
        n = 5
        dates = pd.date_range("2025-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            f"ADX_{ADX_PERIOD}": [30.0] * n,
        }, index=dates)

        from core.ta_compute import _check_adx_trend
        passed, note = _check_adx_trend(df)
        assert passed is True
        assert "trending" in note.lower()

    def test_criterion_4_adx_ranging(self):
        """ADX below threshold → criterion #4 should fail."""
        n = 5
        dates = pd.date_range("2025-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            f"ADX_{ADX_PERIOD}": [15.0] * n,
        }, index=dates)

        from core.ta_compute import _check_adx_trend
        passed, note = _check_adx_trend(df)
        assert passed is False

    def test_criterion_5_momentum_aligned_bullish(self):
        """RSI > 50 AND MACD hist > 0 → pass."""
        n = 5
        dates = pd.date_range("2025-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            f"RSI_{RSI_PERIOD}": [60.0] * n,
            f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}": [0.5] * n,
        }, index=dates)

        from core.ta_compute import _check_momentum_alignment
        passed, note = _check_momentum_alignment(df)
        assert passed is True
        assert "bullish" in note.lower()

    def test_criterion_5_momentum_divergent(self):
        """RSI > 50 but MACD hist < 0 → fail (divergence)."""
        n = 5
        dates = pd.date_range("2025-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            f"RSI_{RSI_PERIOD}": [60.0] * n,
            f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}": [-0.5] * n,
        }, index=dates)

        from core.ta_compute import _check_momentum_alignment
        passed, note = _check_momentum_alignment(df)
        assert passed is False
        assert "divergence" in note.lower()

    def test_criterion_6_rrr_with_valid_atr(self):
        """Valid ATR → criterion #6 should pass (by construction at 1.5x)."""
        n = 5
        dates = pd.date_range("2025-01-01", periods=n, freq="W")
        df = pd.DataFrame({
            "Close": [100.0] * n,
            f"ATRr_{ATR_PERIOD}": [5.0] * n,
        }, index=dates)

        from core.ta_compute import _check_rrr
        passed, note = _check_rrr(df)
        assert passed is True
        assert "RRR" in note


# ============================================================================
# 5. Formatting helpers
# ============================================================================

class TestFormatHelpers:
    """Verify the text-formatting functions used by llm_inference."""

    def test_format_pattern_summary_empty_patterns(self):
        result = format_pattern_summary({})
        assert "unavailable" in result.lower()

    def test_format_pattern_summary_no_detections(self):
        dates = pd.date_range("2025-01-01", periods=5, freq="W")
        patterns = {"Hammer": pd.Series([0, 0, 0, 0, 0], index=dates)}
        result = format_pattern_summary(patterns)
        assert "no candlestick patterns" in result.lower()

    def test_format_pattern_summary_with_detection(self):
        dates = pd.date_range("2025-01-01", periods=5, freq="W")
        patterns = {"Hammer": pd.Series([0, 0, 0, 100, 0], index=dates)}
        result = format_pattern_summary(patterns)
        assert "Hammer" in result
        assert "Bullish" in result

    def test_format_checklist_details_all_pass(self):
        details = [
            {"criterion": "Test 1", "passed": True, "note": "OK"},
            {"criterion": "Test 2", "passed": True, "note": "OK"},
        ]
        result = format_checklist_details(details)
        assert result.count("PASS") == 2
        assert result.count("FAIL") == 0

    def test_format_checklist_details_mixed(self):
        details = [
            {"criterion": "Test 1", "passed": True, "note": "OK"},
            {"criterion": "Test 2", "passed": False, "note": "Bad"},
        ]
        result = format_checklist_details(details)
        assert "PASS" in result
        assert "FAIL" in result
