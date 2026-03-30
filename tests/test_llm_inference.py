"""
test_llm_inference.py — Unit tests for ``core.llm_inference``.

Testing strategy
----------------
* The Gemini API is **fully mocked** — no real API calls, no key required.
* Tests verify:
    - Prompt assembly logic (correct template selection, placeholder injection).
    - Retry behaviour with exponential back-off (mocked ``time.sleep``).
    - Error propagation after exhausting retries.
    - Per-call ``genai.Client`` instantiation (no global state pollution).

Run::

    pytest tests/test_llm_inference.py -v
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL_NAME,
    GEMINI_RETRY_BACKOFF_BASE,
    REPORT_TAIL_ROWS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def minimal_computed_data() -> dict[str, Any]:
    """Build a minimal ``computed_data`` dict matching compute_all_indicators output.

    Contains just enough data for prompt assembly to succeed without
    requiring pandas-ta or TA-Lib at test time.
    """
    n = REPORT_TAIL_ROWS + 5  # a few extra rows beyond the tail limit
    dates = pd.date_range(
        end=datetime(2026, 3, 28), periods=n, freq="W-FRI", name="Date",
    )
    rng = np.random.default_rng(seed=99)

    df = pd.DataFrame(
        {
            "Open": 100 + rng.normal(0, 1, n),
            "High": 102 + rng.normal(0, 1, n),
            "Low": 98 + rng.normal(0, 1, n),
            "Close": 100 + rng.normal(0, 1, n),
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "EMA_20": 100 + rng.normal(0, 0.5, n),
            "EMA_50": 100 + rng.normal(0, 0.3, n),
            "EMA_100": 100 + rng.normal(0, 0.2, n),
            "RSI_14": 50 + rng.normal(0, 10, n),
            "MACD_12_26_9": rng.normal(0, 0.5, n),
            "MACDs_12_26_9": rng.normal(0, 0.3, n),
            "MACDh_12_26_9": rng.normal(0, 0.2, n),
            "BBL_20_2.0": 98 + rng.normal(0, 0.5, n),
            "BBM_20_2.0": 100 + rng.normal(0, 0.5, n),
            "BBU_20_2.0": 102 + rng.normal(0, 0.5, n),
            "ATRr_14": 2 + rng.uniform(0, 1, n),
            "ADX_14": 20 + rng.normal(0, 5, n),
            "VOL_SMA_10": 2_000_000 + rng.normal(0, 100_000, n),
        },
        index=dates,
    )

    # Fake pattern data (empty = no TA-Lib scenario)
    patterns: dict[str, pd.Series] = {}

    fibonacci = {"23.6%": 197.64, "38.2%": 196.18, "50.0%": 195.0, "61.8%": 193.82, "78.6%": 192.14}
    sr_levels = [95.0, 100.0, 105.0]

    return {
        "indicators": df,
        "patterns": patterns,
        "fibonacci": fibonacci,
        "sr_levels": sr_levels,
    }


@pytest.fixture()
def computed_data_with_patterns(minimal_computed_data) -> dict[str, Any]:
    """Variant with non-empty patterns to test the full-prompt template."""
    dates = minimal_computed_data["indicators"].index
    fake_pattern = pd.Series(
        [0] * (len(dates) - 1) + [100],  # signal on last candle
        index=dates,
        name="Hammer",
    )
    minimal_computed_data["patterns"] = {"Hammer": fake_pattern}
    return minimal_computed_data


def _mock_response(text: str = "## 1. Mocked report\n**Verdict: Neutral**"):
    """Build a mock Gemini response object."""
    resp = MagicMock()
    resp.text = text
    return resp


# ============================================================================
# 1. Prompt assembly
# ============================================================================

class TestBuildPrompt:
    """Verify prompt assembly selects the right template and injects data."""

    def test_no_patterns_uses_fallback_template(self, minimal_computed_data):
        from core.llm_inference import _build_prompt

        prompt = _build_prompt(minimal_computed_data, "AAPL")
        assert "No Candlestick Pattern Data" in prompt
        assert "AAPL" in prompt

    def test_with_patterns_uses_full_template(self, computed_data_with_patterns):
        from core.llm_inference import _build_prompt

        prompt = _build_prompt(computed_data_with_patterns, "AAPL")
        assert "CANDLESTICK PATTERN DETECTIONS" in prompt
        assert "AAPL" in prompt

    def test_prompt_contains_raw_data_json(self, minimal_computed_data):
        from core.llm_inference import _build_prompt

        prompt = _build_prompt(minimal_computed_data, "AAPL")
        # JSON records contain the column names
        assert "Open" in prompt
        assert "Close" in prompt

    def test_prompt_contains_checklist_score(self, minimal_computed_data):
        from core.llm_inference import _build_prompt

        prompt = _build_prompt(minimal_computed_data, "AAPL")
        assert "/ 6" in prompt  # "Score: N / 6"

    def test_prompt_contains_sr_levels(self, minimal_computed_data):
        from core.llm_inference import _build_prompt

        prompt = _build_prompt(minimal_computed_data, "AAPL")
        assert "100.00" in prompt  # one of our fake S&R levels

    def test_prompt_contains_fibonacci_levels(self, minimal_computed_data):
        from core.llm_inference import _build_prompt

        prompt = _build_prompt(minimal_computed_data, "AAPL")
        assert "50.0%" in prompt

    def test_prompt_tails_to_report_tail_rows(self, minimal_computed_data):
        from core.llm_inference import _build_prompt

        prompt = _build_prompt(minimal_computed_data, "AAPL")
        # The JSON should contain exactly REPORT_TAIL_ROWS records.
        # Count occurrences of "Open" in the JSON (one per record).
        import json
        # Extract the JSON block between the delimiters
        # Just verify it's parseable as a list with the right length
        assert prompt.count('"Open"') == REPORT_TAIL_ROWS


# ============================================================================
# 2. Gemini API call — happy path
# ============================================================================

class TestCallGeminiHappyPath:
    """Verify successful single-call flow with mocked API."""

    @patch("core.llm_inference.genai.Client")
    def test_returns_response_text(self, MockClient, minimal_computed_data):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response("## Report")
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        result = generate_full_report(minimal_computed_data, "AIzaFakeKey123456789012345678", "AAPL")
        assert result == "## Report"

    @patch("core.llm_inference.genai.Client")
    def test_client_instantiated_with_api_key(self, MockClient, minimal_computed_data):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response()
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        api_key = "AIzaFakeKey123456789012345678"
        generate_full_report(minimal_computed_data, api_key, "AAPL")
        MockClient.assert_called_once_with(api_key=api_key)

    @patch("core.llm_inference.genai.Client")
    def test_correct_model_name_used(self, MockClient, minimal_computed_data):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response()
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        generate_full_report(minimal_computed_data, "AIzaFakeKey123456789012345678", "AAPL")
        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs[1]["model"] == GEMINI_MODEL_NAME


# ============================================================================
# 3. Retry behaviour
# ============================================================================

class TestRetryBehaviour:
    """Verify exponential back-off and retry logic (plan.md §8)."""

    @patch("core.llm_inference.time.sleep", return_value=None)
    @patch("core.llm_inference.genai.Client")
    def test_retries_on_transient_error(self, MockClient, mock_sleep, minimal_computed_data):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            RuntimeError("Rate limited"),
            _mock_response("## OK"),
        ]
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        result = generate_full_report(minimal_computed_data, "AIzaFakeKey123456789012345678", "AAPL")
        assert result == "## OK"
        assert mock_client.models.generate_content.call_count == 2
        # Should have slept once (backoff_base^1 = 2.0s)
        mock_sleep.assert_called_once_with(GEMINI_RETRY_BACKOFF_BASE ** 1)

    @patch("core.llm_inference.time.sleep", return_value=None)
    @patch("core.llm_inference.genai.Client")
    def test_raises_runtime_error_after_max_retries(
        self, MockClient, mock_sleep, minimal_computed_data
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Always fails")
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        with pytest.raises(RuntimeError, match="failed after"):
            generate_full_report(minimal_computed_data, "AIzaFakeKey123456789012345678", "AAPL")

        assert mock_client.models.generate_content.call_count == GEMINI_MAX_RETRIES

    @patch("core.llm_inference.time.sleep", return_value=None)
    @patch("core.llm_inference.genai.Client")
    def test_backoff_delays_increase_exponentially(
        self, MockClient, mock_sleep, minimal_computed_data
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Fail")
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        with pytest.raises(RuntimeError):
            generate_full_report(minimal_computed_data, "AIzaFakeKey123456789012345678", "AAPL")

        # Retries = GEMINI_MAX_RETRIES (3), so sleep called 2 times (not on last attempt)
        expected_sleeps = [
            call(GEMINI_RETRY_BACKOFF_BASE ** i) for i in range(1, GEMINI_MAX_RETRIES)
        ]
        mock_sleep.assert_has_calls(expected_sleeps)

    @patch("core.llm_inference.time.sleep", return_value=None)
    @patch("core.llm_inference.genai.Client")
    def test_empty_response_triggers_retry(self, MockClient, mock_sleep, minimal_computed_data):
        """An empty string response should be treated as an error and retried."""
        mock_client = MagicMock()
        empty_resp = MagicMock()
        empty_resp.text = ""
        ok_resp = _mock_response("## Good report")
        mock_client.models.generate_content.side_effect = [empty_resp, ok_resp]
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        result = generate_full_report(minimal_computed_data, "AIzaFakeKey123456789012345678", "AAPL")
        assert result == "## Good report"
        assert mock_client.models.generate_content.call_count == 2


# ============================================================================
# 4. Per-call client isolation
# ============================================================================

class TestClientIsolation:
    """Verify that each call creates a new client (plan.md §8)."""

    @patch("core.llm_inference.genai.Client")
    def test_new_client_per_call(self, MockClient, minimal_computed_data):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response()
        MockClient.return_value = mock_client

        from core.llm_inference import generate_full_report

        key = "AIzaFakeKey123456789012345678"
        generate_full_report(minimal_computed_data, key, "AAPL")
        generate_full_report(minimal_computed_data, key, "MSFT")

        # Client should be instantiated twice (once per call)
        assert MockClient.call_count == 2


# ============================================================================
# 5. Format helpers (private, tested indirectly but also directly)
# ============================================================================

class TestFormatHelpers:
    """Verify the S&R and Fibonacci formatters in llm_inference."""

    def test_format_sr_empty(self):
        from core.llm_inference import _format_sr_levels
        assert "No S&R" in _format_sr_levels([])

    def test_format_sr_with_levels(self):
        from core.llm_inference import _format_sr_levels
        result = _format_sr_levels([100.0, 200.0])
        assert "100.00" in result
        assert "200.00" in result

    def test_format_fib_empty(self):
        from core.llm_inference import _format_fib_levels
        assert "No Fibonacci" in _format_fib_levels({})

    def test_format_fib_with_levels(self):
        from core.llm_inference import _format_fib_levels
        result = _format_fib_levels({"38.2%": 150.0, "61.8%": 120.0})
        assert "38.2%" in result
        assert "150.00" in result
