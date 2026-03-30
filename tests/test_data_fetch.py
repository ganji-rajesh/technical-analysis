"""
test_data_fetch.py — Unit tests for ``core.data_fetch``.

Testing strategy
----------------
* All ``yfinance.download`` calls are **mocked** — tests must run offline
  and deterministically, without depending on market-data availability.
* Tests cover the three documented error-handling paths (plan.md §8):
    1. Empty DataFrame → ``ValueError``
    2. Insufficient rows (< 52) → ``ValueError``
    3. Network failure with retry → ``ConnectionError``
* A ``conftest``-style fixture builds a realistic 104-row weekly OHLCV
  DataFrame used across all happy-path tests.

Run::

    pytest tests/test_data_fetch.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.settings import DEFAULT_INTERVAL, DEFAULT_PERIOD, MIN_DATA_POINTS


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def make_ohlcv_df():
    """Factory fixture that builds a synthetic weekly OHLCV DataFrame.

    Parameters
    ----------
    n_rows:
        Number of weekly candles to generate (default 104 ≈ 2 years).
    start_price:
        Opening price of the first candle.
    """

    def _factory(n_rows: int = 104, start_price: float = 100.0) -> pd.DataFrame:
        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(
            end=datetime(2026, 3, 28),
            periods=n_rows,
            freq="W-FRI",
            name="Date",
        )

        close = start_price + np.cumsum(rng.normal(0, 2, size=n_rows))
        open_ = close + rng.normal(0, 0.5, size=n_rows)
        high = np.maximum(open_, close) + rng.uniform(0.5, 3, size=n_rows)
        low = np.minimum(open_, close) - rng.uniform(0.5, 3, size=n_rows)
        volume = rng.integers(1_000_000, 10_000_000, size=n_rows).astype(float)

        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    return _factory


@pytest.fixture()
def valid_ohlcv(make_ohlcv_df) -> pd.DataFrame:
    """A 104-row DataFrame that passes all validation checks."""
    return make_ohlcv_df(n_rows=104)


# ============================================================================
# Happy-path tests
# ============================================================================

class TestFetchWeeklyDataHappyPath:
    """Verify that a well-formed yfinance response flows through correctly."""

    @patch("core.data_fetch.yf.download")
    def test_returns_dataframe_with_correct_columns(self, mock_dl, valid_ohlcv):
        mock_dl.return_value = valid_ohlcv
        from core.data_fetch import fetch_weekly_data

        result = fetch_weekly_data("AAPL")

        assert isinstance(result, pd.DataFrame)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    @patch("core.data_fetch.yf.download")
    def test_returns_at_least_min_data_points(self, mock_dl, valid_ohlcv):
        mock_dl.return_value = valid_ohlcv
        from core.data_fetch import fetch_weekly_data

        result = fetch_weekly_data("AAPL")
        assert len(result) >= MIN_DATA_POINTS

    @patch("core.data_fetch.yf.download")
    def test_result_is_chronologically_sorted(self, mock_dl, valid_ohlcv):
        # Shuffle the index before passing to fetch
        shuffled = valid_ohlcv.sample(frac=1, random_state=7)
        mock_dl.return_value = shuffled
        from core.data_fetch import fetch_weekly_data

        result = fetch_weekly_data("AAPL")
        assert result.index.is_monotonic_increasing

    @patch("core.data_fetch.yf.download")
    def test_nan_rows_are_dropped(self, mock_dl, valid_ohlcv):
        df = valid_ohlcv.copy()
        # Inject 3 NaN rows
        df.iloc[0, 0] = np.nan
        df.iloc[5, 2] = np.nan
        df.iloc[10, 4] = np.nan
        mock_dl.return_value = df
        from core.data_fetch import fetch_weekly_data

        result = fetch_weekly_data("AAPL")
        assert not result.isna().any().any()
        assert len(result) == len(valid_ohlcv) - 3

    @patch("core.data_fetch.yf.download")
    def test_default_period_and_interval_passed_to_yfinance(self, mock_dl, valid_ohlcv):
        mock_dl.return_value = valid_ohlcv
        from core.data_fetch import fetch_weekly_data

        fetch_weekly_data("AAPL")
        mock_dl.assert_called_once()
        call_kwargs = mock_dl.call_args[1]
        assert call_kwargs["period"] == DEFAULT_PERIOD
        assert call_kwargs["interval"] == DEFAULT_INTERVAL

    @patch("core.data_fetch.yf.download")
    def test_multi_index_columns_are_flattened(self, mock_dl, valid_ohlcv):
        """yfinance sometimes returns MultiIndex columns for single tickers."""
        multi_cols = pd.MultiIndex.from_tuples(
            [(c, "AAPL") for c in valid_ohlcv.columns],
            names=["Price", "Ticker"],
        )
        multi_df = valid_ohlcv.copy()
        multi_df.columns = multi_cols
        mock_dl.return_value = multi_df
        from core.data_fetch import fetch_weekly_data

        result = fetch_weekly_data("AAPL")
        assert not isinstance(result.columns, pd.MultiIndex)
        assert "Open" in result.columns


# ============================================================================
# Error-path tests
# ============================================================================

class TestFetchWeeklyDataErrors:
    """Verify documented error handling (plan.md §8)."""

    @patch("core.data_fetch.yf.download")
    def test_empty_dataframe_raises_value_error(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        from core.data_fetch import fetch_weekly_data

        with pytest.raises(ValueError, match="No data returned"):
            fetch_weekly_data("INVALIDTICKER")

    @patch("core.data_fetch.yf.download")
    def test_insufficient_rows_raises_value_error(self, mock_dl, make_ohlcv_df):
        short_df = make_ohlcv_df(n_rows=30)
        mock_dl.return_value = short_df
        from core.data_fetch import fetch_weekly_data

        with pytest.raises(ValueError, match="Insufficient data"):
            fetch_weekly_data("PENNY")

    @patch("core.data_fetch.yf.download")
    def test_missing_columns_raises_value_error(self, mock_dl, valid_ohlcv):
        bad_df = valid_ohlcv.drop(columns=["Volume"])
        mock_dl.return_value = bad_df
        from core.data_fetch import fetch_weekly_data

        with pytest.raises(ValueError, match="missing columns"):
            fetch_weekly_data("AAPL")

    @patch("core.data_fetch.time.sleep", return_value=None)  # skip retry delay
    @patch("core.data_fetch.yf.download")
    def test_network_failure_retries_then_raises_connection_error(
        self, mock_dl, mock_sleep
    ):
        mock_dl.side_effect = ConnectionError("Network unreachable")
        from core.data_fetch import fetch_weekly_data

        with pytest.raises(ConnectionError, match="Failed to download"):
            fetch_weekly_data("AAPL")

        # Should have attempted twice
        assert mock_dl.call_count == 2
        # Retry delay should have been called once
        mock_sleep.assert_called_once()

    @patch("core.data_fetch.time.sleep", return_value=None)
    @patch("core.data_fetch.yf.download")
    def test_transient_failure_succeeds_on_retry(
        self, mock_dl, mock_sleep, valid_ohlcv
    ):
        """First call fails, second call succeeds."""
        mock_dl.side_effect = [RuntimeError("Timeout"), valid_ohlcv]
        from core.data_fetch import fetch_weekly_data

        result = fetch_weekly_data("AAPL")
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= MIN_DATA_POINTS
        assert mock_dl.call_count == 2


# ============================================================================
# Edge-case tests
# ============================================================================

class TestFetchWeeklyDataEdgeCases:
    """Edge cases around boundary conditions."""

    @patch("core.data_fetch.yf.download")
    def test_exactly_min_data_points_passes(self, mock_dl, make_ohlcv_df):
        """52 rows should be the minimum acceptable count."""
        df = make_ohlcv_df(n_rows=MIN_DATA_POINTS)
        mock_dl.return_value = df
        from core.data_fetch import fetch_weekly_data

        result = fetch_weekly_data("AAPL")
        assert len(result) == MIN_DATA_POINTS

    @patch("core.data_fetch.yf.download")
    def test_one_below_min_data_points_fails(self, mock_dl, make_ohlcv_df):
        """51 rows should fail validation."""
        df = make_ohlcv_df(n_rows=MIN_DATA_POINTS - 1)
        mock_dl.return_value = df
        from core.data_fetch import fetch_weekly_data

        with pytest.raises(ValueError, match="Insufficient data"):
            fetch_weekly_data("AAPL")

    @patch("core.data_fetch.yf.download")
    def test_custom_period_and_interval_forwarded(self, mock_dl, valid_ohlcv):
        mock_dl.return_value = valid_ohlcv
        from core.data_fetch import fetch_weekly_data

        fetch_weekly_data("AAPL", period="5y", interval="1mo")
        call_kwargs = mock_dl.call_args[1]
        assert call_kwargs["period"] == "5y"
        assert call_kwargs["interval"] == "1mo"
