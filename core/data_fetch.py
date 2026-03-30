"""
data_fetch.py — yfinance data-fetching layer for the TA Report Generator.

Responsibility (plan.md §4.1):
    Download and validate weekly OHLCV data for a given ticker symbol.

Error-handling strategy (plan.md §8):
    - Invalid / non-existent ticker  → ``ValueError`` with user-friendly message.
    - Insufficient history (< 52 wk) → ``ValueError``.
    - Network failure                → single retry after a configurable delay,
                                       then raise ``ConnectionError``.

This module performs **no** technical-indicator computation — that is the
exclusive domain of ``ta_compute.py``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from config.settings import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    MIN_DATA_POINTS,
    YFINANCE_RETRY_DELAY_SECONDS,
)

if TYPE_CHECKING:
    pass  # reserved for future type-only imports

logger = logging.getLogger(__name__)

# ============================================================================
# Constants (module-private)
# ============================================================================

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"Open", "High", "Low", "Close", "Volume"}
)


# ============================================================================
# Public API
# ============================================================================

def fetch_weekly_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Download weekly OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Validated, upper-cased ticker symbol (e.g. ``"AAPL"``, ``"RELIANCE.NS"``).
    period:
        Look-back period string accepted by yfinance (default ``"2y"``).
    interval:
        Candle interval (default ``"1wk"``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame indexed by ``Date`` with columns
        ``Open, High, Low, Close, Volume``.

    Raises
    ------
    ValueError
        If the ticker returns no data or fewer than ``MIN_DATA_POINTS`` rows.
    ConnectionError
        If yfinance fails after one retry.
    """
    df = _download_with_retry(ticker, period, interval)
    df = _clean_dataframe(df, ticker)
    _validate_minimum_rows(df, ticker)

    logger.info(
        "Fetched %d weekly candles for %s (period=%s).",
        len(df),
        ticker,
        period,
    )
    return df


# ============================================================================
# Internal helpers
# ============================================================================

def _download_with_retry(
    ticker: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Attempt a yfinance download with a single retry on failure.

    Plan.md §8: "Retry once after 5 s delay; surface error if still failing."
    """
    last_error: Exception | None = None

    for attempt in range(1, 3):  # attempts 1 and 2
        try:
            df: pd.DataFrame = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,       # suppress tqdm inside Gradio server
            )

            # yfinance may return a MultiIndex on columns for single tickers.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel("Ticker")

            return df

        except Exception as exc:  # noqa: BLE001 — broad catch deliberate
            last_error = exc
            if attempt == 1:
                logger.warning(
                    "yfinance download failed for %s (attempt %d/2): %s. "
                    "Retrying in %.1f s…",
                    ticker,
                    attempt,
                    exc,
                    YFINANCE_RETRY_DELAY_SECONDS,
                )
                time.sleep(YFINANCE_RETRY_DELAY_SECONDS)

    raise ConnectionError(
        f"Failed to download data for '{ticker}' after 2 attempts. "
        f"Last error: {last_error}"
    ) from last_error


def _clean_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate columns, drop NaN rows, and sort chronologically."""
    # --- Empty DataFrame guard ---
    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Ensure the symbol is correct and includes an exchange suffix "
            "if needed (e.g. RELIANCE.NS for NSE India)."
        )

    # --- Ensure required OHLCV columns exist ---
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Downloaded data for '{ticker}' is missing columns: "
            f"{', '.join(sorted(missing))}."
        )

    # --- Drop rows with any NaN in core OHLCV fields ---
    pre_drop_len = len(df)
    df = df.dropna(subset=list(_REQUIRED_COLUMNS))
    dropped = pre_drop_len - len(df)
    if dropped:
        logger.debug(
            "Dropped %d NaN rows from %s data (of %d total).",
            dropped,
            ticker,
            pre_drop_len,
        )

    # --- Chronological sort (yfinance usually returns sorted, but be safe) ---
    df = df.sort_index()

    return df


def _validate_minimum_rows(df: pd.DataFrame, ticker: str) -> None:
    """Raise ``ValueError`` if the DataFrame has fewer rows than required."""
    if len(df) < MIN_DATA_POINTS:
        raise ValueError(
            f"Insufficient data for '{ticker}': only {len(df)} weekly candles "
            f"available (minimum {MIN_DATA_POINTS} required). "
            "Try a different symbol or check the exchange suffix."
        )
