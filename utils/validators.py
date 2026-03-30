"""
validators.py — Input validation utilities for the TA Report Generator.

Provides pure-function validators for ticker symbols and Gemini API keys.
Every validator returns a ``ValidationResult`` named-tuple so callers get
both a boolean flag *and* a user-facing message without resorting to
exception-based control flow for expected invalid inputs.

Security note (plan.md §9):
    - Ticker symbols are sanitised with a strict regex to prevent injection.
    - API keys are checked for format plausibility only — actual auth is
      verified by the Gemini client at call time.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from config.settings import (
    API_KEY_MIN_LENGTH,
    API_KEY_PREFIX,
    TICKER_REGEX,
)


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

class ValidationResult(NamedTuple):
    """Immutable result of a validation check."""

    is_valid: bool
    message: str


# ---------------------------------------------------------------------------
# Ticker validation
# ---------------------------------------------------------------------------

# Pre-compile the regex once at module load for performance.
_TICKER_PATTERN: re.Pattern[str] = re.compile(TICKER_REGEX)


def validate_ticker(ticker: str | None) -> ValidationResult:
    """Validate a user-supplied ticker symbol.

    Rules (derived from plan.md §9):
        1. Must not be empty / whitespace-only.
        2. Must match ``^[A-Z0-9.^=\\-]{1,20}$`` after upper-casing.
        3. Must not exceed 20 characters.

    Parameters
    ----------
    ticker:
        Raw ticker string from the Gradio text-box.

    Returns
    -------
    ValidationResult
        ``(True, <sanitised_ticker>)`` on success, or
        ``(False, <error_message>)`` on failure.
    """
    if not ticker or not ticker.strip():
        return ValidationResult(
            is_valid=False,
            message="Ticker symbol is required. Please enter a valid symbol (e.g. AAPL, RELIANCE.NS).",
        )

    sanitised = ticker.strip().upper()

    if len(sanitised) > 20:
        return ValidationResult(
            is_valid=False,
            message=f"Ticker '{sanitised}' exceeds the 20-character limit.",
        )

    if not _TICKER_PATTERN.fullmatch(sanitised):
        return ValidationResult(
            is_valid=False,
            message=(
                f"Invalid ticker format: '{sanitised}'. "
                "Only uppercase letters, digits, dots, carets (^), "
                "equals (=), and hyphens (-) are allowed."
            ),
        )

    return ValidationResult(is_valid=True, message=sanitised)


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

def validate_api_key(api_key: str | None) -> ValidationResult:
    """Validate the format of a Gemini API key.

    This is a *format-only* check — it does **not** make a network call to
    verify the key with Google.  Actual authentication is deferred to the
    first ``genai.Client`` usage.

    Rules:
        1. Must not be empty / whitespace-only.
        2. Must be at least ``API_KEY_MIN_LENGTH`` characters.
        3. Must start with the expected ``API_KEY_PREFIX`` (``AIza``).

    Parameters
    ----------
    api_key:
        Raw API key string from the Gradio password text-box.

    Returns
    -------
    ValidationResult
        ``(True, "API key format is valid.")`` on success, or
        ``(False, <error_message>)`` on failure.
    """
    if not api_key or not api_key.strip():
        return ValidationResult(
            is_valid=False,
            message="Gemini API key is required. Paste your key in the left panel.",
        )

    key = api_key.strip()

    if len(key) < API_KEY_MIN_LENGTH:
        return ValidationResult(
            is_valid=False,
            message=(
                f"API key appears too short ({len(key)} chars). "
                f"A valid Gemini key is at least {API_KEY_MIN_LENGTH} characters."
            ),
        )

    if not key.startswith(API_KEY_PREFIX):
        return ValidationResult(
            is_valid=False,
            message=(
                f"API key should start with '{API_KEY_PREFIX}'. "
                "Please verify you've pasted the correct Gemini key."
            ),
        )

    return ValidationResult(is_valid=True, message="API key format is valid.")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def sanitise_ticker(ticker: str) -> str:
    """Return the upper-cased, stripped ticker or raise ``ValueError``.

    This is a thin wrapper for call-sites that prefer exception-based flow
    (e.g. inside the pipeline orchestrator) rather than inspecting a
    ``ValidationResult``.

    Raises
    ------
    ValueError
        If the ticker fails validation.
    """
    result = validate_ticker(ticker)
    if not result.is_valid:
        raise ValueError(result.message)
    return result.message  # The sanitised ticker string on success.
