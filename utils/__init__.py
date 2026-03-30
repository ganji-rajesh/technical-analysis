"""
utils — Shared utility modules for the TA Report Generator.

Public API re-exported here for convenience:
    - ``validate_ticker``, ``validate_api_key``, ``sanitise_ticker``
    - ``build_all_charts`` (and individual chart builders)
"""

from utils.validators import (
    ValidationResult,
    sanitise_ticker,
    validate_api_key,
    validate_ticker,
)
from utils.chart_builder import (
    build_all_charts,
    build_fib_overlay,
    build_macd_chart,
    build_price_chart,
    build_rsi_chart,
    build_sr_overlay,
    build_volume_chart,
)

__all__ = [
    # validators
    "ValidationResult",
    "validate_ticker",
    "validate_api_key",
    "sanitise_ticker",
    # chart_builder
    "build_all_charts",
    "build_price_chart",
    "build_volume_chart",
    "build_rsi_chart",
    "build_macd_chart",
    "build_sr_overlay",
    "build_fib_overlay",
]
