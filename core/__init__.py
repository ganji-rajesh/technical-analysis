"""
core — Domain logic for the TA Report Generator.

Public API re-exported here for convenience so that upstream layers
(``main_pipeline``, ``app``) can import directly from the package::

    from core import fetch_weekly_data, compute_all_indicators, generate_full_report
"""

from core.data_fetch import fetch_weekly_data
from core.ta_compute import (
    compute_all_indicators,
    compute_fibonacci,
    compute_grand_checklist,
    compute_support_resistance,
)
from core.llm_inference import generate_full_report

__all__ = [
    # data_fetch
    "fetch_weekly_data",
    # ta_compute
    "compute_all_indicators",
    "compute_fibonacci",
    "compute_support_resistance",
    "compute_grand_checklist",
    # llm_inference
    "generate_full_report",
]
