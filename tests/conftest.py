"""
conftest.py — Shared pytest configuration and fixtures.

Ensures the project root (``ta_report_app/``) is on ``sys.path`` so that
``from config.settings import …`` and ``from core.data_fetch import …``
resolve correctly regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to sys.path so imports work from the tests/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
