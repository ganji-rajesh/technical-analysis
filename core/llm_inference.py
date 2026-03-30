"""
llm_inference.py — Gemini API inference layer for the TA Report Generator.

Responsibility (plan.md §4.3):
    Accepts the fully computed indicator payload, serialises it into a
    structured prompt (via ``prompts.py``), and makes a **single** Gemini
    API call to produce the complete 6-section TA report.

Design decisions
----------------
* **Single comprehensive call** — no multithreading.  The LLM sees all
  indicator context at once so its reasoning about inter-section
  relationships (e.g. Dow Theory vs. Momentum) stays cohesive.
* **Client instantiated per call** — avoids global-state mutations inside
  async / multi-session Gradio workloads (plan.md §8).
* **Exponential back-off** — transient API errors (rate-limit, timeout) are
  retried up to ``GEMINI_MAX_RETRIES`` times (plan.md §8).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

from config.settings import (
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL_NAME,
    GEMINI_RETRY_BACKOFF_BASE,
    GEMINI_TEMPERATURE,
    REPORT_TAIL_ROWS,
)
from core.prompts import (
    REPORT_PROMPT,
    REPORT_PROMPT_NO_PATTERNS,
    SYSTEM_INSTRUCTION,
)
from core.ta_compute import (
    compute_grand_checklist,
    format_checklist_details,
    format_pattern_summary,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def generate_full_report(
    computed_data: dict[str, Any],
    api_key: str,
    ticker: str,
) -> str:
    """Generate a complete TA report via a single Gemini API call.

    Parameters
    ----------
    computed_data:
        Output of ``ta_compute.compute_all_indicators``.
        Expected keys: ``indicators``, ``patterns``, ``fibonacci``, ``sr_levels``.
    api_key:
        Validated Gemini API key.
    ticker:
        Ticker symbol (for prompt injection and report header).

    Returns
    -------
    str
        Markdown-formatted TA report.

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted.
    """
    prompt = _build_prompt(computed_data, ticker)
    report = _call_gemini(api_key, prompt)
    return report


# ============================================================================
# Prompt assembly
# ============================================================================

def _build_prompt(computed_data: dict[str, Any], ticker: str) -> str:
    """Assemble the full prompt string from computed data and templates."""
    df: pd.DataFrame = computed_data["indicators"]
    patterns: dict = computed_data["patterns"]
    fib_levels: dict[str, float] = computed_data["fibonacci"]
    sr_levels: list[float] = computed_data["sr_levels"]

    # --- Serialise indicator DataFrame tail to JSON ---
    tail_df = df.tail(REPORT_TAIL_ROWS).copy()

    # Convert datetime index to string for JSON serialisation.
    if hasattr(tail_df.index, "strftime"):
        tail_df.index = tail_df.index.strftime("%Y-%m-%d")

    raw_data_json = tail_df.to_json(orient="records", indent=2, default_handler=str)

    # --- Format pattern summary ---
    pattern_summary = format_pattern_summary(patterns)

    # --- Format S&R and Fibonacci levels ---
    sr_levels_text = _format_sr_levels(sr_levels)
    fib_levels_text = _format_fib_levels(fib_levels)

    # --- Grand Checklist ---
    checklist = compute_grand_checklist(df, patterns, sr_levels)
    checklist_score = checklist["score"]
    checklist_details = format_checklist_details(checklist["details"])

    # --- Select template (full vs. no-patterns fallback) ---
    if patterns:
        prompt = REPORT_PROMPT.format(
            ticker=ticker,
            raw_data_json=raw_data_json,
            pattern_summary=pattern_summary,
            sr_levels_text=sr_levels_text,
            fib_levels_text=fib_levels_text,
            checklist_score=checklist_score,
            checklist_details=checklist_details,
        )
    else:
        prompt = REPORT_PROMPT_NO_PATTERNS.format(
            ticker=ticker,
            raw_data_json=raw_data_json,
            sr_levels_text=sr_levels_text,
            fib_levels_text=fib_levels_text,
            checklist_score=checklist_score,
            checklist_details=checklist_details,
        )

    return prompt


def _format_sr_levels(sr_levels: list[float]) -> str:
    """Format S&R levels into a bulleted list."""
    if not sr_levels:
        return "No S&R levels identified."
    return "\n".join(f"  • {level:,.2f}" for level in sr_levels)


def _format_fib_levels(fib_levels: dict[str, float]) -> str:
    """Format Fibonacci levels into a bulleted list."""
    if not fib_levels:
        return "No Fibonacci levels computed."
    return "\n".join(
        f"  • {label}: {price:,.2f}" for label, price in fib_levels.items()
    )


# ============================================================================
# Gemini API interaction
# ============================================================================

def _call_gemini(api_key: str, prompt: str) -> str:
    """Call Gemini with exponential back-off retry.

    A fresh ``genai.Client`` is instantiated per call to avoid global state
    pollution across concurrent Gradio sessions (plan.md §8).

    Raises
    ------
    RuntimeError
        If all retries are exhausted.
    """
    client = genai.Client(api_key=api_key)

    last_error: Exception | None = None

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                ),
            )

            # --- Extract text from response ---
            text = response.text
            if not text or not text.strip():
                raise ValueError("Gemini returned an empty response.")

            logger.info(
                "Gemini report generated successfully on attempt %d/%d.",
                attempt,
                GEMINI_MAX_RETRIES,
            )
            return text

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < GEMINI_MAX_RETRIES:
                wait = GEMINI_RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Gemini API error on attempt %d/%d: %s. "
                    "Retrying in %.1f s…",
                    attempt,
                    GEMINI_MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Gemini API failed after %d attempts. Last error: %s",
                    GEMINI_MAX_RETRIES,
                    exc,
                )

    raise RuntimeError(
        f"Gemini report generation failed after {GEMINI_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    ) from last_error
