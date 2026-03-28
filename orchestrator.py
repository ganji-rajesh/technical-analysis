"""
orchestrator.py
---------------
Central controller of the multi-agent pipeline.

Responsibilities:
  - Parse and validate user ticker input
  - Coordinate agents in strict order: Research → Analysis → Writing
  - Pass structured outputs between agents
  - Bubble up agent errors immediately with clear, user-facing messages
  - Report granular progress at each sub-step via optional callback
  - Enforce pipeline-level guardrails (ticker limits, environment checks)

No business logic lives here — only coordination and error surfacing.

Fixes applied vs original:
  - All print() replaced with structured logging
  - HF_TOKEN validated BEFORE research (not discovered mid-pipeline)
  - Ticker validation: rejects non-alphanumeric, overly-long, or numeric-only inputs
  - parse_tickers() strips .NS/.BO suffixes for dedup (TCS == TCS.NS treated same)
  - Pipeline timing per step — logged for performance visibility
  - Structured PipelineResult dataclass replaces raw dict return
  - Progress callback receives (step, pct_complete, message) — richer UI updates
  - Agent errors re-raised with step context (which agent failed + why)
  - Dry-run mode — validates input and env without making API calls
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from agents.research_agent import run_research_agent
from agents.analysis_agent  import run_analysis_agent
from agents.writing_agent   import run_writing_agent
from agents.utils           import get_api_key

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_TICKERS         = 5     # Hard limit — beyond this LLM costs spike sharply
MIN_TICKER_LEN      = 1     # e.g. single-char tickers do exist (rare)
MAX_TICKER_LEN      = 15    # yfinance max ticker length with suffix
VALID_TICKER_RE     = re.compile(r"^[A-Z0-9.\-&]+$")  # allowed chars in a ticker


# ── Progress Callback Type ─────────────────────────────────────────────────────

# Signature: callback(step: str, pct: int, message: str)
# step    — agent name: "ORCHESTRATOR" | "RESEARCH" | "ANALYSIS" | "WRITING"
# pct     — 0-100 completion percentage for progress bars
# message — human-readable status message
ProgressCallback = Callable[[str, int, str], None]


# ── Pipeline Result Dataclass ──────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Structured return type for run_pipeline().

    Using a dataclass instead of a raw dict provides:
      - IDE autocomplete and type hints for consumers (app.py, tests)
      - Explicit field names that can't silently change
      - Default values that are safe (empty string, not None)
    """
    report:        str        = ""
    research_data: dict       = field(default_factory=dict)
    analysis_data: dict       = field(default_factory=dict)
    tickers:       list       = field(default_factory=list)
    timing:        dict       = field(default_factory=dict)   # step → elapsed seconds


# ── Ticker Parser & Validator ──────────────────────────────────────────────────

def _is_valid_ticker(ticker: str) -> bool:
    """
    Validate a single cleaned ticker string.

    Rules:
      - Length between MIN_TICKER_LEN and MAX_TICKER_LEN
      - Only alphanumeric + . - & characters (matches NSE/BSE conventions)
      - Not purely numeric (e.g. "1234" is not a ticker)

    Returns True if valid, False otherwise.
    """
    if not (MIN_TICKER_LEN <= len(ticker) <= MAX_TICKER_LEN):
        return False
    if not VALID_TICKER_RE.match(ticker):
        return False
    if ticker.replace(".", "").replace("-", "").isdigit():
        return False
    return True


def _normalize_for_dedup(ticker: str) -> str:
    """
    Strip exchange suffixes before deduplication so TCS == TCS.NS == TCS.BO
    are treated as the same ticker and only kept once.

    The research_agent handles the full suffix normalisation — this is
    dedup-only and does NOT modify the ticker passed downstream.
    """
    for suffix in (".NS", ".BO", ".BSE", ".NSE"):
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def parse_tickers(raw_input: str) -> list[str]:
    """
    Parse, clean, validate, and deduplicate a comma-separated ticker string.

    Steps:
      1. Split on commas and semicolons (handles sloppy user input)
      2. Strip whitespace, uppercase
      3. Validate each ticker format
      4. Deduplicate (treating TCS == TCS.NS)
      5. Enforce MAX_TICKERS limit

    Args:
        raw_input: Free-form user input, e.g. "TCS, infy; HDFCBANK, TCS.NS"

    Returns:
        Ordered list of clean, unique, validated tickers.
        e.g. ["TCS", "INFY", "HDFCBANK"]

    Raises:
        ValueError — empty input, no valid tickers, too many tickers, invalid symbols
    """
    if not raw_input or not raw_input.strip():
        raise ValueError(
            "No tickers provided. "
            "Please enter at least one stock ticker (e.g. TCS, INFY, HDFCBANK)."
        )

    # Split on both comma and semicolon for user-friendly parsing
    raw_parts = re.split(r"[,;]", raw_input)
    cleaned   = [t.strip().upper() for t in raw_parts if t.strip()]

    if not cleaned:
        raise ValueError(
            "No tickers could be parsed from the input. "
            "Expected format: TCS, INFY, HDFCBANK"
        )

    # Validate each ticker
    invalid = [t for t in cleaned if not _is_valid_ticker(t)]
    if invalid:
        raise ValueError(
            f"Invalid ticker symbol(s): {', '.join(invalid)}. "
            "Tickers must contain only letters, digits, dots, or hyphens "
            "(e.g. TCS, HDFCBANK, BAJAJ-AUTO)."
        )

    # Deduplicate preserving first-seen order (TCS.NS treated same as TCS)
    seen_normalized: set  = set()
    tickers: list[str]    = []
    for t in cleaned:
        norm = _normalize_for_dedup(t)
        if norm not in seen_normalized:
            tickers.append(t)
            seen_normalized.add(norm)
        else:
            logger.info("Duplicate ticker removed: %s (same as %s)", t, norm)

    if len(tickers) > MAX_TICKERS:
        raise ValueError(
            f"Too many tickers: {len(tickers)} provided, maximum is {MAX_TICKERS}. "
            f"Please reduce your selection. Received: {', '.join(tickers)}"
        )

    logger.info("Parsed tickers: %s", tickers)
    return tickers


# ── Environment Preflight Check ────────────────────────────────────────────────

def preflight_check() -> None:
    """
    Validate all environment prerequisites before the pipeline starts.

    Currently checks:
      - GEMINI_API_KEY is set and non-empty

    Designed to be extended with additional checks (e.g. network connectivity,
    yfinance reachability) as the project grows.

    Raises:
        EnvironmentError — any required environment variable is missing
    """
    logger.info("Running preflight environment checks...")
    get_api_key()   # raises EnvironmentError with actionable message if not set
    logger.info("✅ Preflight checks passed.")


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

def run_pipeline(
    raw_input: str,
    progress_callback: Optional[ProgressCallback] = None,
    dry_run: bool = False,
) -> PipelineResult:
    """
    Execute the full multi-agent research pipeline.

    Pipeline steps:
      0. Preflight  — validate environment (GEMINI_API_KEY, etc.)
      1. Parse      — clean + validate ticker input
      2. Research   — fetch yfinance data for all tickers
      3. Analysis   — compute metrics + LLM interpretation
      4. Writing    — generate final Markdown report

    Args:
        raw_input:         Comma-separated ticker string from the user.
                           e.g. "TCS, INFY" or "HDFCBANK, ICICIBANK, KOTAKBANK"
        progress_callback: Optional callable(step, pct, message).
                           Called at key milestones for live UI updates.
                           step ∈ {"ORCHESTRATOR","RESEARCH","ANALYSIS","WRITING"}
                           pct  ∈ 0–100 (pipeline % complete)
        dry_run:           If True, validates input and environment only.
                           No API calls are made. Useful for testing and CI.

    Returns:
        PipelineResult dataclass with report, research_data, analysis_data,
        tickers, and per-step timing.

    Raises:
        EnvironmentError — GEMINI_API_KEY not set (surfaces before any work is done)
        ValueError       — Invalid ticker input or data issues
        RuntimeError     — Agent failure after retries (includes step context)
    """
    pipeline_start = time.perf_counter()
    timing: dict   = {}

    def log(step: str, pct: int, message: str) -> None:
        """Emit a structured log entry and optionally notify the UI callback."""
        logger.info("[%s | %d%%] %s", step, pct, message)
        if progress_callback:
            try:
                progress_callback(step, pct, message)
            except Exception as cb_exc:
                # Callback failures must never crash the pipeline
                logger.warning("Progress callback raised an exception: %s", cb_exc)

    def run_step(step_name: str, fn, *args):
        """
        Execute a pipeline step with timing and structured error wrapping.
        Re-raises agent errors with added step context for clearer debugging.
        """
        t0 = time.perf_counter()
        try:
            result = fn(*args)
            timing[step_name] = round(time.perf_counter() - t0, 2)
            logger.info("⏱  %s completed in %.1fs", step_name, timing[step_name])
            return result
        except (ValueError, EnvironmentError):
            raise   # Pass through known, user-facing errors as-is
        except Exception as exc:
            timing[step_name] = round(time.perf_counter() - t0, 2)
            raise RuntimeError(
                f"[{step_name}] Agent failed after {timing[step_name]:.1f}s: {exc}"
            ) from exc

    # ── Step 0: Preflight ──
    log("ORCHESTRATOR", 0, "Validating environment...")
    run_step("preflight", preflight_check)

    # ── Step 1: Parse Tickers ──
    log("ORCHESTRATOR", 5, "Parsing ticker input...")
    tickers = run_step("parse", parse_tickers, raw_input)
    log("ORCHESTRATOR", 10, f"Tickers confirmed: {', '.join(tickers)}")

    if dry_run:
        logger.info("Dry-run mode — stopping after input validation.")
        log("ORCHESTRATOR", 100, "✅ Dry-run complete. Input and environment are valid.")
        return PipelineResult(tickers=tickers, timing=timing)

    # ── Step 2: Research Agent ──
    log("RESEARCH", 15, f"Fetching market data for {len(tickers)} stock(s)...")
    research_data = run_step("research", run_research_agent, tickers)
    fetched = list(research_data["stocks"].keys())
    log("RESEARCH", 40, f"✅ Data fetched for: {', '.join(fetched)}")

    # ── Step 3: Analysis Agent ──
    log("ANALYSIS", 42, "Computing statistical metrics...")
    analysis_data = run_step("analysis", run_analysis_agent, research_data)
    n_analyzed = len(analysis_data["individual_analyses"])
    log("ANALYSIS", 70, f"✅ Analysis complete for {n_analyzed} stock(s).")

    # ── Step 4: Writing Agent ──
    log("WRITING", 72, "Generating investment report (parallel LLM calls)...")
    report = run_step("writing", run_writing_agent, analysis_data, research_data)
    log("WRITING", 98, f"✅ Report generated ({len(report):,} chars).")

    # ── Done ──
    total_elapsed = round(time.perf_counter() - pipeline_start, 2)
    timing["total"] = total_elapsed
    log("ORCHESTRATOR", 100, f"🎯 Pipeline complete in {total_elapsed:.1f}s.")

    return PipelineResult(
        report=report,
        research_data=research_data,
        analysis_data=analysis_data,
        tickers=tickers,
        timing=timing,
    )


# ── Streaming Pipeline (for real-time Gradio progress) ─────────────────────────

def run_pipeline_streaming(raw_input: str):
    """
    Generator version of run_pipeline — yields progress tuples between each
    agent step so Gradio can render real-time UI updates.

    The blocking run_pipeline() calls its progress_callback, but since the
    entire function is one blocking call the Gradio generator never yields
    between steps. This generator solves that by yielding control back to
    the caller after *every* step.

    Yields:
        Tuple[str, int, str, PipelineResult | None]:
            (step_name, pct_complete, message, result)
        result is None for all intermediate yields.
        The final yield has result set to the completed PipelineResult.

    Raises:
        Same exceptions as run_pipeline().
    """
    pipeline_start = time.perf_counter()
    timing: dict = {}

    def timed(step_name, fn, *args):
        """Run a pipeline step with timing and error wrapping."""
        t0 = time.perf_counter()
        try:
            result = fn(*args)
            timing[step_name] = round(time.perf_counter() - t0, 2)
            logger.info("⏱  %s completed in %.1fs", step_name, timing[step_name])
            return result
        except (ValueError, EnvironmentError):
            raise
        except Exception as exc:
            timing[step_name] = round(time.perf_counter() - t0, 2)
            raise RuntimeError(
                f"[{step_name}] Agent failed after {timing[step_name]:.1f}s: {exc}"
            ) from exc

    # ── Step 0: Preflight ──
    yield ("ORCHESTRATOR", 0, "Validating environment...", None)
    timed("preflight", preflight_check)

    # ── Step 1: Parse Tickers ──
    yield ("ORCHESTRATOR", 5, "Parsing ticker input...", None)
    tickers = timed("parse", parse_tickers, raw_input)
    yield ("ORCHESTRATOR", 10, f"Tickers confirmed: {', '.join(tickers)}", None)

    # ── Step 2: Research Agent ──
    yield ("RESEARCH", 15, f"Fetching market data for {len(tickers)} stock(s)...", None)
    research_data = timed("research", run_research_agent, tickers)
    fetched = list(research_data["stocks"].keys())
    yield ("RESEARCH", 40, f"✅ Data fetched for: {', '.join(fetched)}", None)

    # ── Step 3: Analysis Agent ──
    yield ("ANALYSIS", 42, "Computing metrics & LLM interpretation...", None)
    analysis_data = timed("analysis", run_analysis_agent, research_data)
    n = len(analysis_data["individual_analyses"])
    yield ("ANALYSIS", 70, f"✅ Analysis complete for {n} stock(s).", None)

    # ── Step 4: Writing Agent ──
    yield ("WRITING", 72, "Generating investment report (parallel LLM calls)...", None)
    report = timed("writing", run_writing_agent, analysis_data, research_data)
    yield ("WRITING", 98, f"✅ Report generated ({len(report):,} chars).", None)

    # ── Done ──
    timing["total"] = round(time.perf_counter() - pipeline_start, 2)
    logger.info("🎯 Pipeline complete in %.1fs.", timing["total"])

    yield ("ORCHESTRATOR", 100, f"🎯 Pipeline complete in {timing['total']:.1f}s.", PipelineResult(
        report=report,
        research_data=research_data,
        analysis_data=analysis_data,
        tickers=tickers,
        timing=timing,
    ))
