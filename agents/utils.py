"""
agents/utils.py
---------------
Shared utilities for all agents in the multi-agent pipeline.

Single source of truth for:
  - GEMINI_API_KEY retrieval and validation
  - Google Gemini API client (call_llm)
  - Retry decorator (with_retry)

Uses google-genai SDK — the official Google Generative AI client that accepts
system instructions and user messages, working with all Gemini models.

All agents import from here — no copy-pasting between files.

Usage:
    from agents.utils import call_llm, get_api_key, with_retry
"""

import logging
import os
from functools import wraps
from time import sleep

from google import genai

# ── Logging ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("agents.utils")

# ── Configuration ──────────────────────────────────────────────────────────────

# Default model: gemini-2.0-flash is fast, cost-effective, and well-suited
# for analytical and writing tasks.
# Override via: $env:GEMINI_MODEL = "gemini-2.5-pro-preview-05-06"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# LLM generation defaults (all overridable per call)
LLM_TEMPERATURE        = 0.3
LLM_TIMEOUT_SECONDS    = 120
LLM_MAX_RETRY_ATTEMPTS = 3
LLM_RETRY_WAIT_SECONDS = 5     # Gemini cold-starts are much faster than HF
MIN_VALID_RESPONSE_LEN = 30    # chars below which a response is treated as failed


# ── API Key Management ─────────────────────────────────────────────────────────

def get_api_key() -> str:
    """
    Read and validate GEMINI_API_KEY from the environment.

    Designed to be called at the START of any agent entrypoint so the pipeline
    fails immediately with an actionable message — before any computation or
    API calls are made.

    Returns:
        Non-empty Google Gemini API key string.

    Raises:
        EnvironmentError — if GEMINI_API_KEY is not set or is empty.
    """
    token = os.environ.get("GEMINI_API_KEY", "").strip()
    if not token:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set or is empty.\n\n"
            "To fix this:\n"
            "  Windows PowerShell   : $env:GEMINI_API_KEY = \"your-key-here\"\n"
            "  Windows CMD          : set GEMINI_API_KEY=your-key-here\n"
            "  Linux / Mac          : export GEMINI_API_KEY=your-key-here\n\n"
            "Get your free API key at: https://aistudio.google.com/apikey"
        )
    return token



# ── Retry Decorator ────────────────────────────────────────────────────────────

def with_retry(
    max_attempts: int = LLM_MAX_RETRY_ATTEMPTS,
    wait_seconds: float = LLM_RETRY_WAIT_SECONDS,
    reraise_on: tuple = (),
    exponential: bool = False,
):
    """
    Retry decorator with configurable wait strategy.

    Args:
        max_attempts:  Total number of attempts (including first try).
        wait_seconds:  Base seconds to wait between attempts.
                       If exponential=False: flat wait (e.g. 5s, 5s, 5s).
                       If exponential=True:  wait = wait_seconds ** (attempt-1)
                       (e.g. base=2 → 1s, 2s, 4s).
        reraise_on:    Tuple of exception types that should NOT be retried
                       (e.g. auth errors that won't self-resolve).
        exponential:   If True, use exponential backoff instead of flat wait.

    Usage:
        @with_retry(max_attempts=3, wait_seconds=5)
        def my_llm_call(): ...

        @with_retry(max_attempts=3, wait_seconds=2, exponential=True)
        def my_api_call(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    exc_str = str(exc)
                    # Immediately re-raise exceptions that retrying won't fix
                    if reraise_on and isinstance(exc, reraise_on):
                        raise
                    # Never retry on auth/config failures — they won't self-resolve
                    if (
                        "401" in exc_str
                        or "API_KEY" in exc_str
                        or "GEMINI_API_KEY" in exc_str
                        or "EnvironmentError" in type(exc).__name__
                        or "404" in exc_str          # model not found = config error
                        or "403" in exc_str          # permission error
                        or "INVALID_ARGUMENT" in exc_str
                    ):
                        raise
                    last_exc = exc
                    wait = wait_seconds ** (attempt - 1) if exponential else wait_seconds
                    logger.warning(
                        "[%s] Attempt %d/%d failed: %s. Retrying in %ds...",
                        fn.__name__, attempt, max_attempts, exc, wait
                    )
                    if attempt < max_attempts:
                        sleep(wait)
            raise last_exc
        return wrapper
    return decorator


# ── LLM Client ────────────────────────────────────────────────────────────────

@with_retry(max_attempts=LLM_MAX_RETRY_ATTEMPTS, wait_seconds=LLM_RETRY_WAIT_SECONDS)
def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 600,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Send a prompt to the Google Gemini API and return plain-text response.

    Uses google-genai SDK with system instruction support and
    structured generation config.

    Args:
        prompt:      Task instruction for the model.
        system:      System/role prompt (optional, sent as system_instruction).
        max_tokens:  Maximum new tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        Generated text string — non-empty and validated.

    Raises:
        EnvironmentError — GEMINI_API_KEY not set (not retried)
        RuntimeError     — API failure, timeout, or empty response
    """
    api_key = get_api_key()

    logger.debug(
        "Calling Gemini generate_content — model: %s | max_tokens: %d",
        GEMINI_MODEL, max_tokens
    )

    try:
        client = genai.Client(api_key=api_key)

        # Build generation config
        config = genai.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Add system instruction if provided
        if system:
            config.system_instruction = system

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        text = response.text

    except EnvironmentError:
        raise  # Pass through — not retried by with_retry
    except Exception as exc:
        exc_str = str(exc).lower()
        if "api key" in exc_str or "api_key" in exc_str or "401" in str(exc):
            raise EnvironmentError(
                "Invalid or missing GEMINI_API_KEY. "
                "Check your key at https://aistudio.google.com/apikey"
            ) from exc
        elif "403" in str(exc) or "permission" in exc_str:
            raise RuntimeError(
                "Access forbidden (HTTP 403). "
                "Your Gemini API key may lack the required permissions."
            ) from exc
        elif "404" in str(exc):
            raise RuntimeError(
                f"Model '{GEMINI_MODEL}' not found (HTTP 404).\n"
                "Switch via: $env:GEMINI_MODEL = 'gemini-2.0-flash'\n"
                "Other available models:\n"
                "  gemini-2.5-pro-preview-05-06\n"
                "  gemini-2.0-flash-lite"
            ) from exc
        elif "429" in str(exc) or "quota" in exc_str or "rate" in exc_str:
            raise RuntimeError(
                "Gemini rate limit / quota exceeded (HTTP 429). "
                "Wait a moment or check your API quota at "
                "https://aistudio.google.com/apikey"
            ) from exc
        elif "timeout" in exc_str or "timed out" in exc_str:
            raise RuntimeError(
                f"Gemini API timed out after {LLM_TIMEOUT_SECONDS}s. "
                "Retrying automatically."
            ) from exc
        elif "block" in exc_str or "safety" in exc_str:
            raise RuntimeError(
                "Gemini blocked the response due to safety filters. "
                "Try rephrasing the prompt or adjusting safety settings."
            ) from exc
        else:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    # ── Validate Response ──
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    if len(text) < MIN_VALID_RESPONSE_LEN:
        raise RuntimeError(
            f"Gemini response too short ({len(text)} chars). Content: '{text}'"
        )

    logger.debug("LLM response received — %d chars", len(text))
    return text
