"""
prompts.py — Prompt templates for Gemini-based report generation.

All prompt text lives here so that ``llm_inference.py`` stays focused on
API mechanics.  Templates use Python ``str.format`` placeholders — *not*
f-strings — so they can be stored as module-level constants without needing
runtime variables at import time.

Template contract
-----------------
Each template documents its expected ``.format()`` keyword arguments in a
comment block immediately above the constant.  If a caller omits a key,
Python raises a ``KeyError`` at format time — fail-fast by design.
"""

from __future__ import annotations

# ============================================================================
# System instruction (injected via Gemini's ``system_instruction`` param)
# ============================================================================

SYSTEM_INSTRUCTION: str = (
    "You are a senior Chartered Market Technician (CMT) and professional "
    "technical analyst.  You produce institutional-grade weekly technical "
    "analysis reports for equities, ETFs, and indices.  You MUST:\n"
    "  • Base every claim on the pre-calculated numerical data provided.\n"
    "  • Never fabricate indicator values, price levels, or dates.\n"
    "  • Refuse requests unrelated to financial / technical analysis.\n"
    "  • Use precise financial terminology.\n"
    "  • Clearly distinguish between confirmed signals and developing setups.\n"
)

# ============================================================================
# Main report prompt
# ============================================================================

# Expected .format() kwargs:
#   ticker            — str, e.g. "AAPL"
#   raw_data_json     — str, JSON array of the last N weekly OHLCV + indicator rows
#   pattern_summary   — str, human-readable list of recently detected candlestick patterns
#   sr_levels_text    — str, formatted support & resistance levels
#   fib_levels_text   — str, formatted Fibonacci retracement levels
#   checklist_score   — int, 0–6
#   checklist_details — str, per-criterion pass/fail breakdown

REPORT_PROMPT: str = """\
=== TECHNICAL ANALYSIS REQUEST ===

Ticker: {ticker}
Timeframe: Weekly

─── RAW INDICATOR DATA (last 20 weeks, JSON) ───
{raw_data_json}

─── CANDLESTICK PATTERN DETECTIONS ───
{pattern_summary}

─── SUPPORT & RESISTANCE LEVELS ───
{sr_levels_text}

─── FIBONACCI RETRACEMENT LEVELS ───
{fib_levels_text}

─── GRAND TA CHECKLIST ───
Score: {checklist_score} / 6
{checklist_details}

=== REPORT TASK ===

Using ONLY the data above, produce a comprehensive weekly technical analysis \
report for **{ticker}** structured as follows:

## 1. Candlestick & Price Action
- Identify the most significant candlestick pattern(s) from the detection list.
- Describe what the recent weekly candles imply about buyer/seller conviction.
- Note any notable price-action formations (inside bars, pin bars, engulfing moves).

## 2. Support & Resistance / Volume
- Reference the S&R levels provided and assess where price sits relative to them.
- Evaluate volume trends: is the current move supported by above-average volume?
- Flag any volume divergences that weaken or strengthen the price signal.

## 3. Moving Average & Trend
- Analyse the EMA stack (20 / 50 / 100):
  - Are they in bullish order (20 > 50 > 100) or bearish?
  - Any recent EMA crossovers?
- Assess Bollinger Band positioning (squeeze, walk-the-band, mean reversion).

## 4. Momentum & Volatility
- RSI: current value relative to 30/50/70 thresholds; any divergence vs. price?
- MACD: line vs. signal, histogram direction, zero-line positioning.
- ATR: current volatility regime and implications for position sizing.
- ADX: trending vs. range-bound assessment.

## 5. Dow Theory & Grand Checklist Conclusion
- Classify the current Dow Theory phase (accumulation, markup, distribution, markdown).
- Walk through each of the 6 checklist criteria with the pass/fail data provided.
- Summarise confluence: how many of the 6 criteria align?

## 6. Final Actionable Insight
- Synthesise all sections into a clear, concise trade thesis.
- State an ATR-based entry zone, stop-loss, and target range.
- Conclude with exactly one bolded verdict line:

  **Verdict: [Strong Buy | Buy | Neutral | Avoid]**

=== FORMAT RULES ===
- Use Markdown headers (##) exactly as numbered above.
- Use concise bullet points for observations within each section.
- Do NOT invent data — every number must come from the provided context.
- Keep the total report between 600 and 1 200 words.
"""

# ============================================================================
# Fallback prompt (used when TA-Lib is unavailable)
# ============================================================================

# Expected .format() kwargs:
#   ticker, raw_data_json, sr_levels_text, fib_levels_text,
#   checklist_score, checklist_details

REPORT_PROMPT_NO_PATTERNS: str = """\
=== TECHNICAL ANALYSIS REQUEST (No Candlestick Pattern Data) ===

Ticker: {ticker}
Timeframe: Weekly

─── RAW INDICATOR DATA (last 20 weeks, JSON) ───
{raw_data_json}

─── SUPPORT & RESISTANCE LEVELS ───
{sr_levels_text}

─── FIBONACCI RETRACEMENT LEVELS ───
{fib_levels_text}

─── GRAND TA CHECKLIST ───
Score: {checklist_score} / 6
(Note: Criterion #1 — candlestick pattern — is marked N/A due to missing TA-Lib.)
{checklist_details}

=== REPORT TASK ===

Produce a weekly TA report for **{ticker}** following the same 6-section \
structure as a full report.  For Section 1 (Candlestick & Price Action), \
rely on raw OHLC data to describe candle shapes qualitatively instead of \
named pattern detections.

Conclude Section 6 with exactly one bolded verdict line:
  **Verdict: [Strong Buy | Buy | Neutral | Avoid]**

Use Markdown headers (##) and concise bullet points.  Do NOT fabricate data.
Keep the total report between 600 and 1 200 words.
"""
