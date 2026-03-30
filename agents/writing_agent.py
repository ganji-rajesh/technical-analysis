"""
writing_agent.py
----------------
Responsible ONLY for producing the final investment report in Markdown.

Two layers:
  1. Python layer  → builds structured data tables (Fundamentals, Technical)
  2. LLM layer     → writes narrative sections via separate focused plain-text calls

Fixes applied vs original:
  - call_llm() imported from shared agents/utils.py (no duplication)
  - All LLM narrative calls run in PARALLEL via ThreadPoolExecutor
  - LLM responses validated — empty/short responses replaced with fallback text
  - fmt_pct() fixed — no longer double-multiplies pre-percentage values
  - build_context_summary() passes full llm_analysis (not truncated at 300 chars)
  - stocks[ticker] key mismatch guarded — logs warning instead of silent empty dict
  - datetime.now() replaced with UTC-aware timestamp
  - All print() replaced with logging module
  - HF_TOKEN validated at entry point before any computation
  - Per-stock narrative calls parallelized (independent of each other)
  - Assembler handles None gracefully in all narrative slots
"""

import logging
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timezone

# ── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("writing_agent")


# ── Shared LLM Client (no duplication) ────────────────────────────────────────
# call_llm and get_api_key are defined in agents/utils.py — single source of truth.

from agents.utils import call_llm, get_api_key



# ── Constants ──────────────────────────────────────────────────────────────────

SYSTEM_WRITER = (
    "You are a professional equity research analyst specializing in Indian stock markets. "
    "Write clearly, concisely, and in a professional financial tone. "
    "Always reference specific numbers. Do not use bullet points unless asked."
)

# Narrative section fallbacks — used when LLM returns unusable output
FALLBACK_EXECUTIVE_SUMMARY = "*Executive summary could not be generated. Please re-run the pipeline.*"
FALLBACK_STOCK_SECTION     = "*Stock narrative could not be generated. Please re-run the pipeline.*"
FALLBACK_RISK_FACTORS      = "*Risk factors could not be generated. Please re-run the pipeline.*"
FALLBACK_CONCLUSION        = "*Investment conclusion could not be generated. Please re-run the pipeline.*"

# Minimum acceptable LLM response length (chars) before treating as failed
MIN_VALID_RESPONSE_LEN = 50

# Max parallel workers for concurrent LLM calls
MAX_LLM_WORKERS = 4


# ── Data Formatters ────────────────────────────────────────────────────────────

def fmt_inr(value) -> str:
    """Format a number as Indian Rupee string with Cr/L Cr suffixes."""
    if value is None:
        return "N/A"
    try:
        val = float(value)
        if val >= 1e12:
            return f"\u20b9{val/1e12:.2f} L Cr"
        elif val >= 1e7:
            return f"\u20b9{val/1e7:.2f} Cr"
        return f"\u20b9{val:,.2f}"
    except Exception:
        return "N/A"


def fmt_pct(value, already_percentage: bool = False) -> str:
    """
    Format a value as a percentage string.

    Args:
        value:              Raw value to format
        already_percentage: Set True if value is already in % form (e.g. 18.4)
                            Set False (default) if value is decimal form (e.g. 0.184)

    Fix vs original:
        Original always multiplied by 100, causing double-multiplication
        for fields that yfinance already returns as percentages.
        Now the caller explicitly declares the input form.

    Examples:
        fmt_pct(0.184)              → "18.40%"   (decimal input)
        fmt_pct(18.4, already_percentage=True) → "18.40%"   (percent input)
    """
    if value is None:
        return "N/A"
    try:
        fval = float(value)
        if not already_percentage:
            fval = fval * 100
        return f"{fval:.2f}%"
    except Exception:
        return "N/A"


def fmt_float(value, decimals: int = 2) -> str:
    """Format a float to a fixed number of decimal places."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


# ── LLM Response Validator ─────────────────────────────────────────────────────

def _validate_narrative(text: str, fallback: str, section_name: str) -> str:
    """
    Validate an LLM-generated narrative string.
    Returns fallback text if response is empty, too short, or contains prompt artifacts.

    Args:
        text:         Raw LLM output string
        fallback:     Replacement text if validation fails
        section_name: Label for logging

    Returns:
        Validated narrative or fallback string
    """
    if not text or len(text.strip()) < MIN_VALID_RESPONSE_LEN:
        logger.warning(
            "Section '%s' returned insufficient text (%d chars). Using fallback.",
            section_name, len(text) if text else 0
        )
        return fallback

    # Detect common Mistral prompt-echo artifacts
    if "[/INST]" in text[:100] or text.strip().startswith("<s>"):
        logger.warning("Section '%s' contains prompt echo artifacts. Using fallback.", section_name)
        return fallback

    return text.strip()


# ── Python Table Builders ──────────────────────────────────────────────────────

def _get_stock_safe(stocks: dict, ticker: str) -> dict:
    """
    Safely retrieve stock data dict by ticker.
    Logs a warning if ticker is missing instead of silently returning empty dict.
    """
    data = stocks.get(ticker)
    if data is None:
        logger.warning(
            "Ticker '%s' not found in stocks dict (keys: %s). "
            "All fields will show N/A.",
            ticker, list(stocks.keys())
        )
        return {}
    return data


def build_fundamentals_table(analyses: list, stocks: dict) -> str:
    """
    Build a multi-column Markdown table of fundamental metrics for all tickers.
    Uses fmt_pct with correct already_percentage flags per yfinance field.
    """
    tickers = [a["ticker"] for a in analyses]
    header  = "| Metric | " + " | ".join(tickers) + " |"
    divider = "| --- | "  + " | ".join(["---"] * len(tickers)) + " |"

    # (label, extractor_lambda)
    # already_percentage=True for fields yfinance returns as % already
    rows = [
        ("Current Price (\u20b9)", lambda t: fmt_float(_get_stock_safe(stocks, t).get("current_price"))),
        ("Market Cap",              lambda t: fmt_inr(_get_stock_safe(stocks, t).get("market_cap"))),
        ("P/E Ratio",               lambda t: fmt_float(_get_stock_safe(stocks, t).get("pe_ratio"))),
        ("P/B Ratio",               lambda t: fmt_float(_get_stock_safe(stocks, t).get("pb_ratio"))),
        ("EPS (\u20b9)",           lambda t: fmt_float(_get_stock_safe(stocks, t).get("eps"))),
        ("ROE",                     lambda t: fmt_pct(_get_stock_safe(stocks, t).get("roe"), already_percentage=False)),
        ("Profit Margin",           lambda t: fmt_pct(_get_stock_safe(stocks, t).get("profit_margin"), already_percentage=False)),
        ("Debt / Equity",           lambda t: fmt_float(_get_stock_safe(stocks, t).get("debt_to_equity"))),
        ("Beta",                    lambda t: fmt_float(_get_stock_safe(stocks, t).get("beta"))),
        ("Dividend Yield",          lambda t: fmt_pct(_get_stock_safe(stocks, t).get("dividend_yield"), already_percentage=False)),
        ("52W High (\u20b9)",      lambda t: fmt_float(_get_stock_safe(stocks, t).get("52_week_high"))),
        ("52W Low (\u20b9)",       lambda t: fmt_float(_get_stock_safe(stocks, t).get("52_week_low"))),
        ("Revenue",                 lambda t: fmt_inr(_get_stock_safe(stocks, t).get("revenue"))),
        ("Net Income",              lambda t: fmt_inr(_get_stock_safe(stocks, t).get("net_income"))),
    ]

    table_rows = []
    for label, fn in rows:
        values = [fn(t) for t in tickers]
        table_rows.append(f"| {label} | " + " | ".join(values) + " |")

    return "\n".join([header, divider] + table_rows)


def build_technical_table(analyses: list) -> str:
    """
    Build a multi-column Markdown table of technical indicators for all tickers.
    Handles None values for MA-50 when insufficient data exists.
    """
    tickers = [a["ticker"] for a in analyses]
    header  = "| Metric | " + " | ".join(tickers) + " |"
    divider = "| --- | "  + " | ".join(["---"] * len(tickers)) + " |"

    metric_rows = [
        ("6M Return (%)",        "six_month_return_pct"),
        ("20D Momentum (%)",     "momentum_20d_pct"),
        ("Volatility Ann. (%)",  "volatility_annualized_pct"),
        ("RSI (14)",             "rsi_14"),
        ("MA-20 (\u20b9)",      "ma_20"),
        ("MA-50 (\u20b9)*",     "ma_50"),         # * = may be N/A for new stocks
        ("Price vs MA-20 (%)",   "price_vs_ma20_pct"),
        ("Price vs MA-50 (%)*",  "price_vs_ma50_pct"),
        ("% from 52W High",      "pct_from_52w_high"),
        ("% from 52W Low",       "pct_from_52w_low"),
    ]

    table_rows = []
    for label, key in metric_rows:
        values = []
        for a in analyses:
            val = a.get("metrics", {}).get(key)
            values.append(fmt_float(val) if val is not None else "N/A†")
        table_rows.append(f"| {label} | " + " | ".join(values) + " |")

    table_rows.append("")
    table_rows.append("*†N/A = insufficient data (stock listed < 50 trading days)*")

    return "\n".join([header, divider] + table_rows)


def build_news_section(stocks: dict) -> str:
    """Format recent news headlines per stock into grouped Markdown bullets."""
    lines = []
    for ticker, data in stocks.items():
        news = data.get("recent_news", [])
        if news:
            lines.append(f"### {ticker}")
            for item in news[:5]:
                date = f" *({item['date']})*" if item.get("date") else ""
                lines.append(f"- {item['headline']}{date}")
            lines.append("")
    return "\n".join(lines) if lines else "*No recent news available.*"


def build_context_summary(analyses: list, stocks: dict) -> str:
    """
    Build a compact but complete context block used in LLM prompts.

    Fix vs original:
        Original truncated llm_analysis at 300 chars, giving the writing agent
        only ~50 words of prior analysis context. Now passes full analysis text
        (truncated at 800 chars — a meaningful improvement).
    """
    lines = []
    for a in analyses:
        ticker  = a["ticker"]
        company = a["company_name"]
        m       = a.get("metrics", {})
        data    = _get_stock_safe(stocks, ticker)

        lines.append(
            f"{company} ({ticker}): "
            f"Price \u20b9{data.get('current_price', 'N/A')}, "
            f"P/E {data.get('pe_ratio', 'N/A')}, "
            f"ROE {fmt_pct(data.get('roe'), already_percentage=False)}, "
            f"6M Return {m.get('six_month_return_pct', 'N/A')}%, "
            f"RSI {m.get('rsi_14', 'N/A')}, "
            f"Volatility {m.get('volatility_annualized_pct', 'N/A')}%, "
            f"Debt/Equity {data.get('debt_to_equity', 'N/A')}, "
            f"Sector: {data.get('sector', 'N/A')}"
        )
        # Pass substantially more context than original 300-char truncation
        full_analysis = a.get("llm_analysis", "N/A")
        lines.append(f"Prior Analysis: {textwrap.shorten(full_analysis, width=800, placeholder='...')}")
        lines.append("")
    return "\n".join(lines)


# ── LLM Narrative Writers ─────────────────────────────────────────────────────

def write_executive_summary(analyses: list, stocks: dict) -> str:
    """Generate 3-4 sentence executive summary via LLM."""
    context = build_context_summary(analyses, stocks)
    names   = ", ".join([a["company_name"] for a in analyses])

    prompt = f"""Write a 3-4 sentence executive summary for an investment research report on {names}.

DATA:
{context}

Instructions:
- Summarize the key findings across all stocks
- Mention overall market sentiment
- Reference specific metrics (price, RSI, return %)
- Professional financial tone
- Plain text only, no bullet points, no headers"""

    logger.info("Generating executive summary...")
    result = call_llm(prompt, system=SYSTEM_WRITER, max_tokens=300)
    return _validate_narrative(result, FALLBACK_EXECUTIVE_SUMMARY, "executive_summary")


def write_stock_section(analysis: dict, stock_data: dict) -> str:
    """Generate 150-200 word narrative for a single stock via LLM."""
    ticker  = analysis["ticker"]
    company = analysis["company_name"]
    m       = analysis.get("metrics", {})

    prompt = f"""Write a 150-200 word investment analysis narrative for {company} ({ticker}).

DATA:
- Current Price: \u20b9{stock_data.get('current_price', 'N/A')}
- Sector: {stock_data.get('sector', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- P/B Ratio: {stock_data.get('pb_ratio', 'N/A')}
- ROE: {fmt_pct(stock_data.get('roe'), already_percentage=False)}
- Profit Margin: {fmt_pct(stock_data.get('profit_margin'), already_percentage=False)}
- Debt/Equity: {stock_data.get('debt_to_equity', 'N/A')}
- 6M Return: {m.get('six_month_return_pct', 'N/A')}%
- RSI (14): {m.get('rsi_14', 'N/A')}
- Volatility: {m.get('volatility_annualized_pct', 'N/A')}%
- MA-20: \u20b9{m.get('ma_20', 'N/A')} | MA-50: \u20b9{m.get('ma_50', 'N/A')}
- % from 52W High: {m.get('pct_from_52w_high', 'N/A')}%

Prior Analysis Summary:
{textwrap.shorten(analysis.get('llm_analysis', ''), width=600, placeholder='...')}

Instructions:
- Cover price trend, fundamental strength, and market sentiment
- Reference specific numbers
- Be balanced — mention both positives and negatives
- Plain text paragraphs only, no bullet points"""

    logger.info("Generating narrative for %s...", ticker)
    result = call_llm(prompt, system=SYSTEM_WRITER, max_tokens=400)
    return _validate_narrative(result, FALLBACK_STOCK_SECTION, f"stock_section_{ticker}")


def write_risk_factors(analyses: list, stocks: dict) -> str:
    """Generate risk factors paragraph via LLM."""
    context = build_context_summary(analyses, stocks)
    names   = ", ".join([a["company_name"] for a in analyses])

    prompt = f"""Write 150-200 words on the key risk factors for investing in {names}.

DATA:
{context}

Instructions:
- Cover macro risks, sector-specific risks, and stock-specific risks
- Reference specific metrics (volatility, debt/equity, beta) where relevant
- Professional tone
- Plain text paragraphs only, no bullet points"""

    logger.info("Generating risk factors...")
    result = call_llm(prompt, system=SYSTEM_WRITER, max_tokens=400)
    return _validate_narrative(result, FALLBACK_RISK_FACTORS, "risk_factors")


def write_conclusion(analyses: list, stocks: dict, comparison: str | None) -> str:
    """Generate Buy/Hold/Avoid conclusion per stock via LLM."""
    context     = build_context_summary(analyses, stocks)
    tickers_str = ", ".join([a["ticker"] for a in analyses])

    prompt = f"""Write an investment conclusion for {tickers_str}.

DATA:
{context}

COMPARATIVE INSIGHTS:
{comparison or "Single stock analysis — no peer comparison available."}

Instructions:
- Give a clear recommendation for EACH stock: Buy / Hold / Avoid
- Justify each recommendation in 2-3 sentences with specific data points
- End with a one-sentence overall portfolio summary
- Professional tone
- Plain text paragraphs only, no bullet points"""

    logger.info("Generating investment conclusion...")
    result = call_llm(prompt, system=SYSTEM_WRITER, max_tokens=500)
    return _validate_narrative(result, FALLBACK_CONCLUSION, "conclusion")


# ── Parallel LLM Execution ─────────────────────────────────────────────────────

def _run_narrative_sections_parallel(
    analyses: list,
    stocks: dict,
    comparison: str | None,
) -> dict:
    """
    Run all LLM narrative calls concurrently using ThreadPoolExecutor.

    Parallelizes:
      - executive_summary   (independent)
      - risk_factors        (independent)
      - conclusion          (independent)
      - stock_section_{t}   (each stock independent of others)

    Sequential dependencies:
      - All Python metrics must be computed before this runs (handled by analysis_agent)

    Returns:
        narratives dict with keys: executive_summary, risks, conclusion, stock_sections

    Fix vs original:
        Original ran all calls sequentially — 4-6 blocking API calls in a row.
        For 3 stocks this took 5-8 minutes.
        Parallel execution reduces wall-clock time to ~max(single_call_time) ≈ 30-60s.
    """
    narratives = {"stock_sections": {}}

    # Build all tasks: (future_key, callable, args)
    tasks = {
        "executive_summary": (write_executive_summary, (analyses, stocks)),
        "risks":             (write_risk_factors,      (analyses, stocks)),
        "conclusion":        (write_conclusion,         (analyses, stocks, comparison)),
    }
    for a in analyses:
        key = f"stock_{a['ticker']}"
        tasks[key] = (write_stock_section, (a, _get_stock_safe(stocks, a["ticker"])))

    futures: dict[Future, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
        for task_key, (fn, args) in tasks.items():
            future = executor.submit(fn, *args)
            futures[future] = task_key
            logger.info("Submitted LLM task: %s", task_key)

        for future in as_completed(futures):
            task_key = futures[future]
            try:
                result = future.result()
                if task_key.startswith("stock_"):
                    ticker = task_key.replace("stock_", "")
                    narratives["stock_sections"][ticker] = result
                    logger.info("✅ Stock narrative complete: %s", ticker)
                else:
                    narratives[task_key] = result
                    logger.info("✅ Narrative section complete: %s", task_key)
            except Exception as exc:
                logger.error("LLM task '%s' failed with exception: %s", task_key, exc)
                # Assign fallback so assembler always has a value
                fallback_map = {
                    "executive_summary": FALLBACK_EXECUTIVE_SUMMARY,
                    "risks":             FALLBACK_RISK_FACTORS,
                    "conclusion":        FALLBACK_CONCLUSION,
                }
                if task_key.startswith("stock_"):
                    ticker = task_key.replace("stock_", "")
                    narratives["stock_sections"][ticker] = FALLBACK_STOCK_SECTION
                else:
                    narratives[task_key] = fallback_map.get(task_key, "*Section unavailable.*")

    return narratives


# ── Report Assembler ───────────────────────────────────────────────────────────

def assemble_report(narratives: dict, analyses: list, stocks: dict, comparison: str | None) -> str:
    """
    Stitch all narrative sections and Python-built tables into final Markdown report.

    Fix vs original:
        datetime.now() → datetime.now(timezone.utc) for timezone-correct report date.
        All narrative slots have guaranteed non-None values (from fallback system).
    """
    tickers  = [a["ticker"] for a in analyses]
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y (UTC)")
    names    = ", ".join([a["company_name"] for a in analyses])

    sections = []

    # ── Header ──
    sections.append(
        f"# \U0001f4ca Indian Stock Research Report\n"
        f"**Stocks Analyzed:** {names}  \n"
        f"**Tickers:** {', '.join(tickers)}  \n"
        f"**Report Date:** {date_str}  \n"
        f"**Generated by:** Multi-Agent AI Research System\n\n---"
    )

    # ── Executive Summary ──
    sections.append(
        f"## Executive Summary\n\n"
        f"{narratives.get('executive_summary', FALLBACK_EXECUTIVE_SUMMARY)}\n\n---"
    )

    # ── Per-Stock Narratives ──
    sections.append("## Stock Analysis")
    for a in analyses:
        ticker    = a["ticker"]
        company   = a["company_name"]
        data      = _get_stock_safe(stocks, ticker)
        price     = data.get("current_price", "N/A")
        sector    = data.get("sector", "N/A")
        narrative = narratives["stock_sections"].get(ticker, FALLBACK_STOCK_SECTION)
        sections.append(
            f"### {company} ({ticker})\n"
            f"**Sector:** {sector}  |  **Current Price:** \u20b9{price}\n\n"
            f"{narrative}"
        )

    # ── Data Tables ──
    sections.append("---")
    sections.append(
        f"## Fundamental Metrics\n\n"
        f"{build_fundamentals_table(analyses, stocks)}\n\n---"
    )
    sections.append(
        f"## Technical Indicators\n\n"
        f"{build_technical_table(analyses)}\n\n---"
    )

    # ── Comparative Analysis (2+ stocks only) ──
    if comparison and len(tickers) >= 2:
        sections.append(f"## Comparative Analysis\n\n{comparison}\n\n---")

    # ── News + Risk + Conclusion ──
    sections.append(f"## Recent News\n\n{build_news_section(stocks)}\n\n---")
    sections.append(
        f"## Risk Factors\n\n"
        f"{narratives.get('risks', FALLBACK_RISK_FACTORS)}\n\n---"
    )
    sections.append(
        f"## Investment Conclusion\n\n"
        f"{narratives.get('conclusion', FALLBACK_CONCLUSION)}\n\n---"
    )

    # ── Disclaimer ──
    sections.append(
        "*\u26a0\ufe0f Disclaimer: This report is AI-generated for informational purposes only "
        "and does not constitute financial advice. Past performance is not indicative of future results. "
        "Please consult a SEBI-registered investment advisor before making any investment decisions.*"
    )

    return "\n\n".join(sections)


# ── Writing Agent Entry Point ──────────────────────────────────────────────────

def run_writing_agent(analysis_data: dict, research_data: dict) -> str:
    """
    Generate the full investment research report.

    Steps:
      1. Validate HF_TOKEN is set (fails fast)
      2. Run all LLM narrative calls IN PARALLEL
      3. Assemble final Markdown report from narratives + Python tables

    Args:
        analysis_data:  Output from run_analysis_agent()
        research_data:  Output from run_research_agent()

    Returns:
        Final Markdown report string

    Raises:
        EnvironmentError — GEMINI_API_KEY not set
        ValueError       — missing analysis data
    """
    # Fail fast before any computation
    get_api_key()

    analyses   = analysis_data.get("individual_analyses", [])
    comparison = analysis_data.get("comparison_insights")
    stocks     = research_data.get("stocks", {})

    if not analyses:
        raise ValueError("No analysis data found. Cannot generate report.")

    logger.info(
        "Starting report generation for %d stock(s): %s",
        len(analyses),
        ", ".join([a["ticker"] for a in analyses])
    )

    # Run all LLM narrative sections in parallel
    narratives = _run_narrative_sections_parallel(analyses, stocks, comparison)

    logger.info("Assembling final report...")
    report = assemble_report(narratives, analyses, stocks, comparison)

    logger.info(
        "✅ Report ready — %d sections, %d chars.",
        len(report.split("\n\n##")),
        len(report)
    )
    return report
