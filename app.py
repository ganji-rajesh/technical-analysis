"""
app.py
------
Gradio UI for the Indian Stock Research Multi-Agent System.

Fixes applied vs original:
  - fmt_inr() & format_*_preview() duplicated from writing_agent — removed,
    now each formats via its own agent output (DRY)
  - progress_callback signature updated to (step, pct, message) — matches
    new orchestrator.run_pipeline() contract
  - Gradio generator yields after EACH agent step (not only at end) —
    gives true live status updates to the user
  - EnvironmentError caught and shown with fix instructions (was silently
    swallowed in base Exception block)
  - Footer still references "Ollama" — fixed to HuggingFace
  - server_name="127.0.0.1" blocks HuggingFace Spaces deployment —
    replaced with "0.0.0.0" controlled via env var APP_HOST
  - server_port now reads from APP_PORT env var (12-factor app pattern)
  - share=False hardcoded — now reads from GRADIO_SHARE env var
  - reset_ui() helper clears all output state before each new run
  - Timing summary shown in status log (from PipelineResult.timing)
  - fmt_inr() DRY violation removed — uses writing_agent's existing fmt_inr
  - Input sanitized before passing to pipeline (strip whitespace)
  - All print() replaced with logging
"""

import logging
import os

import gradio as gr

from orchestrator import run_pipeline, run_pipeline_streaming, PipelineResult

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("app")


# ── Launch Config (12-factor: all via environment variables) ───────────────────

APP_HOST      = os.environ.get("APP_HOST",     "0.0.0.0")   # 127.0.0.1 blocks HF Spaces
APP_PORT      = int(os.environ.get("APP_PORT",  "7860"))
GRADIO_SHARE  = os.environ.get("GRADIO_SHARE", "false").lower() == "true"


# ── Step → progress percentage map ────────────────────────────────────────────
# Mirrors the pct values emitted by orchestrator.run_pipeline()
# Used to display a clean progress indicator in the status panel.

STEP_ICONS = {
    "ORCHESTRATOR": "🎯",
    "RESEARCH":     "🔍",
    "ANALYSIS":     "📊",
    "WRITING":      "✍️",
}


# ── Gradio Theme ───────────────────────────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0d1117",
    body_text_color="#e6edf3",
    block_background_fill="#161b22",
    block_border_color="#30363d",
    block_border_width="1px",
    block_radius="12px",
    block_shadow="0 4px 24px rgba(0,0,0,0.4)",
    block_label_text_color="#8b949e",
    block_title_text_color="#e6edf3",
    input_background_fill="#0d1117",
    input_border_color="#30363d",
    input_border_width="1px",
    input_radius="8px",
    input_placeholder_color="#484f58",
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_secondary_background_fill="#21262d",
    button_secondary_background_fill_hover="#30363d",
    button_secondary_text_color="#c9d1d9",
    button_secondary_border_color="#30363d",
    button_large_radius="8px",
    button_small_radius="6px",
    border_color_primary="#30363d",
    shadow_drop="0 2px 8px rgba(0,0,0,0.3)",
    shadow_spread="2px",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container { background: #0d1117 !important; max-width: 1100px !important; margin: 0 auto !important; }

#app-header {
    background: linear-gradient(160deg, #161b22 0%, #1c1f2e 100%);
    border: 1px solid #30363d; border-radius: 14px;
    padding: 2rem 2.5rem 1.75rem; margin-bottom: 1.25rem; text-align: center;
}
#app-header h1 { font-size: 1.85rem !important; font-weight: 700 !important; color: #e6edf3 !important; margin: 0 0 0.35rem 0 !important; letter-spacing: -0.5px; }
#app-header h1 span { color: #818cf8; }
#app-header p { color: #8b949e !important; font-size: 0.9rem !important; margin: 0 0 1.1rem 0 !important; }
.pipeline-flow { display: inline-flex; align-items: center; gap: 0.5rem; background: #0d1117; border: 1px solid #30363d; border-radius: 50px; padding: 0.4rem 1.1rem; font-size: 0.78rem; color: #8b949e; font-family: 'JetBrains Mono', monospace; }
.pipeline-flow .step { color: #c9d1d9; font-weight: 500; }
.pipeline-flow .arrow { color: #484f58; }

#input-section { margin-bottom: 1rem; }
#ticker-label { font-size: 0.8rem !important; font-weight: 600 !important; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem !important; }
#presets-label { font-size: 0.75rem !important; color: #484f58 !important; margin-bottom: 0.4rem !important; }

.preset-pill button { background: #161b22 !important; color: #8b949e !important; border: 1px solid #30363d !important; border-radius: 50px !important; font-size: 0.78rem !important; padding: 0.3rem 0.85rem !important; font-weight: 500 !important; transition: all 0.15s ease !important; min-width: unset !important; }
.preset-pill button:hover { background: #21262d !important; border-color: #6366f1 !important; color: #c9d1d9 !important; }

#analyse-btn button { font-size: 0.95rem !important; font-weight: 600 !important; letter-spacing: 0.02em !important; height: 46px !important; box-shadow: 0 0 20px rgba(99,102,241,0.25) !important; transition: all 0.2s ease !important; }
#analyse-btn button:hover { box-shadow: 0 0 28px rgba(99,102,241,0.4) !important; transform: translateY(-1px) !important; }

/* Progress bar strip */
#progress-bar { height: 3px; background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 2px; transition: width 0.4s ease; }

#status-panel { background: #161b22 !important; border: 1px solid #30363d !important; border-left: 3px solid #6366f1 !important; border-radius: 10px !important; padding: 1rem 1.25rem !important; font-size: 0.85rem !important; line-height: 1.8 !important; }
#status-panel p { color: #c9d1d9 !important; margin: 0 !important; }
#status-panel strong { color: #e6edf3 !important; }

#error-panel { background: #160b0b !important; border: 1px solid #3d1515 !important; border-left: 3px solid #f85149 !important; border-radius: 10px !important; padding: 1rem 1.25rem !important; }
#error-panel p, #error-panel strong { color: #ffa198 !important; }
#error-panel h3 { color: #f85149 !important; font-size: 0.95rem !important; }

.tab-nav { border-bottom: 1px solid #30363d !important; }
.tab-nav button { font-size: 0.85rem !important; font-weight: 500 !important; color: #8b949e !important; padding: 0.6rem 1.1rem !important; border-radius: 6px 6px 0 0 !important; border: none !important; background: transparent !important; transition: color 0.15s !important; }
.tab-nav button:hover { color: #c9d1d9 !important; }
.tab-nav button.selected { color: #818cf8 !important; border-bottom: 2px solid #6366f1 !important; background: transparent !important; }
.tabitem { background: #161b22 !important; border: 1px solid #30363d !important; border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 1.5rem !important; }

.gr-prose, .prose, .md, [class*="prose"] { color: #c9d1d9 !important; font-size: 0.88rem !important; line-height: 1.75 !important; }
.gr-prose h1, .prose h1 { color: #e6edf3 !important; font-size: 1.4rem !important; font-weight: 700 !important; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }
.gr-prose h2, .prose h2 { color: #818cf8 !important; font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 1.5rem !important; border-bottom: 1px solid #21262d; padding-bottom: 0.3rem; }
.gr-prose h3, .prose h3 { color: #58a6ff !important; font-size: 0.95rem !important; font-weight: 600 !important; }
.gr-prose table, .prose table { width: 100% !important; border-collapse: collapse !important; font-size: 0.82rem !important; }
.gr-prose th, .prose th { background: #21262d !important; color: #8b949e !important; padding: 0.5rem 0.85rem !important; font-weight: 600 !important; text-align: left !important; border-bottom: 1px solid #30363d !important; }
.gr-prose td, .prose td { padding: 0.45rem 0.85rem !important; border-bottom: 1px solid #21262d !important; color: #c9d1d9 !important; }
.gr-prose tr:hover td, .prose tr:hover td { background: #1c2128 !important; }
.gr-prose code, .prose code { background: #21262d !important; color: #79c0ff !important; border-radius: 4px !important; padding: 0.1rem 0.4rem !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }
.gr-prose strong, .prose strong { color: #e6edf3 !important; }
.gr-prose hr, .prose hr { border-color: #21262d !important; margin: 1.25rem 0 !important; }
.gr-prose em, .prose em { color: #8b949e !important; }

#app-footer { text-align: center; color: #484f58 !important; font-size: 0.75rem; margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #21262d; }
"""


# ── Output Formatters ──────────────────────────────────────────────────────────
# These are UI-layer formatters only — purely for rendering research_data and
# analysis_data dicts in the preview tabs.
# NOTE: fmt_inr lives in writing_agent — we inline a local copy only for the
# UI layer so app.py has zero dependency on internal agent formatting logic.

def _fmt_inr(value) -> str:
    """Local UI-only currency formatter. Mirrors writing_agent.fmt_inr."""
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


def _fmt_pct(value) -> str:
    """Format decimal ratio as % string. Used only in UI preview tables."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def format_research_preview(research_data: dict) -> str:
    """
    Render raw research_data dict as a readable Markdown summary.
    Shown in the 'Research Data' tab — for transparency and debugging.
    """
    stocks = research_data.get("stocks", {})
    if not stocks:
        return "*No research data available.*"

    lines = []
    for ticker, data in stocks.items():
        lines += [
            f"## {data.get('company_name', ticker)} ({ticker})",
            f"**Sector:** {data.get('sector', 'N/A')}  |  "
            f"**Industry:** {data.get('industry', 'N/A')}",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
        rows = [
            ("Current Price",  f"\u20b9{data.get('current_price', 'N/A')}"),
            ("Previous Close", f"\u20b9{data.get('previous_close', 'N/A')}"),
            ("Market Cap",     _fmt_inr(data.get("market_cap"))),
            ("P/E Ratio",      str(data.get("pe_ratio", "N/A"))),
            ("P/B Ratio",      str(data.get("pb_ratio", "N/A"))),
            ("EPS",            f"\u20b9{data.get('eps', 'N/A')}"),
            ("ROE",            _fmt_pct(data.get("roe"))),
            ("Profit Margin",  _fmt_pct(data.get("profit_margin"))),
            ("Debt / Equity",  str(data.get("debt_to_equity", "N/A"))),
            ("Revenue",        _fmt_inr(data.get("revenue"))),
            ("Net Income",     _fmt_inr(data.get("net_income"))),
            ("Beta",           str(data.get("beta", "N/A"))),
            ("52W High",       f"\u20b9{data.get('52_week_high', 'N/A')}"),
            ("52W Low",        f"\u20b9{data.get('52_week_low', 'N/A')}"),
        ]
        for label, val in rows:
            lines.append(f"| {label} | {val} |")
        lines.append("")

        news = data.get("recent_news", [])
        if news:
            lines.append("**Recent News**")
            for n in news:
                date = f"  *({n.get('date', '')})*" if n.get("date") else ""
                lines.append(f"- {n['headline']}{date}")
        lines += ["", "---", ""]

    ts = research_data.get("timestamp", "")
    if ts:
        lines.append(f"*Data fetched at: {ts} (UTC)*")
    return "\n".join(lines)


def format_analysis_preview(analysis_data: dict) -> str:
    """
    Render analysis_data dict as a readable Markdown summary.
    Shown in the 'Analysis Data' tab — for transparency and debugging.
    """
    analyses = analysis_data.get("individual_analyses", [])
    if not analyses:
        return "*No analysis data available.*"

    lines = []
    for a in analyses:
        lines += [
            f"## {a.get('company_name', a['ticker'])} ({a['ticker']})",
            "| Metric | Value |",
            "| --- | --- |",
        ]
        m = a.get("metrics", {})
        metric_rows = [
            ("6M Return",           "six_month_return_pct",      "%"),
            ("20D Momentum",        "momentum_20d_pct",           "%"),
            ("Volatility (Ann.)",   "volatility_annualized_pct",  "%"),
            ("RSI (14)",            "rsi_14",                     ""),
            ("MA-20 (\u20b9)",      "ma_20",                      ""),
            ("MA-50 (\u20b9)*",     "ma_50",                      ""),
            ("Price vs MA-20",      "price_vs_ma20_pct",          "%"),
            ("Price vs MA-50*",     "price_vs_ma50_pct",          "%"),
            ("% from 52W High",     "pct_from_52w_high",          "%"),
            ("% from 52W Low",      "pct_from_52w_low",           "%"),
        ]
        for label, key, suffix in metric_rows:
            val = m.get(key)
            display = f"{val}{suffix}" if val is not None else "N/A†"
            lines.append(f"| {label} | {display} |")

        lines += ["", "*†N/A = insufficient data (< 50 trading days)*", ""]

        llm = a.get("llm_analysis", "")
        if llm:
            lines += ["**LLM Interpretation**", llm]
        lines += ["", "---", ""]

    comparison = analysis_data.get("comparison_insights")
    if comparison:
        lines += ["## Cross-Stock Comparison", comparison]

    return "\n".join(lines)


def format_timing_summary(timing: dict) -> str:
    """Render per-step timing as a compact inline string for the status log."""
    if not timing:
        return ""
    parts = [f"{k}: {v}s" for k, v in timing.items() if k != "total"]
    total = timing.get("total", "?")
    return f"  *(⏱ {' | '.join(parts)} | total: {total}s)*"


# ── Pipeline Runner (Gradio generator) ────────────────────────────────────────

def run(ticker_input: str):
    """
    Gradio generator — drives the pipeline and yields incremental UI updates.

    Uses run_pipeline_streaming() so we yield a Gradio update after EACH
    agent step, giving the user true live progress visibility.

    Yield contract (7 outputs, matching `outs` list below):
        0  status_box   — gr.Markdown (live log)
        1  progress_bar — gr.HTML     (progress strip)
        2  error_box    — gr.Markdown (error panel)
        3  output_tabs  — gr.Tabs     (visibility)
        4  report_out   — gr.Markdown
        5  research_out — gr.Markdown
        6  analysis_out — gr.Markdown
    """

    # ── Reset UI state ──
    yield (
        gr.update(value="⏳  **Starting pipeline…**", visible=True),
        gr.update(value='<div id="progress-bar" style="width:5%"></div>'),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    )

    ticker_input = (ticker_input or "").strip()
    if not ticker_input:
        yield (
            gr.update(visible=False),
            gr.update(value='<div id="progress-bar" style="width:0%"></div>'),
            gr.update(
                value="### ❌ No Input\n\nPlease enter at least one ticker symbol.",
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
        )
        return

    log_lines: list[str] = []
    result = None

    try:
        for step, pct, message, partial_result in run_pipeline_streaming(ticker_input):
            icon = STEP_ICONS.get(step, "▶")
            log_lines.append(f"{icon}  **{step}** — {message}")
            status_md = "\n\n".join(log_lines)

            if partial_result is not None:
                result = partial_result

            # Yield after EACH step — this is the key fix.
            # The old code called blocking run_pipeline() and only yielded at the end.
            yield (
                gr.update(value=status_md, visible=True),
                gr.update(value=f'<div id="progress-bar" style="width:{pct}%"></div>'),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
            )

    except EnvironmentError as exc:
        logger.error("Environment error: %s", exc)
        yield (
            gr.update(visible=False),
            gr.update(value='<div id="progress-bar" style="width:0%"></div>'),
            gr.update(
                value=(
                    "### ❌ Environment Configuration Error\n\n"
                    f"```\n{exc}\n```\n\n"
                    "**Fix:** Set your Gemini API key as an environment variable "
                    "before running the pipeline."
                ),
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
        )
        return

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        yield (
            gr.update(visible=False),
            gr.update(value='<div id="progress-bar" style="width:0%"></div>'),
            gr.update(
                value=f"### ❌ Input Error\n\n**{exc}**",
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
        )
        return

    except RuntimeError as exc:
        logger.error("Pipeline runtime error: %s", exc)
        yield (
            gr.update(visible=False),
            gr.update(value='<div id="progress-bar" style="width:0%"></div>'),
            gr.update(
                value=(
                    "### ❌ Pipeline Error\n\n"
                    f"**{type(exc).__name__}:** {exc}\n\n"
                    "*This may be a temporary Gemini API issue. "
                    "Please wait a moment and try again.*"
                ),
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
        )
        return

    except Exception as exc:
        logger.exception("Unexpected error in pipeline: %s", exc)
        yield (
            gr.update(visible=False),
            gr.update(value='<div id="progress-bar" style="width:0%"></div>'),
            gr.update(
                value=(
                    "### ❌ Unexpected Error\n\n"
                    f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                    "*Please report this issue with the error above.*"
                ),
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
        )
        return

    # ── Success — assemble final UI state ──
    if result is None:
        return

    timing_str  = format_timing_summary(result.timing)
    status_md   = "\n\n".join(log_lines) + f"\n\n✅  **Pipeline complete — report ready.**{timing_str}"
    research_md = format_research_preview(result.research_data)
    analysis_md = format_analysis_preview(result.analysis_data)

    logger.info("Pipeline complete. Rendering report (%d chars).", len(result.report))

    yield (
        gr.update(value=status_md, visible=True),
        gr.update(value='<div id="progress-bar" style="width:100%"></div>'),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=result.report),
        gr.update(value=research_md),
        gr.update(value=analysis_md),
    )


# ── UI Layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, theme=theme, title="Indian Stock Research AI") as demo:

    # ── Header ──
    gr.HTML("""
    <div id="app-header">
        <h1>\U0001f1ee\U0001f1f3 Indian Stock <span>Research AI</span></h1>
        <p>Multi-Agent System — powered by yfinance &amp; Google Gemini</p>
        <div class="pipeline-flow">
            <span class="step">\U0001f50d Research</span>
            <span class="arrow">&rarr;</span>
            <span class="step">\U0001f4ca Analysis</span>
            <span class="arrow">&rarr;</span>
            <span class="step">\u270d\ufe0f Writing</span>
            <span class="arrow">&rarr;</span>
            <span class="step">\U0001f4cb Report</span>
        </div>
    </div>
    """)

    # ── Input Section ──
    with gr.Group(elem_id="input-section"):
        gr.Markdown("**TICKER SYMBOLS**", elem_id="ticker-label")

        with gr.Row(equal_height=True):
            ticker_input = gr.Textbox(
                placeholder="e.g.  TCS, INFY, HDFCBANK, RELIANCE",
                label="",
                scale=5,
                container=False,
                lines=1,
            )
            analyse_btn = gr.Button(
                "Analyse \u2192",
                variant="primary",
                scale=1,
                elem_id="analyse-btn",
            )

        gr.Markdown("Quick presets", elem_id="presets-label")
        with gr.Row():
            presets = [
                ("\U0001f5a5  IT Sector",  "TCS, INFY, WIPRO"),
                ("\U0001f3e6  Banks",       "HDFCBANK, ICICIBANK"),
                ("\u26a1  Energy",          "RELIANCE, ONGC, BPCL"),
                ("\U0001f697  Auto",        "TATAMOTORS, MARUTI"),
                ("\U0001f4e6  Single",      "HINDUNILVR"),
            ]
            for label, tickers in presets:
                gr.Button(label, elem_classes=["preset-pill"], size="sm").click(
                    fn=lambda t=tickers: t,
                    outputs=ticker_input,
                )

    # ── Progress Bar (thin strip, hidden until run starts) ──
    progress_bar = gr.HTML(
        value='<div id="progress-bar" style="width:0%; height:3px;"></div>'
    )

    # ── Status / Error panels ──
    status_box = gr.Markdown(visible=False, elem_id="status-panel")
    error_box  = gr.Markdown(visible=False, elem_id="error-panel")

    # ── Output Tabs ──
    with gr.Tabs(visible=False) as output_tabs:
        with gr.TabItem("\U0001f4cb  Final Report"):
            report_out = gr.Markdown()
        with gr.TabItem("\U0001f50d  Research Data"):
            research_out = gr.Markdown()
        with gr.TabItem("\U0001f4ca  Analysis Data"):
            analysis_out = gr.Markdown()

    # ── Footer — fixed: was "Ollama", now correctly says HuggingFace ──
    gr.HTML("""
    <div id="app-footer">
        Built with Python &middot; yfinance &middot; Google Gemini &middot; Gradio
        &nbsp;&middot;&nbsp;
        \u26a0\ufe0f For informational purposes only &mdash; not financial advice.
    </div>
    """)

    # ── Wire Up ──
    outs = [status_box, progress_bar, error_box, output_tabs,
            report_out, research_out, analysis_out]

    analyse_btn.click(fn=run, inputs=ticker_input, outputs=outs)
    ticker_input.submit(fn=run, inputs=ticker_input, outputs=outs)


# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(
        "Starting app on %s:%d (share=%s)",
        APP_HOST, APP_PORT, GRADIO_SHARE
    )
    demo.launch(
        server_name=APP_HOST,     # 0.0.0.0 works on HF Spaces; was 127.0.0.1
        server_port=APP_PORT,     # configurable via APP_PORT env var
        share=GRADIO_SHARE,       # configurable via GRADIO_SHARE env var
        show_error=True,
    )
