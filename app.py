"""
app.py — Gradio UI entry point for the TA Report Generator.

Responsibility (plan.md §4.5):
    Present a two-tab interface:
        Tab 1 — "📄 TA Investment Report"  (Gemini-generated Markdown report)
        Tab 2 — "📊 Computed Data & Charts" (Plotly charts + indicator DataFrames)

    A persistent left-side panel holds the Gemini API key input and model info.

Layout reference: plan.md §4.5.1
Security:         plan.md §9  — API key stored in Gradio ``State`` only.
Progress:         plan.md §4.5 — ``gr.Progress`` provides stage-level feedback.

Usage::

    python app.py
    # or
    gradio app.py
"""

from __future__ import annotations

import logging
import sys

import gradio as gr

from config.settings import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    GEMINI_MODEL_NAME,
    GRADIO_THEME,
    INTERVAL_CHOICES,
    INTERVAL_LABELS,
    PERIOD_CHOICES,
    PERIOD_LABELS,
    REPORT_TAIL_ROWS,
)
from main_pipeline import PipelineResult, run_pipeline
from utils.validators import validate_api_key

# ============================================================================
# Logging configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ============================================================================
# Gradio callback helpers
# ============================================================================

def _save_api_key(api_key: str) -> tuple[str, str]:
    """Validate and acknowledge the API key save.

    Returns
    -------
    tuple[str, str]
        (status_markdown, validated_key_for_state)
    """
    result = validate_api_key(api_key)
    if result.is_valid:
        return "🟢 API key saved", api_key.strip()
    return f"🔴 {result.message}", ""


def _run_analysis(
    ticker: str,
    api_key_state: str,
    period: str,
    interval: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple:
    """Execute the full pipeline and unpack results for Gradio outputs."""
    if not api_key_state:
        error_msg = "⚠️ **Please save a valid Gemini API key** in the left panel before analysing."
        empty = None
        return (
            error_msg, error_msg,  # status_msg, report_output
            empty,                 # ohlcv_table
            empty, empty, empty, empty,  # patterns, indicators, sr, fib
            empty, empty, empty, empty,  # price, volume, rsi, macd
        )

    # --- Wrap pipeline with Gradio progress ---
    def _progress_cb(frac: float, desc: str) -> None:
        progress(frac, desc=desc)

    try:
        result: PipelineResult = run_pipeline(
            ticker=ticker,
            api_key=api_key_state,
            period=period,
            interval=interval,
            progress_cb=_progress_cb,
        )
    except (ValueError, ConnectionError, RuntimeError) as exc:
        error_md = f"❌ **Error:** {exc}"
        empty_fig = None
        empty_df = None
        return (
            error_md,
            error_md,
            empty_fig, empty_fig, empty_fig, empty_fig,
            empty_df, empty_df, empty_df, empty_df,
        )

    # --- Build status banner ---
    checklist = result.checklist
    score = checklist["score"]
    max_score = checklist["max_score"]
    interval_label = INTERVAL_LABELS.get(interval, interval)
    period_label   = PERIOD_LABELS.get(period, period)
    status_md = (
        f"✅ Analysis complete for **{result.ticker}** "
        f"— Grand Checklist: **{score}/{max_score}** "
        f"— {interval_label} / {period_label} "
        f"— {result.elapsed_seconds:.1f}s"
    )

    return (
        status_md,
        result.report_md,
        # ── Section 1: Raw OHLCV ──
        result.ohlcv_df,
        # ── Section 2: Computed data ──
        result.patterns_df,
        result.indicators_df,
        result.sr_levels_df,
        result.fib_levels_df,
        # ── Section 3: Charts ──
        result.charts.get("price", None),
        result.charts.get("volume", None),
        result.charts.get("rsi", None),
        result.charts.get("macd", None),
    )


# ============================================================================
# Gradio UI layout (plan.md §4.5.1)
# ============================================================================

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    theme = getattr(gr.themes, GRADIO_THEME, gr.themes.Soft)()

    with gr.Blocks(
        theme=theme,
        title="Technical Analysis Report Generator",
        css=_custom_css(),
    ) as demo:

        # ── Session state (API key — never written to disk, plan.md §9) ────
        api_key_state = gr.State(value="")

        with gr.Row():

            # ════════════════════════════════════════════════════════════════
            # LEFT PANEL — Configuration
            # ════════════════════════════════════════════════════════════════
            with gr.Column(scale=1, min_width=240, elem_id="config-panel"):
                gr.Markdown("### ⚙️ Configuration")

                api_key_input = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    placeholder="AIza…",
                    elem_id="api-key-input",
                )
                save_btn = gr.Button(
                    "💾 Save Key",
                    variant="secondary",
                    size="sm",
                    elem_id="save-key-btn",
                )
                api_status = gr.Markdown("🔴 No key saved", elem_id="api-status")

                gr.Markdown("---")

                # ── Data Configuration ──────────────────────────────────
                gr.Markdown("### 📊 Data Settings")

                period_input = gr.Dropdown(
                    choices=[(PERIOD_LABELS[p], p) for p in PERIOD_CHOICES],
                    value=DEFAULT_PERIOD,
                    label="📅 Time Period",
                    info="How far back to fetch historical data",
                    elem_id="period-dropdown",
                    interactive=True,
                )
                interval_input = gr.Dropdown(
                    choices=[(INTERVAL_LABELS[i], i) for i in INTERVAL_CHOICES],
                    value=DEFAULT_INTERVAL,
                    label="⏱️ Data Frequency",
                    info="Candle size / bar interval",
                    elem_id="interval-dropdown",
                    interactive=True,
                )

                gr.Markdown("---")
                gr.Markdown(f"**Model:** `{GEMINI_MODEL_NAME}`")
                gr.Markdown(f"**Context:** Last {REPORT_TAIL_ROWS} bars → LLM")

                # Wire save-key callback
                save_btn.click(
                    fn=_save_api_key,
                    inputs=[api_key_input],
                    outputs=[api_status, api_key_state],
                )

            # ════════════════════════════════════════════════════════════════
            # MAIN CONTENT AREA
            # ════════════════════════════════════════════════════════════════
            with gr.Column(scale=4):

                # ── Header + ticker input ──────────────────────────────────
                gr.Markdown(
                    "## 📈 Technical Analysis Report Generator\n"
                    "Enter a ticker symbol and click **Analyze** to generate a "
                    "comprehensive weekly TA report powered by Gemini."
                )

                with gr.Row():
                    ticker_input = gr.Textbox(
                        placeholder="Enter ticker symbol (e.g. AAPL, RELIANCE.NS, ^NSEI)",
                        label="Ticker Symbol",
                        scale=4,
                        elem_id="ticker-input",
                    )
                    submit_btn = gr.Button(
                        "🔍 Analyze",
                        variant="primary",
                        scale=1,
                        elem_id="analyze-btn",
                    )

                status_msg = gr.Markdown(
                    value="",
                    elem_id="status-msg",
                )

                # ── Tabbed output ──────────────────────────────────────────
                with gr.Tabs():

                    # ─── Tab 1: TA Report ──────────────────────────────────
                    with gr.Tab("📄 TA Investment Report", id="tab-report"):
                        report_output = gr.Markdown(
                            value=(
                                "*Your report will appear here after analysis. "
                                "Enter a ticker and click* **🔍 Analyze** *to begin.*"
                            ),
                            elem_id="report-output",
                        )

                    # ─── Tab 2: Charts & Data ──────────────────────────────
                    with gr.Tab("📊 Computed Data & Charts", id="tab-data"):

                        gr.Markdown(
                            "*Click a section header to expand or collapse it.*",
                            elem_id="tab2-hint",
                        )

                        # ════ Section 1: Raw Market Data ════════════════════
                        with gr.Accordion(
                            label="📥  Section 1 — Raw Market Data (OHLCV from yFinance)",
                            open=True,
                            elem_id="acc-raw",
                        ):
                            gr.Markdown(
                                "The unmodified price/volume data downloaded directly "
                                "from Yahoo Finance. No indicators applied."
                            )
                            ohlcv_table = gr.Dataframe(
                                label="Raw OHLCV Data (all fetched rows)",
                                interactive=False,
                                wrap=False,
                                elem_id="table-ohlcv",
                            )

                        # ════ Section 2: Computed Indicators & Patterns ══════
                        with gr.Accordion(
                            label="🔢  Section 2 — Computed Technical Indicators & Patterns",
                            open=False,
                            elem_id="acc-computed",
                        ):
                            gr.Markdown("### 🕯️ Detected Candlestick Patterns")
                            patterns_table = gr.Dataframe(
                                label="Candlestick Patterns (Last 5 Bars)",
                                interactive=False,
                                elem_id="table-patterns",
                            )

                            gr.Markdown("### 📋 Full Indicator Data")
                            indicators_table = gr.Dataframe(
                                label=f"Indicator Data (Last {REPORT_TAIL_ROWS} Bars)",
                                interactive=False,
                                wrap=True,
                                elem_id="table-indicators",
                            )

                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("### 🛡️ Support & Resistance")
                                    sr_table = gr.Dataframe(
                                        label="S&R Levels",
                                        interactive=False,
                                        elem_id="table-sr",
                                    )
                                with gr.Column():
                                    gr.Markdown("### 📐 Fibonacci Retracement")
                                    fib_table = gr.Dataframe(
                                        label="Fibonacci Levels",
                                        interactive=False,
                                        elem_id="table-fib",
                                    )

                        # ════ Section 3: Interactive Charts ═════════════════
                        with gr.Accordion(
                            label="📊  Section 3 — Interactive Charts",
                            open=False,
                            elem_id="acc-charts",
                        ):
                            with gr.Row():
                                price_chart = gr.Plot(
                                    label="Price + EMA + Bollinger Bands",
                                    elem_id="chart-price",
                                )
                                volume_chart = gr.Plot(
                                    label="Volume Analysis",
                                    elem_id="chart-volume",
                                )
                            with gr.Row():
                                rsi_chart = gr.Plot(
                                    label="RSI (14)",
                                    elem_id="chart-rsi",
                                )
                                macd_chart = gr.Plot(
                                    label="MACD (12, 26, 9)",
                                    elem_id="chart-macd",
                                )

                # ── Wire the Analyze button ────────────────────────────────
                _outputs = [
                    status_msg,
                    report_output,
                    # Section 1
                    ohlcv_table,
                    # Section 2
                    patterns_table,
                    indicators_table,
                    sr_table,
                    fib_table,
                    # Section 3
                    price_chart,
                    volume_chart,
                    rsi_chart,
                    macd_chart,
                ]

                submit_btn.click(
                    fn=_run_analysis,
                    inputs=[ticker_input, api_key_state, period_input, interval_input],
                    outputs=_outputs,
                )

                # Also allow Enter key in the ticker textbox to trigger analysis.
                ticker_input.submit(
                    fn=_run_analysis,
                    inputs=[ticker_input, api_key_state, period_input, interval_input],
                    outputs=_outputs,
                )

    return demo


# ============================================================================
# Custom CSS
# ============================================================================

def _custom_css() -> str:
    """Return custom CSS for minor layout polish."""
    return """
    #config-panel {
        border-right: 1px solid var(--border-color-primary);
        padding-right: 16px;
    }
    #report-output {
        min-height: 400px;
        padding: 16px;
    }
    #status-msg {
        margin-top: 8px;
        margin-bottom: 8px;
    }
    #analyze-btn {
        min-height: 42px;
    }
    """


# ============================================================================
# Entrypoint
# ============================================================================

def main() -> None:
    """Launch the Gradio server."""
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
