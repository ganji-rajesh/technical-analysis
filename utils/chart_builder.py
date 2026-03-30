"""
chart_builder.py — Plotly chart generation for the "Computed Data & Charts" tab.

Each public function returns a ``plotly.graph_objects.Figure`` that Gradio's
``gr.Plot`` component can render directly.  All styling constants are imported
from ``config.settings`` so visual tweaks stay centralised.

Design decisions
----------------
* **Candlestick + overlay chart** — combines OHLC candles, EMA lines, and
  Bollinger Bands into a single figure with Plotly subplots so the user
  sees price action *in context* rather than toggling between charts.
* **Separate RSI / MACD panels** — momentum oscillators have their own Y-axis
  scale and reference lines, making a standalone figure more readable.
* **Volume bar chart** — colour-coded green/red bars (close > open → green)
  with a 10-period SMA overlay for quick visual confirmation of volume spikes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import (
    ADX_RANGING_THRESHOLD,
    ADX_TRENDING_THRESHOLD,
    CHART_HEIGHT,
    CHART_TEMPLATE,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    VOLUME_SMA_PERIOD,
)

if TYPE_CHECKING:
    import pandas as pd


# ============================================================================
# Internal helpers
# ============================================================================

def _apply_base_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply shared layout defaults to any figure."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
    return fig


def _date_index(df: pd.DataFrame) -> pd.Index:
    """Return the datetime index regardless of whether it is named 'Date'."""
    if df.index.name and "date" in df.index.name.lower():
        return df.index
    if "Date" in df.columns:
        return df["Date"]
    return df.index


# ============================================================================
# 1. Price + EMA + Bollinger Bands (candlestick overlay)
# ============================================================================

def build_price_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Candlestick chart overlaid with EMA lines and Bollinger Bands.

    Expected DataFrame columns (appended by ``ta_compute``):
        ``Open``, ``High``, ``Low``, ``Close``,
        ``EMA_20``, ``EMA_50``, ``EMA_100``,
        ``BBL_20_2.0``, ``BBM_20_2.0``, ``BBU_20_2.0``

    Parameters
    ----------
    df:
        Enriched OHLCV DataFrame from the computation layer.
    ticker:
        Ticker symbol for the chart title.

    Returns
    -------
    go.Figure
    """
    dates = _date_index(df)

    fig = go.Figure()

    # --- Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # --- EMA overlays ---
    ema_styles: list[tuple[str, str, float]] = [
        ("EMA_20", "#ffeb3b", 1.2),   # Yellow, thin
        ("EMA_50", "#ff9800", 1.5),    # Orange, medium
        ("EMA_100", "#e91e63", 1.8),   # Pink, thick
    ]
    for col, colour, width in ema_styles:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=colour, width=width),
                )
            )

    # --- Bollinger Bands (shaded region) ---
    bbl_col = "BBL_20_2.0"
    bbu_col = "BBU_20_2.0"
    bbm_col = "BBM_20_2.0"
    if bbu_col in df.columns and bbl_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[bbu_col],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[bbl_col],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.15)",
                name="Bollinger Band",
            )
        )
    if bbm_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[bbm_col],
                mode="lines",
                name="BB Mid",
                line=dict(color="rgba(173,216,230,0.6)", width=1, dash="dot"),
            )
        )

    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_base_layout(fig, f"{ticker} — Price · EMA · Bollinger Bands")


# ============================================================================
# 2. Volume Analysis
# ============================================================================

def build_volume_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Bar chart of weekly volume with a 10-period SMA overlay.

    Bars are coloured green when Close ≥ Open and red otherwise.

    Parameters
    ----------
    df:
        Enriched OHLCV DataFrame.
    ticker:
        Ticker symbol for the chart title.

    Returns
    -------
    go.Figure
    """
    dates = _date_index(df)

    bar_colours = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["Close"], df["Open"])
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=dates,
            y=df["Volume"],
            marker_color=bar_colours,
            name="Volume",
            opacity=0.7,
        )
    )

    # Volume SMA overlay
    vol_sma_col = f"VOL_SMA_{VOLUME_SMA_PERIOD}"
    if vol_sma_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[vol_sma_col],
                mode="lines",
                name=f"Vol SMA({VOLUME_SMA_PERIOD})",
                line=dict(color="#ffa726", width=2),
            )
        )

    fig.update_layout(yaxis_title="Volume")
    return _apply_base_layout(fig, f"{ticker} — Volume Analysis")


# ============================================================================
# 3. RSI
# ============================================================================

def build_rsi_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """RSI line chart with overbought / oversold reference bands.

    Parameters
    ----------
    df:
        Must contain ``RSI_14`` column.
    ticker:
        Ticker symbol for the chart title.

    Returns
    -------
    go.Figure
    """
    dates = _date_index(df)
    rsi_col = "RSI_14"

    fig = go.Figure()

    if rsi_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[rsi_col],
                mode="lines",
                name="RSI(14)",
                line=dict(color="#7c4dff", width=2),
            )
        )

    # Overbought / oversold reference lines
    fig.add_hline(
        y=RSI_OVERBOUGHT, line_dash="dash", line_color="#ef5350",
        annotation_text="Overbought", annotation_position="top left",
    )
    fig.add_hline(
        y=RSI_OVERSOLD, line_dash="dash", line_color="#26a69a",
        annotation_text="Oversold", annotation_position="bottom left",
    )
    # Neutral midline
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.4)

    fig.update_layout(yaxis=dict(title="RSI", range=[0, 100]))
    return _apply_base_layout(fig, f"{ticker} — RSI (14)")


# ============================================================================
# 4. MACD
# ============================================================================

def build_macd_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """MACD line, signal line, and histogram.

    Expected columns (pandas-ta defaults):
        ``MACD_12_26_9``, ``MACDs_12_26_9``, ``MACDh_12_26_9``

    Parameters
    ----------
    df:
        Enriched DataFrame.
    ticker:
        Ticker symbol for the chart title.

    Returns
    -------
    go.Figure
    """
    dates = _date_index(df)

    macd_col = "MACD_12_26_9"
    signal_col = "MACDs_12_26_9"
    hist_col = "MACDh_12_26_9"

    fig = go.Figure()

    # Histogram (green / red)
    if hist_col in df.columns:
        hist_colours = [
            "#26a69a" if v >= 0 else "#ef5350" for v in df[hist_col]
        ]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df[hist_col],
                marker_color=hist_colours,
                name="Histogram",
                opacity=0.5,
            )
        )

    # MACD line
    if macd_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[macd_col],
                mode="lines",
                name="MACD",
                line=dict(color="#42a5f5", width=2),
            )
        )

    # Signal line
    if signal_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df[signal_col],
                mode="lines",
                name="Signal",
                line=dict(color="#ff7043", width=1.5, dash="dot"),
            )
        )

    fig.add_hline(y=0, line_color="gray", line_dash="dot", opacity=0.4)
    fig.update_layout(yaxis_title="MACD Value")
    return _apply_base_layout(fig, f"{ticker} — MACD (12, 26, 9)")


# ============================================================================
# 5. Support & Resistance horizontal lines on price chart
# ============================================================================

def build_sr_overlay(
    base_fig: go.Figure,
    sr_levels: list[float],
) -> go.Figure:
    """Add horizontal S&R lines to an existing price figure *in-place*.

    Parameters
    ----------
    base_fig:
        An existing ``go.Figure`` (typically the price chart).
    sr_levels:
        List of price levels identified by ``ta_compute.compute_support_resistance``.

    Returns
    -------
    go.Figure
        The mutated figure (same object, returned for chaining).
    """
    for level in sr_levels:
        base_fig.add_hline(
            y=level,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.45)",
            annotation_text=f"S/R {level:,.2f}",
            annotation_position="right",
            annotation_font_size=9,
        )
    return base_fig


# ============================================================================
# 6. Fibonacci Retracement overlay
# ============================================================================

def build_fib_overlay(
    base_fig: go.Figure,
    fib_levels: dict[str, float],
) -> go.Figure:
    """Add Fibonacci retracement lines to an existing price figure *in-place*.

    Parameters
    ----------
    base_fig:
        An existing ``go.Figure`` (typically the price chart).
    fib_levels:
        Dict mapping level labels (e.g. ``"38.2%"``) to price values,
        as returned by ``ta_compute.compute_fibonacci``.

    Returns
    -------
    go.Figure
        The mutated figure (same object, returned for chaining).
    """
    fib_colours = {
        "23.6%": "#b2dfdb",
        "38.2%": "#80cbc4",
        "50.0%": "#4db6ac",
        "61.8%": "#26a69a",
        "78.6%": "#00897b",
    }

    for label, price in fib_levels.items():
        colour = fib_colours.get(label, "#26a69a")
        base_fig.add_hline(
            y=price,
            line_dash="dot",
            line_color=colour,
            annotation_text=f"Fib {label}: {price:,.2f}",
            annotation_position="left",
            annotation_font_size=9,
        )
    return base_fig


# ============================================================================
# 7. Convenience — build all charts at once
# ============================================================================

def build_all_charts(
    df: pd.DataFrame,
    ticker: str,
    sr_levels: list[float] | None = None,
    fib_levels: dict[str, float] | None = None,
) -> dict[str, go.Figure]:
    """Build every chart required by the Gradio Tab-2 layout.

    Parameters
    ----------
    df:
        Fully enriched DataFrame from ``ta_compute.compute_all_indicators``.
    ticker:
        Ticker symbol for chart titles.
    sr_levels:
        Optional list of S&R price levels.
    fib_levels:
        Optional dict of Fibonacci level labels → prices.

    Returns
    -------
    dict[str, go.Figure]
        Keys: ``"price"``, ``"volume"``, ``"rsi"``, ``"macd"``.
        The price chart includes S&R and Fibonacci overlays when provided.
    """
    price_fig = build_price_chart(df, ticker)

    if sr_levels:
        build_sr_overlay(price_fig, sr_levels)

    if fib_levels:
        build_fib_overlay(price_fig, fib_levels)

    return {
        "price": price_fig,
        "volume": build_volume_chart(df, ticker),
        "rsi": build_rsi_chart(df, ticker),
        "macd": build_macd_chart(df, ticker),
    }
