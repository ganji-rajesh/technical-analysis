# Technical Analysis Report Generator — Architecture Plan

## TL;DR
A Gradio-based web app that takes a ticker symbol from the user, fetches weekly historical data via yfinance, computes a specific suite of technical indicators and 15+ candlestick patterns using pandas-ta and TA-Lib, then uses the Gemini API (via a single, comprehensive call) to generate a structured TA report — all presented in a clean two-tab UI with a left-side API key panel.

---

## 1. Project Overview

| Attribute        | Detail                                              |
|------------------|-----------------------------------------------------|
| Purpose          | Generate automated Technical Analysis (TA) reports |
| Primary Users    | Retail traders, analysts, finance students          |
| Core Libraries   | yfinance, pandas, pandas-ta, TA-Lib, Gemini API, Gradio |
| Inference        | Gemini API (Single Comprehensive Prompt)            |
| UI Framework     | Gradio (Blocks API)                                 |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GRADIO UI (app.py)                             │
│                                                                         │
│  ┌──────────────┐   ┌──────────────────────────────────────────────┐   │
│  │  Left Panel  │   │              Main Content Area               │   │
│  │              │   │                                              │   │
│  │  Gemini API  │   │  ┌──────────────────────────────────────┐   │   │
│  │  Key Input   │   │  │         Chat Window (Center)          │   │   │
│  │              │   │  │   [ Enter Ticker Symbol... ] [Send]  │   │   │
│  │  [Save Key]  │   │  └──────────────────────────────────────┘   │   │
│  │              │   │                                              │   │
│  │  Model Info  │   │  ┌────────────────┐  ┌──────────────────┐  │   │
│  │  Status LED  │   │  │  Tab 1:        │  │  Tab 2:          │  │   │
│  │              │   │  │  📄 TA Report  │  │  📊 Data & Charts│  │   │
│  └──────────────┘   │  └────────────────┘  └──────────────────┘  │   │
│                      └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
  API Key Storage            Ticker Symbol Input
         │                          │
         └──────────┬───────────────┘
                    ▼
        ┌───────────────────────┐
        │   Orchestrator Layer  │  (main_pipeline.py)
        │   - Validates inputs  │
        │   - Manages pipeline  │
        └───────────┬───────────┘
                    │
        ┌───────────┴
        ▼                        
┌───────────────┐      ┌──────────────────────────────┐
│  Data Layer   │      │  Inference Layer             │
│ data_fetch.py │      │  core/llm_inference.py       │
│               │      │                              │
│ - yfinance    │      │ - Raw numeric data feed      │
│ - Weekly OHLCV│      │ - Single Gemini prompt       │
│ - Validation  │      │                              │
└───────┬───────┘      └──────────────▲───────────────┘
        │                             │
        ▼                             │
┌───────┴───────────────────────┐     │
│     Computation Layer         ├─────┘
│     ta_compute.py             │
│                               │
│ - Selective pandas-ta tools   │
│ - TA-Lib (15+ patterns)       │
│ - Fibonacci levels            │
│ - S&R level detection         │
│ - Volume analysis             │
└───────────────────────────────┘
```

---

## 3. Directory Structure

```
ta_report_app/
│
├── app.py                     # Gradio UI entry point
├── main_pipeline.py           # Orchestrator: connects all layers
│
├── core/
│   ├── __init__.py
│   ├── data_fetch.py          # yfinance data fetching
│   ├── ta_compute.py          # Technical indicator computation
│   ├── llm_inference.py       # Gemini API single-call logic
│   └── prompts.py             # Prompt templates for Gemini
│
├── utils/
│   ├── __init__.py
│   ├── validators.py          # Input validation (ticker, API key)
│   └── chart_builder.py       # Plotly chart generation for Tab 2
│
├── config/
│   └── settings.py            # Constants, model names, defaults
│
├── tests/
│   ├── test_data_fetch.py
│   ├── test_ta_compute.py
│   └── test_llm_inference.py
│
├── requirements.txt
└── README.md
```

---

## 4. Layer-by-Layer Breakdown

### 4.1 Data Layer — `core/data_fetch.py`

**Responsibility:** Fetch and validate weekly OHLCV data from yfinance.

```python
# Pseudocode
def fetch_weekly_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Downloads weekly historical OHLCV data for a given ticker.
    Period default: 2 years (sufficient for Dow Theory primary trend analysis).
    Returns a cleaned DataFrame indexed by Date.
    """
    import yfinance as yf
    data = yf.download(ticker, period=period, interval="1wk", auto_adjust=True)
    # Drop rows with missing OHLCV
    data.dropna(inplace=True)
    # Validate minimum data points (at least 52 weeks)
    if len(data) < 52:
        raise ValueError(f"Insufficient data for {ticker}. Try a different symbol.")
    return data
```

**Key Decisions:**
- Use `interval="1wk"` for swing trading focus
- `auto_adjust=True` adjusts for splits/dividends
- Minimum 2 years of data to support Dow Theory primary trend identification
- Validate ticker symbol existence before passing downstream

---

### 4.2 Computation Layer — `core/ta_compute.py`

**Responsibility:** Compute all technical indicators and candlestick pattern signals.

#### 4.2.1 Indicators Computed (via pandas-ta)

| Indicator          | Parameters                        | Signal Logic                                      |
|--------------------|-----------------------------------|---------------------------------------------------|
| EMA                | 20, 50, 100-period                | Price vs EMA, EMA crossover detection             |
| RSI                | 14-period                         | <30 oversold, >70 overbought                      |
| MACD               | 12, 26, 9 EMA                     | MACD line vs signal line crossover                |
| Bollinger Bands    | 20 SMA, ±2 std dev                | Price at upper/lower band                         |
| ATR                | 14-period                         | Volatility-based dynamic stop-loss sizing         |
| ADX                | 14-period                         | >25 = trending, <20 = ranging                     |
| Volume SMA         | 10-period                         | Current volume vs 10-day avg                      |
| Fibonacci Levels   | Computed from swing high/low      | 23.6%, 38.2%, 50%, 61.8%, 78.6%                   |
| S&R Levels         | Rolling pivot detection (custom)  | Identified from 3+ price action zone confluence   |

#### 4.2.2 Candlestick Patterns (15 TA-Lib Functions)

| Pattern                | TA-Lib Function          | Signal Type       |
|------------------------|--------------------------|-------------------|
| Marubozu               | `CDLMARUBOZU`            | Strong trend      |
| Doji                   | `CDLDOJI`                | Indecision        |
| Spinning Top           | `CDLSPINNINGTOP`         | Indecision        |
| Hammer                 | `CDLHAMMER`              | Bullish reversal  |
| Inverted Hammer        | `CDLINVERTEDHAMMER`      | Bullish reversal  |
| Hanging Man            | `CDLHANGINGMAN`          | Bearish reversal  |
| Shooting Star          | `CDLSHOOTINGSTAR`        | Bearish reversal  |
| Bullish Engulfing      | `CDLENGULFING`           | Bullish reversal  |
| Bearish Engulfing      | `CDLENGULFING`           | Bearish reversal  |
| Piercing Pattern       | `CDLPIERCING`            | Bullish reversal  |
| Dark Cloud Cover       | `CDLDARKCLOUDCOVER`      | Bearish reversal  |
| Morning Star           | `CDLMORNINGSTAR`         | Bullish reversal  |
| Evening Star           | `CDLEVENINGSTAR`         | Bearish reversal  |
| Harami                 | `CDLHARAMI`              | Reversal signal   |
| Three White Soldiers   | `CDL3WHITESOLDIERS`      | Bullish reversal  |

```python
# Pseudocode
def compute_all_indicators(df: pd.DataFrame) -> dict:
    import pandas_ta as ta
    import talib

    results = {}

    # Explicitly calculate minimal required indicators (avoiding strategy("all"))
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    # Custom Volume SMA logic
    df['VOL_SMA_10'] = df['Volume'].rolling(window=10).mean()

    # TA-Lib candlestick patterns (15 patterns)
    op, hi, lo, cl = df['Open'], df['High'], df['Low'], df['Close']
    results['patterns'] = {
        'Hammer':          talib.CDLHAMMER(op, hi, lo, cl),
        'InvertedHammer':  talib.CDLINVERTEDHAMMER(op, hi, lo, cl),
        'ShootingStar':    talib.CDLSHOOTINGSTAR(op, hi, lo, cl),
        'Engulfing':       talib.CDLENGULFING(op, hi, lo, cl),
        'MorningStar':     talib.CDLMORNINGSTAR(op, hi, lo, cl),
        'EveningStar':     talib.CDLEVENINGSTAR(op, hi, lo, cl),
        'Doji':            talib.CDLDOJI(op, hi, lo, cl),
        'Marubozu':        talib.CDLMARUBOZU(op, hi, lo, cl),
        'HangingMan':      talib.CDLHANGINGMAN(op, hi, lo, cl),
        'Harami':          talib.CDLHARAMI(op, hi, lo, cl),
        'Piercing':        talib.CDLPIERCING(op, hi, lo, cl),
        'DarkCloud':       talib.CDLDARKCLOUDCOVER(op, hi, lo, cl),
        'SpinningTop':     talib.CDLSPINNINGTOP(op, hi, lo, cl),
        'ThreeWhiteSoldiers': talib.CDL3WHITESOLDIERS(op, hi, lo, cl),
    }

    results['indicators'] = df  # DataFrame with TA columns
    results['fibonacci']  = compute_fibonacci(df) # Custom project component
    results['sr_levels']  = compute_support_resistance(df) # Custom project component
    return results
```

---

### 4.3 Inference Layer — `core/llm_inference.py`

**Responsibility:** Single API call structure ensuring holistic analysis of all sections at once, processing raw numerical inputs directly. No multithreading, allowing the LLM to comprehend interconnected concepts (e.g., Dow Theory vs Momentum) cohesively.

```python
# Pseudocode
def generate_full_report(computed_data: dict, api_key: str, ticker: str) -> str:
    import google.generativeai as genai
    import json
    
    # Convert dataframe to JSON representation for context
    raw_data = computed_data['indicators'].tail(20).to_json(orient="records")
    
    prompt = f"""... [Inject System Prompt + {raw_data} + Checklist Score] ..."""
    
    # Instantiating a client avoids global state mutations inside async workloads
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text
```

**Prompt Template Structure:**
```
System Context:
  You are a professional technical analyst generating a structured trading report
  for {ticker} based on pre-calculated technical signals.

Data Context:
  {raw_data_json}
  Grand Checklist Score: {checklist_score}/6

Task:
  Synthesize this data into a comprehensive report comprising:
  1. Candlestick & Price Action
  2. Support & Resistance / Volume
  3. Moving Average & Trend
  4. Momentum & Volatility
  5. Dow Theory Phase Checklist Conclusion
  6. Final Actionable Insight

Output Format Specifications:
  - Use Markdown headers (##) for each of the 6 sections
  - Use concise bullet points for key observations within sections
  - Conclude section 6 strictly with a bolded verdict line: **Verdict: [Strong Buy/Buy/Neutral/Avoid]**
```

---

### 4.5 UI Layer — `app.py`

**Framework:** Gradio Blocks API

#### 4.5.1 Layout Architecture

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    with gr.Row():

        # ── Left Panel ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### ⚙️ Configuration")
            api_key_input = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="AIza..."
            )
            save_btn      = gr.Button("Save Key", variant="secondary")
            api_status    = gr.Markdown("🔴 No key saved")
            gr.Markdown("---")
            gr.Markdown("**Model:** gemini-2.0-flash")
            gr.Markdown("**Data:** Weekly OHLCV (2yr)")

        # ── Main Content Area ──────────────────────────────────
        with gr.Column(scale=4):

            # Chat Window (Top)
            gr.Markdown("## 📈 Technical Analysis Report Generator")
            with gr.Row():
                ticker_input = gr.Textbox(
                    placeholder="Enter ticker symbol (e.g. AAPL, RELIANCE.NS)",
                    label="Ticker Symbol",
                    scale=4
                )
                submit_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)

            status_msg = gr.Markdown("")
            
            # Use gr.Progress for UI feedback during heavy computations
            # Flow: Fetching Data -> Computing Indicators -> Generating LLM Report -> Done

            # Tabbed Output (Below Chat)
            with gr.Tabs():

                with gr.Tab("📄 TA Investment Report"):
                    report_output = gr.Markdown(
                        value="*Your report will appear here after analysis...*"
                    )

                with gr.Tab("📊 Computed Data & Charts"):
                    with gr.Row():
                        price_chart   = gr.Plot(label="Price + EMA + Bollinger Bands")
                        volume_chart  = gr.Plot(label="Volume Analysis")
                    with gr.Row():
                        rsi_chart     = gr.Plot(label="RSI")
                        macd_chart    = gr.Plot(label="MACD")
                    patterns_table    = gr.Dataframe(label="Detected Candlestick Patterns")
                    indicators_table  = gr.Dataframe(label="Full Indicator Data (Last 20 Weeks)")
                    sr_table          = gr.Dataframe(label="Support & Resistance Levels")
                    fib_table         = gr.Dataframe(label="Fibonacci Retracement Levels")
```

---

## 5. Data Flow Diagram

```
User Input (Ticker + API Key)
         │
         ▼
┌─────────────────────┐
│  validators.py      │  ← Validate ticker format, API key format
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  data_fetch.py      │  ← yfinance.download(ticker, interval="1wk")
│  Output: OHLCV df   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ta_compute.py      │  ← pandas-ta + TA-Lib
│  Output: enriched   │    Adds: EMA, RSI, MACD, BB, ATR, ADX,
│  DataFrame + dicts  │    Candlestick signals, S&R, Fibonacci
└────────┬────────────┘
         │
         ├─────────────────────────────────┐
         ▼                                 ▼
┌──────────────────┐             ┌────────────────────────┐
│ chart_builder.py │             │ core/llm_inference.py  │
│ (Plotly charts   │             │ Single API Call to     │
│  for Tab 2)      │             │ Gemini 2.0 Flash via   │
└────────┬─────────┘             │ direct raw data prompt │
         │                       └─────────┬──────────────┘
         └──────────────┬──────────────────┘
                        ▼
              ┌─────────────────┐
              │  Gradio UI      │
              │  Tab 1: Report  │
              │  Tab 2: Data    │
              └─────────────────┘
```

---

## 6. Grand TA Checklist Implementation

The 6-point checklist is scored programmatically and passed to the Dow/Checklist prompt:

| # | Criterion                          | Data Source                        | Auto-Score Method                          |
|---|------------------------------------|------------------------------------|---------------------------------------------|
| 1 | Recognizable candlestick pattern   | TA-Lib pattern outputs             | Any non-zero signal in last 3 candles       |
| 2 | S&R confirms trade + stop-loss     | S&R computation                    | Price within 2% of identified S&R level     |
| 3 | Volume above 10-day average        | pandas-ta volume SMA               | `current_vol > vol_sma_10`                  |
| 4 | Aligns with Dow Theory trend/phase | ADX + price structure              | ADX trend direction + phase classification  |
| 5 | RSI + MACD confirm direction       | RSI + MACD values                  | Signal alignment check; flag if divergent   |
| 6 | RRR ≥ 1.5                          | ATR-based target/stop calculation  | `(target - entry) / (entry - stop) >= 1.5` |

Score summary and trade recommendation (Strong Buy / Buy / Neutral / Avoid) are appended to the final report.

---

## 7. Technical Stack & Dependencies

```
# requirements.txt

yfinance>=0.2.40
pandas>=2.0.0
pandas-ta>=0.3.14b
TA-Lib>=0.4.28
google-generativeai>=0.5.0
gradio>=4.30.0
plotly>=5.20.0
numpy>=1.26.0
```

> **Note on TA-Lib:** Requires a C library pre-installation.
> - Linux: `sudo apt-get install libta-lib-dev`
> - macOS: `brew install ta-lib`
> - Windows: Use the unofficial wheel from Christoph Gohlke's repository

---

## 8. Error Handling & Edge Cases

| Scenario                             | Handling Strategy                                              |
|--------------------------------------|----------------------------------------------------------------|
| Invalid ticker symbol                | yfinance returns empty df → catch + user-friendly error msg   |
| Insufficient historical data (<52wk) | Raise ValueError, prompt user to check symbol/exchange suffix |
| Missing API key                      | Disable Analyze button until key is saved                      |
| Gemini API rate limit / timeout      | Retry with exponential backoff (max 3 retries per request)     |
| TA-Lib C library not found           | Graceful fallback to pandas-ta only; disable pattern columns  |
| API Key Configuration Conflict       | Client explicitly instantiated per call via `genai.Client()`   |
| Network failure (yfinance)           | Retry once after 5s delay; surface error if still failing      |

---

## 9. Security Considerations

- API key is stored **only in session memory** via Gradio `State` — never written to disk or logs
- Input ticker sanitized with regex `^[A-Z0-9.^=\-]{1,20}$` to prevent injection
- Gemini prompts include a system instruction to refuse non-financial queries
- No user data is persisted between sessions

---

## 10. Performance Notes

- **Single comprehensive API logic** removes unnecessary threading complexity and preserves conceptual cohesiveness in output generation.
- **Weekly data** (vs daily) keeps DataFrame size manageable (~104 rows for 2yr), reducing LLM token usage
- **Selective pandas-ta calculation** avoids the heavy overhead typical with calculating all 130+ indicators in `strategy("all")`.
- Plotly charts are rendered client-side in Gradio, adding no server-side overhead

---

## 11. Future Enhancements

- [ ] **Backtesting & Signal Validation Engine:** Verify historical accuracy of the system’s trade signals on established setups.
- [ ] Add daily timeframe toggle for intraday traders
- [ ] Export report as PDF with charts embedded
- [ ] Multi-ticker comparison mode
- [ ] Save/load analysis history per session
- [ ] Add screener mode: scan a watchlist and rank by checklist score
- [ ] Support OpenAI / Claude API key as alternative LLM backend
