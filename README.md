# 📈 Technical Analysis Report Generator

An automated, AI-powered investment research tool that fetches market data, computes a comprehensive set of technical indicators, and generates a structured TA report using **Google Gemini 2.0 Flash**.

---

## ✨ Features

- **Live market data** via `yfinance` (Daily / Weekly / Monthly granularity)
- **15+ technical indicators** — EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Volume SMA, Fibonacci retracement, Support & Resistance levels
- **14 candlestick patterns** detected via TA-Lib (Hammer, Engulfing, Doji, Morning Star, …)
- **Grand TA Checklist** — 6-point rule-based scoring system for disciplined trade evaluation
- **Single-call LLM inference** via Gemini 2.0 Flash for cohesive, hallucination-resistant reports
- **Interactive Gradio UI** with dark-themed Plotly charts and tabular data views

---

## 🏗️ Project Structure

```
technical_analysis/
├── app.py               # Gradio UI entry point
├── main_pipeline.py     # Orchestrator — ties all layers together
├── requirements.txt     # Python dependencies
├── plan.md              # Architecture & design decisions
│
├── config/
│   └── settings.py      # All constants & tuneable defaults
│
├── core/
│   ├── data_fetch.py    # yfinance data acquisition
│   ├── ta_compute.py    # Indicator & pattern computation
│   ├── llm_inference.py # Gemini API client & retry logic
│   └── prompts.py       # Prompt templates for the LLM
│
├── utils/
│   ├── chart_builder.py # Plotly chart factories
│   └── validators.py    # Input validation (ticker, API key)
│
└── tests/
    ├── test_data_fetch.py
    ├── test_ta_compute.py
    └── test_llm_inference.py
```

---

## ⚙️ Setup

### 1. Prerequisites

**TA-Lib C library** must be installed before `pip install`:

| OS | Command |
|---|---|
| Linux | `sudo apt-get install libta-lib-dev` |
| macOS | `brew install ta-lib` |
| Windows | Download `.whl` from [Christoph Gohlke's builds](https://github.com/cgohlke/talib-build/) |

### 2. Create & activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Gemini API key

Generate a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🚀 Usage

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

1. **Paste your Gemini API key** in the left panel and click **💾 Save Key**
2. **Enter a ticker symbol** (e.g. `AAPL`, `RELIANCE.NS`, `^NSEI`)
3. Choose your preferred **time period** and **data frequency**
4. Click **🔍 Analyze** — the report and charts will appear in the tabs below

> ⚠️ **Disclaimer:** This tool is for educational purposes only. It does not constitute financial advice.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Market data fetching |
| `pandas-ta` | Technical indicator computation |
| `TA-Lib` | Candlestick pattern detection |
| `google-genai` | Gemini LLM inference |
| `gradio` | Web UI framework |
| `plotly` | Interactive charting |

---

## 📄 License

This project is for academic and educational use (UoH MBA Programme).
