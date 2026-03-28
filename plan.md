 Purpose:
The project lets you type one or more NSE/BSE stock tickers (e.g., TCS or HDFCBANK, ICICIBANK) and automatically generates a full investment research report within minutes . It covers fundamentals, technical indicators, news sentiment, cross-stock comparisons, and an AI-written Buy/Hold/Avoid recommendation.

🏗️ Architecture & Agents
The system is built around three specialized agents coordinated by an orchestrator :

Research Agent — Fetches price history, fundamentals, and recent news using yfinance

Analysis Agent — Computes RSI, Moving Averages, volatility, and uses an  LLM  via api key to interpret the numbers

Writing Agent — Uses LLM via api key to write each section of the report in clean Markdown

Orchestrator (orchestrator.py) — Wires all agents together in a fail-fast pipeline with progress callbacks

📁 Project Structure
The repo is lean and well-organized with just 6 core files :

text
├── app.py              # Gradio UI (dark professional theme)
├── orchestrator.py     # Pipeline coordinator
├── requirements.txt    # Dependencies
└── agents/
    ├── research_agent.py
    ├── analysis_agent.py
    └── writing_agent.py
| Layer              | Technology                                      |
| ------------------ | ----------------------------------------------- |
| UI                 | Gradio (runs at localhost:7860)                 |
| Stock Data         | yfinance (NSE/BSE)                              |
| LLM Runtime        | LLMs via api key               |

| Data Processing    | NumPy, Pandas                                   |
| Language           | Python 3.11+                                    |