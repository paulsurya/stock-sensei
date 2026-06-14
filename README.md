# Stock Sensei

A machine learning web app that predicts stock direction (Up/Down) using 7 technical indicators — powered by XGBoost, yfinance, and Flask.

## Features

- **Live Dashboard** — Track real-time prices and model predictions for a configurable watchlist. Auto-refreshes every 25 seconds via JS polling.
- **Rate-Limited Data Fetching** — Calls yfinance once every 20–30 seconds per ticker to avoid API throttling.
- **7 Technical Indicators** — RSI(14), MA Cross (SMA5 − SMA20), Volatility (14d σ), Volume Ratio (vs 20d avg), VWAP Diff, Daily Range, and Delivery % (heuristic).
- **XGBoost Classifier** — Binary classifier trained on 470K+ NSE rows. Backtested hit rate ~50% with a conservative Sell bias.
- **Manual Predict Form** — Enter indicator values by hand to test hypothetical scenarios.
- **Prediction Logging** — Every prediction is stored in SQLite with full inputs and results.

## Tech Stack

| Layer | Technology |
|---|---|
| Data | yfinance, pandas, numpy |
| ML | xgboost, scikit-learn |
| Backend | Python 3.11+, Flask, SQLite, joblib |
| Frontend | HTML5, CSS3 (custom design system), Vanilla JS |

## Project Structure

```
stock-sensei/
├── app.py                  # Flask routes (predict, dashboard, history, API)
├── model.py                # Feature engineering (7 technical indicators)
├── predict.py              # Model loader + prediction wrapper
├── dataFetcher.py          # Rate-limited yfinance fetcher (historical + intraday)
├── train.py                # Training script (XGBClassifier)
├── requirements.txt
├── models/
│   └── final_model.pkl     # Pre-trained XGBoost classifier
├── templates/
│   ├── layout.html         # Base template (nav + footer)
│   ├── index.html          # Landing / overview page
│   ├── dashboard.html      # Live ticker dashboard with auto-refresh
│   ├── predict.html        # Manual indicator input form
│   ├── result.html         # Prediction result display
│   └── history.html        # Past predictions table
├── static/CSS/
│   └── style.css           # Complete dark-theme design system
├── cache/
│   ├── historical/         # Daily OHLCV CSVs (auto-downloaded)
│   └── live/               # Intraday CSVs (auto-downloaded)
├── predictions.db          # SQLite prediction log
└── watchlist.json          # User's tracked tickers
```

## Getting Started

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

- `/dashboard` — Live watchlist with auto-refreshing prices + predictions
- `/predict` — Manual indicator entry form
- `/history` — All past predictions

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/refresh?ticker=X` | Fetch fresh data + prediction for a ticker (JSON) |
| GET | `/api/watchlist` | Get current watchlist |
| POST | `/api/watchlist/add` | Add ticker to watchlist `{"ticker": "AAPL"}` |
| POST | `/api/watchlist/remove` | Remove ticker `{"ticker": "AAPL"}` |

## License

MIT

---

> Built by [Paul Surya P](https://github.com/paulsurya) · This project is for educational purposes only and does not constitute financial advice.
