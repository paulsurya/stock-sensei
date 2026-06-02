# 📈 Stock Sensei

A machine learning web app that predicts whether a stock will trend **Up or Down** using technical indicators — powered by XGBoost, yfinance, and Flask.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How to Contribute](#how-to-contribute)

## Overview

Enter any stock ticker (e.g. `TCS.NS`, `INFY.NS`, `AAPL`) and get an instant Up/Down trend prediction with a live price chart and model confidence score. The model is trained on RSI, MACD, and Moving Average features and backtested on held-out data before serving predictions through a clean Flask interface.

## Features

- 📡 Live stock data fetched via `yfinance` — no API key required
- 🔬 Technical indicators: RSI, MACD, SMA-20, SMA-50, EMA-20
- 🤖 XGBoost binary classifier with backtested accuracy
- 💾 Model persisted as `.pkl` — no retraining on each request
- 📈 Interactive price chart rendered in the browser
- 🗃 SQLite logging of every prediction request

## Tech Stack

| Layer | Technology |
|---|---|
| Data | `yfinance`, `pandas`, `numpy` |
| ML | `xgboost`, `scikit-learn` |
| Backend | `Python 3.11+`, `Flask`, `SQLite` |
| Frontend | `HTML5`, `CSS3`, `JavaScript` |

## How to Contribute

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Make your changes and commit — `git commit -m "add: your feature"`
4. Push to your branch — `git push origin feature/your-feature`
5. Open a Pull Request and describe what you changed

Please keep PRs focused and small. For large changes, open an issue first to discuss the approach.

---

> Built by [paulsurya](https://github.com/paulsurya) · This project is for educational purposes only and does not constitute financial advice.
