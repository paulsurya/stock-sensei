import json
import os
import logging
from flask import Flask, render_template, request, jsonify
from predict import predict
from model import compute_features_from_latest, features
from dataFetcher import RateLimitedFetcher
import pandas as pd
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')
WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), 'watchlist.json')
CACHE_HISTORICAL = os.path.join(os.path.dirname(__file__), 'cache', 'historical')
CACHE_LIVE = os.path.join(os.path.dirname(__file__), 'cache', 'live')

fetcher = RateLimitedFetcher()


def load_watchlist():
    if os.path.exists(WATCHLIST_PATH):
        with open(WATCHLIST_PATH) as f:
            return json.load(f)
    default = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL"]
    save_watchlist(default)
    return default


def save_watchlist(tickers):
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    with open(WATCHLIST_PATH, 'w') as f:
        json.dump(tickers, f, indent=2)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            rsi           REAL,
            ma_cross      REAL,
            volatility    REAL,
            volume_ratio  REAL,
            vwap_diff     REAL,
            daily_range   REAL,
            delivery_pct  REAL,
            signal        TEXT,
            confidence    TEXT,
            strength      TEXT,
            advice        TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_latest_features_from_cache(ticker):
    safe = ticker.replace('.', '_')
    path = os.path.join(CACHE_HISTORICAL, f"{safe}_daily.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        feat_dict = compute_features_from_latest(df)
        return feat_dict
    except Exception as e:
        logger.warning(f"Failed to compute features for {ticker}: {e}")
        return None


def get_live_data(ticker):
    safe = ticker.replace('.', '_')
    path = os.path.join(CACHE_LIVE, f"{safe}_intraday.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        first = df.iloc[0]
        last = df.iloc[-1]
        change = last['Close'] - first['Open']
        change_pct = (change / first['Open']) * 100
        return {
            'ltp': round(last['Close'], 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'high': round(df['High'].max(), 2),
            'low': round(df['Low'].min(), 2),
            'volume': int(df['Volume'].sum()),
            'candles': len(df),
        }
    except Exception as e:
        logger.warning(f"Failed to get live data for {ticker}: {e}")
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    if request.method == 'POST':
        inputs = {
            'rsi'         : float(request.form['rsi']),
            'ma_cross'    : float(request.form['ma_cross']),
            'volatility'  : float(request.form['volatility']),
            'volume_ratio': float(request.form['volume_ratio']),
            'vwap_diff'   : float(request.form['vwap_diff']),
            'daily_range' : float(request.form['daily_range']),
            'delivery_pct': float(request.form['delivery_pct'])
        }

        result = predict(inputs)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions
            (rsi, ma_cross, volatility, volume_ratio, vwap_diff, daily_range, delivery_pct, signal, confidence, strength, advice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            inputs['rsi'], inputs['ma_cross'], inputs['volatility'],
            inputs['volume_ratio'], inputs['vwap_diff'], inputs['daily_range'],
            inputs['delivery_pct'], result['signal'], result['confidence'],
            result['strength'], result['advice']
        ))
        conn.commit()
        conn.close()

        return render_template('result.html', result=result, inputs=inputs)

    return render_template('predict.html')


@app.route('/history')
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)


@app.route('/dashboard')
def dashboard():
    watchlist = load_watchlist()
    ticker_data = []
    for t in watchlist:
        live = get_live_data(t)
        feats = get_latest_features_from_cache(t)
        pred = None
        if feats:
            try:
                pred = predict(feats)
            except Exception as e:
                logger.warning(f"Prediction failed for {t}: {e}")
        ticker_data.append({
            'ticker': t,
            'live': live,
            'features': feats,
            'prediction': pred,
        })
    return render_template('dashboard.html', tickers=ticker_data, features_list=features)


@app.route('/api/refresh', methods=['GET'])
def api_refresh():
    ticker = request.args.get('ticker', '')
    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400

    hist = fetcher.fetch_historical(ticker)
    intra = fetcher.fetch_intraday(ticker)

    feats = None
    if hist is not None:
        feats = compute_features_from_latest(hist)

    live = None
    if intra is not None:
        live = _intraday_summary(intra)

    pred = None
    if feats:
        try:
            pred = predict(feats)
        except Exception as e:
            logger.warning(f"Prediction failed for {ticker}: {e}")

    return jsonify({
        'ticker': ticker,
        'live': live,
        'features': feats,
        'prediction': pred,
    })


@app.route('/api/watchlist', methods=['GET'])
def api_get_watchlist():
    return jsonify(load_watchlist())


@app.route('/api/watchlist/add', methods=['POST'])
def api_add_ticker():
    data = request.get_json()
    ticker = data.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'error': 'Missing ticker'}), 400
    watchlist = load_watchlist()
    if ticker not in watchlist:
        watchlist.append(ticker)
        save_watchlist(watchlist)
    return jsonify({'watchlist': watchlist})


@app.route('/api/watchlist/remove', methods=['POST'])
def api_remove_ticker():
    data = request.get_json()
    ticker = data.get('ticker', '').strip().upper()
    watchlist = load_watchlist()
    if ticker in watchlist:
        watchlist.remove(ticker)
        save_watchlist(watchlist)
    return jsonify({'watchlist': watchlist})


def _intraday_summary(df):
    if df.empty:
        return None
    first = df.iloc[0]
    last = df.iloc[-1]
    change = last['Close'] - first['Open']
    change_pct = (change / first['Open']) * 100 if first['Open'] else 0
    return {
        'ltp': round(last['Close'], 2),
        'change': round(change, 2),
        'change_pct': round(change_pct, 2),
        'high': round(df['High'].max(), 2),
        'low': round(df['Low'].min(), 2),
        'volume': int(df['Volume'].sum()),
    }


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
