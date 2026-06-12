from flask import Flask, render_template, request
from predict import predict
import sqlite3
import os

app = Flask(__name__)

DB_PATH = './predictions.db'

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

if __name__ == '__main__':
    init_db()
    app.run(debug=True)