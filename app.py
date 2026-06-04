import pandas as pd
import yfinance as yf

# api_key = "sk-live-OJH6gt8Gr1EKFqaVUb9ImziBcKvSY8S9WseXd58h"

def get_hist_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d", interval="1m")
    return hist

ticker = "RELIANCE.NS"
df = pd.DataFrame(get_hist_data(ticker))
ticker = ticker.split(".")[0]
df.to_csv(f"./cache/historical/{ticker}.csv", index=False)