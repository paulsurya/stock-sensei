import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill()
    df['MA_7'] = df['Close'].rolling(7).mean().fillna(df['Close'].expanding().mean())
    df['Price Range'] = df['High'] - df['Low']
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

features = ['Open', 'Close', 'Volume', 'MA_7', 'Price Range']