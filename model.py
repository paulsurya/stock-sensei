import pandas as pd
import numpy as np

features = ['rsi', 'ma_cross', 'volatility', 'volume_ratio', 'vwap_diff', 'daily_range', 'delivery_pct']


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    result['rsi'] = _compute_rsi(df['Close'], 14)

    sma5 = df['Close'].rolling(5).mean()
    sma20 = df['Close'].rolling(20).mean()
    result['ma_cross'] = sma5 - sma20

    result['volatility'] = df['Close'].rolling(14).std()

    vol_avg20 = df['Volume'].rolling(20).mean()
    result['volume_ratio'] = df['Volume'] / vol_avg20.replace(0, np.nan)

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vol_cum = (typical_price * df['Volume']).cumsum()
    vol_sum = df['Volume'].cumsum()
    vwap = vol_cum / vol_sum.replace(0, np.nan)
    result['vwap_diff'] = df['Close'] - vwap

    result['daily_range'] = df['High'] - df['Low']

    close_position = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
    vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    result['delivery_pct'] = close_position.fillna(0.5) * vol_ratio.fillna(1.0)
    result['delivery_pct'] = result['delivery_pct'].clip(0, 1)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cols_lower = {c.lower(): c for c in df.columns}
    df.rename(columns={
        cols_lower.get('open', 'Open'): 'Open',
        cols_lower.get('high', 'High'): 'High',
        cols_lower.get('low', 'Low'): 'Low',
        cols_lower.get('close', 'Close'): 'Close',
        cols_lower.get('volume', 'Volume'): 'Volume',
    }, inplace=True)

    df['Volume'] = df['Volume'].replace(0, np.nan).ffill()

    feat = compute_features(df)
    feat['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    result = feat.dropna(subset=features + ['Target']).copy()
    return result


def compute_features_from_latest(df: pd.DataFrame) -> dict:
    feat = compute_features(df)
    last = feat.iloc[-1:][features]
    if last.isnull().any().any():
        return None
    return last.iloc[0].to_dict()
