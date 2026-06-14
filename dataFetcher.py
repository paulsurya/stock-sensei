import time
import random
import os
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_HISTORICAL = os.path.join(os.path.dirname(__file__), 'cache', 'historical')
CACHE_LIVE = os.path.join(os.path.dirname(__file__), 'cache', 'live')


class RateLimitedFetcher:
    def __init__(self, min_delay: float = 20, max_delay: float = 30):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_call: dict[str, float] = {}

    def _rate_limit(self, ticker: str):
        now = time.time()
        last = self._last_call.get(ticker, 0)
        elapsed = now - last
        delay = random.uniform(self.min_delay, self.max_delay)
        if elapsed < delay:
            sleep_for = delay - elapsed
            logger.debug(f"Rate limiting {ticker}: sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
        self._last_call[ticker] = time.time()

    def fetch_historical(self, ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame | None:
        self._rate_limit(ticker)
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No historical data for {ticker}")
                return None
            os.makedirs(CACHE_HISTORICAL, exist_ok=True)
            path = os.path.join(CACHE_HISTORICAL, f"{ticker.replace('.', '_')}_daily.csv")
            df.to_csv(path)
            logger.info(f"Saved historical data for {ticker} -> {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch historical {ticker}: {e}")
            return None

    def fetch_intraday(self, ticker: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame | None:
        self._rate_limit(ticker)
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No intraday data for {ticker}")
                return None
            os.makedirs(CACHE_LIVE, exist_ok=True)
            path = os.path.join(CACHE_LIVE, f"{ticker.replace('.', '_')}_intraday.csv")
            df.to_csv(path)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch intraday {ticker}: {e}")
            return None

    def fetch_all(self, tickers: list[str], period: str = "3mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
        results = {}
        for ticker in tickers:
            df = self.fetch_historical(ticker, period, interval)
            if df is not None:
                results[ticker] = df
        return results
