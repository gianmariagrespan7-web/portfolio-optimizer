"""Market data download and cleaning."""

import re

import pandas as pd
import yfinance as yf


def parse_tickers(text):
    """Turn 'aapl, msft nvda' into ['AAPL', 'MSFT', 'NVDA'], without duplicates."""
    raw = re.split(r"[,\s]+", text.upper())
    tickers = []
    for t in raw:
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def download_prices(tickers, start, end):
    """Download adjusted close prices.

    Returns a DataFrame: rows = dates, columns = tickers.
    """
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,  # prices adjusted for splits and dividends
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    prices = data["Close"]
    # With a single ticker yfinance may return a Series: make it a table
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


def clean_prices(prices, max_gap=5):
    """Remove tickers without data and handle missing values.

    Returns (clean_prices, list_of_invalid_tickers).
    """
    # 1. Tickers with no prices at all are invalid
    invalid = [c for c in prices.columns if prices[c].isna().all()]
    prices = prices.drop(columns=invalid)

    # 2. Short gaps (e.g. different holidays): carry the last price forward,
    #    for at most max_gap days
    prices = prices.ffill(limit=max_gap)

    # 3. Keep only dates where every asset has a price
    prices = prices.dropna()

    return prices, invalid