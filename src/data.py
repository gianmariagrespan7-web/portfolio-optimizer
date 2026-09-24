"""Market data download and cleaning."""

import re
import time

import pandas as pd
import yfinance as yf

# Letters, digits and the symbols Yahoo uses (e.g. BRK-B, ^GSPC, EURUSD=X)
VALID_TICKER = re.compile(r"^[A-Z0-9.\-=^]{1,15}$")


class DataDownloadError(Exception):
    """Raised when market data cannot be downloaded."""


def parse_tickers(text):
    """Split user input into tickers.

    Returns (valid_tickers, malformed_tickers), both without duplicates.
    'aapl, msft nvda' -> (['AAPL', 'MSFT', 'NVDA'], [])
    """
    raw = re.split(r"[,\s]+", text.upper())
    valid, malformed = [], []
    for t in raw:
        if not t or t in valid or t in malformed:
            continue
        if VALID_TICKER.match(t):
            valid.append(t)
        else:
            malformed.append(t)
    return valid, malformed


def download_prices(tickers, start, end, retries=2):
    """Download adjusted close prices (rows = dates, columns = tickers).

    Tries again once if Yahoo returns nothing, then raises DataDownloadError.
    Raising (instead of returning an empty table) matters: Streamlit does not
    cache exceptions, so the next attempt really downloads again.
    """
    last_error = None
    for attempt in range(retries):
        try:
            data = yf.download(
                list(tickers),
                start=start,
                end=end,
                auto_adjust=True,  # prices adjusted for splits and dividends
                progress=False,
            )
        except Exception as e:  # network errors, rate limits, ...
            last_error = e
        else:
            if data is not None and not data.empty:
                prices = data["Close"]
                # With a single ticker yfinance may return a Series
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(name=tickers[0])
                return prices
        time.sleep(1 + attempt)  # short pause before trying again

    raise DataDownloadError(
        "No data received from Yahoo Finance. Check the tickers and the dates; "
        "if they are correct, Yahoo may be temporarily unavailable. "
        "Please try again in a minute."
    ) from last_error


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