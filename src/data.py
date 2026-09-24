"""Download e pulizia dei dati di mercato."""

import re

import pandas as pd
import yfinance as yf


def parse_tickers(text):
    """Trasforma 'aapl, msft nvda' in ['AAPL', 'MSFT', 'NVDA'], senza duplicati."""
    raw = re.split(r"[,\s]+", text.upper())
    tickers = []
    for t in raw:
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def download_prices(tickers, start, end):
    """Scarica i prezzi di chiusura aggiustati.

    Ritorna un DataFrame: righe = date, colonne = ticker.
    """
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,  # prezzi aggiustati per split e dividendi
        progress=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    prices = data["Close"]
    # Con un solo ticker yfinance puo restituire una Series: la rendiamo tabella
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


def clean_prices(prices, max_gap=5):
    """Rimuove i ticker senza dati e sistema i valori mancanti.

    Ritorna (prezzi_puliti, lista_ticker_non_validi).
    """
    # 1. Ticker senza nessun prezzo = ticker non validi
    invalid = [c for c in prices.columns if prices[c].isna().all()]
    prices = prices.drop(columns=invalid)

    # 2. Buchi brevi (es. festivita diverse): ripete l'ultimo prezzo noto,
    #    ma al massimo per max_gap giorni
    prices = prices.ffill(limit=max_gap)

    # 3. Tiene solo le date in cui tutti i titoli hanno un prezzo
    prices = prices.dropna()

    return prices, invalid
