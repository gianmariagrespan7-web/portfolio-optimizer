"""Calcolo di rendimenti e volatilita."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # giorni di borsa in un anno


def daily_returns(prices):
    """Rendimenti giornalieri semplici: (P_t / P_t-1) - 1."""
    return prices.pct_change().dropna()


def cumulative_returns(returns):
    """Rendimento cumulato nel tempo: prodotto di (1 + r) meno 1."""
    return (1 + returns).cumprod() - 1


def annualized_return(returns):
    """Media aritmetica dei rendimenti giornalieri x 252 (usata da Markowitz)."""
    return returns.mean() * TRADING_DAYS


def cagr(prices):
    """Tasso di crescita annuo composto (media geometrica)."""
    years = (len(prices) - 1) / TRADING_DAYS
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def annualized_volatility(returns):
    """Deviazione standard giornaliera x radice di 252."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def asset_summary(prices):
    """Tabella riassuntiva per ogni titolo (valori in decimali)."""
    returns = daily_returns(prices)
    return pd.DataFrame({
        "Rendimento cumulato": cumulative_returns(returns).iloc[-1],
        "Rendimento medio annuo": annualized_return(returns),
        "CAGR": cagr(prices),
        "Volatilita annua": annualized_volatility(returns),
    })
