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


def covariance_matrix(returns):
    """Matrice di covarianza annualizzata (covarianza giornaliera x 252)."""
    return returns.cov() * TRADING_DAYS


def portfolio_return(weights, mean_returns):
    """Rendimento atteso del portafoglio: w' * mu."""
    return float(weights @ mean_returns)


def portfolio_volatility(weights, cov_matrix):
    """Volatilita del portafoglio: radice di w' * Sigma * w."""
    return float(np.sqrt(weights @ cov_matrix @ weights))


def sharpe_ratio(port_return, port_vol, risk_free_rate):
    """(Rp - Rf) / sigma_p, con valori annualizzati."""
    return (port_return - risk_free_rate) / port_vol


def portfolio_performance(weights, returns, risk_free_rate):
    """Ritorna (rendimento atteso, volatilita, Sharpe) del portafoglio."""
    mean_returns = annualized_return(returns).to_numpy()
    cov = covariance_matrix(returns).to_numpy()
    port_ret = portfolio_return(weights, mean_returns)
    port_vol = portfolio_volatility(weights, cov)
    return port_ret, port_vol, sharpe_ratio(port_ret, port_vol, risk_free_rate)


def correlation_matrix(returns):
    """Matrice di correlazione dei rendimenti giornalieri (valori tra -1 e 1)."""
    return returns.corr()


def average_correlation(corr):
    """Correlazione media tra tutte le coppie di titoli (esclusa la diagonale)."""
    n = corr.shape[0]
    upper = corr.to_numpy()[np.triu_indices(n, k=1)]
    return float(upper.mean())
