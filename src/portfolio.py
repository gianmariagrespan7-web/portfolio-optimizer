"""Return, risk and performance metrics for assets and portfolios."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # trading days in a year


# ---------- Asset-level metrics ----------

def daily_returns(prices):
    """Simple daily returns: P_t / P_(t-1) - 1."""
    return prices.pct_change().dropna()


def cumulative_returns(returns):
    """Compounded return over time: product of (1 + r) minus 1."""
    return (1 + returns).cumprod() - 1


def annualized_return(returns):
    """Arithmetic mean of daily returns x 252 (expected return in Markowitz)."""
    return returns.mean() * TRADING_DAYS


def cagr(prices):
    """Compound annual growth rate (geometric mean)."""
    years = (len(prices) - 1) / TRADING_DAYS
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def annualized_volatility(returns):
    """Daily standard deviation x square root of 252."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def asset_summary(prices):
    """Summary table per asset (values in decimals)."""
    returns = daily_returns(prices)
    return pd.DataFrame({
        "Cumulative return": cumulative_returns(returns).iloc[-1],
        "Mean annual return": annualized_return(returns),
        "CAGR": cagr(prices),
        "Annual volatility": annualized_volatility(returns),
    })


# ---------- Covariance and correlation ----------

def covariance_matrix(returns):
    """Annualized covariance matrix (daily covariance x 252)."""
    return returns.cov() * TRADING_DAYS


def correlation_matrix(returns):
    """Correlation matrix of daily returns (values between -1 and 1)."""
    return returns.corr()


def average_correlation(corr):
    """Average correlation across all asset pairs (diagonal excluded)."""
    n = corr.shape[0]
    upper = corr.to_numpy()[np.triu_indices(n, k=1)]
    return float(upper.mean())


# ---------- Portfolio metrics ----------

def portfolio_return(weights, mean_returns):
    """Expected portfolio return: w' * mu."""
    return float(weights @ mean_returns)


def portfolio_volatility(weights, cov_matrix):
    """Portfolio volatility: sqrt(w' * Sigma * w)."""
    return float(np.sqrt(weights @ cov_matrix @ weights))


def sharpe_ratio(port_return, port_vol, risk_free_rate):
    """(Rp - Rf) / sigma_p, using annualized values."""
    return (port_return - risk_free_rate) / port_vol


def portfolio_performance(weights, returns, risk_free_rate):
    """Return (expected return, volatility, Sharpe ratio) of a portfolio."""
    mean_returns = annualized_return(returns).to_numpy()
    cov = covariance_matrix(returns).to_numpy()
    port_ret = portfolio_return(weights, mean_returns)
    port_vol = portfolio_volatility(weights, cov)
    return port_ret, port_vol, sharpe_ratio(port_ret, port_vol, risk_free_rate)


# ---------- Historical performance ----------

def portfolio_daily_returns(returns, weights):
    """Daily returns of a constant-weight portfolio (rebalanced daily)."""
    return returns @ weights


def growth_of_one(daily_rets):
    """Value over time of $1 invested at the start."""
    return (1 + daily_rets).cumprod()


def drawdown_series(daily_rets):
    """Loss (in decimals) relative to the highest value reached so far."""
    wealth = growth_of_one(daily_rets)
    return wealth / wealth.cummax() - 1


def max_drawdown(daily_rets):
    """Worst peak-to-trough loss."""
    return float(drawdown_series(daily_rets).min())


def performance_summary(daily_rets, risk_free_rate):
    """Historical performance metrics of a daily return series."""
    wealth = growth_of_one(daily_rets)
    years = len(daily_rets) / TRADING_DAYS
    ann_ret = daily_rets.mean() * TRADING_DAYS
    vol = daily_rets.std() * np.sqrt(TRADING_DAYS)
    return {
        "Cumulative return (%)": (wealth.iloc[-1] - 1) * 100,
        "CAGR (%)": (wealth.iloc[-1] ** (1 / years) - 1) * 100,
        "Volatility (%)": vol * 100,
        "Sharpe ratio": (ann_ret - risk_free_rate) / vol,
        "Max drawdown (%)": max_drawdown(daily_rets) * 100,
    }