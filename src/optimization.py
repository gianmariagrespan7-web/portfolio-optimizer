"""Ottimizzazione di portafoglio: long-only, pesi che sommano a 1."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio import (
    annualized_return,
    covariance_matrix,
    portfolio_return,
    portfolio_volatility,
)


def equal_weight(n):
    """Stesso peso 1/n su ogni titolo."""
    return np.full(n, 1 / n)


def _optimize(objective, n, args=(), extra_constraints=None):
    """Minimizza 'objective' con i vincoli: somma pesi = 1 e 0 <= peso <= 1.

    extra_constraints: vincoli aggiuntivi (es. rendimento target).
    """
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if extra_constraints:
        constraints += extra_constraints
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        objective,
        x0=equal_weight(n),  # punto di partenza
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Ottimizzazione non riuscita: {result.message}")

    # Pulizia: toglie residui numerici minuscoli (es. -0.0000001)
    weights = np.clip(result.x, 0, 1)
    weights[weights < 1e-4] = 0
    return weights / weights.sum()


def min_volatility(cov_matrix):
    """Portafoglio a minima volatilita."""
    n = cov_matrix.shape[0]
    return _optimize(portfolio_volatility, n, args=(cov_matrix,))


def max_sharpe(mean_returns, cov_matrix, risk_free_rate):
    """Portafoglio a massimo Sharpe ratio."""
    n = len(mean_returns)

    def negative_sharpe(w):
        ret = portfolio_return(w, mean_returns)
        vol = portfolio_volatility(w, cov_matrix)
        return -(ret - risk_free_rate) / vol

    return _optimize(negative_sharpe, n)


def optimal_portfolios(returns, risk_free_rate):
    """Calcola i tre portafogli. Ritorna {nome: array di pesi}."""
    mean_returns = annualized_return(returns).to_numpy()
    cov = covariance_matrix(returns).to_numpy()
    n = len(mean_returns)

    return {
        "Equal Weight": equal_weight(n),
        "Min Volatility": min_volatility(cov),
        "Max Sharpe": max_sharpe(mean_returns, cov, risk_free_rate),
    }


def efficient_frontier(mean_returns, cov_matrix, n_points=40):
    """Per ogni rendimento target trova il portafoglio a volatilita minima.

    Va dal portafoglio Min Volatility (sotto e inefficiente)
    fino al titolo con il rendimento atteso piu alto.
    """
    n = len(mean_returns)
    ret_min = portfolio_return(min_volatility(cov_matrix), mean_returns)
    ret_max = float(np.max(mean_returns))

    points = []
    for target in np.linspace(ret_min, ret_max, n_points):
        # Vincolo extra: rendimento del portafoglio = target
        # (t=target "congela" il valore di target per questa iterazione)
        target_constraint = {
            "type": "eq",
            "fun": lambda w, t=target: portfolio_return(w, mean_returns) - t,
        }
        try:
            w = _optimize(
                portfolio_volatility, n,
                args=(cov_matrix,), extra_constraints=[target_constraint],
            )
        except ValueError:
            continue  # se un punto non converge, lo saltiamo

        points.append({
            "Volatilita": portfolio_volatility(w, cov_matrix),
            "Rendimento": portfolio_return(w, mean_returns),
        })
    return pd.DataFrame(points)


def random_portfolios(mean_returns, cov_matrix, risk_free_rate,
                      n_portfolios=3000, seed=42):
    """Genera portafogli con pesi casuali (distribuzione di Dirichlet)."""
    rng = np.random.default_rng(seed)  # seed fisso = risultati riproducibili
    n = len(mean_returns)
    all_weights = rng.dirichlet(np.ones(n), n_portfolios)

    rets = [portfolio_return(w, mean_returns) for w in all_weights]
    vols = [portfolio_volatility(w, cov_matrix) for w in all_weights]

    df = pd.DataFrame({"Volatilita": vols, "Rendimento": rets})
    df["Sharpe"] = (df["Rendimento"] - risk_free_rate) / df["Volatilita"]
    return df
