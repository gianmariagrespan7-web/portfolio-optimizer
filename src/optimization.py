"""Portfolio optimization: long-only, fully invested, optional max weight per asset."""

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
    """Same weight 1/n on every asset."""
    return np.full(n, 1 / n)


def _optimize(objective, n, max_weight=1.0, args=(), extra_constraints=None):
    """Minimize `objective` subject to: sum(w) = 1 and 0 <= w_i <= max_weight."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if extra_constraints:
        constraints += extra_constraints
    bounds = [(0.0, max_weight)] * n

    result = minimize(
        objective,
        x0=equal_weight(n),  # starting point
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    # Clean-up: remove tiny numerical leftovers (e.g. -0.0000001)
    weights = np.clip(result.x, 0, max_weight)
    weights[weights < 1e-4] = 0
    return weights / weights.sum()


def min_volatility(cov_matrix, max_weight=1.0):
    """Minimum volatility portfolio."""
    n = cov_matrix.shape[0]
    return _optimize(portfolio_volatility, n, max_weight, args=(cov_matrix,))


def max_sharpe(mean_returns, cov_matrix, risk_free_rate, max_weight=1.0):
    """Maximum Sharpe ratio portfolio (minimizes the negative Sharpe)."""
    n = len(mean_returns)

    def negative_sharpe(w):
        ret = portfolio_return(w, mean_returns)
        vol = portfolio_volatility(w, cov_matrix)
        return -(ret - risk_free_rate) / vol

    return _optimize(negative_sharpe, n, max_weight)


def optimal_portfolios(returns, risk_free_rate, max_weight=1.0):
    """Compute the three portfolios. Returns {name: weights array}."""
    mean_returns = annualized_return(returns).to_numpy()
    cov = covariance_matrix(returns).to_numpy()
    n = len(mean_returns)

    return {
        "Equal Weight": equal_weight(n),
        "Min Volatility": min_volatility(cov, max_weight),
        "Max Sharpe": max_sharpe(mean_returns, cov, risk_free_rate, max_weight),
    }


def max_feasible_return(mean_returns, max_weight=1.0):
    """Highest return reachable under the cap: fill the best assets first."""
    remaining = 1.0
    total = 0.0
    for mu in np.sort(mean_returns)[::-1]:
        w = min(max_weight, remaining)
        total += w * mu
        remaining -= w
        if remaining <= 1e-12:
            break
    return total


def efficient_frontier(mean_returns, cov_matrix, max_weight=1.0, n_points=40):
    """For each target return, find the minimum volatility portfolio.

    Runs from the Min Volatility portfolio (below it the curve is inefficient)
    up to the highest return reachable under the constraints.
    """
    n = len(mean_returns)
    ret_min = portfolio_return(min_volatility(cov_matrix, max_weight), mean_returns)
    ret_max = max_feasible_return(mean_returns, max_weight)

    points = []
    for target in np.linspace(ret_min, ret_max, n_points):
        # Extra constraint: portfolio return = target
        # (t=target "freezes" the value of target for this iteration)
        target_constraint = {
            "type": "eq",
            "fun": lambda w, t=target: portfolio_return(w, mean_returns) - t,
        }
        try:
            w = _optimize(
                portfolio_volatility, n, max_weight,
                args=(cov_matrix,), extra_constraints=[target_constraint],
            )
        except ValueError:
            continue  # skip points that do not converge

        points.append({
            "Volatility": portfolio_volatility(w, cov_matrix),
            "Return": portfolio_return(w, mean_returns),
        })
    return pd.DataFrame(points)


def random_portfolios(mean_returns, cov_matrix, risk_free_rate,
                      max_weight=1.0, n_portfolios=3000, seed=42):
    """Random long-only portfolios (Dirichlet weights) that respect the cap."""
    rng = np.random.default_rng(seed)  # fixed seed = reproducible results
    n = len(mean_returns)

    # Draw more than needed, then keep only portfolios within the weight cap
    all_weights = rng.dirichlet(np.ones(n), n_portfolios * 10)
    all_weights = all_weights[all_weights.max(axis=1) <= max_weight][:n_portfolios]

    rets = [portfolio_return(w, mean_returns) for w in all_weights]
    vols = [portfolio_volatility(w, cov_matrix) for w in all_weights]

    df = pd.DataFrame({"Volatility": vols, "Return": rets})
    df["Sharpe"] = (df["Return"] - risk_free_rate) / df["Volatility"]
    return df