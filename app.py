"""Portfolio Optimizer: Streamlit dashboard."""

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import DataDownloadError, clean_prices, download_prices, parse_tickers
from src.optimization import (
    efficient_frontier,
    optimal_portfolios,
    random_portfolios,
)
from src.portfolio import (
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
    asset_summary,
    average_correlation,
    correlation_matrix,
    covariance_matrix,
    cumulative_returns,
    daily_returns,
    drawdown_series,
    growth_of_one,
    historical_cvar,
    historical_var,
    performance_summary,
    portfolio_daily_returns,
    portfolio_performance,
    risk_summary,
)

BENCHMARK = "SPY"
DEFAULT_TICKERS = "AAPL, MSFT, NVDA, AMZN, GOOGL, META"
MAX_TICKERS = 20        # keeps optimization fast and downloads reasonable
MIN_OBSERVATIONS = 60   # minimum trading days for meaningful estimates

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")


# ---------- Cached computations ----------

@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def load_prices(tickers, start, end):
    # Raises DataDownloadError on failure: exceptions are not cached
    return download_prices(tickers, start, end)


@st.cache_data(show_spinner="Computing the efficient frontier...")
def compute_frontier(returns, risk_free_rate, max_weight):
    mean_returns = annualized_return(returns).to_numpy()
    cov = covariance_matrix(returns).to_numpy()
    frontier = efficient_frontier(mean_returns, cov, max_weight)
    cloud = random_portfolios(mean_returns, cov, risk_free_rate, max_weight)
    return frontier, cloud


# ---------- Chart helper ----------

def style_time_chart(fig, y_title, prefix="", suffix=""):
    """Consistent style for time-series charts, with 2 decimals in the tooltip."""
    for trace in fig.data:
        trace.hovertemplate = f"{trace.name}: {prefix}%{{y:.2f}}{suffix}<extra></extra>"
    fig.update_layout(
        xaxis_title=None, yaxis_title=y_title,
        legend_title=None, hovermode="x unified",
    )
    return fig


# ---------- Sidebar ----------

with st.sidebar:
    st.header("Settings")
    ticker_text = st.text_input("Tickers (comma separated)", value=DEFAULT_TICKERS)
    start = st.date_input("Start date", value=date.today() - timedelta(days=5 * 365))
    end = st.date_input("End date", value=date.today())
    risk_free_pct = st.number_input(
        "Risk-free rate (% p.a.)",
        min_value=0.0, max_value=20.0, value=4.0, step=0.1,
        help="For USD assets, use the current 3-month US T-bill yield.",
    )
    risk_free_rate = risk_free_pct / 100

    st.subheader("Optimization constraints")
    max_weight_pct = st.slider(
        "Max weight per asset (%)",
        min_value=5, max_value=100, value=100, step=5,
        help="Caps the weight of any single asset in the optimized portfolios.",
    )
    max_weight = max_weight_pct / 100
    st.caption("Long-only (no short selling), no leverage, weights sum to 100%.")

# ---------- Header ----------

st.title("📈 Portfolio Optimizer")
st.caption(
    "Mean-variance portfolio analysis with historical market data. "
    "Educational project, not investment advice."
)

# ---------- Input validation ----------

tickers, malformed = parse_tickers(ticker_text)
if malformed:
    st.warning(f"Invalid ticker format, ignored: {', '.join(malformed)}")
if not tickers:
    st.info("Enter at least one valid ticker in the sidebar (e.g. AAPL, MSFT).")
    st.stop()
if len(tickers) > MAX_TICKERS:
    st.error(f"Too many tickers ({len(tickers)}). The maximum is {MAX_TICKERS}.")
    st.stop()
if start >= date.today():
    st.error("The start date cannot be today or in the future.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

# ---------- Data download and cleaning ----------

try:
    raw = load_prices(tuple(tickers), start, end)
except DataDownloadError as e:
    st.error(str(e))
    st.button("Try again")  # clicking any button reruns the app (and the download)
    st.stop()

prices, invalid = clean_prices(raw)
if invalid:
    st.warning(
        f"No data for: {', '.join(invalid)}. The ticker may not exist on "
        "Yahoo Finance, or its download failed."
    )
    if st.button("Retry download"):
        load_prices.clear()  # forget cached downloads and try again
        st.rerun()
if prices.empty:
    st.error("No valid data for the selected tickers and period.")
    st.stop()
if (prices.index[0].date() - start).days > 10:
    st.info(
        f"Common data for all assets starts on {prices.index[0]:%d %b %Y} "
        "(one asset probably has a shorter history)."
    )

if len(prices) < MIN_OBSERVATIONS:
    st.error(
        f"Only {len(prices)} trading days of common data. At least "
        f"{MIN_OBSERVATIONS} are needed for reliable estimates: choose a longer period."
    )
    st.stop()
if len(prices) < TRADING_DAYS:
    st.warning(
        "Less than one year of data: annualized figures are extrapolated "
        "and may be unreliable."
    )

n_assets = prices.shape[1]
if n_assets == 1:
    st.info(
        "Only one asset selected: correlation, optimization and the efficient "
        "frontier need at least two."
    )
elif n_assets * max_weight < 1:
    st.error(
        f"With {n_assets} assets the max weight must be at least "
        f"{math.ceil(100 / n_assets)}%, otherwise weights cannot sum to 100%."
    )
    st.stop()

# ---------- Core computations ----------

returns = daily_returns(prices)
stats = asset_summary(prices)

portfolios = {}
if n_assets > 1:
    try:
        portfolios = optimal_portfolios(returns, risk_free_rate, max_weight)
    except Exception as e:
        st.warning(f"Portfolio optimization failed ({e}). Optimized portfolios are not shown.")

# Benchmark data (used by the Benchmark and Risk Metrics tabs)
bench_rets = None
try:
    bench_prices, _ = clean_prices(load_prices((BENCHMARK,), start, end))
    if not bench_prices.empty:
        bench_rets = daily_returns(bench_prices)[BENCHMARK]
except DataDownloadError:
    pass  # handled inside the tabs

# ---------- Summary KPIs ----------

k1, k2, k3, k4 = st.columns(4)
k1.metric("Assets", n_assets)
k2.metric("Period", f"{prices.index[0]:%b %Y} - {prices.index[-1]:%b %Y}")
k3.metric("Trading days", len(prices))
if portfolios:
    _, _, best_sharpe = portfolio_performance(
        portfolios["Max Sharpe"], returns, risk_free_rate
    )
    k4.metric("Max Sharpe ratio", f"{best_sharpe:.2f}")

(tab_assets, tab_corr, tab_frontier, tab_optimal,
 tab_custom, tab_bench, tab_risk) = st.tabs([
    "Asset Performance",
    "Correlation",
    "Efficient Frontier",
    "Optimal Portfolios",
    "Custom Portfolio",
    "Benchmark",
    "Risk Metrics",
])

# Note: tabs can be filled in any order. The Custom Portfolio tab is filled
# first because the other tabs use its weights.

# ---------- Custom Portfolio ----------

with tab_custom:
    st.markdown("Set your own weights (in %) and compare them with the optimized portfolios.")

    default_weight = round(100 / n_assets, 2)
    cols = st.columns(4)
    weights_pct = []
    for i, ticker in enumerate(prices.columns):  # same order as the data columns
        w = cols[i % 4].number_input(
            ticker, min_value=0.0, max_value=100.0,
            value=default_weight, step=1.0, key=f"w_{ticker}",
        )
        weights_pct.append(w)

    total = sum(weights_pct)
    custom_weights = None
    if abs(total - 100) > 0.1:
        st.error(f"Weights sum to {total:.2f}%: they must sum to 100%.")
    else:
        custom_weights = np.array(weights_pct) / total  # sum exactly 1
        ret, vol, shp = portfolio_performance(custom_weights, returns, risk_free_rate)

        m1, m2, m3 = st.columns(3)
        m1.metric("Expected annual return", f"{ret:.2%}")
        m2.metric("Annual volatility", f"{vol:.2%}")
        m3.metric("Sharpe ratio", f"{shp:.2f}")

        avg_vol = custom_weights @ annualized_volatility(returns).to_numpy()
        st.caption(
            f"Weighted average of single-asset volatilities: {avg_vol:.2%}. "
            f"Diversification lowers portfolio volatility by "
            f"{(avg_vol - vol) * 100:.2f} percentage points."
        )

# ---------- Daily returns of all portfolios (+ benchmark) ----------

series = {}
if custom_weights is not None:
    series["Your portfolio"] = portfolio_daily_returns(returns, custom_weights)
for name, w in portfolios.items():
    series[name] = portfolio_daily_returns(returns, w)
if bench_rets is not None:
    series[BENCHMARK] = bench_rets

# dropna aligns the dates: keep only days common to all series
all_rets = pd.DataFrame(series).dropna()

# ---------- Asset Performance ----------

with tab_assets:
    st.markdown("**Cumulative return**")
    fig = px.line(cumulative_returns(returns) * 100)
    st.plotly_chart(style_time_chart(fig, "Cumulative return (%)", suffix="%"))

    st.markdown("**Asset statistics** (annualized with 252 trading days)")
    st.dataframe((stats * 100).round(2).add_suffix(" (%)"))

    with st.expander("Adjusted prices"):
        fig = px.line(prices)
        st.plotly_chart(style_time_chart(fig, "Adjusted price"))
        st.dataframe(prices.tail(10).round(2))

# ---------- Correlation ----------

with tab_corr:
    if n_assets < 2:
        st.info("Add at least two assets to see correlations.")
    else:
        corr = correlation_matrix(returns)
        st.metric("Average pairwise correlation", f"{average_correlation(corr):.2f}")

        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig)
        st.caption(
            "Correlation of daily returns. Lower correlation means a stronger "
            "diversification benefit."
        )

        with st.expander("Covariance matrix (annualized)"):
            st.dataframe(covariance_matrix(returns).round(4))
            st.caption(
                "Diagonal = variances. The square root of the diagonal "
                "is the annual volatility."
            )

# ---------- Optimal Portfolios ----------

with tab_optimal:
    if not portfolios:
        st.info("Optimized portfolios are not available for this selection.")
    else:
        st.caption(
            f"Long-only, max {max_weight_pct}% per asset, weights sum to 100%. "
            "Solved with scipy (SLSQP)."
        )

        rows = {}
        for name, w in portfolios.items():
            ret, vol, shp = portfolio_performance(w, returns, risk_free_rate)
            rows[name] = {
                "Expected return (%)": ret * 100,
                "Volatility (%)": vol * 100,
                "Sharpe ratio": shp,
            }
        st.dataframe(pd.DataFrame(rows).T.round(2))

        weights_df = pd.DataFrame(portfolios, index=prices.columns) * 100
        fig = px.bar(weights_df, barmode="group")
        for trace in fig.data:
            trace.hovertemplate = f"{trace.name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>"
        fig.update_layout(xaxis_title=None, yaxis_title="Weight (%)", legend_title=None)
        st.plotly_chart(fig)

        with st.expander("Weights table (%)"):
            st.dataframe(weights_df.round(2))

# ---------- Efficient Frontier ----------

with tab_frontier:
    if not portfolios:
        st.info("The efficient frontier is not available for this selection.")
    else:
        try:
            frontier, cloud = compute_frontier(returns, risk_free_rate, max_weight)
        except Exception:
            frontier, cloud = pd.DataFrame(), pd.DataFrame()

        if frontier.empty:
            st.warning("The efficient frontier could not be computed for this selection.")
        else:
            fig = go.Figure()

            # 1. Random portfolios, colored by Sharpe ratio
            fig.add_trace(go.Scatter(
                x=cloud["Volatility"] * 100, y=cloud["Return"] * 100,
                mode="markers", name="Random portfolios", opacity=0.5,
                marker=dict(size=4, color=cloud["Sharpe"], colorscale="Viridis",
                            showscale=True, colorbar=dict(title="Sharpe")),
                hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            ))

            # 2. Efficient frontier
            fig.add_trace(go.Scatter(
                x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
                mode="lines", name="Efficient frontier",
                line=dict(color="#2c3e50", width=3),
                hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            ))

            # 3. Capital Market Line: from Rf, tangent at the Max Sharpe portfolio
            x_cml = np.array([0, stats["Annual volatility"].max() * 100])
            fig.add_trace(go.Scatter(
                x=x_cml, y=risk_free_pct + best_sharpe * x_cml,
                mode="lines", name="Capital Market Line",
                line=dict(color="gray", dash="dash"), hoverinfo="skip",
            ))

            # 4. Optimal portfolios (stars) and custom portfolio (X)
            markers = [
                ("Min Volatility", portfolios["Min Volatility"], "blue", "star"),
                ("Max Sharpe", portfolios["Max Sharpe"], "red", "star"),
            ]
            if custom_weights is not None:
                markers.append(("Your portfolio", custom_weights, "green", "x"))

            for name, w, color, symbol in markers:
                r, v, s = portfolio_performance(w, returns, risk_free_rate)
                fig.add_trace(go.Scatter(
                    x=[v * 100], y=[r * 100], mode="markers", name=name,
                    marker=dict(size=18, color=color, symbol=symbol,
                                line=dict(width=1, color="black")),
                    hovertemplate=(f"{name}<br>Volatility: %{{x:.2f}}%<br>"
                                   f"Return: %{{y:.2f}}%<br>Sharpe: {s:.2f}<extra></extra>"),
                ))

            # 5. Single assets, drawn last so their labels stay on top
            fig.add_trace(go.Scatter(
                x=stats["Annual volatility"] * 100,
                y=stats["Mean annual return"] * 100,
                mode="markers+text", name="Single assets",
                text=stats.index, textposition="middle right",
                textfont=dict(size=13),
                marker=dict(size=11, color="orange", symbol="diamond",
                            line=dict(width=1, color="black")),
                hovertemplate="%{text}<br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
            ))

            fig.update_layout(
                xaxis_title="Annual volatility (%)",
                yaxis_title="Expected annual return (%)",
                height=620,
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig)
            st.caption(
                "Each dot is a portfolio. The efficient frontier is the upper-left edge: "
                "the highest expected return for each level of risk. The Capital Market "
                "Line touches the frontier at the Max Sharpe portfolio."
            )

# ---------- Benchmark ----------

with tab_bench:
    if bench_rets is None:
        st.warning(f"Could not download {BENCHMARK}: comparison not available.")
    elif all_rets.empty:
        st.info("No portfolio to compare yet.")
    else:
        st.markdown("**Growth of $1 invested**")
        fig = px.line(all_rets.apply(growth_of_one))
        st.plotly_chart(style_time_chart(fig, "Value ($)", prefix="$"))

        st.markdown("**Historical performance**")
        summary = pd.DataFrame({
            name: performance_summary(all_rets[name], risk_free_rate)
            for name in all_rets.columns
        }).T
        st.dataframe(summary.round(2))

        st.markdown("**Drawdown** (loss from the previous peak)")
        fig = px.line(all_rets.apply(drawdown_series) * 100)
        st.plotly_chart(style_time_chart(fig, "Drawdown (%)", suffix="%"))

        if custom_weights is None:
            st.info("Your custom portfolio is not shown: its weights do not sum to 100%.")

        st.warning(
            "In-sample analysis: the optimized portfolios are built and evaluated on "
            "the same data, so this comparison is optimistic and does not predict "
            "future performance."
        )

# ---------- Risk Metrics ----------

with tab_risk:
    if all_rets.empty:
        st.info("No portfolio to analyze yet.")
    else:
        confidence = st.select_slider(
            "Confidence level",
            options=[0.90, 0.95, 0.99], value=0.95,
            format_func=lambda x: f"{x:.0%}",
        )
        level = f"{confidence:.0%}"

        bench_for_beta = all_rets[BENCHMARK] if BENCHMARK in all_rets else None
        risk_df = pd.DataFrame({
            name: risk_summary(all_rets[name], confidence, bench_for_beta)
            for name in all_rets.columns
        }).T
        st.dataframe(risk_df.round(2))
        st.caption(
            f"VaR and CVaR are 1-day losses, shown as positive numbers. "
            f"Beta is measured against {BENCHMARK} on daily returns."
        )
        if bench_for_beta is None:
            st.info(f"Beta is not shown because {BENCHMARK} could not be downloaded.")

        st.markdown("**Distribution of daily returns**")
        choice = st.selectbox("Portfolio", list(all_rets.columns))
        rets = all_rets[choice]
        var = historical_var(rets, confidence)
        cvar = historical_cvar(rets, confidence)

        fig = px.histogram(rets * 100, nbins=80)
        fig.add_vline(x=-var * 100, line_dash="dash", line_color="orange",
                      annotation_text=f"VaR {level}")
        fig.add_vline(x=-cvar * 100, line_dash="dash", line_color="red",
                      annotation_text=f"CVaR {level}",
                      annotation_position="bottom left")
        fig.update_layout(xaxis_title="Daily return (%)",
                          yaxis_title="Number of days", showlegend=False)
        st.plotly_chart(fig)
        st.caption(
            f"On {level} of days the loss was smaller than {var:.2%} (VaR). "
            f"On the remaining worst days, the average loss was {cvar:.2%} (CVaR)."
        )

        with st.expander("Limitations of these metrics"):
            st.markdown(
                "- **Historical window:** all metrics depend on one past period "
                "and do not predict future crises.\n"
                "- **Parametric VaR** assumes normal returns; real returns have "
                "fat tails, so it tends to underestimate extreme losses.\n"
                "- **VaR** says where the tail starts, not how deep it is; "
                "**CVaR** addresses this.\n"
                "- **1-day horizon:** scaling to 10 days with the square root of "
                "time assumes independent returns.\n"
                "- **Beta** changes over time and depends on the chosen benchmark."
            )

st.divider()
st.caption(
    "Data: Yahoo Finance via yfinance. Constant-weight portfolios rebalanced daily, "
    "no transaction costs. Past performance does not predict future results."
)