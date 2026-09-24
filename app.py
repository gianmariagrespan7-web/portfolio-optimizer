from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import clean_prices, download_prices, parse_tickers
from src.optimization import (
    efficient_frontier,
    optimal_portfolios,
    random_portfolios,
)
from src.portfolio import (
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
    performance_summary,
    portfolio_daily_returns,
    portfolio_performance,
)

BENCHMARK = "SPY"

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("Portfolio Optimizer")
st.caption("MVP: analisi e ottimizzazione di portafoglio con dati storici")


# Streamlit riesegue tutto lo script a ogni interazione.
# La cache evita di riscaricare gli stessi dati (validi per 1 ora).
@st.cache_data(ttl=3600, show_spinner="Download dei dati...")
def load_prices(tickers, start, end):
    return download_prices(tickers, start, end)


# Anche la frontiera e in cache: sono ~40 ottimizzazioni + 3000 portafogli
@st.cache_data(show_spinner="Calcolo della frontiera efficiente...")
def compute_frontier(returns, risk_free_rate):
    mean_returns = annualized_return(returns).to_numpy()
    cov = covariance_matrix(returns).to_numpy()
    frontier = efficient_frontier(mean_returns, cov)
    cloud = random_portfolios(mean_returns, cov, risk_free_rate)
    return frontier, cloud


# --- SIDEBAR ---
st.sidebar.header("Impostazioni")
ticker_text = st.sidebar.text_input(
    "Ticker (separati da virgola)",
    value="AAPL, MSFT, NVDA, AMZN, GOOGL, META",
)
start = st.sidebar.date_input(
    "Data inizio", value=date.today() - timedelta(days=5 * 365)
)
end = st.sidebar.date_input("Data fine", value=date.today())
risk_free_pct = st.sidebar.number_input(
    "Risk-free rate annuo (%)",
    min_value=0.0, max_value=20.0, value=4.0, step=0.1,
)
risk_free_rate = risk_free_pct / 100

tickers = parse_tickers(ticker_text)

# --- CONTROLLI SUGLI INPUT ---
if not tickers:
    st.info("Inserisci almeno un ticker nella sidebar.")
    st.stop()
if start >= end:
    st.error("La data di inizio deve essere prima della data di fine.")
    st.stop()

# --- DOWNLOAD ---
try:
    raw = load_prices(tuple(tickers), start, end)
except Exception as e:
    st.error(f"Errore nel download dei dati: {e}")
    st.stop()

if raw.empty:
    st.error("Nessun dato scaricato. Controlla i ticker e la connessione.")
    st.stop()

prices, invalid = clean_prices(raw)

if invalid:
    st.warning(f"Ticker non trovati e ignorati: {', '.join(invalid)}")
if prices.empty:
    st.error("Nessun dato valido per il periodo scelto.")
    st.stop()
if (prices.index[0].date() - start).days > 10:
    st.info(
        f"I dati comuni a tutti i titoli iniziano il {prices.index[0]:%d.%m.%Y} "
        "(probabilmente un titolo e quotato da meno tempo)."
    )

# --- OUTPUT: DATI ---
st.subheader("Dati scaricati")
col1, col2, col3 = st.columns(3)
col1.metric("Titoli", prices.shape[1])
col2.metric("Giorni di borsa", prices.shape[0])
col3.metric("Dal", f"{prices.index[0]:%d.%m.%Y}")

st.subheader("Prezzi aggiustati")
fig = px.line(prices, labels={"value": "Prezzo", "Date": "Data"})
st.plotly_chart(fig)

with st.expander("Mostra tabella prezzi"):
    st.dataframe(prices.tail(10).round(2))

# --- OUTPUT: RENDIMENTI ---
returns = daily_returns(prices)

st.subheader("Rendimento cumulato")
cum = cumulative_returns(returns) * 100
fig2 = px.line(cum, labels={"value": "Rendimento cumulato (%)", "Date": "Data"})
st.plotly_chart(fig2)

st.subheader("Statistiche per titolo")
st.caption("Valori in %. Rendimenti e volatilita annualizzati con 252 giorni di borsa.")
stats = asset_summary(prices)
st.dataframe((stats * 100).round(2))

# --- CORRELAZIONE E COVARIANZA ---
if prices.shape[1] > 1:
    st.subheader("Correlazione tra i titoli")
    corr = correlation_matrix(returns)

    st.metric("Correlazione media tra i titoli", f"{average_correlation(corr):.2f}")

    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",  # rosso = positiva, blu = negativa
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig3)
    st.caption(
        "Correlazione dei rendimenti giornalieri. Piu e bassa, "
        "maggiore e il beneficio della diversificazione."
    )

    with st.expander("Mostra matrice di covarianza (annualizzata)"):
        st.dataframe(covariance_matrix(returns).round(4))
        st.caption(
            "Sulla diagonale: varianze dei singoli titoli. "
            "La radice quadrata della diagonale e la volatilita annua."
        )

# --- PORTAFOGLI OTTIMIZZATI ---
portfolios = {}
if prices.shape[1] > 1:
    st.subheader("Portafogli ottimizzati")
    st.caption("Long-only, pesi tra 0% e 100%, somma = 100%. Ottimizzazione con scipy (SLSQP).")

    try:
        portfolios = optimal_portfolios(returns, risk_free_rate)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Tabella di confronto delle metriche
    rows = {}
    for name, w in portfolios.items():
        ret, vol, shp = portfolio_performance(w, returns, risk_free_rate)
        rows[name] = {
            "Rendimento atteso (%)": ret * 100,
            "Volatilita (%)": vol * 100,
            "Sharpe ratio": shp,
        }
    st.dataframe(pd.DataFrame(rows).T.round(2))

    # Pesi di ogni portafoglio
    weights_df = pd.DataFrame(portfolios, index=prices.columns) * 100
    fig4 = px.bar(
        weights_df,
        barmode="group",
        labels={"value": "Peso (%)", "index": "Titolo", "variable": "Portafoglio"},
    )
    st.plotly_chart(fig4)

    with st.expander("Mostra tabella dei pesi (%)"):
        st.dataframe(weights_df.round(2))

# --- FRONTIERA EFFICIENTE ---
if prices.shape[1] > 1:
    st.subheader("Frontiera efficiente")

    frontier, cloud = compute_frontier(returns, risk_free_rate)

    fig5 = go.Figure()

    # 1. Nuvola di portafogli casuali, colorati per Sharpe
    fig5.add_trace(go.Scatter(
        x=cloud["Volatilita"] * 100, y=cloud["Rendimento"] * 100,
        mode="markers", name="Portafogli casuali",
        marker=dict(size=4, color=cloud["Sharpe"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe")),
    ))

    # 2. Frontiera efficiente
    fig5.add_trace(go.Scatter(
        x=frontier["Volatilita"] * 100, y=frontier["Rendimento"] * 100,
        mode="lines", name="Frontiera efficiente",
        line=dict(color="black", width=3),
    ))

    # 3. Capital Market Line: da Rf, tangente al portafoglio Max Sharpe
    _, _, max_shp = portfolio_performance(
        portfolios["Max Sharpe"], returns, risk_free_rate
    )
    x_cml = np.array([0, frontier["Volatilita"].max() * 100])
    fig5.add_trace(go.Scatter(
        x=x_cml, y=risk_free_pct + max_shp * x_cml,
        mode="lines", name="Capital Market Line",
        line=dict(color="gray", dash="dash"),
    ))

    # 4. Singoli titoli
    fig5.add_trace(go.Scatter(
        x=stats["Volatilita annua"] * 100,
        y=stats["Rendimento medio annuo"] * 100,
        mode="markers+text", name="Singoli titoli",
        text=stats.index, textposition="top center",
        marker=dict(size=10, color="orange", symbol="diamond"),
    ))

    # 5. Portafogli ottimali (stelle)
    for name, color in [("Min Volatility", "blue"), ("Max Sharpe", "red")]:
        r, v, _ = portfolio_performance(portfolios[name], returns, risk_free_rate)
        fig5.add_trace(go.Scatter(
            x=[v * 100], y=[r * 100], mode="markers", name=name,
            marker=dict(size=18, color=color, symbol="star",
                        line=dict(width=1, color="black")),
        ))

    fig5.update_layout(
        xaxis_title="Volatilita annua (%)",
        yaxis_title="Rendimento atteso annuo (%)",
        height=600,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig5)
    st.caption(
        "Ogni punto e un portafoglio. La frontiera e il bordo superiore sinistro: "
        "il massimo rendimento per ogni livello di rischio."
    )

# --- PORTAFOGLIO MANUALE ---
st.subheader("Il tuo portafoglio")
st.caption("Inserisci il peso di ogni titolo in %. Il totale deve essere 100%.")

# L'ordine dei pesi deve seguire l'ordine delle colonne dei dati
assets = list(prices.columns)
default_weight = round(100 / len(assets), 2)

cols = st.columns(4)
weights_pct = []
for i, ticker in enumerate(assets):
    w = cols[i % 4].number_input(
        ticker, min_value=0.0, max_value=100.0,
        value=default_weight, step=1.0, key=f"w_{ticker}",
    )
    weights_pct.append(w)

total = sum(weights_pct)
if abs(total - 100) > 0.1:
    st.error(f"La somma dei pesi e {total:.2f}%: deve essere 100%.")
    st.stop()

# Normalizziamo per ottenere una somma esattamente uguale a 1
weights = np.array(weights_pct) / total

port_ret, port_vol, sharpe = portfolio_performance(weights, returns, risk_free_rate)

m1, m2, m3 = st.columns(3)
m1.metric("Rendimento atteso annuo", f"{port_ret:.2%}")
m2.metric("Volatilita annua", f"{port_vol:.2%}")
m3.metric("Sharpe ratio", f"{sharpe:.2f}")

avg_vol = weights @ annualized_volatility(returns).to_numpy()
st.caption(
    f"Media pesata delle volatilita dei singoli titoli: {avg_vol:.2%}. "
    f"Grazie alla diversificazione la volatilita del portafoglio e piu bassa "
    f"di {(avg_vol - port_vol) * 100:.2f} punti percentuali."
)

# --- CONFRONTO CON IL BENCHMARK ---
st.subheader(f"Confronto con il benchmark ({BENCHMARK})")

try:
    bench_raw = load_prices((BENCHMARK,), start, end)
    bench_prices, _ = clean_prices(bench_raw)
except Exception:
    bench_prices = pd.DataFrame()

if bench_prices.empty:
    st.warning(f"Impossibile scaricare {BENCHMARK}: confronto non disponibile.")
else:
    # Rendimenti giornalieri di ogni portafoglio + benchmark
    series = {"Il tuo portafoglio": portfolio_daily_returns(returns, weights)}
    for name, w in portfolios.items():
        series[name] = portfolio_daily_returns(returns, w)
    series[BENCHMARK] = daily_returns(bench_prices)[BENCHMARK]

    # dropna allinea le date: teniamo solo i giorni comuni a tutti
    all_rets = pd.DataFrame(series).dropna()

    st.markdown("**Crescita di 1 $ investito**")
    fig6 = px.line(
        all_rets.apply(growth_of_one),
        labels={"value": "Valore ($)", "index": "Data", "variable": "Portafoglio"},
    )
    st.plotly_chart(fig6)

    st.markdown("**Metriche di performance storica**")
    summary = pd.DataFrame({
        name: performance_summary(all_rets[name], risk_free_rate)
        for name in all_rets.columns
    }).T
    st.dataframe(summary.round(2))

    st.markdown("**Drawdown (perdita rispetto al massimo precedente)**")
    fig7 = px.line(
        all_rets.apply(drawdown_series) * 100,
        labels={"value": "Drawdown (%)", "index": "Data", "variable": "Portafoglio"},
    )
    st.plotly_chart(fig7)

    st.warning(
        "Attenzione: i portafogli ottimizzati sono calcolati sugli stessi dati "
        "su cui vengono valutati (analisi in-sample). Il confronto con il "
        "benchmark e quindi ottimistico e non indica performance future."
    )