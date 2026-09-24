from datetime import date, timedelta

import numpy as np
import plotly.express as px
import streamlit as st

from src.data import clean_prices, download_prices, parse_tickers
from src.portfolio import (
    annualized_volatility,
    asset_summary,
    average_correlation,
    correlation_matrix,
    covariance_matrix,
    cumulative_returns,
    daily_returns,
    portfolio_performance,
)

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("Portfolio Optimizer")
st.caption("MVP: analisi e ottimizzazione di portafoglio con dati storici")


# Streamlit riesegue tutto lo script a ogni interazione.
# La cache evita di riscaricare gli stessi dati (validi per 1 ora).
@st.cache_data(ttl=3600, show_spinner="Download dei dati...")
def load_prices(tickers, start, end):
    return download_prices(tickers, start, end)


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
st.dataframe((asset_summary(prices) * 100).round(2))

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

# --- PORTAFOGLIO ---
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
