from datetime import date, timedelta

import plotly.express as px
import streamlit as st

from src.data import clean_prices, download_prices, parse_tickers

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

# --- OUTPUT ---
st.subheader("Dati scaricati")
col1, col2, col3 = st.columns(3)
col1.metric("Titoli", prices.shape[1])
col2.metric("Giorni di borsa", prices.shape[0])
col3.metric("Dal", f"{prices.index[0]:%d.%m.%Y}")

st.subheader("Prezzi aggiustati")
fig = px.line(prices, labels={"value": "Prezzo", "Date": "Data"})
st.plotly_chart(fig)

st.subheader("Andamento normalizzato (base 100)")
normalized = prices / prices.iloc[0] * 100
fig2 = px.line(normalized, labels={"value": "Valore (base 100)", "Date": "Data"})
st.plotly_chart(fig2)

with st.expander("Mostra tabella dati"):
    st.dataframe(prices.tail(10).round(2))
