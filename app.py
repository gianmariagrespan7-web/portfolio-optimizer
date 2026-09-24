import streamlit as st
import pandas as pd
import numpy as np
import scipy
import plotly
import yfinance as yf

# Configurazione della pagina (deve essere il primo comando Streamlit)
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("Portfolio Optimizer")
st.caption("MVP: analisi e ottimizzazione di portafoglio con dati storici")

# Sidebar: qui metteremo ticker, date, risk-free rate
st.sidebar.header("Impostazioni")
st.sidebar.info("Gli input arriveranno nello STEP 2.")

# Controllo ambiente: mostra le versioni delle librerie
st.subheader("Controllo ambiente")
st.write({
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "plotly": plotly.__version__,
    "yfinance": yf.__version__,
})
st.success("Setup completato: tutte le librerie sono installate.")
