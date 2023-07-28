import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import yfinance as yf

# company dictionary 
ticker_mapping = {
    "Google": "GOOG",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Gold": "GC=F",
    "SP500": "^GSPC"}


# set the page
st.set_page_config(page_title="Valor de mercado", page_icon="📊", layout="wide")

# get the ticker from the user
ticker = st.sidebar.selectbox("Selecciona la compañia:", ticker_mapping.keys())
# setup for the ticker
if ticker == "Google":
    st.sidebar.image("img/logo1.png")
    start_date = "2016-06-15"
    num_ac = 12609
elif ticker == "Microsoft":
    st.sidebar.image("img/logo2.png")
    start_date = "2016-03-02"
    num_ac = 7432
elif ticker == "Tesla":
    st.sidebar.image("img/logo3.png")
    start_date = "2016-03-02"
    num_ac = 3169
elif ticker == "Gold":
    st.sidebar.image("img/logo4.png")
    start_date = "2016-02-02"
    num_ac = 0
else:
    st.sidebar.image("img/logo5.png")
    start_date = "2016-02-02"
    num_ac = 0

# Get the historical data for the specified ticker and start date
df = yf.Ticker(ticker_mapping[ticker]).history(start=start_date, end=None)

st.title("Análisis de mercado")

placeholder = st.empty()


market_cap = (df['Close'].iloc[-1]  * num_ac)/1000000
rpd = ((df['Close'].iloc[-1] - df['Open'].iloc[-1])/df['Open'].iloc[-1])*100
rpd2=((df['Close'].iloc[-2] - df['Open'].iloc[-2])/df['Open'].iloc[-2])*100


with placeholder.container():
    kpi2, kpi3, kpi4, kpi5 = st.columns(4)

    kpi2.metric(label="Precio cierre", value=round(df["Close"].iloc[-1],2), delta= round(df['Close'].iloc[-1]-df['Close'].iloc[-2], 2))
    kpi3.metric(label="Precio apertura", value=round(df["Open"].iloc[-1],2), delta= round(df['Open'].iloc[-1]-df['Open'].iloc[-2], 2))
    kpi4.metric(label="Rendimiento porcentual diario", value=f" {round(rpd, 2)} %", delta = round(rpd- rpd2, 2))
    kpi5.metric(label="Capitalización bursátil", value=f" $ {round(market_cap,2)} T", delta= round(market_cap - df['Close'].iloc[-2]  * num_ac))


    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("### Volumen de Operaciones")
        fig = px.bar(data_frame=df, y='Volume')
        st.write(fig)
    with fig_col2:
        st.markdown("### Tendencia del Precio")
        fig2 = px.line(data_frame=df, y='Close')
        st.write(fig2)

# Display the head of the dataframe
st.markdown("## Predicciones del mercado")
st.dataframe(df.tail())
