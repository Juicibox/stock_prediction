import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf

import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler



# company dictionary 
ticker_mapping = {
    "Google": "GOOG",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Gold": "GC=F",
    "SP500": "^GSPC"}


# set the page

st.set_page_config(page_title="Valor de mercado",page_icon="logo.png", layout="wide")


#load array
with open('sequen_goog', 'rb') as f:
    seque_goo = pickle.load(f)
with open('sequen_msft', 'rb') as f:
    seque_ms = pickle.load(f)
with open('sequen_tsla', 'rb') as f:
    seque_ts = pickle.load(f)
with open('sequen_oro', 'rb') as f:
    seque_oro = pickle.load(f)
with open('sequen_500', 'rb') as f:
    seque_500 = pickle.load(f)

# get the ticker from the user
ticker = st.sidebar.selectbox("Selecciona la Acción:", ticker_mapping.keys())
# setup for the ticker
if ticker == "Google":
    st.sidebar.image("img/logo1.png")
    start_date = "2018-06-15"
    num_ac = 12609
    sequences = seque_goo
    final_model = tf.keras.models.load_model('model/model_goog.h5')
elif ticker == "Microsoft":
    st.sidebar.image("img/logo2.png")
    start_date = "2019-03-02"
    num_ac = 7432
    sequences = seque_ms
    final_model = tf.keras.models.load_model('model/model_msft.h5')
elif ticker == "Tesla":
    st.sidebar.image("img/logo3.png")
    start_date = "2018-03-02"
    num_ac = 3169
    sequences = seque_ts
    final_model = tf.keras.models.load_model('model/model_tsla.h5')
elif ticker == "Gold":
    st.sidebar.image("img/logo4.png")
    start_date = "2016-02-02"
    num_ac = 0
    sequences = seque_oro
    final_model = tf.keras.models.load_model('model/model_oro.h5')
else:
    st.sidebar.image("img/logo5.png")
    start_date = "2018-02-02"
    num_ac = 0
    sequences = seque_500
    final_model = tf.keras.models.load_model('model/model_500.h5')

# Get the historical data for the specified ticker and start date
df = yf.Ticker(ticker_mapping[ticker]).history(start=start_date, end=None)

st.title("Análisis del Mercado de Valores")

placeholder = st.empty()

if not df.empty and len(df) > 1:
    market_cap = (df['Close'].iloc[-1] * num_ac) / 1000000
    rpd = ((df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1]) * 100
    rpd2 = ((df['Close'].iloc[-2] - df['Open'].iloc[-2]) / df['Open'].iloc[-2]) * 100

    with placeholder.container():
        kpi2, kpi3, kpi4, kpi5 = st.columns(4)

        kpi2.metric(label="Precio Cierre", value=round(df["Close"].iloc[-1], 2), delta=round(df['Close'].iloc[-1] - df['Close'].iloc[-2], 2))
        kpi3.metric(label="Precio Apertura", value=round(df["Open"].iloc[-1], 2), delta=round(df['Open'].iloc[-1] - df['Open'].iloc[-2], 2))
        kpi4.metric(label="Rendimiento Porcentual Diario", value=f" {round(rpd, 2)} %", delta=round(rpd - rpd2, 2))
        kpi5.metric(label="Capitalización Bursátil", value=f" $ {round(market_cap, 2)} T", delta=round(market_cap - df['Close'].iloc[-2] * num_ac))
else:
    st.error("No hay suficientes datos disponibles para realizar el análisis.")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("### Volumen de Operaciones")
        fig = px.bar(data_frame=df, y='Volume')
        st.write(fig)
    with fig_col2:
        st.markdown("### Tendencia del Precio")
        fig2 = px.line(data_frame=df, y='Close')
        st.write(fig2)

def predic():
    last_sequence = sequences[-1:, 1:, :]

    # Generate predictions for the next 10 days
    PRED_DAYS = 10
    for i in range(PRED_DAYS):
        pred_i = final_model.predict(last_sequence, verbose=0)
        # Append array1 to array2
        last_sequence = np.concatenate((last_sequence, pred_i[np.newaxis, :, :]), axis=1)
        # Remove the first item
        last_sequence = last_sequence[:, 1:, :]
    df2 = df.copy(deep=True)
    scaler = MinMaxScaler(feature_range=(0, 15)).fit(df2.Low.values.reshape(-1, 1))
    pred_days = scaler.inverse_transform(last_sequence.reshape(PRED_DAYS, 4))

    df_pred = pd.DataFrame(
        data=pred_days,
        columns=['Open', 'High', 'Low', 'Close'])
    df_fin = df_pred
    df_fin['Date'] = pd.date_range(start='2024-06-11', periods=10)

    return df_fin

# Display the head of the dataframe
st.markdown("## Predicciones del Mercado")
df_fin = predic()
df_fin = df_fin.set_index('Date')
pre1, pre2 = st.columns(2)
with pre1:
    st.markdown("### Tabla de Predicciones")
    st.dataframe(df_fin)

with pre2:
    st.markdown("### Tendencia de la Predicción")
    fig4 = px.line(data_frame=df_fin, y='Close')
    fig4.update_xaxes(tickmode='linear', dtick='D1')
    st.write(fig4)

