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
    "Gold": "GC"}
#select company
ticker = st.selectbox("Select a ticker symbol:", ticker_mapping.keys())
#select date 
start_date = "2016-06-15"

# Get the historical data for the specified ticker and start date
df = yf.Ticker(ticker_mapping[ticker]).history(start=start_date, end=None)

# Display the head of the dataframe
st.dataframe(df.head())
