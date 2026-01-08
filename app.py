import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

st.set_page_config(page_title="Time Series Forecast App", layout="centered")

st.title("📈 Time Series Forecasting App")
st.write("Upload your data and get predictions")

# --- Upload data ---
uploaded_file = st.file_uploader("Upload CSV (date,value)", type=["csv"])

model_choice = st.selectbox(
    "Choose model",
    ["LSTM", "SARIMA", "Prophet"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    ts = pd.Series(df["value"].values, index=df["date"])

    st.line_chart(ts)

    if st.button("Run Forecast"):
        if model_choice == "LSTM":
            bundle = joblib.load("models/lstm_model.pkl")
            model = bundle["model"]
            scaler = bundle["scaler"]

            # Prepare last window
            WINDOW = 6
            arr = ts.values.reshape(-1,1)
            arr_s = scaler.transform(arr)
            last = arr_s[-WINDOW:].reshape(1, WINDOW, 1)

            pred_s = model.predict(last)
            pred = scaler.inverse_transform(pred_s)[0][0]

        elif model_choice == "SARIMA":
            bundle = joblib.load("models/sarima_model.pkl")
            model = bundle["model"]
            pred = model.get_forecast(steps=1).predicted_mean.iloc[0]

        else:  # Prophet
            bundle = joblib.load("models/prophet_model.pkl")
            model = bundle["model"]

            future = model.make_future_dataframe(periods=1, freq="MS")
            forecast = model.predict(future)
            pred = forecast["yhat"].iloc[-1]

        st.success(f"📊 Forecasted value: **{pred:.2f}**")
