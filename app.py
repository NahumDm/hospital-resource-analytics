# ---------------------------
# Streamlit Auto-Clean & Forecast App
# ---------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="🧼 Auto-Clean Time Series Forecast App", layout="wide")
st.title("🧼 Auto-Clean Time Series Forecast App")
st.write("Upload a CSV — data will be cleaned automatically, then predict using your ML models!")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ---------------------------
# Paths to saved models
# ---------------------------
LSTM_PATH = "models/lstm_model.pkl"
SARIMA_PATH = "models/sarima_model.pkl"
PROPHET_PATH = "models/prophet_model.pkl"

# ---------------------------
# Auto-clean function
# ---------------------------
def auto_clean_timeseries(df):
    date_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                date_col = col
                df[col] = parsed
                break
        except Exception:
            pass
    if date_col is None:
        raise ValueError("No valid date column detected")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        for col in df.columns:
            if col != date_col:
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().sum() > len(df) * 0.5:
                    df[col] = coerced
                    numeric_cols.append(col)
    if not numeric_cols:
        raise ValueError("No numeric column detected")
    
    value_col = numeric_cols[0]
    df = df[[date_col, value_col]].copy()
    df.dropna(inplace=True)
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=date_col, keep="last")
    
    ts = pd.Series(data=df[value_col].values, index=df[date_col]).sort_index()
    return ts, date_col, value_col

# ---------------------------
# SARIMA evaluation
# ---------------------------
def evaluate_sarima(ts, future_periods=None):
    model = joblib.load(SARIMA_PATH)
    n = len(ts)
    train, test = ts, ts  # full series as SARIMA model already trained
    forecast_hist = model.forecast(steps=len(test))
    
    rmse = np.sqrt(mean_squared_error(test, forecast_hist))
    r2 = r2_score(test, forecast_hist)
    
    # Future forecast
    if future_periods:
        future_index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(), periods=future_periods, freq='MS')
        forecast_future = model.forecast(steps=future_periods)
    else:
        forecast_future, future_index = None, None
    
    return forecast_hist, forecast_future, rmse, r2, future_index

# ---------------------------
# Prophet evaluation
# ---------------------------
def evaluate_prophet(ts, future_periods=None):
    model = joblib.load(PROPHET_PATH)
    df = ts.reset_index()
    df.columns = ["ds", "y"]
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    
    future = model.make_future_dataframe(periods=len(test) + (future_periods if future_periods else 0), freq='MS')
    forecast = model.predict(future)
    
    y_pred_hist = forecast["yhat"].iloc[:len(test)].values
    y_true = test["y"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_hist))
    r2 = r2_score(y_true, y_pred_hist)
    
    y_pred_future = forecast["yhat"].iloc[len(test):].values if future_periods else None
    future_index = forecast["ds"].iloc[len(test):] if future_periods else None
    
    return y_pred_hist, y_pred_future, rmse, r2, future_index

# ---------------------------
# LSTM evaluation with zigzag
# ---------------------------
def create_sequences(data, window=6):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def evaluate_lstm(ts, future_periods=None, window=3, max_change=0.0001):
    model_dict = joblib.load(LSTM_PATH)
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    scaled = scaler.transform(ts.values.reshape(-1,1))
    X, y = create_sequences(scaled, window)
    
    split = int(len(X) * 0.008)
    X_test, y_test = X[split:], y[split:]
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Historical predictions
    y_pred_hist = model.predict(X_test)
    y_pred_hist_inv = scaler.inverse_transform(y_pred_hist)
    
    # Future predictions with zigzag
    last_window = scaled[-window:].copy()
    preds_future_scaled = []
    
    for _ in range(future_periods if future_periods else 0):
        input_window = last_window.reshape(1, window, 1)
        pred = model.predict(input_window)[0,0]
        
        # Zigzag logic
        last_value = last_window[-1]
        pred = np.clip(pred, last_value*(1-max_change), last_value*(1+max_change))
        pred += np.random.uniform(-0.01, 0.01)  # small wiggle
        preds_future_scaled.append(pred)
        last_window = np.roll(last_window, -1)
        last_window[-1] = pred
    
    if preds_future_scaled:
        preds_future_scaled = np.array(preds_future_scaled).reshape(-1,1)
        preds_future = scaler.inverse_transform(preds_future_scaled).flatten()
        future_index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(), periods=future_periods, freq='MS')
    else:
        preds_future, future_index = None, None
    
    rmse = np.sqrt(mean_squared_error(ts.values[-len(y_pred_hist_inv):], scaler.inverse_transform(y_pred_hist).flatten()))
    r2 = r2_score(ts.values[-len(y_pred_hist_inv):], scaler.inverse_transform(y_pred_hist).flatten())
    
    return y_pred_hist_inv.flatten(), preds_future, rmse, r2, future_index

# ---------------------------
# Main App Logic
# ---------------------------
if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.subheader("Raw data preview")
        st.dataframe(raw_df.head())
        
        ts, date_col, value_col = auto_clean_timeseries(raw_df)
        st.success("✅ Data cleaned automatically")
        st.write(f"Detected date column: **{date_col}**")
        st.write(f"Detected value column: **{value_col}**")
        st.write(f"Rows after cleaning: **{len(ts)}**")
        st.line_chart(ts)

        # Future forecast option
        st.subheader("Forecast Options")
        predict_until = st.date_input("Predict until date", value=pd.to_datetime("2027-12-31"))
        future_periods = (pd.to_datetime(predict_until).year - ts.index[-1].year) * 12 + (pd.to_datetime(predict_until).month - ts.index[-1].month)
        if future_periods <= 0:
            future_periods = 0

        # Model selection
        st.subheader("Choose Model to Predict & Evaluate")
        model_choice = st.selectbox("Select a model", ["SARIMA", "Prophet", "LSTM"])

        if st.button("Run Prediction & Accuracy"):
            with st.spinner("Running model..."):
                if model_choice == "SARIMA":
                    hist, future_preds, rmse, r2, future_index = evaluate_sarima(ts, future_periods)
                elif model_choice == "Prophet":
                    hist, future_preds, rmse, r2, future_index = evaluate_prophet(ts, future_periods)
                else:
                    hist, future_preds, rmse, r2, future_index = evaluate_lstm(ts, future_periods)

            st.success(f"✅ {model_choice} Prediction completed!")
            st.metric("RMSE (historical)", round(rmse, 3))
            st.metric("R² Score (historical)", round(r2, 3))

            ts_hist = pd.Series(hist, index=ts.index[-len(hist):])
            if future_preds is not None:
                ts_future = pd.Series(future_preds, index=future_index)
                ts_full = pd.concat([ts_hist, ts_future])
            else:
                ts_full = ts_hist

            st.subheader("Forecast Chart")
            st.line_chart(ts_full)

    except Exception as e:
        st.error(f"❌ Auto-clean or prediction failed: {e}")

else:
    st.info("👆 Upload a CSV file to start")
