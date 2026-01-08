import pandas as pd
import numpy as np
import logging
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# --- Configuration / paths ---
CSV_PATH = r"C:\Users\Abrish\hospital-resource-analytics\ER Wait Time Dataset.csv"


def safe_import(name):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None


def load_data(path=CSV_PATH):
    logger.info(f"Loading dataset from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataframe shape: {df.shape}")
    return df


def train_cross_sectional(df):
    # Use 'Total Wait Time (min)' as target if present
    target = 'Total Wait Time (min)'
    if target not in df.columns:
        # fallback: pick first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError('No numeric columns found for cross-sectional model')
        target = numeric_cols[0]

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target]
    if not feature_cols:
        raise ValueError('No numeric features available')

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    logger.info(f"Cross-sectional: using target='{target}' and {len(feature_cols)} numeric features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info(f"RandomForest — RMSE: {rmse:.2f}, R2: {r2:.3f}")

    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
    logger.info("Top features:\n" + str(feat_imp.head(10)))

    joblib.dump({'model': rf, 'features': feature_cols}, 'rf_cross_sectional.pkl')
    logger.info('Saved rf_cross_sectional.pkl')

    return {'model': rf, 'rmse': rmse, 'r2': r2}


def prepare_monthly_series(df, hospital_id=None):
    # Parse visit date and compute monthly median wait
    if 'Visit Date' not in df.columns:
        raise ValueError('Visit Date column required for time-series models')
    df['Visit Date'] = pd.to_datetime(df['Visit Date'], errors='coerce')
    if hospital_id:
        s = df[df['Hospital ID'] == hospital_id]
    else:
        s = df

    # resample by month
    ts = s.set_index('Visit Date')['Total Wait Time (min)'].resample('MS').median().dropna()
    logger.info(f"Prepared monthly series (hospital_id={hospital_id}) length={len(ts)}")
    return ts


def eval_series_forecast(true, pred):
    # align
    common = true.index.intersection(pred.index)
    if len(common) == 0:
        return None
    y_true = true.loc[common].values
    y_pred = pred.loc[common].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def try_prophet(ts):
    prophet = safe_import('prophet') or safe_import('fbprophet')
    if prophet is None:
        logger.warning('Prophet not installed — skipping')
        return None
    # prophet import name differs
    try:
        from prophet import Prophet
    except Exception:
        from fbprophet import Prophet

    dfprop = ts.reset_index()
    dfprop.columns = ['ds', 'y']
    # train/test: use 80/20 split
    n = len(dfprop)
    if n < 12:
        logger.warning('Time series too short for Prophet — skipping')
        return None
    split = int(n * 0.8)
    train = dfprop.iloc[:split]
    test = dfprop.iloc[split:]
    periods = len(test)
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=periods, freq='MS')
    forecast = m.predict(future)
    fc = forecast.set_index('ds')['yhat']
    # get predictions for test period
    pred = fc.loc[test['ds']]
    rmse = eval_series_forecast(test.set_index('ds')['y'], pred)
    logger.info(f'Prophet RMSE: {rmse:.2f}')
    joblib.dump({'model': m}, 'prophet_model.pkl')
    return {'rmse': rmse}


def try_sarima(ts):
    statsmodels = safe_import('statsmodels')
    if statsmodels is None:
        logger.warning('statsmodels not installed — skipping SARIMA')
        return None
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if len(ts) < 24:
        logger.warning('Time series too short for SARIMA — skipping')
        return None
    # use 80/20 train/test split
    n = len(ts)
    split = int(n * 0.8)
    train = ts.iloc[:split]
    test = ts.iloc[split:]
    # simple seasonal order (p,d,q)x(P,D,Q,s) chosen lightly
    try:
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=len(test)).predicted_mean
        rmse = eval_series_forecast(test, pred)
        logger.info(f'SARIMA RMSE: {rmse:.2f}')
        joblib.dump({'model': res}, 'sarima_model.pkl')
        return {'rmse': rmse}
    except Exception as e:
        logger.error(f'SARIMA failed: {e}')
        return None


def try_lstm(ts):
    tf = safe_import('tensorflow')
    if tf is None:
        logger.warning('TensorFlow not installed — skipping LSTM')
        return None
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    if len(ts) < 36:
        logger.warning('Time series too short for LSTM — skipping')
        return None
    arr = ts.values.reshape(-1,1).astype('float32')
    scaler = MinMaxScaler()
    arr_s = scaler.fit_transform(arr)

    # windowing
    WINDOW = 6
    X, Y = [], []
    for i in range(len(arr_s)-WINDOW):
        X.append(arr_s[i:i+WINDOW, 0])
        Y.append(arr_s[i+WINDOW, 0])
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

    preds_s = model.predict(X_test)
    preds = scaler.inverse_transform(preds_s)
    true = scaler.inverse_transform(y_test.reshape(-1,1))
    rmse = np.sqrt(mean_squared_error(true, preds))
    logger.info(f'LSTM RMSE: {rmse:.2f}')
    joblib.dump({'model': model, 'scaler': scaler}, 'lstm_model.pkl')
    return {'rmse': rmse}


def main():
    df = load_data()

    # Cross-sectional model
    try:
        cs_res = train_cross_sectional(df)
    except Exception as e:
        logger.error(f'Cross-sectional model failed: {e}')
        cs_res = None

    # Time series: overall monthly median
    try:
        ts = prepare_monthly_series(df)
        # baseline: simple persistence forecast (last value)
        if len(ts) >= 12:
            n = len(ts)
            split = int(n * 0.8)
            train = ts.iloc[:split]
            test = ts.iloc[split:]
            persistence_pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)
            persistence_rmse = eval_series_forecast(test, persistence_pred)
            logger.info(f'Persistence baseline RMSE: {persistence_rmse:.2f}')
        else:
            logger.info('Time series too short for baseline/test')

        prophet_res = try_prophet(ts)
        sarima_res = try_sarima(ts)
        lstm_res = try_lstm(ts)
    except Exception as e:
        logger.error(f'Time-series experiments failed: {e}')


if __name__ == '__main__':
    main()
