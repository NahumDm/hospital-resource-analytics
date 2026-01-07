import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Consultation dataset ---
logging.info("Loading Consultation dataset...")
try:
    # Adjust file path and engine if needed
    consultation = pd.read_csv(r"C:\Users\Abrish\hospital-resource-analytics\Consultation.csv")
except Exception as e:
    logging.error(f"Failed to load Consultation dataset: {e}")
    raise

logging.info(f"Consultation columns: {list(consultation.columns)}")

# --- Parse month robustly ---
def parse_month(x):
    try:
        x = str(x).strip()
        # Try YYYY-MM first
        try:
            return pd.Period(x, freq='M')
        except:
            pass
        # Try Mon-YY format
        try:
            return pd.Period(pd.to_datetime(x, format='%b-%y', errors='coerce'), freq='M')
        except:
            pass
        return pd.NaT
    except:
        return pd.NaT

if 'CALENDAR_MONTH_END_DATE' in consultation.columns:
    consultation['Month'] = consultation['CALENDAR_MONTH_END_DATE'].apply(parse_month)
    consultation = consultation.dropna(subset=['Month'])

# --- Select numeric columns ---
numeric_cols = consultation.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    logging.error("No numeric columns in dataset to train ML model.")
    raise ValueError("No numeric columns found")

# --- Use first numeric column as target, rest as features ---
target_col = numeric_cols[0]
feature_cols = [c for c in numeric_cols if c != target_col]

X = consultation[feature_cols].fillna(0)
y = consultation[target_col].fillna(0)

logging.info(f"Using '{target_col}' as target and {len(feature_cols)} features")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data split into training and testing sets.")

# --- Train Random Forest Regressor ---
logging.info("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
logging.info(f"Model Evaluation: RMSE = {rmse:.2f}, R² = {r2:.2f}")

# --- Feature importance ---
importances = model.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
logging.info("\nTop Important Features:\n" + str(feat_imp.head(10)))

# --- Save model ---
joblib.dump(model, 'rf_model.pkl')
logging.info("ML model trained and saved successfully.")

# --- First 10 predictions vs actual ---
preds_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test.values})
logging.info("First 10 predictions vs actual:\n" + str(preds_df.head(10)))
