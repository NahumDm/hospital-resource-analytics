import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Auto-Clean Time Series App")

st.title("🧼 Auto-Clean Time Series Forecast App")
st.write("Upload a CSV — data will be cleaned automatically")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def auto_clean_timeseries(df):
    # ---- Detect date column ----
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

    # ---- Detect numeric value column ----
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        # try coercing non-numeric columns
        for col in df.columns:
            if col != date_col:
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().sum() > len(df) * 0.5:
                    df[col] = coerced
                    numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No numeric column detected")

    value_col = numeric_cols[0]  # auto-pick first valid numeric column

    # ---- Clean dataframe ----
    df = df[[date_col, value_col]].copy()
    df.dropna(inplace=True)

    # ---- Remove duplicate dates ----
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=date_col, keep="last")

    # ---- Build time series ----
    ts = pd.Series(
        data=df[value_col].values,
        index=df[date_col]
    ).sort_index()

    return ts, date_col, value_col


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

    except Exception as e:
        st.error(f"❌ Auto-clean failed: {e}")

else:
    st.info("👆 Upload a CSV file to start")
