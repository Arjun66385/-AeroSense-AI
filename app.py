
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="AeroSense AI",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AeroSense AI")
st.subheader("Smart City Air Quality Forecasting System")

st.write(
    "An educational machine-learning prototype that forecasts "
    "the next hour's AQI using historical measurements."
)

MODEL_PATH = Path("models/aerosense_rf_model.pkl")
FEATURE_PATH = Path("models/aerosense_features.pkl")
DATA_PATH = Path("aqi_processed.csv")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features

def get_aqi_advisory(aqi):
    if aqi <= 50:
        return "Good", "Air quality is good. Minimal impact is expected."
    elif aqi <= 100:
        return "Satisfactory", "Sensitive individuals should monitor symptoms."
    elif aqi <= 200:
        return "Moderate", "Sensitive groups should follow official guidance."
    elif aqi <= 300:
        return "Poor", "Consider reducing prolonged outdoor activity."
    elif aqi <= 400:
        return "Very Poor", "Reduce prolonged outdoor exposure."
    else:
        return "Severe", "Minimize outdoor exposure where possible."

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error("Model files are missing. Add the trained model files.")
    st.stop()

if not DATA_PATH.exists():
    st.error("aqi_processed.csv is missing. Add the processed dataset.")
    st.stop()

model, feature_columns = load_model()
data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

data = data.sort_values("timestamp").reset_index(drop=True)

st.info(
    f"Dataset period: {data['timestamp'].min()} to "
    f"{data['timestamp'].max()}. This is a historical-data prototype, "
    "not a live AQI service."
)

# Recreate features using the saved historical data
hourly = data.set_index("timestamp")[["aqi"]].sort_index()

hourly = hourly[~hourly.index.duplicated(keep="last")]

hourly = hourly.reindex(
    pd.date_range(
        hourly.index.min(),
        hourly.index.max(),
        freq="h",
        name="timestamp"
    )
)

hourly["aqi_lag_1"] = hourly["aqi"].shift(1)
hourly["aqi_lag_2"] = hourly["aqi"].shift(2)
hourly["aqi_lag_3"] = hourly["aqi"].shift(3)
hourly["aqi_lag_24"] = hourly["aqi"].shift(24)
hourly["rolling_mean_6"] = (
    hourly["aqi"].shift(1).rolling(6, min_periods=6).mean()
)

valid = hourly.dropna(subset=feature_columns + ["aqi"])

if valid.empty:
    st.error("Not enough complete historical data to make a prediction.")
    st.stop()

latest = valid.iloc[-1]
features = latest[feature_columns].to_frame().T

prediction = float(np.clip(model.predict(features)[0], 0, 500))
category, advisory = get_aqi_advisory(prediction)

st.caption(
    f"Forecast target: the hour after {valid.index[-1]} "
    "(historical demonstration)."
)

col1, col2, col3 = st.columns(3)

col1.metric("Predicted next-hour AQI", f"{prediction:.1f}")
col2.metric("Estimated AQI category", category)
col3.metric("Latest observed AQI", f"{latest['aqi']:.1f}")

st.subheader("💡 General Health Advisory")
st.write(advisory)

st.caption(
    "This advisory is informational and does not replace official "
    "air-quality alerts or medical advice."
)

st.subheader("📈 Historical AQI Trend")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(data["timestamp"], data["aqi"])
ax.set_xlabel("Timestamp")
ax.set_ylabel("AQI")
ax.set_title("Historical Hourly AQI")
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.subheader("📊 Dataset Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Valid observations", f"{data['aqi'].notna().sum():,}")
c2.metric("Average AQI", f"{data['aqi'].mean():.1f}")
c3.metric("Maximum AQI", f"{data['aqi'].max():.1f}")

st.subheader("🤖 Model Information")
st.write("Model: Random Forest Regressor")
st.write("Forecast horizon: One hour")
st.write("Input features: Historical AQI lags and a six-hour rolling mean")
