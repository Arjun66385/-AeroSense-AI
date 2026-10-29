import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from pathlib import Path
from datetime import timedelta


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AeroSense AI | Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DARK AI DASHBOARD THEME
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1728 100%);
        color: #e8f0ff;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1728;
        border-right: 1px solid #1d3553;
    }

    h1, h2, h3 {
        color: #e8f0ff !important;
    }

    p, label {
        color: #c4d2e8;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #102846, #0d3b50);
        border: 1px solid #1b7894;
        margin-bottom: 24px;
    }

    .hero-title {
        color: #55e6ff;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #d4e8ff;
        font-size: 16px;
    }

    .section-label {
        color: #55e6ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .info-card {
        padding: 18px;
        border-radius: 14px;
        background: #101f33;
        border: 1px solid #223a56;
        margin-bottom: 12px;
    }

    .team-text {
        color: #d8e6fa;
        font-size: 14px;
        line-height: 1.9;
    }

    .footer {
        text-align: center;
        color: #8ca3bf;
        font-size: 12px;
        padding: 25px 0 10px 0;
        border-top: 1px solid #223a56;
        margin-top: 30px;
    }

    [data-testid="stMetric"] {
        background-color: #101f33;
        border: 1px solid #223a56;
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bfdc;
    }

    [data-testid="stMetricValue"] {
        color: #55e6ff;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT FILES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "aerosense_rf_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "aerosense_features.pkl"
DATA_PATH = BASE_DIR / "aqi_processed.csv"


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


# =========================================================
# AQI CATEGORY AND ADVISORY
# =========================================================

def get_aqi_advisory(aqi):

    if aqi <= 50:
        return (
            "Good",
            "Air quality is considered good. "
            "Continue normal outdoor activities."
        )

    elif aqi <= 100:
        return (
            "Satisfactory",
            "Air quality is generally acceptable. "
            "Sensitive individuals should monitor symptoms."
        )

    elif aqi <= 200:
        return (
            "Moderate",
            "People with asthma or other respiratory conditions "
            "should follow official health guidance."
        )

    elif aqi <= 300:
        return (
            "Poor",
            "Consider reducing prolonged outdoor activity, "
            "especially if you are sensitive to air pollution."
        )

    elif aqi <= 400:
        return (
            "Very Poor",
            "Reduce prolonged outdoor exposure and follow "
            "local air-quality advisories."
        )

    else:
        return (
            "Severe",
            "Minimize outdoor exposure where possible and "
            "follow official health guidance."
        )


# =========================================================
# SIDEBAR
# =========================================================


with st.sidebar:
    st.markdown("## 🌍 AeroSense AI")
    st.caption("AIR QUALITY INTELLIGENCE")

    st.divider()

    st.markdown("### 🏆 Project Team")
    st.markdown("""
    <div class="team-text">
        <b>TECH TITAN</b><br>
        1. Arjun Bajpai<br>
        2. Aman Mishra<br>
        3. Dhananjay Mishra
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎓 Institution")
    st.write("Bansal Institute of Engineering & Technology")
    st.caption("Lucknow, Uttar Pradesh")
    st.write("Academic Session: 2026–2027")

    st.divider()

    st.caption("Model: Random Forest Regressor")
    st.caption("Forecast horizon: One hour")

    st.divider()

    st.markdown("### 🔄 Data Controls")

    if st.button("📂 Reload Dataset", use_container_width=True):
        st.rerun()


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="section-label">SMART CITY • MACHINE LEARNING</div>
    <div class="hero-title">🌍 AeroSense AI</div>
    <div class="hero-subtitle">
        Smart City Air Quality Forecasting & Health Advisory System
    </div>
    <br>
    <div class="hero-subtitle">
        Transforming historical air-quality measurements into
        data-driven forecasts and environmental insights.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

required_files = [
    MODEL_PATH,
    FEATURE_PATH,
    DATA_PATH
]

missing_files = [
    str(path.relative_to(BASE_DIR))
    for path in required_files
    if not path.exists()
]

if missing_files:

    st.error(
        "Required project files are missing:\n\n"
        + "\n".join(f"- {file}" for file in missing_files)
    )

    st.info(
        "Check that the model files are inside the models folder "
        "and that app.py and aqi_processed.csv are in the repository root."
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    model, feature_columns = load_model()

    data = pd.read_csv(DATA_PATH)

    required_columns = {"timestamp", "aqi"}

    if not required_columns.issubset(data.columns):

        st.error(
            "The dataset must contain timestamp and aqi columns."
        )

        st.stop()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["aqi"] = pd.to_numeric(
        data["aqi"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", "aqi"]
    )

    data = data.sort_values(
        "timestamp"
    ).drop_duplicates(
        subset=["timestamp"],
        keep="last"
    ).reset_index(drop=True)

    if data.empty:

        st.error("The dataset contains no valid AQI observations.")

        st.stop()

except Exception as error:

    st.error(f"Unable to load project data: {error}")

    st.stop()


# =========================================================
# DATASET INFORMATION
# =========================================================

first_timestamp = data["timestamp"].min()
last_timestamp = data["timestamp"].max()

st.info(
    f"📅 Historical dataset: "
    f"{first_timestamp:%d %b %Y, %H:%M} — "
    f"{last_timestamp:%d %b %Y, %H:%M}"
)

st.caption(
    "This application uses historical dataset observations. "
    "It does not provide live government AQI readings."
)


# =========================================================
# PREPARE HOURLY AQI SERIES
# =========================================================

hourly = data.set_index("timestamp")["aqi"].sort_index()

hourly = hourly[~hourly.index.duplicated(keep="last")]

hourly = hourly.reindex(
    pd.date_range(
        start=hourly.index.min(),
        end=hourly.index.max(),
        freq="h"
    )
)

if len(hourly) < 24 or hourly.tail(24).isna().any():

    st.error(
        "At least 24 consecutive hourly AQI observations "
        "are required for this forecast."
    )

    st.stop()


# =========================================================
# PREPARE NEXT-HOUR FORECAST FEATURES
# =========================================================

latest_aqi = float(hourly.iloc[-1])

forecast_timestamp = hourly.index[-1] + timedelta(hours=1)

forecast_features = {
    "aqi_lag_1": float(hourly.iloc[-1]),
    "aqi_lag_2": float(hourly.iloc[-2]),
    "aqi_lag_3": float(hourly.iloc[-3]),
    "aqi_lag_24": float(hourly.iloc[-24]),
    "rolling_mean_6": float(hourly.tail(6).mean())
}

try:

    X_forecast = pd.DataFrame(
        [forecast_features],
        columns=feature_columns
    )

    prediction = float(model.predict(X_forecast)[0])

    if not np.isfinite(prediction):

        st.error("The model returned an invalid prediction.")

        st.stop()

    prediction = float(np.clip(prediction, 0, 500))

except Exception as error:

    st.error(f"Unable to generate the forecast: {error}")

    st.stop()


forecast_category, forecast_advisory = get_aqi_advisory(
    prediction
)

latest_category, _ = get_aqi_advisory(
    latest_aqi
)

aqi_change = prediction - latest_aqi


# =========================================================
# FORECAST STATUS
# =========================================================

st.markdown("### 🔮 Next-Hour Forecast")

st.caption(
    f"Forecast target: {forecast_timestamp:%d %b %Y, %H:%M}"
)

st.caption(
    "The forecast is generated using the trained model "
    "and the most recent 24 hours of historical observations."
)
# Record when the forecast was calculated
if "forecast_calculated_at" not in st.session_state:
    st.session_state["forecast_calculated_at"] = datetime.now().astimezone()

st.caption(
    "🕒 Last Updated: "
    + st.session_state["forecast_calculated_at"].strftime(
        "%d %b %Y, %I:%M:%S %p %Z"
    )
)
# =========================================================
# REFRESH PREDICTION
# =========================================================

if st.button("🔄 Refresh Prediction", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()


# =========================================================
# KEY PERFORMANCE INDICATORS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="Latest Recorded AQI",
        value=f"{latest_aqi:.1f}",
        help="The latest available AQI measurement in the dataset."
    )

with col2:

    st.metric(
        label="Predicted Next-Hour AQI",
        value=f"{prediction:.1f}",
        help="The Random Forest model's next-hour prediction."
    )

with col3:

    st.metric(
        label="Forecast Change",
        value=f"{aqi_change:+.1f}",
        delta=f"{aqi_change:+.1f} AQI points",
        delta_color="inverse"
    )

with col4:

    st.metric(
        label="Historical Observations",
        value=f"{len(data):,}"
    )


# =========================================================
# AQI CATEGORY SUMMARY
# =========================================================

st.markdown("### 🌡️ Air Quality Overview")

category_col, forecast_col = st.columns(2)

with category_col:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-label">LATEST OBSERVATION</div>
            <h2>{latest_category}</h2>
            <p>Recorded AQI: {latest_aqi:.1f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with forecast_col:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-label">PREDICTED CATEGORY</div>
            <h2>{forecast_category}</h2>
            <p>Predicted AQI: {prediction:.1f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEALTH ADVISORY
# =========================================================

st.markdown("### 🩺 General Health Advisory")

st.markdown(
    f"""
    <div class="info-card">
        <h3>Forecast category: {forecast_category}</h3>
        <p>{forecast_advisory}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Advisories are informational. Follow official air-quality "
    "alerts and medical advice where appropriate."
)


# =========================================================
# HISTORICAL AQI CHART
# =========================================================

st.markdown("### 📈 Historical AQI Trend")

fig, ax = plt.subplots(figsize=(13, 4.5))

fig.patch.set_facecolor("#101f33")
ax.set_facecolor("#101f33")

ax.plot(
    data["timestamp"],
    data["aqi"],
    linewidth=2,
    label="Recorded AQI"
)

ax.axhline(
    y=prediction,
    linestyle="--",
    linewidth=1.5,
    label=f"Next-hour prediction: {prediction:.1f}"
)

ax.set_xlabel("Timestamp", color="#d4e8ff")
ax.set_ylabel("AQI", color="#d4e8ff")

ax.tick_params(colors="#c4d2e8")

ax.set_title(
    "Historical Hourly AQI and Next-Hour Forecast",
    color="#e8f0ff",
    pad=15
)

ax.grid(
    True,
    alpha=0.18
)

for spine in ax.spines.values():
    spine.set_color("#344b66")

legend = ax.legend(
    facecolor="#172a42",
    edgecolor="#344b66"
)

for text_item in legend.get_texts():
    text_item.set_color("#e8f0ff")

fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig, use_container_width=True)

plt.close(fig)


# =========================================================
# DATASET STATISTICS
# =========================================================

st.markdown("### 📊 Dataset Analytics")

stat1, stat2, stat3 = st.columns(3)

with stat1:

    st.metric(
        "Average Historical AQI",
        f"{data['aqi'].mean():.1f}"
    )

with stat2:

    st.metric(
        "Maximum Historical AQI",
        f"{data['aqi'].max():.1f}"
    )

with stat3:

    st.metric(
        "Minimum Historical AQI",
        f"{data['aqi'].min():.1f}"
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("🤖 Model & Technical Information"):

    st.write("**Algorithm:** Random Forest Regressor")

    st.write("**Prediction target:** AQI for the next hour")

    st.write("**Input features:**")

    for feature in feature_columns:
        st.write(f"- `{feature}`")

    st.write(
        "**Forecast method:** Historical AQI lag features "
        "and a six-hour rolling mean."
    )

    st.write(
        "**Project type:** Educational machine-learning prototype."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <b>AeroSense AI</b> | Developed by Team TECH TITAN<br>
    Arjun Bajpai • Aman Mishra • Dhananjay Mishra<br>
    Bansal Institute of Engineering & Technology, Lucknow<br>
    Academic Session 2026–2027<br><br>
    Smart Cities • Environmental Intelligence • Machine Learning
</div>
""", unsafe_allow_html=True)
