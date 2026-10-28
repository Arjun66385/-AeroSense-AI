# 🌍 AeroSense AI
### Smart City Air Quality Forecasting & Health Advisory System

An AI-powered mini project that analyzes historical Air Quality Index (AQI) data, forecasts air quality, and provides health-oriented advisories through an interactive Streamlit dashboard.

**Developed by:** Team TECH TITAN  
**Team Members:** Aman Mishra and Dhananjay Mishra  
**Institution:** Bansal Institute of Engineering & Technology, Lucknow  
**Academic Session:** 2026–2027

---

## 📌 Project Overview

AeroSense AI demonstrates how machine learning and data visualization can support air quality monitoring and awareness.

The application processes historical AQI observations, uses engineered time-series features to predict AQI, displays historical trends, and presents health advisories based on AQI categories.

## ✨ Key Features

- 📊 Interactive air quality dashboard
- 🤖 Machine learning-based AQI forecasting
- 📈 Historical AQI trend visualization
- 🩺 AQI-based health advisory messages
- 🌱 Air quality awareness for smart cities

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib / Plotly, depending on the implemented visualizations

## 📁 Project Structure

```text
AeroSense-AI/
├── app.py
├── aqi_processed.csv
├── requirements.txt
└── models/
    ├── aerosense_rf_model.pkl
    └── aerosense_features.pkl
```

## 🚀 Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/AeroSense-AI.git
   ```

2. Open the project folder:

   ```bash
   cd AeroSense-AI
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in your terminal.

## 📊 Data and Scope

This educational prototype uses historical AQI observations for Lucknow from September 2026. Predictions and visualizations depend on the supplied dataset and trained model.

The application should not be interpreted as a live government AQI monitoring service or a substitute for professional medical advice.

## 🔮 Future Scope

- Integration with live air quality APIs or sensor networks
- Forecasting across multiple cities
- Longer-term model evaluation
- Improved model accuracy and monitoring
- Mobile-friendly alerts and notifications

## 👥 Team

**TECH TITAN**

- Aman Mishra
- Dhananjay Mishra

**Bansal Institute of Engineering & Technology, Lucknow**

---

*Academic mini project | 2026–2027*
