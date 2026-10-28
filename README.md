# 🌍 AeroSense AI
### Smart City Air Quality Forecasting & Health Advisory System

An AI-powered academic mini project that analyzes historical Air Quality Index (AQI) data, forecasts air quality using machine learning, and provides health-oriented advisories through an interactive Streamlit dashboard.

**Developed by:** Team TECH TITAN  
**Team Members:** Arjun Bajpai, Aman Mishra, and Dhananjay Mishra  
**Institution:** Bansal Institute of Engineering & Technology, Lucknow  
**Academic Session:** 2026–2027

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Dataset Information](#-dataset-information)
- [Machine Learning and Evaluation](#-machine-learning-and-evaluation)
- [AQI Categories](#-aqi-categories)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Health Advisory System](#-health-advisory-system)
- [Future Scope](#-future-scope)
- [Limitations](#-limitations)
- [Project Team](#-project-team)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Project Overview

Air pollution is an important environmental concern that affects public health and urban living conditions. Monitoring air quality and understanding its possible future trends can help people become more aware of environmental risks.

**AeroSense AI** is a machine learning-based air quality forecasting and health advisory application developed to demonstrate how historical environmental data, predictive modelling, and interactive visualizations can support air quality awareness.

The application processes historical AQI observations, uses engineered time-series features for prediction, displays air quality trends, and presents general health advisories based on AQI categories.

The project combines data analysis, machine learning, and web application development in an accessible dashboard.

---

## 🎯 Project Objectives

The main objectives of AeroSense AI are:

1. To analyze historical Air Quality Index data.
2. To explore changes and patterns in air quality over time.
3. To demonstrate machine learning-based AQI forecasting.
4. To visualize historical air quality trends through an interactive dashboard.
5. To provide general health awareness messages based on AQI categories.
6. To demonstrate the practical application of Python and machine learning in an environmental monitoring use case.
7. To develop a simple, user-friendly interface using Streamlit.

---

## ✨ Key Features

### 📊 Interactive Dashboard
Displays air quality information in a convenient and easy-to-understand interface.

### 🤖 Machine Learning-Based Forecasting
Uses a trained machine learning model and engineered features to generate AQI predictions.

### 📈 Historical Trend Analysis
Visualizes historical AQI observations to help users explore changes in air quality.

### 🩺 AQI-Based Health Advisories
Provides general health-oriented messages corresponding to AQI categories.

### 🌱 Environmental Awareness
Demonstrates how data-driven tools can support public awareness of urban air pollution.

### 💻 Web-Based Interface
Uses Streamlit to provide an interactive application accessible through a web browser.

### 📉 Model Evaluation
Includes recorded evaluation results for a persistence baseline, Linear Regression, and Random Forest models.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib / Plotly | Data visualization, depending on the implemented code |
| Joblib / Pickle | Loading saved model artifacts, depending on the implementation |
| CSV | Storage of processed AQI observations |
| Git and GitHub | Version control and project hosting |

---

## 🏗️ Project Architecture

The project follows a simple data-processing and prediction workflow.

```text
Historical AQI Dataset
         |
         v
Data Loading and Processing
         |
         v
Feature Engineering
         |
         v
Trained Machine Learning Model
         |
         v
AQI Forecast Generation
         |
         v
AQI Category Classification
         |
         v
Health Advisory Generation
         |
         v
Interactive Streamlit Dashboard
```

### Workflow Explanation

**1. Data Loading**

The application reads processed AQI observations from the supplied CSV dataset.

**2. Data Processing**

The dataset is prepared for analysis and prediction using the processing and feature-engineering logic implemented in the application.

**3. Feature Engineering**

Relevant input features are prepared in the format expected by the trained model.

**4. AQI Forecasting**

The saved machine learning model generates predictions using the input features supplied by the application.

**5. AQI Classification**

The resulting AQI values are interpreted using the AQI category ranges adopted by the project.

**6. Health Advisory**

The application presents general health awareness messages based on the applicable AQI category.

**7. Visualization**

Streamlit presents the results, historical trends, and other implemented dashboard elements.

---

## 📁 Project Structure

The recommended repository structure is:

```text
AeroSense-AI/
│
├── app.py
├── aqi_processed.csv
├── requirements.txt
├── README.md
│
└── models/
    ├── aerosense_rf_model.pkl
    └── aerosense_features.pkl
```

### File Descriptions

| File or Directory | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `aqi_processed.csv` | Processed historical AQI dataset |
| `requirements.txt` | Python dependencies required by the project |
| `README.md` | Project documentation |
| `models/aerosense_rf_model.pkl` | Saved Random Forest model artifact |
| `models/aerosense_features.pkl` | Saved feature information or related model metadata |

**Important:** The actual file paths and serialization format must match the loading logic in `app.py`. Keep the model files in the locations expected by your code.

---

## 📊 Dataset Information

The project uses historical hourly AQI observations for Lucknow from September 2026.

The dataset supports the demonstration of historical trend analysis and AQI forecasting.

### Dataset Usage

- Loading historical AQI observations.
- Examining air quality patterns.
- Preparing inputs for the forecasting model.
- Visualizing historical trends.
- Demonstrating AQI-based health advisories.

### Dataset Scope

The results depend on the quality, coverage, and structure of the supplied dataset.

The current project is an educational prototype based on historical data. It does not automatically represent current outdoor air quality unless a live data source is explicitly integrated.

---

## 🤖 Machine Learning and Evaluation

AeroSense AI demonstrates the use of machine learning techniques for AQI forecasting.

The recorded evaluation results are shown below.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence Baseline | 1.014 | 1.399 | 0.993 |
| Linear Regression | 0.911 | 1.200 | 0.995 |
| Random Forest | 1.355 | 1.784 | 0.988 |

### Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values. Lower values generally indicate smaller prediction errors.

**Root Mean Squared Error (RMSE)**

Measures prediction error while penalizing larger errors more heavily. Lower values generally indicate better predictive accuracy.

**R-squared (R²)**

Measures how well a model explains variation in the target values under the evaluation setup. Values closer to 1 indicate a stronger fit, but do not by themselves guarantee reliable future forecasts.

### Interpretation

Based on the recorded evaluation results, Linear Regression has the lowest MAE and RMSE among the three listed approaches.

The Random Forest model has the highest error of the three in these recorded results.

These metrics should be interpreted in the context of the dataset, train-test split, feature engineering, and evaluation method. Time-series evaluation should preserve chronological order to reduce the risk of data leakage.

**Model consistency note:** The deployed application's forecasting algorithm must be identified from the model-loading and prediction code in `app.py`. The presence of a Random Forest model file does not, by itself, prove that every displayed prediction is generated by that model.

---

## 🌡️ AQI Categories

The following category ranges are used for the project's general AQI interpretation.

| AQI Range | Category | General Interpretation |
|---|---|---|
| 0–50 | Good | Generally associated with lower air pollution |
| 51–100 | Satisfactory | Generally acceptable air quality, with possible concerns for sensitive individuals |
| 101–200 | Moderate | May cause discomfort for sensitive individuals |
| 201–300 | Poor | May cause breathing discomfort with prolonged exposure |
| 301–400 | Very Poor | May cause increased respiratory discomfort |
| 401–500 | Severe | May pose significant health risks, particularly for sensitive groups |

These ranges follow the commonly used Indian AQI category structure. Health effects can vary by pollutant, exposure duration, and individual circumstances.

---

## 🚀 Installation and Setup

Follow these steps to run AeroSense AI on your local computer.

### Prerequisites

Install the following:

- Python
- pip
- Git (optional, for cloning the repository)
- A web browser

### Step 1: Clone the Repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/AeroSense-AI.git
```

### Step 2: Open the Project Directory

```bash
cd AeroSense-AI
```

### Step 3: Create a Virtual Environment (Recommended)

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify the Project Files

Make sure the following files and folders are present:

```text
app.py
aqi_processed.csv
requirements.txt
models/
```

The `models/` directory should contain the saved model artifacts expected by the application.

---

## ▶️ Running the Application

After installing the dependencies, execute:

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal, commonly:

```text
http://localhost:8501
```

Open the displayed address in your browser to access the dashboard.

If the application fails to start, check that:

- All required dependencies are installed.
- The CSV dataset exists at the expected path.
- The saved model files exist at the expected paths.
- The model-loading code matches the saved artifact format.
- The Python environment is activated correctly.

---

## 🩺 Health Advisory System

AeroSense AI provides general health-oriented information based on the AQI category.

The purpose of these messages is to help users understand the potential significance of different air quality levels.

Possible advisory themes include:

- General air quality awareness.
- Reducing prolonged outdoor exertion during poor air quality.
- Extra caution for people sensitive to air pollution.
- Following official public health and air quality guidance.

The exact advisory messages depend on the implementation in `app.py`.

**Disclaimer:** The health advisories are educational and informational. They are not medical diagnoses or personalized medical recommendations. Users should consult official air quality information and qualified health professionals for guidance appropriate to their circumstances.

---

## 🔮 Future Scope

AeroSense AI can be extended with the following improvements:

### 1. Live AQI Integration
Connect to reliable air quality APIs or sensor networks to retrieve current observations.

### 2. Multi-City Forecasting
Expand the system to analyze and forecast air quality for multiple cities.

### 3. Larger Historical Datasets
Use longer time periods to examine seasonal trends and improve the robustness of model evaluation.

### 4. Improved Model Validation
Compare forecasting approaches using chronological train-test splits and appropriate time-series validation.

### 5. Real-Time Alerts
Introduce optional notifications when air quality crosses configured thresholds.

### 6. Mobile-Friendly Interface
Improve the layout for mobile devices and smaller screens.

### 7. Explainable Machine Learning
Add feature-importance analysis or other suitable techniques to help users understand model behavior.

### 8. Interactive Forecast Visualizations
Display predicted and observed values together, where suitable historical evaluation data is available.

### 9. Model Monitoring
Track prediction errors and identify changes in data patterns that could reduce model reliability.

---

## ⚠️ Limitations

- The current prototype uses a supplied historical dataset.
- It should not be treated as a live government air quality monitoring service.
- Forecast accuracy depends on the dataset, features, model, and evaluation methodology.
- Performance measured on historical data may not generalize to future conditions.
- The application does not replace official environmental monitoring or professional medical advice.
- Live alerts, multi-city support, and sensor integration require additional implementation.

---

## 👥 Project Team

### Team TECH TITAN

The project was developed as an academic mini project by the following team members:

1. **Arjun Bajpai**
2. **Aman Mishra**
3. **Dhananjay Mishra**

**Institution:** Bansal Institute of Engineering & Technology, Lucknow  
**Academic Session:** 2026–2027

---

## 🙏 Acknowledgements

We express our gratitude to our institution and faculty for providing the opportunity to develop this academic mini project.

This project helped us explore practical applications of Python programming, data analysis, machine learning, and interactive web development in the field of environmental awareness.

---

## 📄 Project Information

| Detail | Information |
|---|---|
| Project Name | AeroSense AI |
| Project Type | Academic Mini Project |
| Domain | Artificial Intelligence and Environmental Data Analysis |
| Team Name | TECH TITAN |
| Institution | Bansal Institute of Engineering & Technology, Lucknow |
| Academic Session | 2026–2027 |

---

**AeroSense AI — Using Data and Machine Learning to Support Air Quality Awareness.**

*Academic Mini Project | Team TECH TITAN | 2026–2027*
