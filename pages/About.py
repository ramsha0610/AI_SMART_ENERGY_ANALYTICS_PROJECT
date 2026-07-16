"""About page — project context, dataset, roadmap, and developer info."""

from __future__ import annotations

import streamlit as st

import config
from utils.helpers import get_logger
from utils.visualization import (
    badges,
    hero,
    inject_global_css,
    register_plotly_template,
    section_header,
)

logger = get_logger(__name__)

register_plotly_template()
inject_global_css()

hero("ℹ️ About", f"{config.APP_NAME} · v{config.APP_VERSION}")

section_header("Project Description")
st.markdown(
    """
    **AI Smart Energy Analytics** is a production-oriented platform that analyses
    smart-meter data to forecast demand, profile and cluster households, detect
    anomalous usage, and generate actionable, quantified energy-saving
    recommendations — presented through a modern Streamlit dashboard.
    """
)

section_header("Dataset Overview")
st.markdown(
    """
    Built for the **London Smart Meter** dataset ("Smart meters in London"):

    - **Half-hourly consumption** (`kWh`) for thousands of London households.
    - **Household metadata** including ACORN socio-economic groups and tariff type.
    - **Hourly weather** observations to model weather-driven demand.
    """
)

section_header("AI Models")
st.markdown(
    """
    - **Forecasting** — Random Forest & XGBoost on lagged, calendar, and weather features.
    - **Clustering** — KMeans on household consumption profiles.
    - **Anomaly detection** — Isolation Forest on consumption residuals.
    - **Recommendations** — cluster-baseline comparison with quantified energy, cost, and CO2 savings.
    """
)

section_header("Folder Structure")
st.code(
    """
AI_Smart_Energy_Analytics/
├── app.py                 # Entry point + Home + navigation
├── config.py              # Central config: paths, theme, constants
├── train_models.py        # Model training orchestration
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml # Native theme
├── assets/                # Images, logos
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Cleaned, analysis-ready data
├── models/                # Trained model artefacts
├── pages/                 # Streamlit pages
│   ├── Dashboard.py
│   ├── Forecast.py
│   ├── Household.py
│   ├── Clustering.py
│   ├── Anomaly.py
│   ├── Recommendations.py
│   └── About.py
├── utils/                 # Reusable logic
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── prediction.py
│   ├── recommendation.py
│   ├── visualization.py
│   └── helpers.py
├── reports/               # Generated reports
├── logs/                  # Application logs
└── tests/                 # Test suite
    """,
    language="text",
)

section_header("Technology Stack")
badges(
    [
        "Python 3.11+",
        "Streamlit",
        "Pandas",
        "NumPy",
        "Plotly",
        "scikit-learn",
        "XGBoost",
        "Joblib",
        "Matplotlib",
    ]
)

section_header("Developer")
st.markdown("""
- **Author:** Ramsha Firdous
- **Contact:** [ramshafirdous666@gmail.com](mailto:ramshafirdous666@gmail.com)
- **Version:** 1.0.0 (Production Release)
""")
