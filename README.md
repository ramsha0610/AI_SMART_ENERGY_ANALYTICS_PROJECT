# ⚡ AI Smart Energy Analytics System

A production-ready analytics platform for the **London Smart Meter** dataset.
It forecasts electricity demand, profiles and clusters households, detects
abnormal usage, and generates **quantified, AI-powered energy-saving
recommendations** — all inside a modern, dark-mode Streamlit dashboard.

> **Version:** 1.0.0 · Production Release
> **Developer:** Ramsha Firdous · <ramshafirdous666@gmail.com>

---

## ✨ Overview

| Capability | Description |
| --- | --- |
| 📊 **Dashboard** | Fleet-wide KPIs, trends, weather impact, energy distribution. |
| 📈 **Forecast** | Per-household consumption forecasting with confidence bands. |
| 🏡 **Household Analysis** | Daily / weekly / monthly profiles, peak hour, cluster assignment. |
| 📉 **Clustering** | KMeans segmentation of households by consumption behaviour. |
| ⚠️ **Anomaly Detection** | Isolation Forest surfacing abnormal usage and potential faults. |
| 💡 **AI Recommendations** | Personalised, quantified energy / cost / CO₂ savings. |

### Dataset

- **167M+** half-hourly smart meter readings processed
- **5,561** London households
- Hourly weather observations (temperature, humidity, wind, pressure)
- Household metadata including ACORN socio-economic groups and tariff type

---

## 🏗️ Architecture

```text
Streamlit UI (app.py + pages/)
        │  calls
        ▼
Utility facades (utils/)
  data_loader → preprocess → feature_engineering → prediction → recommendation
        │                                              │
        ▼                                              ▼
   data/ (raw, processed)                         models/ (trained artefacts)

config.py   → single source of truth for paths, theme, colours, constants
utils/helpers.py → logging, formatting, conversion helpers
utils/visualization.py → Plotly theming & reusable UI components
```

**Key design principles**

- No hard-coded paths — everything flows from `config.py`.
- The UI never imports ML libraries directly; it talks to the `utils.prediction` facade.
- Reusable, documented, type-hinted functions across all modules.
- Memory-safe data pipeline handling 167M+ rows via incremental batch processing.

---

## 📁 Folder Structure

```text
AI_Smart_Energy_Analytics/
├── app.py                   # Entry point + Home page + navigation
├── config.py                # Central config: paths, theme, constants
├── train_models.py          # Model training orchestration
├── rebuild_pipeline.py      # Memory-safe data pipeline builder
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml   # Streamlit theme configuration
├── assets/                  # Images, logos
├── data/
│   ├── raw/                 # Original dataset (London Smart Meter)
│   └── processed/           # Cleaned, feature-engineered dataset
├── models/                  # Trained model artefacts (.joblib)
├── pages/                   # Streamlit multi-page layout
│   ├── Dashboard.py
│   ├── Forecast.py
│   ├── Household.py
│   ├── Clustering.py
│   ├── Anomaly.py
│   ├── Recommendations.py
│   └── About.py
├── utils/                   # Reusable logic
│   ├── data_loader.py       # Data loading with caching
│   ├── preprocess.py        # Cleaning, validation, outlier flagging
│   ├── feature_engineering.py  # Calendar, lag, rolling features
│   ├── prediction.py        # ML model loading & inference facade
│   ├── recommendation.py    # Quantified recommendation engine
│   ├── visualization.py     # Plotly theming & UI components
│   └── helpers.py           # Logging, formatting, conversion utils
├── reports/                 # Generated reports (model_results.csv, etc.)
├── logs/                    # Application logs
└── tests/                   # Test suite
```

---

## 🤖 Machine Learning Models

| Model | Algorithm | Purpose | Metrics |
| --- | --- | --- | --- |
| **Best Regressor** | Random Forest | Consumption forecasting | R² = 0.986, MAE = 0.003 |
| **XGBoost** | XGBRegressor | Alternative forecaster | R² = 0.982, MAE = 0.009 |
| **Clustering** | KMeans (k=4) | Household segmentation | Silhouette-score based |
| **Anomaly Detection** | Isolation Forest | Abnormal usage detection | 2% contamination rate |

### Pipeline

1. **Data Ingestion** — 112 block CSV files (~167M rows) streamed in chunks
2. **Cleaning** — deduplication, negative/null removal, outlier flagging
3. **Weather Merge** — `merge_asof` with hourly weather observations
4. **Feature Engineering** — calendar (8), lag (4), rolling (8), interaction (3), household (3) features
5. **Sampling** — 3M row stratified sample for the final dataset; 500K for model training
6. **Training** — Random Forest, XGBoost, KMeans, Isolation Forest
7. **Inference** — lazy model loading, feature alignment, confidence intervals

---

## 🚀 Installation

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download the **"Smart meters in London"** dataset and place the raw CSVs in `data/raw/`:

- `halfhourly_dataset/block_*.csv` (or `halfhourly_dataset.csv`)
- `informations_households.csv`
- `weather_hourly_darksky.csv`

The `data/`, `models/`, `reports/`, and `logs/` directories are created automatically on startup.

---

## ▶️ Running

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default `http://localhost:8501`).
All pages load with trained models and real data.

### Model Training

To retrain all models from scratch:

```bash
python train_models.py
```

This loads the processed dataset, samples 500K rows, and trains all four models with validation metrics.

### Testing

```bash
pytest -q
```

---

## 📊 Dashboard Pages

- **Home** — Landing page with capabilities overview, workflow diagram, tech stack badges
- **Dashboard** — Fleet-wide KPIs, consumption trends, energy distribution, weather impact
- **Forecast** — Per-household forecasting with confidence intervals and historical comparison
- **Household Analysis** — Individual household profiles with daily/weekly/monthly patterns
- **Clustering** — KMeans cluster visualization, per-cluster statistics, silhouette score
- **Anomaly Detection** — Timeline view with anomaly markers, risk summary, alerts table
- **AI Recommendations** — Ranked, quantified energy/cost/CO₂ savings per household
- **About** — Project description, architecture, tech stack, developer info

---

## 📈 Results

- **Random Forest** (Best Model): MAE = 0.0034, RMSE = 0.035, R² = 0.986
- **XGBoost**: MAE = 0.0087, RMSE = 0.040, R² = 0.982
- **KMeans**: 4 clusters identified across 5,561 households
- **Isolation Forest**: 2% contamination rate for anomaly flagging
- **Recommendation Engine**: 6 rule-based strategies generating quantified savings

---

## 🛠️ Technology Stack

Python 3.11+ · Streamlit · Pandas · NumPy · Plotly · scikit-learn · XGBoost · Joblib

---

## 🌟 Future Enhancements

- Real-time streaming ingestion and alerts
- LSTM / Transformer-based forecasting models
- User authentication and per-tenant dashboards
- Automated PDF/HTML report generation
- Model registry, experiment tracking, and CI/CD
- Docker deployment and cloud hosting

---

## 📄 License

This project is provided for educational and demonstration purposes.

---

## 👤 Developer

- **Author:** Ramsha Firdous
- **Contact:** [ramshafirdous666@gmail.com](mailto:ramshafirdous666@gmail.com)

> **Note:** Large model files (best_regressor.joblib, random_forest_regressor.joblib) and
> processed datasets are tracked with **Git LFS**. Ensure `git lfs install` is run before cloning.
> Raw data blocks are excluded — download the London Smart Meter dataset separately.

---

## Screenshots

> _Screenshot placeholders — add screenshots of each dashboard page here._

| Home | Dashboard | Forecast |
| --- | --- | --- |
| _placeholder_ | _placeholder_ | _placeholder_ |

| Household | Clustering | Anomaly Detection |
| --- | --- | --- |
| _placeholder_ | _placeholder_ | _placeholder_ |

| Recommendations | About |
| --- | --- |
| _placeholder_ | _placeholder_ |
