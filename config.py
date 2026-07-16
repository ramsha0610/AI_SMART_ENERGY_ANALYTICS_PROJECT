"""Central configuration for the AI Smart Energy Analytics System.

This module centralizes every path, theme value, colour, and domain constant
used across the application. Nothing else in the codebase should hard-code a
filesystem path, colour, or magic number — import it from here instead.

Keeping configuration in a single place makes the project easy to deploy in
different environments (local, staging, production) and trivial to extend when
new datasets, models, or pages are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Project metadata
# --------------------------------------------------------------------------- #
APP_NAME: Final[str] = "AI Smart Energy Analytics"
APP_ICON: Final[str] = "⚡"
APP_TAGLINE: Final[str] = "Forecast • Analyze • Optimize household energy at scale"
APP_VERSION: Final[str] = "1.0.0"
AUTHOR: Final[str] = "Ramsha Firdous"
CONTACT_EMAIL: Final[str] = "ramshafirdous666@gmail.com"

# --------------------------------------------------------------------------- #
# Filesystem paths (resolved relative to this file so they work anywhere)
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(__file__).resolve().parent

ASSETS_DIR: Final[Path] = BASE_DIR / "assets"
DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
LOGS_DIR: Final[Path] = BASE_DIR / "logs"

# All directories that must exist at runtime. `ensure_directories` creates them.
REQUIRED_DIRS: Final[tuple[Path, ...]] = (
    ASSETS_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)


def ensure_directories() -> None:
    """Create every required project directory if it does not already exist.

    Safe to call repeatedly — existing directories are left untouched. Called
    once at application startup from :mod:`app`.
    """
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset locations & schema (London Smart Meter dataset)
# --------------------------------------------------------------------------- #
# The London Smart Meter dataset ("Smart meters in London") ships as several
# CSV files. These constants describe the expected raw and processed files so
# that data loading never hard-codes a filename.
# The raw meter readings may arrive either as a single CSV or as a folder of
# ``block_*.csv`` files (the Kaggle layout). Both layouts are auto-discovered.
RAW_METER_FILE: Final[Path] = RAW_DATA_DIR / "halfhourly_dataset.csv"
RAW_METER_DIR: Final[Path] = RAW_DATA_DIR / "halfhourly_dataset"
RAW_METER_BLOCK_GLOB: Final[str] = "block_*.csv"
RAW_HOUSEHOLD_INFO_FILE: Final[Path] = RAW_DATA_DIR / "informations_households.csv"
RAW_WEATHER_FILE: Final[Path] = RAW_DATA_DIR / "weather_hourly_darksky.csv"
PROCESSED_METER_FILE: Final[Path] = PROCESSED_DATA_DIR / "consumption_clean.parquet"

# Final feature-rich dataset produced by the data pipeline.
FINAL_DATASET_PARQUET: Final[Path] = PROCESSED_DATA_DIR / "final_dataset.parquet"
FINAL_DATASET_CSV: Final[Path] = PROCESSED_DATA_DIR / "final_dataset.csv"
DATA_QUALITY_REPORT_FILE: Final[Path] = REPORTS_DIR / "data_quality_report.md"

# Canonical column names used throughout the app. Rename raw columns to these
# during preprocessing so downstream code depends only on these constants.
COL_HOUSEHOLD_ID: Final[str] = "household_id"
COL_TIMESTAMP: Final[str] = "timestamp"
COL_CONSUMPTION_KWH: Final[str] = "consumption_kwh"
COL_TEMPERATURE: Final[str] = "temperature_c"
COL_ACORN_GROUP: Final[str] = "acorn_group"
COL_ACORN: Final[str] = "acorn"
COL_TARIFF: Final[str] = "tariff"
COL_HUMIDITY: Final[str] = "humidity"
COL_WIND_SPEED: Final[str] = "wind_speed"
COL_PRESSURE: Final[str] = "pressure"
COL_DEW_POINT: Final[str] = "dew_point"

# Raw column names as they appear in the London Smart Meter source files.
RAW_COL_LCLID: Final[str] = "LCLid"
RAW_COL_TSTP: Final[str] = "tstp"
RAW_COL_ENERGY: Final[str] = "energy(kWh/hh)"
RAW_COL_TARIFF: Final[str] = "stdorToU"
RAW_COL_ACORN: Final[str] = "Acorn"
RAW_COL_ACORN_GROUPED: Final[str] = "Acorn_grouped"
RAW_COL_WEATHER_TIME: Final[str] = "time"
RAW_WEATHER_COLUMN_MAP: Final[dict[str, str]] = {
    "temperature": COL_TEMPERATURE,
    "humidity": COL_HUMIDITY,
    "windSpeed": COL_WIND_SPEED,
    "pressure": COL_PRESSURE,
    "dewPoint": COL_DEW_POINT,
}

# --------------------------------------------------------------------------- #
# Model locations
# --------------------------------------------------------------------------- #
FORECAST_MODEL_FILE: Final[Path] = MODELS_DIR / "best_regressor.joblib"
CLUSTERING_MODEL_FILE: Final[Path] = MODELS_DIR / "clustering_model.joblib"
ANOMALY_MODEL_FILE: Final[Path] = MODELS_DIR / "anomaly_model.joblib"
FEATURE_SCALER_FILE: Final[Path] = MODELS_DIR / "feature_scaler.joblib"
REGRESSION_MODEL_FILE: Final[Path] = MODELS_DIR / "best_regressor.joblib"
REGRESSION_FEATURE_NAMES_FILE: Final[Path] = MODELS_DIR / "regression_feature_names.joblib"
MODEL_RESULTS_FILE: Final[Path] = REPORTS_DIR / "model_results.csv"

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
HALF_HOURS_PER_DAY: Final[int] = 48
DAYS_PER_WEEK: Final[int] = 7
DEFAULT_FORECAST_HORIZON_DAYS: Final[int] = 7
DEFAULT_N_CLUSTERS: Final[int] = 4
ANOMALY_CONTAMINATION: Final[float] = 0.02  # expected fraction of anomalies

# --------------------------------------------------------------------------- #
# Data-pipeline configuration
# --------------------------------------------------------------------------- #
# Rows read per chunk when streaming large raw CSVs.
CHUNK_SIZE: Final[int] = 500_000
# Consumption values outside this range (kWh per half-hour) are treated as
# invalid and dropped. Negative energy is physically impossible for a meter.
MIN_VALID_KWH: Final[float] = 0.0
MAX_VALID_KWH: Final[float] = 20.0
# Z-score threshold used to *flag* (not drop) statistical consumption outliers.
OUTLIER_ZSCORE_THRESHOLD: Final[float] = 4.0
# Lag features (in half-hourly steps) and rolling windows for feature building.
LAG_STEPS: Final[tuple[int, ...]] = (1, 2, 24, 48)
ROLLING_WINDOWS: Final[tuple[int, ...]] = (24, 48)
# Scaler artefacts persisted for reuse by ML models.
STANDARD_SCALER_FILE: Final[Path] = MODELS_DIR / "standard_scaler.joblib"
MINMAX_SCALER_FILE: Final[Path] = MODELS_DIR / "minmax_scaler.joblib"

# Carbon intensity of grid electricity used to translate kWh savings into CO2.
# Source placeholder — refine with a regional grid factor later.
KG_CO2_PER_KWH: Final[float] = 0.233
# Average unit price used for cost estimates (GBP per kWh).
PRICE_PER_KWH_GBP: Final[float] = 0.28


# --------------------------------------------------------------------------- #
# Theme & colour palette
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Theme:
    """Immutable colour palette and typography for the dashboard.

    Centralising the palette guarantees a consistent, dark-mode-friendly look
    across every page and makes rebranding a one-line change.
    """

    # Brand / accent colours
    primary: str = "#22D3A6"      # energetic teal-green (energy/eco)
    secondary: str = "#6366F1"    # indigo accent
    accent: str = "#F59E0B"       # amber highlight

    # Surfaces (tuned for a modern dark UI)
    background: str = "#0E1117"
    surface: str = "#161B22"
    surface_alt: str = "#1F2630"
    border: str = "#2A313C"

    # Text
    text: str = "#E6EDF3"
    text_muted: str = "#9AA4B2"

    # Semantic status colours
    success: str = "#22C55E"
    warning: str = "#F59E0B"
    danger: str = "#EF4444"
    info: str = "#38BDF8"

    font_family: str = (
        "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
    )

    # Ordered palette for categorical charts (e.g. clusters).
    categorical: tuple[str, ...] = field(
        default_factory=lambda: (
            "#22D3A6",
            "#6366F1",
            "#F59E0B",
            "#38BDF8",
            "#EF4444",
            "#A855F7",
        )
    )


THEME: Final[Theme] = Theme()

# Plotly template name registered in ``utils.visualization``.
PLOTLY_TEMPLATE: Final[str] = "smart_energy_dark"

# --------------------------------------------------------------------------- #
# Page registry — single source of truth for navigation labels & icons.
# `app.py` builds the sidebar from this so pages and icons never drift.
# --------------------------------------------------------------------------- #
PAGES: Final[dict[str, dict[str, str]]] = {
    "home": {"title": "Home", "icon": "🏠", "module": "app_home"},
    "dashboard": {"title": "Dashboard", "icon": "📊", "module": "pages/Dashboard.py"},
    "forecast": {"title": "Forecast", "icon": "📈", "module": "pages/Forecast.py"},
    "household": {
        "title": "Household Analysis",
        "icon": "🏡",
        "module": "pages/Household.py",
    },
    "clustering": {"title": "Clustering", "icon": "📉", "module": "pages/Clustering.py"},
    "anomaly": {
        "title": "Anomaly Detection",
        "icon": "⚠️",
        "module": "pages/Anomaly.py",
    },
    "recommendations": {
        "title": "AI Recommendations",
        "icon": "💡",
        "module": "pages/Recommendations.py",
    },
    "about": {"title": "About", "icon": "ℹ️", "module": "pages/About.py"},
}
