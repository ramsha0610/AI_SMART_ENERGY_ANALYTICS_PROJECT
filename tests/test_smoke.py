"""Smoke tests for the AI Smart Energy Analytics System.

These tests verify that the architecture imports cleanly, configuration is
sane, and the utility facades return correctly-typed results. They also
exercise the ML prediction and recommendation pipelines.

Run with::

    pytest -q
"""

from __future__ import annotations

import pandas as pd

import config
from utils import (
    data_loader,
    feature_engineering,
    preprocess,
    prediction,
    recommendation,
)
from utils.helpers import (
    format_currency,
    format_energy,
    format_percent,
    get_logger,
    kwh_to_co2,
    kwh_to_cost,
    models_available,
)


def test_config_paths_and_constants() -> None:
    assert config.APP_NAME
    assert config.BASE_DIR.exists()
    assert config.DEFAULT_N_CLUSTERS > 1
    assert 0 < config.ANOMALY_CONTAMINATION < 1
    # Page registry is complete and well-formed.
    for key, meta in config.PAGES.items():
        assert {"title", "icon", "module"} <= meta.keys(), key


def test_ensure_directories_creates_all() -> None:
    config.ensure_directories()
    for directory in config.REQUIRED_DIRS:
        assert directory.exists()


def test_logger_is_singleton() -> None:
    assert get_logger("test") is get_logger("test")


def test_formatting_helpers() -> None:
    assert format_energy(1234.5) == "1,234.5 kWh"
    assert format_currency(9.5) == "£9.50"
    assert format_percent(0.25) == "25.0%"
    assert format_percent(25) == "25.0%"


def test_conversion_helpers() -> None:
    assert kwh_to_co2(10) == 10 * config.KG_CO2_PER_KWH
    assert kwh_to_cost(10) == 10 * config.PRICE_PER_KWH_GBP


def test_models_availability() -> None:
    """models_available() reflects whether trained model files exist."""
    result = models_available()
    assert isinstance(result, bool)


def test_loaders_are_graceful_without_files() -> None:
    """Loaders that read optional files return DataFrames even when missing."""
    assert isinstance(data_loader.load_household_info(), pd.DataFrame)
    assert isinstance(data_loader.load_weather_data(), pd.DataFrame)
    assert isinstance(data_loader.load_processed_data(), pd.DataFrame)
    assert isinstance(data_loader.list_household_ids(), list)


def test_pipeline_on_synthetic_data() -> None:
    """End-to-end pipeline test on a tiny in-memory dataset.

    Exercises cleaning (dedup, negatives, nulls), merging, feature engineering,
    and validation without touching disk (``save=False``).
    """
    hh = ["MAC000001", "MAC000002"]
    ts = pd.date_range("2013-01-01", periods=config.HALF_HOURS_PER_DAY * 3, freq="30min")
    rows = [(h, str(t), str(round(0.2 + 0.1 * (i % 5), 3)))
            for h in hh for i, t in enumerate(ts)]
    meter = pd.DataFrame(rows, columns=["LCLid", "tstp", "energy(kWh/hh)"])
    meter = meter.rename(
        columns={
            "LCLid": config.COL_HOUSEHOLD_ID,
            "tstp": config.COL_TIMESTAMP,
            "energy(kWh/hh)": config.COL_CONSUMPTION_KWH,
        }
    )
    # Inject a duplicate, a negative, and a null to test cleaning.
    meter = pd.concat([meter, meter.iloc[[0]]], ignore_index=True)
    meter.loc[3, config.COL_CONSUMPTION_KWH] = "-1"
    meter.loc[4, config.COL_CONSUMPTION_KWH] = "Null"

    cleaned = preprocess.clean_meter_data(meter)
    assert not cleaned.empty
    # No duplicate (household, timestamp) rows and no negatives survive.
    assert cleaned.duplicated(
        subset=[config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP]
    ).sum() == 0
    assert (cleaned[config.COL_CONSUMPTION_KWH] >= 0).all()

    cleaned = preprocess.flag_outliers(cleaned)
    features = feature_engineering.build_feature_matrix(cleaned)
    features = preprocess.handle_missing_values(features)
    # Calendar + lag + rolling features were added.
    for col in ("hour", "is_weekend", "season", "lag_1", "roll_mean_24"):
        assert col in features.columns, col

    validation = preprocess.validate_dataset(features)
    assert validation["no_duplicate_records"] is True
    assert validation["has_required_columns"] is True


def test_prediction_facade_returns_correct_types() -> None:
    """Prediction functions return expected types."""
    # load_forecast_model returns a model or None
    model = prediction.load_forecast_model()
    # forecast_consumption returns a ForecastResult or None
    result = prediction.forecast_consumption("MAC000001")
    # assign_cluster returns an int or None
    cluster = prediction.assign_cluster("MAC000001")
    assert cluster is None or isinstance(cluster, int)
    # detect_anomalies returns a DataFrame
    anomalies = prediction.detect_anomalies("MAC000001")
    assert isinstance(anomalies, pd.DataFrame)


def test_recommendation_engine_returns_list() -> None:
    """Recommendation engine returns a list of Recommendations."""
    recs = recommendation.generate_recommendations("MAC000001")
    assert isinstance(recs, list)
    summary = recommendation.summarise_savings(recs)
    assert "total_kwh" in summary
    assert "total_cost_gbp" in summary
    assert "total_co2_kg" in summary
