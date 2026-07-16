"""Model loading and inference facade.

Provides functions for loading trained models and running inference:
- Regression models (Random Forest, XGBoost) for forecasting
- KMeans clustering model for household segmentation
- Isolation Forest for anomaly detection

All models are loaded lazily from :data:`config.MODELS_DIR`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config
from utils.data_loader import load_processed_data
from utils.helpers import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ForecastResult:
    """Container for forecast predictions with confidence intervals."""

    predicted_kwh: list[float]
    timestamps: list[datetime]
    confidence: float
    lower_kwh: list[float]
    upper_kwh: list[float]


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model():
    """Load the best trained regression model (XGBoost or Random Forest).

    Returns:
        The fitted model, or ``None`` if no model file exists.
    """
    path = config.REGRESSION_MODEL_FILE
    if not path.is_file():
        # Try individual model files
        for alt in (
            config.MODELS_DIR / "xgboost_regressor.joblib",
            config.MODELS_DIR / "random_forest_regressor.joblib",
        ):
            if alt.is_file():
                path = alt
                break
        else:
            logger.warning("No regression model found at %s", config.REGRESSION_MODEL_FILE)
            return None
    try:
        model = joblib.load(path)
        logger.info("Loaded regression model from %s", path)
        return model
    except Exception as e:
        logger.error("Failed to load regression model: %s", e)
        return None


def load_forecast_model():
    """Load the model used for forecasting (same as best regressor).

    Returns:
        The fitted model, or ``None`` if unavailable.
    """
    return load_model()


def load_clustering_model():
    """Load the trained KMeans clustering model.

    Returns:
        The fitted KMeans model, or ``None`` if unavailable.
    """
    path = config.CLUSTERING_MODEL_FILE
    if not path.is_file():
        logger.warning("Clustering model not found at %s", path)
        return None
    try:
        model = joblib.load(path)
        logger.info("Loaded clustering model from %s", path)
        return model
    except Exception as e:
        logger.error("Failed to load clustering model: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Feature preparation helper
# --------------------------------------------------------------------------- #
def _prepare_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector for model training.

    Excludes identifier and target columns, one-hot encodes categoricals,
    and fills NaN values with 0.

    Args:
        df: The processed dataset with features and target.

    Returns:
        A tuple of (features_df, target_series).
    """
    exclude = {
        config.COL_HOUSEHOLD_ID,
        config.COL_TIMESTAMP,
        config.COL_CONSUMPTION_KWH,
        "season",  # redundant with month
        "is_outlier",
    }

    feature_cols = [c for c in df.columns if c not in exclude]
    features = df[feature_cols].copy()

    # One-hot encode categorical columns
    cat_cols = features.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        features = pd.get_dummies(features, columns=cat_cols, drop_first=True)

    # Fill any remaining NaN
    features = features.fillna(0)

    # Ensure all columns are numeric
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0)

    target = df[config.COL_CONSUMPTION_KWH].copy()

    return features, target


def _prepare_inference_features(
    df: pd.DataFrame, feature_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Prepare features for inference, aligning columns to training schema.

    Args:
        df: Input data with engineered features.
        feature_names: Expected feature column names (from training).

    Returns:
        Feature DataFrame aligned to the expected schema.
    """
    exclude = {
        config.COL_HOUSEHOLD_ID,
        config.COL_TIMESTAMP,
        config.COL_CONSUMPTION_KWH,
        "season",
        "is_outlier",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    features = df[feature_cols].copy()

    cat_cols = features.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        features = pd.get_dummies(features, columns=cat_cols, drop_first=True)

    features = features.fillna(0)
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.fillna(0)

    # Align to expected feature names
    if feature_names is not None:
        for name in feature_names:
            if name not in features.columns:
                features[name] = 0
        features = features[feature_names]

    return features


# --------------------------------------------------------------------------- #
# Inference: Forecasting
# --------------------------------------------------------------------------- #
def forecast_consumption(
    household_id: str, horizon_days: int = 7
) -> Optional[ForecastResult]:
    """Generate consumption forecasts for a household.

    Uses the trained regression model to predict future half-hourly
    consumption. Falls back to historical averages if no model is available.

    Args:
        household_id: The household to forecast for.
        horizon_days: Number of days ahead to forecast.

    Returns:
        A :class:`ForecastResult` or ``None`` if no data is available.
    """
    df = load_processed_data()
    if df.empty:
        return None

    household_data = df[df[config.COL_HOUSEHOLD_ID] == household_id].copy()
    if household_data.empty:
        return None

    household_data[config.COL_TIMESTAMP] = pd.to_datetime(
        household_data[config.COL_TIMESTAMP]
    )
    household_data = household_data.sort_values(config.COL_TIMESTAMP)

    # Load feature names from training (if saved)
    feature_names_path = config.REGRESSION_FEATURE_NAMES_FILE
    feature_names = None
    if feature_names_path.is_file():
        try:
            feature_names = joblib.load(feature_names_path)
        except Exception:
            feature_names = None

    model = load_model()

    n_periods = horizon_days * config.HALF_HOURS_PER_DAY
    last_ts = household_data[config.COL_TIMESTAMP].max()

    # Generate future timestamps
    future_ts = pd.date_range(
        start=last_ts + timedelta(minutes=30),
        periods=n_periods,
        freq="30min",
    )

    if model is not None:
        # Build future feature matrix based on last known data
        last_row = household_data.iloc[[-1]].copy()
        future_rows = []
        for ts in future_ts:
            row = last_row.copy()
            row[config.COL_TIMESTAMP] = ts
            row["hour"] = ts.hour
            row["day"] = ts.day
            row["month"] = ts.month
            row["weekday"] = ts.weekday()
            row["is_weekend"] = int(ts.weekday() >= 5)
            row["quarter"] = ts.quarter
            future_rows.append(row)

        future_df = pd.concat(future_rows, ignore_index=True)
        features = _prepare_inference_features(future_df, feature_names)

        try:
            predictions = model.predict(features)
            predictions = np.maximum(predictions, 0)  # No negative consumption

            # Estimate confidence from training R²
            confidence = 0.85
            results_path = config.MODEL_RESULTS_FILE
            if results_path.is_file():
                try:
                    results_df = pd.read_csv(results_path)
                    best = results_df[results_df.get("is_best", pd.Series(dtype=bool)).fillna(False).astype(bool)]
                    if not best.empty:
                        confidence = float(best["r2"].iloc[0])
                except Exception:
                    pass

            # Confidence intervals (wider for further predictions)
            std_est = np.std(predictions) if len(predictions) > 1 else 0.1
            steps = np.arange(1, n_periods + 1)
            widening = 1.0 + 0.02 * steps  # Widens over time
            lower = np.maximum(predictions - 1.96 * std_est * widening, 0)
            upper = predictions + 1.96 * std_est * widening

            return ForecastResult(
                predicted_kwh=predictions.tolist(),
                timestamps=future_ts.tolist(),
                confidence=confidence,
                lower_kwh=lower.tolist(),
                upper_kwh=upper.tolist(),
            )
        except Exception as e:
            logger.error("Model prediction failed: %s", e)
            # Fall through to historical average fallback

    # Fallback: use historical hourly averages
    hourly_avg = household_data.groupby(
        household_data[config.COL_TIMESTAMP].dt.hour
    )[config.COL_CONSUMPTION_KWH].mean()

    predictions = []
    for ts in future_ts:
        hour = ts.hour
        val = hourly_avg.get(hour, household_data[config.COL_CONSUMPTION_KWH].mean())
        predictions.append(float(val))

    predictions = np.array(predictions)
    std_est = np.std(predictions) if len(predictions) > 1 else 0.1
    lower = np.maximum(predictions - 1.96 * std_est, 0)
    upper = predictions + 1.96 * std_est

    return ForecastResult(
        predicted_kwh=predictions.tolist(),
        timestamps=future_ts.tolist(),
        confidence=0.5,  # Lower confidence for fallback
        lower_kwh=lower.tolist(),
        upper_kwh=upper.tolist(),
    )


# --------------------------------------------------------------------------- #
# Inference: Clustering
# --------------------------------------------------------------------------- #
def assign_cluster(household_id: str) -> Optional[int]:
    """Assign a household to its consumption cluster.

    Loads the pre-computed cluster map if available, otherwise runs the
    clustering model on the household's features.

    Args:
        household_id: The household to classify.

    Returns:
        An integer cluster label, or ``None`` if clustering is unavailable.
    """
    # Try loading the pre-computed household → cluster map first
    map_path = config.MODELS_DIR / "household_cluster_map.joblib"
    if map_path.is_file():
        try:
            cluster_map = joblib.load(map_path)
            if isinstance(cluster_map, pd.DataFrame):
                row = cluster_map[cluster_map["household_id"] == household_id]
                if not row.empty:
                    return int(row["cluster"].iloc[0])
            elif isinstance(cluster_map, dict):
                return cluster_map.get(household_id)
        except Exception as e:
            logger.warning("Could not load cluster map: %s", e)

    # Fall back to running the model on this household's features
    model = load_clustering_model()
    if model is None:
        return None

    df = load_processed_data()
    if df.empty:
        return None

    household_data = df[df[config.COL_HOUSEHOLD_ID] == household_id]
    if household_data.empty:
        return None

    # Compute aggregate features for this household
    features = {
        "avg_consumption": household_data[config.COL_CONSUMPTION_KWH].mean(),
        "peak_consumption": household_data[config.COL_CONSUMPTION_KWH].max(),
        "std_consumption": household_data[config.COL_CONSUMPTION_KWH].std(),
        "total_consumption": household_data[config.COL_CONSUMPTION_KWH].sum(),
    }

    # Load expected feature names
    feat_names_path = config.MODELS_DIR / "cluster_feature_names.joblib"
    if feat_names_path.is_file():
        try:
            expected = joblib.load(feat_names_path)
            feature_vec = pd.DataFrame(
                [{name: features.get(name, 0) for name in expected}]
            )
        except Exception:
            feature_vec = pd.DataFrame([features])
    else:
        feature_vec = pd.DataFrame([features])

    feature_vec = feature_vec.fillna(0)

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_vec)
        cluster = int(model.predict(X_scaled)[0])
        return cluster
    except Exception as e:
        logger.error("Cluster assignment failed: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Inference: Anomaly Detection
# --------------------------------------------------------------------------- #
def detect_anomalies(household_id: str) -> pd.DataFrame:
    """Detect anomalous consumption readings for a household.

    Uses the trained Isolation Forest model. Returns a DataFrame with
    columns: timestamp, consumption_kwh, score, severity.

    Args:
        household_id: The household to analyse.

    Returns:
        A DataFrame of detected anomalies (empty if none found or model
        unavailable).
    """
    model_path = config.ANOMALY_MODEL_FILE
    if not model_path.is_file():
        logger.warning("Anomaly model not found at %s", model_path)
        return pd.DataFrame(columns=["timestamp", config.COL_CONSUMPTION_KWH, "score", "severity"])

    df = load_processed_data()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", config.COL_CONSUMPTION_KWH, "score", "severity"])

    if household_id != "all":
        household_data = df[df[config.COL_HOUSEHOLD_ID] == household_id].copy()
    else:
        household_data = df.copy()

    if household_data.empty:
        return pd.DataFrame(columns=["timestamp", config.COL_CONSUMPTION_KWH, "score", "severity"])

    # Load anomaly feature names
    feat_names_path = config.MODELS_DIR / "anomaly_feature_names.joblib"
    feature_names = None
    if feat_names_path.is_file():
        try:
            feature_names = joblib.load(feat_names_path)
        except Exception:
            feature_names = None

    features = _prepare_inference_features(household_data, feature_names)

    try:
        model = joblib.load(model_path)
        # Isolation Forest: predict returns 1 (normal) or -1 (anomaly)
        labels = model.predict(features)
        # score_samples returns anomaly scores (more negative = more anomalous)
        scores = model.score_samples(features)

        household_data = household_data.copy()
        household_data["anomaly_label"] = labels
        household_data["score"] = scores

        # Filter anomalies (label == -1)
        anomalies = household_data[household_data["anomaly_label"] == -1].copy()

        if anomalies.empty:
            return pd.DataFrame(
                columns=["timestamp", config.COL_CONSUMPTION_KWH, "score", "severity"]
            )

        # Compute severity: normalise scores to 0-1 (more negative → higher severity)
        if len(anomalies) > 0:
            min_score = anomalies["score"].min()
            max_score = anomalies["score"].max()
            score_range = max_score - min_score
            if score_range > 0:
                anomalies["severity"] = 1 - (anomalies["score"] - min_score) / score_range
            else:
                anomalies["severity"] = 1.0
        else:
            anomalies["severity"] = 0.0

        # Return relevant columns
        result_cols = {
            config.COL_TIMESTAMP: "timestamp",
            config.COL_CONSUMPTION_KWH: config.COL_CONSUMPTION_KWH,
            "score": "score",
            "severity": "severity",
        }
        result = anomalies.rename(columns=result_cols)
        keep_cols = [v for v in result_cols.values() if v in result.columns]
        return result[keep_cols]

    except Exception as e:
        logger.error("Anomaly detection failed: %s", e)
        return pd.DataFrame(columns=["timestamp", config.COL_CONSUMPTION_KWH, "score", "severity"])


# --------------------------------------------------------------------------- #
# Training functions
# --------------------------------------------------------------------------- #
def train_classical_models(max_train_rows: int | None = None) -> dict:
    """Train Random Forest and XGBoost regressors for consumption forecasting.

    Trains both models on the processed dataset, evaluates on a hold-out set,
    and saves the best one along with both individual models.

    Args:
        max_train_rows: If set, sample the dataset to this many rows before
            training (useful for large datasets where full training is too slow).

    Returns:
        A dictionary of model names to their evaluation metrics.
    """
    config.ensure_directories()
    frame = load_processed_data()
    if frame.empty:
        raise FileNotFoundError(
            f"Processed dataset not found at {config.FINAL_DATASET_PARQUET}"
        )

    if max_train_rows and len(frame) > max_train_rows:
        logger.info(
            "Sampling %d rows from %d for training",
            max_train_rows, len(frame),
        )
        frame = frame.sample(n=max_train_rows, random_state=42).reset_index(drop=True)

    features, target = _prepare_training_data(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Save feature names for inference alignment
    joblib.dump(list(features.columns), config.REGRESSION_FEATURE_NAMES_FILE)

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = float(np.mean(np.abs(rf_pred - y_test)))
    rf_rmse = float(np.sqrt(np.mean((rf_pred - y_test) ** 2)))
    rf_r2 = float(1 - np.sum((rf_pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    results["RandomForestRegressor"] = {"mae": rf_mae, "rmse": rf_rmse, "r2": rf_r2}

    joblib.dump(rf, config.MODELS_DIR / "random_forest_regressor.joblib")
    logger.info("Random Forest — MAE: %.6f, RMSE: %.6f, R²: %.6f", rf_mae, rf_rmse, rf_r2)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_mae = float(np.mean(np.abs(xgb_pred - y_test)))
    xgb_rmse = float(np.sqrt(np.mean((xgb_pred - y_test) ** 2)))
    xgb_r2 = float(1 - np.sum((xgb_pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    results["XGBoostRegressor"] = {"mae": xgb_mae, "rmse": xgb_rmse, "r2": xgb_r2}

    joblib.dump(xgb, config.MODELS_DIR / "xgboost_regressor.joblib")
    logger.info("XGBoost — MAE: %.6f, RMSE: %.6f, R²: %.6f", xgb_mae, xgb_rmse, xgb_r2)

    # Save the best model
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = rf if best_name == "RandomForestRegressor" else xgb
    joblib.dump(best_model, config.REGRESSION_MODEL_FILE)
    logger.info("Best model: %s (R²=%.6f)", best_name, results[best_name]["r2"])

    # Save results CSV
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "is_best": name == best_name,
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(config.MODEL_RESULTS_FILE, index=False)

    return results


def train_clustering_model() -> None:
    """Train a KMeans clustering model on household consumption profiles.

    Computes aggregate features per household, scales them, fits KMeans,
    and saves both the model and the household→cluster mapping.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    config.ensure_directories()
    frame = load_processed_data()
    if frame.empty:
        raise FileNotFoundError(
            f"Processed dataset not found at {config.FINAL_DATASET_PARQUET}"
        )

    # Compute per-household features
    features_list = []
    for hhid, group in frame.groupby(config.COL_HOUSEHOLD_ID):
        features_list.append({
            "household_id": hhid,
            "avg_consumption": group[config.COL_CONSUMPTION_KWH].mean(),
            "peak_consumption": group[config.COL_CONSUMPTION_KWH].max(),
            "std_consumption": group[config.COL_CONSUMPTION_KWH].std(),
            "total_consumption": group[config.COL_CONSUMPTION_KWH].sum(),
        })

    features_df = pd.DataFrame(features_list)
    feature_cols = [c for c in features_df.columns if c != "household_id"]
    X = features_df[feature_cols].fillna(0)

    # Save feature names for inference
    joblib.dump(feature_cols, config.MODELS_DIR / "cluster_feature_names.joblib")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(config.DEFAULT_N_CLUSTERS, len(features_df))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    joblib.dump(model, config.CLUSTERING_MODEL_FILE)

    # Save household → cluster map
    cluster_map = features_df[["household_id"]].copy()
    cluster_map["cluster"] = labels
    joblib.dump(cluster_map, config.MODELS_DIR / "household_cluster_map.joblib")

    logger.info(
        "Trained KMeans with %d clusters on %d households",
        n_clusters,
        len(features_df),
    )


def train_anomaly_model(
    contamination: float = 0.02,
    random_state: int = 42,
    max_train_rows: int | None = None,
) -> None:
    """Train an Isolation Forest model for anomaly detection.

    Uses the same feature preparation as for regression (excluding target
    and identifiers). Saves the model and the feature names used during
    training.
    """
    config.ensure_directories()
    frame = load_processed_data()
    if frame.empty:
        raise FileNotFoundError(
            f"Processed dataset not found at {config.FINAL_DATASET_PARQUET}"
        )

    if max_train_rows and len(frame) > max_train_rows:
        logger.info(
            "Sampling %d rows from %d for anomaly training",
            max_train_rows, len(frame),
        )
        frame = frame.sample(n=max_train_rows, random_state=42).reset_index(drop=True)

    features, _ = _prepare_training_data(frame)
    features = features.fillna(0)

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    iso_forest.fit(features)

    joblib.dump(iso_forest, config.ANOMALY_MODEL_FILE)

    feature_names = list(features.columns)
    feature_names_path = config.MODELS_DIR / "anomaly_feature_names.joblib"
    joblib.dump(feature_names, feature_names_path)

    logger.info(
        "Trained Isolation Forest (contamination=%.2f) → %s",
        contamination,
        config.ANOMALY_MODEL_FILE,
    )
