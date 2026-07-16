"""Feature engineering for forecasting, clustering, and anomaly detection.

Derives model-ready features from the cleaned consumption data: calendar
features, lag/rolling statistics, weather interactions, and per-household
aggregates. Also fits and persists ``StandardScaler`` / ``MinMaxScaler``
artefacts under :data:`config.MODELS_DIR` for reuse by ML models.

All feature generators are vectorised (``groupby`` transforms, no Python
loops over rows) so the pipeline scales to millions of rows.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config
from utils.helpers import get_logger

logger = get_logger(__name__)

# Seasons keyed by calendar month (Northern Hemisphere).
_SEASON_BY_MONTH: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the timestamp.

    Adds hour, day, month, weekday, weekend flag, week number, quarter,
    and season.

    Args:
        df: Dataset with a ``timestamp`` column.

    Returns:
        The DataFrame with calendar feature columns appended.
    """
    ts = df[config.COL_TIMESTAMP].dt
    df["hour"] = ts.hour.astype("int8")
    df["day"] = ts.day.astype("int8")
    df["month"] = ts.month.astype("int8")
    df["weekday"] = ts.weekday.astype("int8")
    df["is_weekend"] = (ts.weekday >= 5).astype("int8")
    df["week"] = ts.isocalendar().week.astype("int16").to_numpy()
    df["quarter"] = ts.quarter.astype("int8")
    df["season"] = df["month"].map(_SEASON_BY_MONTH).astype("category")
    logger.info("add_calendar_features: added 8 calendar columns")
    return df


# --------------------------------------------------------------------------- #
# Lag & rolling features
# --------------------------------------------------------------------------- #
def add_lag_features(
    df: pd.DataFrame, lags: tuple[int, ...] = config.LAG_STEPS
) -> pd.DataFrame:
    """Add per-household lagged consumption features.

    Args:
        df: Time-sorted dataset grouped-implicitly by household.
        lags: Lag steps (in half-hourly intervals) to compute.

    Returns:
        The DataFrame with ``lag_{n}`` columns appended.
    """
    grouped = df.groupby(config.COL_HOUSEHOLD_ID, observed=True)[config.COL_CONSUMPTION_KWH]
    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)
    logger.info("add_lag_features: added lags %s", lags)
    return df


def add_rolling_features(
    df: pd.DataFrame, windows: tuple[int, ...] = config.ROLLING_WINDOWS
) -> pd.DataFrame:
    """Add per-household rolling mean/std/max/min features.

    Rolling stats are shifted by one step so a row never sees its own value
    (prevents target leakage during model training).

    Args:
        df: Time-sorted dataset.
        windows: Rolling window sizes (in half-hourly intervals).

    Returns:
        The DataFrame with rolling feature columns appended.
    """
    grouped = df.groupby(config.COL_HOUSEHOLD_ID, observed=True)[config.COL_CONSUMPTION_KWH]
    shifted = grouped.shift(1)
    for window in windows:
        roll = shifted.rolling(window=window, min_periods=1)
        df[f"roll_mean_{window}"] = roll.mean()
        df[f"roll_std_{window}"] = roll.std()
        df[f"roll_max_{window}"] = roll.max()
        df[f"roll_min_{window}"] = roll.min()
    logger.info("add_rolling_features: added rolling windows %s", windows)
    return df


# --------------------------------------------------------------------------- #
# Interaction & household features
# --------------------------------------------------------------------------- #
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between weather, calendar, and consumption.

    Adds temperature×consumption, weekend×hour, and season×temperature
    (season encoded numerically). Weather-based interactions are only created
    when the underlying weather columns are present.

    Args:
        df: Dataset with calendar (and optionally weather) features.

    Returns:
        The DataFrame with interaction columns appended.
    """
    if config.COL_TEMPERATURE in df.columns:
        df["temp_x_consumption"] = (
            df[config.COL_TEMPERATURE] * df[config.COL_CONSUMPTION_KWH]
        )
        if "season" in df.columns:
            # ``season`` is set as a category by add_calendar_features.
            df["season_x_temp"] = df["season"].cat.codes * df[config.COL_TEMPERATURE]

    df["weekend_x_hour"] = df["is_weekend"] * df["hour"]
    logger.info("add_interaction_features: added interaction columns")
    return df


def add_household_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-household aggregate features (average, peak, variance).

    Args:
        df: Cleaned consumption dataset.

    Returns:
        The DataFrame with ``hh_avg_consumption``, ``hh_peak_consumption`` and
        ``hh_consumption_variance`` columns broadcast to every row.
    """
    grouped = df.groupby(config.COL_HOUSEHOLD_ID, observed=True)[config.COL_CONSUMPTION_KWH]
    df["hh_avg_consumption"] = grouped.transform("mean")
    df["hh_peak_consumption"] = grouped.transform("max")
    df["hh_consumption_variance"] = grouped.transform("var")
    logger.info("add_household_features: added household aggregate columns")
    return df


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def fit_and_save_scalers(df: pd.DataFrame) -> pd.DataFrame:
    """Fit ``StandardScaler`` and ``MinMaxScaler`` on numeric features and save.

    Both fitted scalers are persisted with joblib under
    :data:`config.MODELS_DIR` for reuse by ML models. The input DataFrame
    is returned unchanged (scaling is applied at model-training time using the
    saved artefacts).

    Args:
        df: The final feature matrix.

    Returns:
        The unchanged input DataFrame (side effect: scaler files written).
    """
    config.ensure_directories()
    numeric = df.select_dtypes(include=[np.number])
    # Exclude the raw target and pure calendar index columns from scaling meta.
    feature_cols = [c for c in numeric.columns if c != config.COL_CONSUMPTION_KWH]
    if not feature_cols:
        logger.warning("fit_and_save_scalers: no numeric feature columns to scale")
        return df

    values = df[feature_cols].to_numpy()
    standard = StandardScaler().fit(values)
    minmax = MinMaxScaler().fit(values)

    joblib.dump(
        {"scaler": standard, "columns": feature_cols}, config.STANDARD_SCALER_FILE
    )
    joblib.dump(
        {"scaler": minmax, "columns": feature_cols}, config.MINMAX_SCALER_FILE
    )
    logger.info(
        "fit_and_save_scalers: fitted on %d features, saved to %s",
        len(feature_cols),
        config.MODELS_DIR,
    )
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final feature matrix from all feature generators.

    Chains calendar, lag, rolling, interaction, and household features, fills
    the NaNs introduced by lags/rolling with 0, and fits/saves the scalers.

    Args:
        df: The cleaned, merged dataset (time-sorted per household).

    Returns:
        The model-ready feature matrix.
    """
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df = add_household_features(df)

    # Lags/rolling create leading NaNs — fill with 0 (no history available).
    engineered = [c for c in df.columns if c.startswith(("lag_", "roll_"))]
    df[engineered] = df[engineered].fillna(0)
    df["season_x_temp"] = df.get("season_x_temp", pd.Series(0, index=df.index)).fillna(0)

    df = fit_and_save_scalers(df)
    logger.info("build_feature_matrix: final shape %s", df.shape)
    return df
