"""Data cleaning, merging, and validation for the London Smart Meter data.

Transforms the raw sources into a single clean, analysis-ready DataFrame:

* parses timestamps and numeric energy values,
* removes duplicates and physically invalid records,
* handles missing values and flags statistical outliers,
* merges household metadata and weather onto every reading,
* optimises dtypes/memory,
* validates the result and writes a Markdown data-quality report.

All operations are vectorised for performance on millions of rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from utils import data_loader, feature_engineering
from utils.helpers import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_meter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw half-hourly meter readings.

    Parses timestamps and energy values, drops null/duplicate/invalid rows,
    validates the energy range, and sorts by household and time.

    Args:
        df: Raw meter data with ``household_id``, ``timestamp`` and
            ``consumption_kwh`` columns.

    Returns:
        A cleaned, chronologically sorted DataFrame.
    """
    start_rows = len(df)
    df = df.copy()

    # --- Type parsing (vectorised) ---------------------------------------- #
    df[config.COL_TIMESTAMP] = pd.to_datetime(
        df[config.COL_TIMESTAMP], errors="coerce", utc=False
    )
    # Energy arrives as strings; 'Null'/blank become NaN.
    df[config.COL_CONSUMPTION_KWH] = pd.to_numeric(
        df[config.COL_CONSUMPTION_KWH], errors="coerce"
    )

    # --- Drop unusable rows ----------------------------------------------- #
    df = df.dropna(
        subset=[config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP, config.COL_CONSUMPTION_KWH]
    )

    # --- Remove duplicate (household, timestamp) readings ----------------- #
    df = df.drop_duplicates(subset=[config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP])

    # --- Validate energy range (drop negatives / impossible spikes) ------- #
    valid = df[config.COL_CONSUMPTION_KWH].between(
        config.MIN_VALID_KWH, config.MAX_VALID_KWH
    )
    df = df[valid]

    # --- Sort chronologically per household ------------------------------- #
    df = df.sort_values([config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP]).reset_index(
        drop=True
    )

    logger.info(
        "clean_meter_data: %s -> %s rows (removed %s)",
        f"{start_rows:,}",
        f"{len(df):,}",
        f"{start_rows - len(df):,}",
    )
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag (but do not drop) statistical consumption outliers per household.

    Adds a boolean ``is_outlier`` column using a per-household z-score so that
    normal high-usage homes are not unfairly flagged.

    Args:
        df: Cleaned meter data.

    Returns:
        The DataFrame with an added ``is_outlier`` column.
    """
    grouped = df.groupby(config.COL_HOUSEHOLD_ID)[config.COL_CONSUMPTION_KWH]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    zscore = (df[config.COL_CONSUMPTION_KWH] - mean) / std
    df["is_outlier"] = (zscore.abs() > config.OUTLIER_ZSCORE_THRESHOLD).fillna(False)
    logger.info("flag_outliers: flagged %s outliers", f"{int(df['is_outlier'].sum()):,}")
    return df


def clean_weather_data(weather: pd.DataFrame) -> pd.DataFrame:
    """Parse and de-duplicate weather observations.

    Args:
        weather: Raw weather data with a ``timestamp`` column.

    Returns:
        A cleaned, time-sorted weather DataFrame (empty if no weather data).
    """
    if weather.empty:
        return weather
    weather = weather.copy()
    weather[config.COL_TIMESTAMP] = pd.to_datetime(
        weather[config.COL_TIMESTAMP], errors="coerce"
    )
    weather = (
        weather.dropna(subset=[config.COL_TIMESTAMP])
        .drop_duplicates(subset=[config.COL_TIMESTAMP])
        .sort_values(config.COL_TIMESTAMP)
        .reset_index(drop=True)
    )
    return weather


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def merge_household(df: pd.DataFrame, household: pd.DataFrame) -> pd.DataFrame:
    """Attach household metadata (ACORN, tariff) to every meter reading."""
    if household.empty:
        return df
    merged = df.merge(household, on=config.COL_HOUSEHOLD_ID, how="left")
    logger.info("merge_household: attached metadata columns")
    return merged


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join hourly weather onto half-hourly readings via a nearest-time merge.

    Uses ``merge_asof`` so each reading picks up the closest weather record,
    bridging the hourly/half-hourly cadence mismatch without row explosion.

    Args:
        df: Cleaned (and already time-sorted) meter data.
        weather: Cleaned, time-sorted weather data.

    Returns:
        The meter data enriched with weather feature columns.
    """
    if weather.empty:
        return df

    # merge_asof requires both sides sorted by the merge key.
    left = df.sort_values(config.COL_TIMESTAMP)
    merged = pd.merge_asof(
        left,
        weather,
        on=config.COL_TIMESTAMP,
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )
    merged = merged.sort_values(
        [config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP]
    ).reset_index(drop=True)
    logger.info("merge_weather: joined %d weather columns", weather.shape[1] - 1)
    return merged


# --------------------------------------------------------------------------- #
# Missing values & memory optimisation
# --------------------------------------------------------------------------- #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining missing values with sensible, leakage-free strategies.

    Weather gaps are forward/backward filled (weather is temporally smooth);
    categorical metadata gaps become an explicit ``"Unknown"`` category.

    Args:
        df: Merged dataset.

    Returns:
        The DataFrame with missing values resolved.
    """
    weather_cols = [c for c in config.RAW_WEATHER_COLUMN_MAP.values() if c in df.columns]
    if weather_cols:
        df[weather_cols] = df[weather_cols].ffill().bfill()

    for col in (config.COL_ACORN, config.COL_ACORN_GROUP, config.COL_TARIFF):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df


def optimise_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and categorise low-cardinality strings.

    Args:
        df: The dataset to optimise.

    Returns:
        The same data in a more memory-efficient dtype layout.
    """
    before = df.memory_usage(deep=True).sum()

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in (config.COL_HOUSEHOLD_ID, config.COL_ACORN, config.COL_ACORN_GROUP, config.COL_TARIFF):
        if col in df.columns:
            df[col] = df[col].astype("category")

    after = df.memory_usage(deep=True).sum()
    logger.info(
        "optimise_memory: %.1f MB -> %.1f MB", before / 1e6, after / 1e6
    )
    return df


# --------------------------------------------------------------------------- #
# Validation & reporting
# --------------------------------------------------------------------------- #
def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Run automated data-quality checks and return a results dictionary.

    Checks for missing values, duplicate (household, timestamp) records,
    presence of required columns, and the consumption range.

    Args:
        df: The final dataset to validate.

    Returns:
        A dictionary of check names to boolean/summary results.
    """
    duplicate_keys = int(
        df.duplicated(subset=[config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP]).sum()
    )
    consumption = df.get(config.COL_CONSUMPTION_KWH, pd.Series(dtype=float))
    results = {
        "row_count": int(len(df)),
        "no_missing_values": bool(df.isna().sum().sum() == 0),
        "no_duplicate_records": duplicate_keys == 0,
        "duplicate_records": duplicate_keys,
        "has_required_columns": all(
            c in df.columns
            for c in (config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP, config.COL_CONSUMPTION_KWH)
        ),
        "consumption_in_range": bool(
            consumption.between(config.MIN_VALID_KWH, config.MAX_VALID_KWH).all()
        )
        if not consumption.empty
        else False,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }
    for name, ok in results.items():
        if isinstance(ok, bool) and not ok:
            logger.warning("Validation check failed: %s", name)
    return results


def generate_data_quality_report(df: pd.DataFrame, validation: dict[str, Any]) -> None:
    """Write a Markdown data-quality report to :data:`config.DATA_QUALITY_REPORT_FILE`.

    Args:
        df: The final processed dataset.
        validation: The result dict returned by :func:`validate_dataset`.
    """
    config.ensure_directories()
    missing = df.isna().sum()
    missing = missing[missing > 0]

    lines: list[str] = [
        "# Data Quality Report",
        "",
        f"_Generated by the AI Smart Energy Analytics pipeline (v{config.APP_VERSION})._",
        "",
        "## Dataset Size",
        f"- Rows: **{len(df):,}**",
        f"- Columns: **{df.shape[1]}**",
        (
            f"- Households: **{df[config.COL_HOUSEHOLD_ID].nunique():,}**"
            if config.COL_HOUSEHOLD_ID in df.columns
            else "- Households: **n/a**"
        ),
        f"- Memory usage: **{validation['memory_mb']} MB**",
        "## Validation Checks",
        f"- No missing values: **{validation['no_missing_values']}**",
        f"- No duplicate (household, timestamp) records: **{validation['no_duplicate_records']}**",
        f"- Required columns present: **{validation['has_required_columns']}**",
        f"- Consumption within [{config.MIN_VALID_KWH}, {config.MAX_VALID_KWH}] kWh: "
        f"**{validation['consumption_in_range']}**",
        "",
        "## Missing Values",
    ]
    if missing.empty:
        lines.append("- None 🎉")
    else:
        lines.extend(f"- `{col}`: {int(cnt):,}" for col, cnt in missing.items())

    lines += ["", "## Data Types", "", "| Column | Dtype |", "| --- | --- |"]
    lines.extend(f"| `{col}` | {dtype} |" for col, dtype in df.dtypes.items())

    # Numeric feature summary.
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        desc = numeric.describe().T
        lines += [
            "",
            "## Feature Summary (numeric)",
            "",
            "| Feature | Mean | Std | Min | Max |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| `{idx}` | {row['mean']:.3f} | {row['std']:.3f} | "
            f"{row['min']:.3f} | {row['max']:.3f} |"
            for idx, row in desc.iterrows()
        )

    # Potential issues.
    lines += ["", "## Potential Issues"]
    issues: list[str] = []
    if not validation["no_missing_values"]:
        issues.append("Dataset still contains missing values.")
    if not validation["no_duplicate_records"]:
        issues.append(f"{validation['duplicate_records']:,} duplicate records remain.")
    if "is_outlier" in df.columns and df["is_outlier"].any():
        issues.append(f"{int(df['is_outlier'].sum()):,} consumption outliers flagged.")
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- None detected.")

    report = "\n".join(lines)
    config.DATA_QUALITY_REPORT_FILE.write_text(report, encoding="utf-8")
    logger.info("Wrote data-quality report to %s", config.DATA_QUALITY_REPORT_FILE)


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #
def build_processed_dataset(save: bool = True) -> pd.DataFrame:
    """Run the full data-engineering pipeline end-to-end.

    Loads the raw sources, cleans and merges them, engineers features,
    validates the result, and (optionally) persists the final dataset as both
    parquet and CSV plus a Markdown data-quality report.

    This is the single public entry point for data engineering::

        from utils.preprocess import build_processed_dataset
        df = build_processed_dataset()

    Args:
        save: When ``True`` (default), write ``final_dataset.parquet``,
            ``final_dataset.csv`` and the data-quality report to disk.

    Returns:
        The final, feature-rich DataFrame.
    """
    config.ensure_directories()
    logger.info("=== build_processed_dataset: starting pipeline ===")

    # 1. Load raw sources.
    meter = data_loader.load_raw_meter_data()
    household = data_loader.load_household_info()
    weather = data_loader.load_weather_data()

    # 2. Clean.
    meter = clean_meter_data(meter)
    meter = flag_outliers(meter)
    weather = clean_weather_data(weather)

    # 3. Merge household metadata + weather.
    df = merge_household(meter, household)
    df = merge_weather(df, weather)

    # 4. Feature engineering.
    df = feature_engineering.build_feature_matrix(df)

    # 5. Resolve remaining missing values and optimise memory.
    df = handle_missing_values(df)
    df = optimise_memory(df)

    # 6. Validate + report.
    validation = validate_dataset(df)
    generate_data_quality_report(df, validation)

    # 7. Persist.
    if save:
        df.to_parquet(config.FINAL_DATASET_PARQUET, index=False)
        df.to_csv(config.FINAL_DATASET_CSV, index=False)
        logger.info(
            "Saved final dataset to %s and %s",
            config.FINAL_DATASET_PARQUET,
            config.FINAL_DATASET_CSV,
        )

    logger.info("=== build_processed_dataset: complete (%s rows) ===", f"{len(df):,}")
    return df
