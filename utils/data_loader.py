"""Data loading layer for the London Smart Meter dataset.

Responsible for reading raw and processed data from disk into tidy pandas
DataFrames. This module is the *only* place that should touch data files
directly — every other module receives DataFrames, keeping I/O concerns
isolated and easy to swap (e.g. move to a database or cloud storage later).

The loaders are robust to the two common layouts of the raw meter data:

* a single ``halfhourly_dataset.csv`` file, or
* a ``halfhourly_dataset/`` folder of ``block_*.csv`` files (Kaggle layout).

Large CSVs are streamed in chunks and results are cached in-process so repeated
calls within a Streamlit session are cheap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

import config
from utils.helpers import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Low-level file helpers
# --------------------------------------------------------------------------- #
def _read_csv_chunked(path: Path, **read_kwargs) -> pd.DataFrame:
    """Read a (potentially large) CSV in chunks and concatenate the result.

    Args:
        path: CSV file to read.
        **read_kwargs: Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns:
        The full DataFrame assembled from all chunks.
    """
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=config.CHUNK_SIZE, **read_kwargs):
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    logger.info("Read %s rows from %s", f"{len(frame):,}", path.name)
    return frame


def _discover_meter_files() -> list[Path]:
    """Locate the raw meter file(s) on disk, supporting both layouts.

    Returns:
        A list of CSV/parquet paths to read (a single file, or every
        ``block_*.csv`` inside the meter folder).

    Raises:
        FileNotFoundError: If no raw meter data can be found.
    """
    if config.RAW_METER_DIR.is_dir():
        blocks = sorted(config.RAW_METER_DIR.rglob(config.RAW_METER_BLOCK_GLOB))
        if blocks:
            logger.info("Discovered %d meter block files in %s", len(blocks), config.RAW_METER_DIR)
            return blocks

    if config.RAW_METER_FILE.is_file():
        return [config.RAW_METER_FILE]

    # Fall back to any parquet export of the raw meter data.
    parquet = config.RAW_METER_FILE.with_suffix(".parquet")
    if parquet.is_file():
        return [parquet]

    raise FileNotFoundError(
        "No raw meter data found. Expected either "
        f"'{config.RAW_METER_FILE}' or a '{config.RAW_METER_DIR}' folder of "
        f"'{config.RAW_METER_BLOCK_GLOB}' files."
    )


def _read_any(path: Path, **read_kwargs) -> pd.DataFrame:
    """Read a CSV or Parquet file based on its extension."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return _read_csv_chunked(path, **read_kwargs)


# --------------------------------------------------------------------------- #
# Raw data loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_raw_meter_data() -> pd.DataFrame:
    """Load raw half-hourly smart-meter readings from all discovered files.

    Reads every block file (or the single meter CSV), concatenates them, and
    renames the raw columns to the canonical schema in :mod:`config`.

    Returns:
        A DataFrame with columns ``household_id``, ``timestamp`` (raw string),
        and ``consumption_kwh`` (raw string — parsed later in preprocessing).

    Raises:
        FileNotFoundError: If no raw meter data is present.
    """
    files = _discover_meter_files()
    usecols = [config.RAW_COL_LCLID, config.RAW_COL_TSTP, config.RAW_COL_ENERGY]

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = _read_any(path, usecols=usecols)
        except ValueError:
            # Column names differ from the expected raw schema — read as-is.
            frame = _read_any(path)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw = raw.rename(
        columns={
            config.RAW_COL_LCLID: config.COL_HOUSEHOLD_ID,
            config.RAW_COL_TSTP: config.COL_TIMESTAMP,
            config.RAW_COL_ENERGY: config.COL_CONSUMPTION_KWH,
        }
    )
    logger.info("Loaded %s raw meter rows from %d file(s)", f"{len(raw):,}", len(files))
    return raw


@lru_cache(maxsize=1)
def load_household_info() -> pd.DataFrame:
    """Load household metadata (ACORN classification, tariff type).

    Returns:
        A DataFrame keyed by ``household_id`` with ``acorn``, ``acorn_group``
        and ``tariff`` columns. Returns an empty, correctly-typed frame if the
        file is missing (household features are then simply skipped).
    """
    path = config.RAW_HOUSEHOLD_INFO_FILE
    if not path.is_file():
        logger.warning("Household info file missing at %s — skipping metadata", path)
        return pd.DataFrame(
            columns=[
                config.COL_HOUSEHOLD_ID,
                config.COL_ACORN,
                config.COL_ACORN_GROUP,
                config.COL_TARIFF,
            ]
        )

    info = pd.read_csv(path)
    info = info.rename(
        columns={
            config.RAW_COL_LCLID: config.COL_HOUSEHOLD_ID,
            config.RAW_COL_ACORN: config.COL_ACORN,
            config.RAW_COL_ACORN_GROUPED: config.COL_ACORN_GROUP,
            config.RAW_COL_TARIFF: config.COL_TARIFF,
        }
    )
    keep = [
        c
        for c in (
            config.COL_HOUSEHOLD_ID,
            config.COL_ACORN,
            config.COL_ACORN_GROUP,
            config.COL_TARIFF,
        )
        if c in info.columns
    ]
    logger.info("Loaded %s household records", f"{len(info):,}")
    return info[keep]


@lru_cache(maxsize=1)
def load_weather_data() -> pd.DataFrame:
    """Load hourly weather observations aligned to the meter data.

    Returns:
        A DataFrame with a ``timestamp`` column and the canonical weather
        feature columns. Returns an empty frame if the file is missing.
    """
    path = config.RAW_WEATHER_FILE
    if not path.is_file():
        logger.warning("Weather file missing at %s — skipping weather features", path)
        return pd.DataFrame(columns=[config.COL_TIMESTAMP, config.COL_TEMPERATURE])

    weather = pd.read_csv(path)
    rename = {config.RAW_COL_WEATHER_TIME: config.COL_TIMESTAMP, **config.RAW_WEATHER_COLUMN_MAP}
    weather = weather.rename(columns=rename)
    keep = [config.COL_TIMESTAMP] + [
        c for c in config.RAW_WEATHER_COLUMN_MAP.values() if c in weather.columns
    ]
    logger.info("Loaded %s weather observations", f"{len(weather):,}")
    return weather[keep]


# --------------------------------------------------------------------------- #
# Processed data loaders
# --------------------------------------------------------------------------- #
def load_processed_data(path: Path = config.FINAL_DATASET_PARQUET) -> pd.DataFrame:
    """Load the cleaned, feature-rich dataset produced by the pipeline.

    Args:
        path: Location of the processed parquet. Defaults to
            :data:`config.FINAL_DATASET_PARQUET`.

    Returns:
        The processed DataFrame, or an empty frame if it has not been built yet
        (so the UI can gracefully show an empty state).
    """
    if not path.is_file():
        logger.info("Processed dataset not found at %s (run build_processed_dataset)", path)
        return pd.DataFrame(
            columns=[config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP, config.COL_CONSUMPTION_KWH]
        )
    frame = pd.read_parquet(path)
    logger.info("Loaded processed dataset (%s rows) from %s", f"{len(frame):,}", path)
    return frame


def list_household_ids() -> list[str]:
    """Return the sorted list of household identifiers in the processed data.

    Used to populate selectors in the UI. Returns an empty list if the
    processed dataset has not been built yet.
    """
    frame = load_processed_data()
    if frame.empty or config.COL_HOUSEHOLD_ID not in frame.columns:
        return []
    return sorted(frame[config.COL_HOUSEHOLD_ID].astype(str).unique().tolist())


def clear_cache() -> None:
    """Clear the in-process raw-data caches (useful after new files land)."""
    load_raw_meter_data.cache_clear()
    load_household_info.cache_clear()
    load_weather_data.cache_clear()
    logger.info("Cleared data-loader caches")
