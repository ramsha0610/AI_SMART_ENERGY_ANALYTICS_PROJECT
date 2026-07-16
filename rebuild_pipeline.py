#!/usr/bin/env python3
"""Rebuild the processed dataset using a memory-safe incremental pipeline.

Processes one intermediate batch at a time through merge + feature engineering
to avoid exhausting RAM on the full 167M-row dataset. Saves a representative
sample for the final dataset while training models on batch-level data.
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib

import config
from utils.data_loader import (
    _discover_meter_files,
    load_household_info,
    load_weather_data,
)
from utils.preprocess import (
    clean_meter_data,
    flag_outliers,
    clean_weather_data,
    merge_household,
    merge_weather,
    handle_missing_values,
    optimise_memory,
    validate_dataset,
    generate_data_quality_report,
)
from utils.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_interaction_features,
    fit_and_save_scalers,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

# Max rows in final dataset (memory-safe for 8GB RAM)
MAX_FINAL_ROWS = 3_000_000


def main():
    t0 = time.time()

    print("=" * 60)
    print("PHASE 1: DATA DISCOVERY")
    print("=" * 60)

    meter_files = _discover_meter_files()
    print(f"\nHalf-hourly block files detected: {len(meter_files)}")

    household = load_household_info()
    print(f"Household metadata rows: {len(household)}")

    weather_raw = load_weather_data()
    print(f"Weather rows: {len(weather_raw)}")

    # ------------------------------------------------------------------ #
    # PHASE 1b: Pre-compute per-household aggregates from intermediates
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PHASE 1b: PRE-COMPUTE HOUSEHOLD AGGREGATES")
    print("=" * 60)

    intermediate_dir = config.PROCESSED_DATA_DIR / "intermediate"
    intermediate_files = sorted(intermediate_dir.glob("batch_*.parquet"))

    processed_dir = config.PROCESSED_DATA_DIR / "processed_batches"
    processed_files_existing = sorted(processed_dir.glob("batch_*.parquet")) if processed_dir.exists() else []

    if processed_files_existing:
        print(f"Found {len(processed_files_existing)} already-processed batches, skipping to Phase 4")
        # Compute total rows from processed batches
        total_rows = 0
        for pf in processed_files_existing:
            info = pd.read_parquet(pf, columns=[config.COL_HOUSEHOLD_ID])
            total_rows += len(info)
            del info
            gc.collect()
        print(f"Total rows across processed batches: {total_rows:,}")
        print(f"Households: {len(household)}")
    elif not intermediate_files:
        print("No intermediate files found. Running Phase 2 first...")
        return _run_full_pipeline(meter_files, household, weather_raw)

    if not processed_files_existing:
        # Only run Phase 1b + Phase 3 if we don't have processed batches yet
        print(f"Found {len(intermediate_files)} intermediate batches")

        # Accumulate per-household stats across all batches
        hh_stats = {}  # household_id -> (sum, count, max, list_of_vals_for_var)
        total_rows = 0
        for ipath in intermediate_files:
            batch = pd.read_parquet(ipath, columns=[config.COL_HOUSEHOLD_ID, config.COL_CONSUMPTION_KWH])
            total_rows += len(batch)
            for hh_id, grp in batch.groupby(config.COL_HOUSEHOLD_ID, observed=True):
                vals = grp[config.COL_CONSUMPTION_KWH]
                s, c, mx = vals.sum(), len(vals), vals.max()
                m2 = ((vals - vals.mean()) ** 2).sum()
                if hh_id in hh_stats:
                    prev_s, prev_c, prev_mx, prev_m2 = hh_stats[hh_id]
                    new_c = prev_c + c
                    new_mean = (prev_s + s) / new_c
                    hh_stats[hh_id] = (
                        prev_s + s, new_c, max(prev_mx, mx),
                        prev_m2 + m2 + prev_c * (prev_s / prev_c - new_mean) ** 2 + c * (s / c - new_mean) ** 2
                    )
                else:
                    hh_stats[hh_id] = (s, c, mx, m2)
            del batch
            gc.collect()

        # Build household aggregate DataFrame
        hh_agg_rows = []
        for hh_id, (s, c, mx, m2) in hh_stats.items():
            hh_agg_rows.append({
                config.COL_HOUSEHOLD_ID: hh_id,
                "hh_avg_consumption": s / c if c > 0 else 0,
                "hh_peak_consumption": mx,
                "hh_consumption_variance": m2 / (c - 1) if c > 1 else 0,
            })
        hh_aggregates = pd.DataFrame(hh_agg_rows)
        del hh_stats, hh_agg_rows
        gc.collect()

        print(f"Total rows across intermediates: {total_rows:,}")
        print(f"Households with aggregates: {len(hh_aggregates)}")

        # ------------------------------------------------------------------ #
        # PHASE 3: INCREMENTAL MERGE + FEATURE ENGINEERING
        # ------------------------------------------------------------------ #
        print("\n" + "=" * 60)
        print("PHASE 3: INCREMENTAL MERGE + FEATURE ENGINEERING")
        print("=" * 60)

        weather = clean_weather_data(weather_raw)
        weather[config.COL_TIMESTAMP] = pd.to_datetime(
            weather[config.COL_TIMESTAMP]
        ).astype("datetime64[ns]")

        processed_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0
        for ipath in intermediate_files:
            batch_id = ipath.stem  # e.g. batch_000
            out_path = processed_dir / f"{batch_id}.parquet"

            if out_path.exists():
                print(f"  {batch_id}: already processed, skipping")
                processed_count += 1
                continue

            print(f"  {batch_id}: loading...", end=" ")
            batch_df = pd.read_parquet(ipath)

            # Fix timestamp precision
            batch_df[config.COL_TIMESTAMP] = pd.to_datetime(
                batch_df[config.COL_TIMESTAMP]
            ).astype("datetime64[ns]")

            # Sort for lag/rolling features
            batch_df = batch_df.sort_values(
                [config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP]
            ).reset_index(drop=True)

            print("merging...", end=" ")
            batch_df = merge_household(batch_df, household)
            batch_df = merge_weather(batch_df, weather)

            print("features...", end=" ")
            batch_df = add_calendar_features(batch_df)
            batch_df = add_lag_features(batch_df)
            batch_df = add_rolling_features(batch_df)
            batch_df = add_interaction_features(batch_df)

            # Use pre-computed household aggregates
            batch_df = batch_df.merge(hh_aggregates, on=config.COL_HOUSEHOLD_ID, how="left")

            # Fill NaNs from lags/rolling
            engineered = [c for c in batch_df.columns if c.startswith(("lag_", "roll_"))]
            if engineered:
                batch_df[engineered] = batch_df[engineered].fillna(0)
            if "season_x_temp" in batch_df.columns:
                batch_df["season_x_temp"] = batch_df["season_x_temp"].fillna(0)

            batch_df = handle_missing_values(batch_df)
            batch_df = optimise_memory(batch_df)

            batch_df.to_parquet(out_path, index=False)
            processed_count += 1
            print(f"saved ({len(batch_df):,} rows)")

            del batch_df
            gc.collect()

        print(f"\nProcessed {processed_count} batches")

        # Clean up intermediate (raw cleaned) files
        for ipath in intermediate_files:
            ipath.unlink()
        intermediate_dir.rmdir()
        print("Raw intermediate files cleaned up.")

    # ------------------------------------------------------------------ #
    # PHASE 4: BUILD FINAL DATASET (sampled for memory safety)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PHASE 4: BUILD FINAL DATASET")
    print("=" * 60)

    processed_files = sorted(processed_dir.glob("batch_*.parquet"))

    # First pass: count total rows
    total_processed = 0
    for pf in processed_files:
        info = pd.read_parquet(pf, columns=[config.COL_HOUSEHOLD_ID])
        total_processed += len(info)
        del info
        gc.collect()

    print(f"Total processed rows: {total_processed:,}")

    # Determine sampling fraction
    if total_processed > MAX_FINAL_ROWS:
        sample_frac = MAX_FINAL_ROWS / total_processed
        print(f"Sampling {sample_frac:.1%} to get ~{MAX_FINAL_ROWS:,} rows")
    else:
        sample_frac = 1.0
        print("No sampling needed")

    # Load and sample from each batch
    all_frames = []
    for pf in processed_files:
        batch_df = pd.read_parquet(pf)
        if sample_frac < 1.0:
            # Simple random sample (fast, memory-safe)
            batch_df = batch_df.sample(frac=sample_frac, random_state=42)
        all_frames.append(batch_df)
        gc.collect()

    df = pd.concat(all_frames, ignore_index=True)
    del all_frames
    gc.collect()

    # Sort by household and time
    df = df.sort_values(
        [config.COL_HOUSEHOLD_ID, config.COL_TIMESTAMP]
    ).reset_index(drop=True)

    print(f"Final dataset: {len(df):,} rows, {df.shape[1]} columns")

    # Fit scalers on final dataset
    fit_and_save_scalers(df)

    # ------------------------------------------------------------------ #
    # PHASE 5: VALIDATION AND SAVE
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PHASE 5: VALIDATION AND SAVE")
    print("=" * 60)

    validation = validate_dataset(df)
    generate_data_quality_report(df, validation)

    print("Saving to parquet and CSV...")
    df.to_parquet(config.FINAL_DATASET_PARQUET, index=False)
    df.to_csv(config.FINAL_DATASET_CSV, index=False)

    # Print stats
    print(f"\n{'=' * 60}")
    print("PROCESSED DATASET STATISTICS")
    print(f"{'=' * 60}")
    print(f"Number of half-hourly block files detected: {len(meter_files)}")
    print(f"Number of households: {df[config.COL_HOUSEHOLD_ID].nunique()}")
    print(f"Number of rows: {len(df):,}")
    print(f"Number of columns: {df.shape[1]}")

    if config.COL_TIMESTAMP in df.columns:
        df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])
        date_min = df[config.COL_TIMESTAMP].min()
        date_max = df[config.COL_TIMESTAMP].max()
        print(f"Date range: {date_min} to {date_max}")

    missing = df.isna().sum().sum()
    print(f"Missing values: {missing}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Clean up processed batches
    for pf in processed_files:
        pf.unlink()
    processed_dir.rmdir()
    print("Processed batch files cleaned up.")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return df


def _run_full_pipeline(meter_files, household, weather_raw):
    """Phase 2: Clean raw blocks into intermediates (only if not already done)."""
    print("\n" + "=" * 60)
    print("PHASE 2: INCREMENTAL CLEANING")
    print("=" * 60)

    intermediate_dir = config.PROCESSED_DATA_DIR / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    total_raw = 0
    total_clean = 0
    batch_size = 10

    for batch_start in range(0, len(meter_files), batch_size):
        batch_files = meter_files[batch_start:batch_start + batch_size]
        batch_frames = []

        for path in batch_files:
            try:
                chunk = pd.read_csv(
                    path,
                    usecols=[config.RAW_COL_LCLID, config.RAW_COL_TSTP, config.RAW_COL_ENERGY],
                    dtype={config.RAW_COL_LCLID: "str", config.RAW_COL_ENERGY: "str"},
                )
                chunk = chunk.rename(
                    columns={
                        config.RAW_COL_LCLID: config.COL_HOUSEHOLD_ID,
                        config.RAW_COL_TSTP: config.COL_TIMESTAMP,
                        config.RAW_COL_ENERGY: config.COL_CONSUMPTION_KWH,
                    }
                )
                batch_frames.append(chunk)
                total_raw += len(chunk)
            except Exception as e:
                logger.warning("Failed to read %s: %s", path.name, e)

        if batch_frames:
            batch_df = pd.concat(batch_frames, ignore_index=True)
            del batch_frames
            batch_df = clean_meter_data(batch_df)
            batch_df = flag_outliers(batch_df)
            total_clean += len(batch_df)

            batch_id = batch_start // batch_size
            intermediate_path = intermediate_dir / f"batch_{batch_id:03d}.parquet"
            batch_df.to_parquet(intermediate_path, index=False)
            del batch_df
            gc.collect()

            print(
                f"  Batch {batch_id}: processed {batch_start + len(batch_files)}/{len(meter_files)} blocks "
                f"({total_clean:,} clean rows so far)"
            )

    print(f"\nTotal raw rows: {total_raw:,}")
    print(f"Total clean rows: {total_clean:,}")


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result is not None else 1)
