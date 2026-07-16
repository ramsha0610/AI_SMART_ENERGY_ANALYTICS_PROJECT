#!/usr/bin/env python3
"""Train all models on the rebuilt processed dataset.

Uses sampling for memory-safe training on the 3M-row dataset.
Prints validation stats and generates reports.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib

import config
from utils.prediction import (
    train_classical_models,
    train_clustering_model,
    train_anomaly_model,
)
from utils.helpers import get_logger

logger = get_logger(__name__)

# Training sample size (fast enough for RF/XGBoost while maintaining quality)
TRAIN_SAMPLE_SIZE = 500_000


def generate_feature_importance(model_path: Path, feature_names_path: Path) -> None:
    """Extract and save feature importance from the best regression model."""
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        logger.warning("Model has no feature_importances_ attribute")
        return

    fi_df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    fi_path = config.REPORTS_DIR / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    logger.info("Saved feature importance to %s", fi_path)
    print(f"\nFeature Importance (top 10):")
    for _, row in fi_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


def generate_model_comparison(results: dict) -> None:
    """Generate model comparison markdown report."""
    report_path = config.REPORTS_DIR / "model_comparison.md"

    lines = [
        "# Model Comparison Report",
        "",
        "## Regression Models",
        "",
        "| Model | MAE | RMSE | R² | Best |",
        "|-------|-----|------|----|------|",
    ]

    best_name = max(results, key=lambda k: results[k]["r2"])
    for name, metrics in results.items():
        is_best = "Yes" if name == best_name else "No"
        lines.append(
            f"| {name} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} "
            f"| {metrics['r2']:.6f} | {is_best} |"
        )

    lines.extend([
        "",
        f"## Best Model: {best_name}",
        "",
        f"- **MAE**: {results[best_name]['mae']:.6f}",
        f"- **RMSE**: {results[best_name]['rmse']:.6f}",
        f"- **R²**: {results[best_name]['r2']:.6f}",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved model comparison to %s", report_path)


def main():
    t0 = time.time()

    print("=" * 60)
    print("MODEL TRAINING ON REBUILT DATASET")
    print("=" * 60)

    # Show dataset info
    print("\nLoading processed dataset...")
    frame = pd.read_parquet(config.FINAL_DATASET_PARQUET)
    n_households = frame[config.COL_HOUSEHOLD_ID].nunique()
    n_rows = len(frame)
    print(f"Number of households: {n_households}")
    print(f"Number of rows: {n_rows:,}")

    if config.COL_TIMESTAMP in frame.columns:
        ts = pd.to_datetime(frame[config.COL_TIMESTAMP])
        print(f"Date range: {ts.min()} to {ts.max()}")
    del frame  # Free memory

    # --------------------------------------------------------------- #
    # 1. REGRESSION MODELS (Random Forest + XGBoost)
    # --------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("TRAINING REGRESSION MODELS")
    print("=" * 60)

    t1 = time.time()
    results = train_classical_models(max_train_rows=TRAIN_SAMPLE_SIZE)
    reg_time = time.time() - t1

    print(f"\n--- Regression Results ---")
    for name, metrics in results.items():
        print(
            f"{name}: MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.6f}"
        )

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_metrics = results[best_name]
    print(f"\nSelected best regression model: {best_name}")
    print(f"  MAE:  {best_metrics['mae']:.6f}")
    print(f"  RMSE: {best_metrics['rmse']:.6f}")
    print(f"  R²:   {best_metrics['r2']:.6f}")
    print(f"  Training time: {reg_time:.1f}s")

    # Generate reports
    generate_model_comparison(results)
    generate_feature_importance(
        config.REGRESSION_MODEL_FILE,
        config.REGRESSION_FEATURE_NAMES_FILE,
    )

    # --------------------------------------------------------------- #
    # 2. CLUSTERING MODEL (KMeans)
    # --------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("TRAINING CLUSTERING MODEL")
    print("=" * 60)

    t2 = time.time()
    train_clustering_model()
    cluster_time = time.time() - t2

    cluster_map = joblib.load(config.MODELS_DIR / "household_cluster_map.joblib")
    print(f"KMeans clusters: {config.DEFAULT_N_CLUSTERS}")
    print(f"Households clustered: {len(cluster_map)}")
    print(f"Training time: {cluster_time:.1f}s")

    # --------------------------------------------------------------- #
    # 3. ANOMALY MODEL (Isolation Forest)
    # --------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("TRAINING ANOMALY MODEL")
    print("=" * 60)

    t3 = time.time()
    train_anomaly_model(max_train_rows=TRAIN_SAMPLE_SIZE)
    anomaly_time = time.time() - t3

    print(f"Isolation Forest trained successfully")
    print(f"Training time: {anomaly_time:.1f}s")

    # --------------------------------------------------------------- #
    # SUMMARY
    # --------------------------------------------------------------- #
    total_time = time.time() - t0

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Number of households used: {n_households}")
    print(f"Number of rows in dataset: {n_rows:,}")
    print(f"Training sample size: {min(TRAIN_SAMPLE_SIZE, n_rows):,}")
    print(f"Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"\nBest regression model: {best_name}")
    print(f"  MAE:  {best_metrics['mae']:.6f}")
    print(f"  RMSE: {best_metrics['rmse']:.6f}")
    print(f"  R²:   {best_metrics['r2']:.6f}")
    print(f"\nModel files updated:")
    for f in [
        config.REGRESSION_MODEL_FILE,
        config.MODELS_DIR / "random_forest_regressor.joblib",
        config.MODELS_DIR / "xgboost_regressor.joblib",
        config.CLUSTERING_MODEL_FILE,
        config.ANOMALY_MODEL_FILE,
    ]:
        exists = "OK" if f.exists() else "MISSING"
        print(f"  [{exists}] {f.name}")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        print(f"\nERROR: {e}")
        sys.exit(1)
