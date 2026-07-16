"""Clustering page — segment households by consumption behaviour.

Displays the clustering scatter plot, per-cluster statistics, and the
distribution of households across clusters. Now uses the trained clustering
model to show actual results.
"""

from __future__ import annotations

import joblib
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import streamlit as st

from config import (
    COL_HOUSEHOLD_ID,
    COL_CONSUMPTION_KWH,
    COL_TIMESTAMP,
    MODELS_DIR,
)
from utils.data_loader import load_processed_data
from utils.helpers import get_logger
from utils.prediction import assign_cluster, load_clustering_model
from utils.visualization import (
    hero,
    inject_global_css,
    kpi_card,
    register_plotly_template,
    section_header,
)

logger = get_logger(__name__)

register_plotly_template()
inject_global_css()

hero("📉 Clustering", "Group households into meaningful consumption segments")

# --------------------------------------------------------------------------- #
# Load data and compute household features
# --------------------------------------------------------------------------- #
def get_household_features() -> pd.DataFrame:
    """Compute features for each household used in clustering."""
    df = load_processed_data()
    if df.empty:
        return pd.DataFrame()

    # Ensure timestamp is datetime
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
    df = df.sort_values([COL_HOUSEHOLD_ID, COL_TIMESTAMP])

    # Group by household and compute features
    household_features = []
    for hhid, group in df.groupby(COL_HOUSEHOLD_ID):
        # Basic consumption stats
        avg_consumption = group[COL_CONSUMPTION_KWH].mean()
        peak_consumption = group[COL_CONSUMPTION_KWH].max()
        # Consumption variability
        std_consumption = group[COL_CONSUMPTION_KWH].std()
        # Daily patterns: average by hour of day
        hourly_avg = group.groupby(group[COL_TIMESTAMP].dt.hour)[COL_CONSUMPTION_KWH].mean()
        # Energy during peak hours (18-22)
        evening_mask = (group[COL_TIMESTAMP].dt.hour >= 18) & (group[COL_TIMESTAMP].dt.hour < 22)
        evening_avg = group.loc[evening_mask, COL_CONSUMPTION_KWH].mean() if evening_mask.any() else 0
        # Weekend vs weekday
        weekend_mask = group[COL_TIMESTAMP].dt.weekday >= 5
        weekday_avg = group.loc[~weekend_mask, COL_CONSUMPTION_KWH].mean()
        weekend_avg = group.loc[weekend_mask, COL_CONSUMPTION_KWH].mean()
        # Total consumption
        total_consumption = group[COL_CONSUMPTION_KWH].sum()

        household_features.append(
            {
                "household_id": hhid,
                "avg_consumption": avg_consumption,
                "peak_consumption": peak_consumption,
                "std_consumption": std_consumption if not pd.isna(std_consumption) else 0,
                "evening_avg": evening_avg,
                "weekday_avg": weekday_avg if not pd.isna(weekday_avg) else 0,
                "weekend_avg": weekend_avg if not pd.isna(weekend_avg) else 0,
                "total_consumption": total_consumption,
            }
        )

    features_df = pd.DataFrame(household_features)
    return features_df


# Load data
with st.spinner("Computing household features..."):
    features_df = get_household_features()
if features_df.empty:
    st.warning("No household data available for clustering.")
    st.stop()

# Load the clustering model and household cluster map
model_path = MODELS_DIR / "clustering_model.joblib"
map_path = MODELS_DIR / "household_cluster_map.joblib"

cluster_labels = None
if map_path.exists():
    try:
        cluster_map = joblib.load(map_path)
        # Merge with features_df
        features_df = features_df.merge(cluster_map, on="household_id", how="left")
        cluster_labels = features_df["cluster"].values
    except Exception as e:
        logger.warning("Could not load cluster map: %s", e)
        cluster_labels = None

# If we don't have a saved map, try to compute clusters on the fly using the model
if cluster_labels is None and model_path.exists():
    try:
        model = joblib.load(model_path)
        # Prepare features for clustering (same as used in training)
        feature_cols = [c for c in features_df.columns if c != "household_id"]
        X = features_df[feature_cols].fillna(0)
        # Scale features (as done in training)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        cluster_labels = model.predict(X_scaled)
        features_df["cluster"] = cluster_labels
    except Exception as e:
        logger.warning("Could not compute clusters with model: %s", e)
        cluster_labels = None

# If we still don't have clusters, assign each household to its own cluster (for demo)
if cluster_labels is None:
    # In a real scenario, we would train the model, but for now, we'll assign each household to cluster 0
    # This is just to have something to show
    features_df["cluster"] = 0
    cluster_labels = np.zeros(len(features_df))
    st.warning("Clustering model not available. Showing all households in a single cluster.")

# --------------------------------------------------------------------------- #
# Controls / KPIs
# --------------------------------------------------------------------------- #
section_header("Overview")
kpi_cols = st.columns(3)

# Number of clusters
n_clusters = len(np.unique(cluster_labels)) if cluster_labels is not None else 0
kpi_cols[0].metric("Number of Clusters", f"{n_clusters}")

# Households clustered
n_households = len(features_df)
kpi_cols[1].metric("Households Clustered", f"{n_households}")

# Silhouette score (if we have at least 2 clusters and more than 1 sample per cluster)
if n_clusters >= 2 and n_households > n_clusters:
    try:
        feature_cols = [c for c in features_df.columns if c not in ["household_id", "cluster"]]
        X = features_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        sil_score = silhouette_score(X_scaled, cluster_labels)
        kpi_cols[2].metric("Silhouette Score", f"{sil_score:.3f}")
    except Exception as e:
        logger.warning("Could not compute silhouette score: %s", e)
        kpi_cols[2].metric("Silhouette Score", "—")
else:
    kpi_cols[2].metric("Silhouette Score", "—")

# --------------------------------------------------------------------------- #
# Scatter + distribution
# --------------------------------------------------------------------------- #
left, right = st.columns((2, 1))

# Scatter plot: average vs peak consumption, colored by cluster
with left:
    section_header("Cluster Scatter (Avg Consumption vs Peak Consumption)")
    if not features_df.empty:
        fig = px.scatter(
            features_df,
            x="avg_consumption",
            y="peak_consumption",
            color="cluster" if "cluster" in features_df.columns else None,
            hover_name="household_id",
            size="total_consumption",
            color_continuous_scale=px.colors.qualitative.Set1,
            labels={
                "avg_consumption": "Average Consumption (kWh)",
                "peak_consumption": "Peak Consumption (kWh)",
                "cluster": "Cluster",
                "total_consumption": "Total Consumption (kWh)",
            },
            title="Households by Average and Peak Consumption",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for scatter plot.")

# Household distribution by cluster
with right:
    section_header("Household Distribution")
    if "cluster" in features_df.columns and not features_df.empty:
        cluster_counts = features_df["cluster"].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={"x": "Cluster", "y": "Number of Households"},
            title="Number of Households per Cluster",
            text=cluster_counts.values,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cluster data available for distribution.")

# --------------------------------------------------------------------------- #
# Cluster Statistics
# --------------------------------------------------------------------------- #
section_header("Cluster Statistics")
if "cluster" in features_df.columns and not features_df.empty:
    # Compute statistics per cluster
    stats_list = []
    for cluster_id in sorted(features_df["cluster"].unique()):
        cluster_data = features_df[features_df["cluster"] == cluster_id]
        stats = {
            "Cluster": cluster_id,
            "Households": len(cluster_data),
            "Avg Consumption (kWh)": cluster_data["avg_consumption"].mean(),
            "Peak Consumption (kWh)": cluster_data["peak_consumption"].mean(),
            "Total Consumption (kWh)": cluster_data["total_consumption"].sum(),
            "Evening Avg (kWh)": (
                cluster_data["evening_avg"].mean()
                if "evening_avg" in cluster_data.columns
                else 0
            ),
        }
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    # Format the dataframe for display
    display_df = stats_df.copy()
    for col in display_df.columns:
        if col not in ["Cluster", "Households"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No cluster statistics available.")