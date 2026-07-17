"""Dashboard page — high-level operational overview.

Presents the top-level KPIs and trend visualisations for the whole fleet of
households. Now displays real data from the processed dataset and trained
models.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from config import COL_HOUSEHOLD_ID, COL_CONSUMPTION_KWH, COL_TIMESTAMP
from utils.data_loader import load_processed_data
from utils.helpers import get_logger
from utils.prediction import load_model
from utils.visualization import (
    hero,
    inject_global_css,
    kpi_card,
    register_plotly_template,
    section_header,
)

import config

try:
    import statsmodels  # noqa: F401
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

logger = get_logger(__name__)

register_plotly_template()
inject_global_css()

hero("Dashboard", "Fleet-wide energy KPIs, trends, and weather impact")

# Load processed data
with st.spinner("Loading dataset..."):
    df = load_processed_data()
if df.empty:
    st.warning("No processed data available. Please run the data pipeline.")
    st.stop()

# Ensure timestamp is datetime
df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
df = df.sort_values([COL_HOUSEHOLD_ID, COL_TIMESTAMP])

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
section_header("Key Performance Indicators")
kpi_cols = st.columns(4)

# Total Households
total_households = df[COL_HOUSEHOLD_ID].nunique()
kpi_cols[0].metric("Total Households", f"{total_households}", help="Number of unique households in the dataset")

# Avg. Daily Consumption
# Calculate daily total per household, then average across all households and days
df["date"] = df[COL_TIMESTAMP].dt.date
daily_totals = df.groupby([COL_HOUSEHOLD_ID, "date"])[COL_CONSUMPTION_KWH].sum().reset_index()
avg_daily_consumption = daily_totals[COL_CONSUMPTION_KWH].mean()
kpi_cols[1].metric(
    "Avg. Daily Consumption",
    f"{avg_daily_consumption:.2f} kWh",
    help="Average daily total consumption per household",
)

# Peak Usage Hour
# Average consumption per hour of day across all records
df["hour"] = df[COL_TIMESTAMP].dt.hour
hourly_avg = df.groupby("hour")[COL_CONSUMPTION_KWH].mean()
peak_hour = int(hourly_avg.idxmax()) if not hourly_avg.empty else 0
peak_value = float(hourly_avg.max()) if not hourly_avg.empty else 0
kpi_cols[2].metric(
    "Peak Usage Hour",
    f"{peak_hour}:00",
    help=f"Hour of day with highest average consumption ({peak_value:.2f} kWh)",
)

# Forecast Accuracy
# Try to load model results to get R² of the best model
try:
    model_results = pd.read_csv(config.MODEL_RESULTS_FILE)
    if not model_results.empty and "is_best" in model_results.columns:
        best_model_row = model_results[model_results["is_best"].fillna(False).astype(bool)]
        if not best_model_row.empty:
            r2_score = float(best_model_row["r2"].iloc[0])
            kpi_cols[3].metric(
                "Forecast Accuracy (R²)",
                f"{r2_score:.2f}",
                help="Coefficient of determination of the best forecasting model",
            )
        else:
            kpi_cols[3].metric("Forecast Accuracy (R²)", "—", help="Model results not available")
    else:
        kpi_cols[3].metric("Forecast Accuracy (R²)", "—", help="Model results not available")
except Exception as e:
    logger.warning("Could not load model results: %s", e)
    kpi_cols[3].metric("Forecast Accuracy (R²)", "—", help="Model results not available")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
left, right = st.columns((2, 1))

# Left chart: Consumption Trend
with left:
    section_header("Consumption Trend")
    # Prepare daily average consumption over time
    daily_avg = df.groupby("date")[COL_CONSUMPTION_KWH].mean().reset_index()
    daily_avg.columns = ["Date", "Average Consumption (kWh)"]
    fig = px.line(
        daily_avg,
        x="Date",
        y="Average Consumption (kWh)",
        title="Average Daily Consumption Trend",
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# Right chart: Energy Distribution (by household average consumption)
with right:
    section_header("Energy Distribution")
    # Calculate average consumption per household; show top 10 + "Other" for readability
    household_avg = df.groupby(COL_HOUSEHOLD_ID)[COL_CONSUMPTION_KWH].mean().reset_index()
    household_avg.columns = ["Household", "Average Consumption (kWh)"]
    household_avg = household_avg.sort_values("Average Consumption (kWh)", ascending=False)
    top_n = 10
    if len(household_avg) > top_n:
        top = household_avg.head(top_n)
        other_sum = household_avg.iloc[top_n:]["Average Consumption (kWh)"].sum()
        other_row = pd.DataFrame({"Household": ["Other"], "Average Consumption (kWh)": [other_sum]})
        pie_df = pd.concat([top, other_row], ignore_index=True)
    else:
        pie_df = household_avg
    fig = px.pie(
        pie_df,
        values="Average Consumption (kWh)",
        names="Household",
        title="Average Consumption Distribution (Top 10 + Other)",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

# Lower row: Weather Impact and Recent Activity
lower_left, lower_right = st.columns(2)

# Weather Impact
with lower_left:
    section_header("Weather Impact")
    # Sample a subset of points for clarity if dataset is large
    sample_df = df.sample(n=min(500, len(df)), random_state=42) if len(df) > 500 else df
    fig = px.scatter(
        sample_df,
        x="temperature_c",
        y=COL_CONSUMPTION_KWH,
        opacity=0.6,
        title="Consumption vs. Temperature",
        labels={"temperature_c": "Temperature (°C)", COL_CONSUMPTION_KWH: "Consumption (kWh)"},
        trendline="ols" if _HAS_STATSMODELS else None,
    )
    st.plotly_chart(fig, use_container_width=True)

# Recent Activity
with lower_right:
    section_header("Recent Activity")
    # Show the latest 5 readings per household
    latest_per_household = (
        df.sort_values(COL_TIMESTAMP)
        .groupby(COL_HOUSEHOLD_ID, observed=True)
        .tail(5)
    )
    # Format for display
    recent_cols = [COL_HOUSEHOLD_ID, COL_TIMESTAMP, COL_CONSUMPTION_KWH]
    if "temperature_c" in latest_per_household.columns:
        recent_cols.append("temperature_c")
    display_df = latest_per_household[recent_cols].copy()
    display_df[COL_TIMESTAMP] = display_df[COL_TIMESTAMP].dt.strftime("%Y-%m-%d %H:%M")
    rename_map = {
        COL_HOUSEHOLD_ID: "Household ID",
        COL_TIMESTAMP: "Timestamp",
        COL_CONSUMPTION_KWH: "Consumption (kWh)",
    }
    if "temperature_c" in display_df.columns:
        rename_map["temperature_c"] = "Temperature (°C)"
    display_df = display_df.rename(columns=rename_map)
    st.dataframe(
        display_df.head(20),  # Limit to 20 rows for display
        use_container_width=True,
        hide_index=True,
    )