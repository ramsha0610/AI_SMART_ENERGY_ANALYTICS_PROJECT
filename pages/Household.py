"""Household Analysis page — deep-dive into a single household.

Lets an analyst search for a household and review its consumption summary,
temporal usage patterns, peak hour, and assigned cluster. Now uses real data.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from config import (
    COL_HOUSEHOLD_ID,
    COL_CONSUMPTION_KWH,
    COL_TIMESTAMP,
)
from utils.data_loader import load_processed_data, list_household_ids
from utils.helpers import format_energy, get_logger
from utils.prediction import assign_cluster
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

hero("Household Analysis", "Understand an individual household's usage profile")

# --------------------------------------------------------------------------- #
# Search / selection
# --------------------------------------------------------------------------- #
section_header("Select Household")
households = list_household_ids() or ["(no households loaded yet)"]
household_id = st.selectbox("Search household", households)

# --------------------------------------------------------------------------- #
# Summary KPIs
# --------------------------------------------------------------------------- #
section_header("Consumption Summary")

# Load household data
with st.spinner("Loading household data..."):
    df = load_processed_data()
if not df.empty:
    household_data = df[df[COL_HOUSEHOLD_ID] == household_id].copy()
    if not household_data.empty:
        # Ensure timestamp is datetime
        household_data[COL_TIMESTAMP] = pd.to_datetime(household_data[COL_TIMESTAMP])
        household_data = household_data.sort_values(COL_TIMESTAMP)

        # Total consumption
        total_consumption = household_data[COL_CONSUMPTION_KWH].sum()
        total_consumption_str = f"{total_consumption:,.1f} kWh"

        # Daily average: total consumption divided by number of days
        date_range = household_data[COL_TIMESTAMP].max() - household_data[COL_TIMESTAMP].min()
        days = max(date_range.days, 1)  # Avoid division by zero
        daily_avg = total_consumption / days
        daily_avg_str = f"{daily_avg:.1f} kWh/day"

        # Peak hour: hour of day with highest average consumption
        household_data["hour"] = household_data[COL_TIMESTAMP].dt.hour
        hourly_avg = household_data.groupby("hour")[COL_CONSUMPTION_KWH].mean()
        if not hourly_avg.empty:
            peak_hour = int(hourly_avg.idxmax())
            peak_hour_str = f"{peak_hour}:00"
        else:
            peak_hour_str = "—"

        # Cluster
        cluster = assign_cluster(household_id)
        cluster_label = "—" if cluster is None else str(cluster)
    else:
        # No data for this household
        total_consumption_str = "—"
        daily_avg_str = "—"
        peak_hour_str = "—"
        cluster_label = "—"
else:
    # No data at all
    total_consumption_str = "—"
    daily_avg_str = "—"
    peak_hour_str = "—"
    cluster_label = "—"

# Display KPIs
summary_cols = st.columns(4)
summary = [
    ("Total Consumption", total_consumption_str, "energy"),
    ("Daily Average", daily_avg_str, "calendar"),
    ("Peak Hour", peak_hour_str, "clock"),
    ("Cluster", cluster_label, "clustering"),
]
for col, (label, value, icon) in zip(summary_cols, summary):
    with col:
        kpi_card(label, value, icon=icon)

# --------------------------------------------------------------------------- #
# Usage patterns
# --------------------------------------------------------------------------- #
section_header("Usage Patterns")

if not df.empty and not household_data.empty:
    # Prepare data for plotting
    household_data = household_data.copy()  # Ensure we have a copy
    household_data["hour"] = household_data[COL_TIMESTAMP].dt.hour
    household_data["weekday"] = household_data[COL_TIMESTAMP].dt.weekday  # Monday=0, Sunday=6
    household_data["day"] = household_data[COL_TIMESTAMP].dt.day
    household_data["month"] = household_data[COL_TIMESTAMP].dt.month

    # Daily pattern: average consumption by hour of day
    hourly_profile = household_data.groupby("hour")[COL_CONSUMPTION_KWH].mean().reset_index()
    hourly_profile.columns = ["Hour", "Average Consumption (kWh)"]

    # Weekly pattern: average consumption by day of week
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly_profile = household_data.groupby("weekday")[COL_CONSUMPTION_KWH].mean().reset_index()
    weekly_profile["Weekday"] = weekly_profile["weekday"].map(lambda x: weekday_names[x])
    weekly_profile = weekly_profile[["Weekday", COL_CONSUMPTION_KWH]].rename(
        columns={COL_CONSUMPTION_KWH: "Average Consumption (kWh)"}
    )

    # Monthly pattern: average consumption by day of month
    monthly_profile = household_data.groupby("day")[COL_CONSUMPTION_KWH].mean().reset_index()
    monthly_profile.columns = ["Day of Month", "Average Consumption (kWh)"]
else:
    # Empty dataframes for plotting
    hourly_profile = pd.DataFrame({"Hour": [], "Average Consumption (kWh)": []})
    weekly_profile = pd.DataFrame({"Weekday": [], "Average Consumption (kWh)": []})
    monthly_profile = pd.DataFrame({"Day of Month": [], "Average Consumption (kWh)": []})

# Create tabs
tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])

with tab1:
    st.subheader("Average Consumption by Hour of Day")
    if not hourly_profile.empty:
        fig = px.bar(
            hourly_profile,
            x="Hour",
            y="Average Consumption (kWh)",
            title="Daily Consumption Pattern",
            labels={"Hour": "Hour of Day (0-23)", "Average Consumption (kWh)": "Average kWh per Half-Hour"},
        )
        fig.update_layout(
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for daily pattern.")

with tab2:
    st.subheader("Average Consumption by Day of Week")
    if not weekly_profile.empty:
        fig = px.bar(
            weekly_profile,
            x="Weekday",
            y="Average Consumption (kWh)",
            title="Weekly Consumption Pattern",
            labels={"Weekday": "Day of Week", "Average Consumption (kWh)": "Average kWh per Half-Hour"},
        )
        fig.update_layout(
            xaxis=dict(categoryorder="array", categoryarray=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for weekly pattern.")

with tab3:
    st.subheader("Average Consumption by Day of Month")
    if not monthly_profile.empty:
        fig = px.bar(
            monthly_profile,
            x="Day of Month",
            y="Average Consumption (kWh)",
            title="Monthly Consumption Pattern",
            labels={"Day of Month": "Day of Month", "Average Consumption (kWh)": "Average kWh per Half-Hour"},
        )
        fig.update_layout(
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for monthly pattern.")

# Note: We removed the empty_state call at the end since we now show data or informative messages