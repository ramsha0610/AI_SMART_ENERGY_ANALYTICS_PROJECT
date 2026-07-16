"""Anomaly Detection page — surface abnormal energy usage.

Shows summary risk cards, a timeline of flagged anomalies, and an alert table.
Now uses the trained anomaly detection model to show actual results.
"""

from __future__ import annotations

from datetime import datetime

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
from utils.prediction import detect_anomalies
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

hero("⚠️ Anomaly Detection", "Detect abnormal consumption and potential faults")

# --------------------------------------------------------------------------- #
# Selection + summary
# --------------------------------------------------------------------------- #
households = list_household_ids() or ["(all households)"]
household_id = st.selectbox("Household", households)

# Load household data
df = load_processed_data()
if df.empty:
    st.warning("No data available for anomaly detection.")
    st.stop()

# Ensure timestamp is datetime
df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
df = df.sort_values([COL_HOUSEHOLD_ID, COL_TIMESTAMP])

# Get data for the selected household
if household_id == "(all households)":
    household_data = df.copy()
else:
    household_data = df[df[COL_HOUSEHOLD_ID] == household_id].copy()

if household_data.empty:
    st.warning(f"No data available for household {household_id}.")
    st.stop()

# Run anomaly detection
with st.spinner("Running anomaly detection..."):
    alerts = detect_anomalies(household_id)
if alerts is None:
    alerts = pd.DataFrame(columns=["timestamp", COL_CONSUMPTION_KWH, "score", "severity"])

# Ensure alerts have the expected columns
if not isinstance(alerts, pd.DataFrame):
    alerts = pd.DataFrame(columns=["timestamp", COL_CONSUMPTION_KWH, "score", "severity"])

# Compute summary metrics
anomaly_count = len(alerts) if not alerts.empty else 0
max_severity = (
    alerts["severity"].max()
    if not alerts.empty and "severity" in alerts.columns
    else 0
)
# Determine risk level based on anomaly count and severity
if anomaly_count == 0:
    risk_level = "Low"
elif anomaly_count < 5 or max_severity < 0.3:
    risk_level = "Low"
elif anomaly_count < 20 or max_severity < 0.7:
    risk_level = "Medium"
else:
    risk_level = "High"

# Last checked: we can use the current time or the latest timestamp in the data
last_checked = datetime.now().strftime("%Y-%m-%d %H:%M")
# Alternatively, we could use the latest timestamp in the data:
# last_checked = household_data[COL_TIMESTAMP].max().strftime("%Y-%m-%d %H:%M")

# --------------------------------------------------------------------------- #
# Risk Summary
# --------------------------------------------------------------------------- #
section_header("Risk Summary")
kpi_cols = st.columns(4)
kpis = [
    ("Anomalies Detected", f"{anomaly_count}", "🚩"),
    ("Highest Severity", f"{max_severity:.2f}", "🔴"),
    ("Risk Level", risk_level, "🌡️"),
    ("Last Checked", last_checked, "🕒"),
]
for col, (label, value, icon) in zip(kpi_cols, kpis):
    with col:
        kpi_card(label, value, icon=icon)

# --------------------------------------------------------------------------- #
# Timeline + alerts
# --------------------------------------------------------------------------- #
section_header("Anomaly Timeline")
if not household_data.empty:
    # Create a time series of consumption
    fig = go.Figure()

    # Add the consumption time series
    fig.add_trace(
        go.Scatter(
            x=household_data[COL_TIMESTAMP],
            y=household_data[COL_CONSUMPTION_KWH],
            mode="lines",
            name="Consumption",
            line=dict(color="#6366F1", width=1),
        )
    )

    # Highlight anomalies if any
    if not alerts.empty:
        # Ensure alerts have timestamp and consumption
        if COL_TIMESTAMP in alerts.columns and COL_CONSUMPTION_KWH in alerts.columns:
            fig.add_trace(
                go.Scatter(
                    x=alerts[COL_TIMESTAMP],
                    y=alerts[COL_CONSUMPTION_KWH],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(
                        color="#EF4444",
                        size=8,
                        symbol="x",
                        line=dict(width=1, color="#FFFFFF"),
                    ),
                )
            )

    fig.update_layout(
        title="Consumption Over Time with Anomalies Highlighted",
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="smart_energy_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available to display timeline.")

# --------------------------------------------------------------------------- #
# Alerts table
# --------------------------------------------------------------------------- #
section_header("Alerts")
if not alerts.empty:
    # Format the alert table for display
    display_alerts = alerts.copy()
    if COL_TIMESTAMP in display_alerts.columns:
        display_alerts[COL_TIMESTAMP] = pd.to_datetime(
            display_alerts[COL_TIMESTAMP]
        ).dt.strftime("%Y-%m-%d %H:%M")
    if COL_CONSUMPTION_KWH in display_alerts.columns:
        display_alerts[COL_CONSUMPTION_KWH] = display_alerts[COL_CONSUMPTION_KWH].apply(
            lambda x: f"{x:.3f} kWh"
        )
    if "score" in display_alerts.columns:
        display_alerts["score"] = display_alerts["score"].apply(
            lambda x: f"{x:.3f}"
        )
    if "severity" in display_alerts.columns:
        display_alerts["severity"] = display_alerts["severity"].apply(
            lambda x: f"{x:.2f}"
        )

    # Rename columns for display
    display_alerts = display_alerts.rename(
        columns={
            COL_TIMESTAMP: "Timestamp",
            COL_CONSUMPTION_KWH: "Consumption",
            "score": "Anomaly Score",
            "severity": "Severity",
        }
    )
    # Reorder columns
    cols = [c for c in ["Timestamp", "Consumption", "Anomaly Score", "Severity"] if c in display_alerts.columns]
    display_alerts = display_alerts[cols]

    st.dataframe(
        display_alerts,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No anomalies detected for the selected household.")