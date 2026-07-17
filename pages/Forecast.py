"""Forecast page — interface for consumption forecasting.

Provides the controls and result layout for per-household forecasting.
Now uses the trained model to generate real forecasts.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from config import COL_HOUSEHOLD_ID, COL_CONSUMPTION_KWH, COL_TIMESTAMP, DEFAULT_FORECAST_HORIZON_DAYS
from utils.data_loader import load_processed_data, list_household_ids
from utils.helpers import format_energy, get_logger
from utils.prediction import forecast_consumption
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

hero("Forecast", "Predict future household electricity consumption")

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
section_header("Forecast Settings")
households = list_household_ids() or ["(no households loaded yet)"]

c1, c2, c3 = st.columns((2, 2, 1))
with c1:
    household_id = st.selectbox("Household", households)
with c2:
    start = st.date_input("Forecast start date", value=date.today())
with c3:
    horizon = st.number_input(
        "Horizon (days)",
        min_value=1,
        max_value=30,
        value=DEFAULT_FORECAST_HORIZON_DAYS,
    )

run = st.button("Run Forecast", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
section_header("Results")
if run:
    with st.spinner("Generating forecast..."):
        result = forecast_consumption(household_id, int(horizon))
else:
    result = None

# Display KPIs
res_cols = st.columns(2)
with res_cols[0]:
    if result is None:
        kpi_card("Predicted Consumption", "—", icon="forecast")
    else:
        total_predicted = sum(result.predicted_kwh)
        kpi_card(
            "Predicted Consumption",
            f"{total_predicted:,.1f} kWh",
            icon="forecast",
        )
with res_cols[1]:
    if result is None:
        kpi_card("Confidence", "—", icon="activity")
    else:
        kpi_card(
            "Confidence",
            f"{result.confidence:.0%}",
            icon="activity",
            # Optionally add a delta if we want to show confidence change
        )

# Display forecast chart
section_header("Forecast vs. Actual")
if result is None:
    # Show placeholder if no result
    fig = go.Figure()
    fig.add_annotation(
        text="Click 'Run Forecast' to generate predictions",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        title="Forecast vs. Actual",
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
    if run:
        st.warning("Forecast model is not available. Please train the models first.")
else:
    # We have a result, let's create a forecast chart
    # We'll also try to get actual historical data for comparison

    # Load historical data for the household
    df = load_processed_data()
    if not df.empty:
        df = df[df[COL_HOUSEHOLD_ID] == household_id].copy()
        if not df.empty:
            # Ensure timestamp is datetime
            df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
            # Sort by time
            df = df.sort_values(COL_TIMESTAMP)

            # We'll show the last 7 days of actual data as context
            cutoff_date = df[COL_TIMESTAMP].max() - timedelta(days=7)
            historical = df[df[COL_TIMESTAMP] >= cutoff_date]
        else:
            historical = pd.DataFrame()
    else:
        historical = pd.DataFrame()

    # Create the figure
    fig = go.Figure()

    # Add historical data if available
    if not historical.empty:
        fig.add_trace(
            go.Scatter(
                x=historical[COL_TIMESTAMP],
                y=historical[COL_CONSUMPTION_KWH],
                mode="lines",
                name="Actual (Historical)",
                line=dict(color="#4F6BED", width=2),
            )
        )

    # Add forecast data
    if result.timestamps and result.predicted_kwh:
        fig.add_trace(
            go.Scatter(
                x=result.timestamps,
                y=result.predicted_kwh,
                mode="lines",
                name="Forecast",
                line=dict(color="#F59E0B", width=2, dash="dash"),
            )
        )

        # Add confidence interval if available
        if result.lower_kwh and result.upper_kwh:
            fig.add_trace(
                go.Scatter(
                    x=result.timestamps + result.timestamps[::-1],  # x, then x reversed
                    y=result.upper_kwh + result.lower_kwh[::-1],
                    fill="toself",
                    fillcolor="rgba(245, 158, 11, 0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="95% Confidence Interval",
                )
            )

    # Update layout
    fig.update_layout(
        title="Forecast vs. Actual Consumption",
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show a table with forecast details
    with st.expander("Forecast Details"):
        forecast_df = pd.DataFrame({
            "Timestamp": result.timestamps,
            "Predicted (kWh)": result.predicted_kwh,
            "Lower Bound (kWh)": result.lower_kwh,
            "Upper Bound (kWh)": result.upper_kwh,
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)