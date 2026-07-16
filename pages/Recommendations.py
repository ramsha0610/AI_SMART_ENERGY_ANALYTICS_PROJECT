"""AI Recommendations page — turn insight into action.

Shows quantified, ranked energy-saving recommendations plus headline savings
(energy, cost, carbon). Now uses the implemented recommendation engine.
"""

from __future__ import annotations

import streamlit as st

from utils.data_loader import list_household_ids
from utils.helpers import format_energy, format_currency, get_logger
from utils.recommendation import generate_recommendations, summarise_savings
from utils.visualization import (
    hero,
    inject_global_css,
    kpi_card,
    register_plotly_template,
    section_header,
    empty_state,
)

logger = get_logger(__name__)

register_plotly_template()
inject_global_css()

hero("💡 AI Recommendations", "Personalised, quantified energy-saving actions")

# --------------------------------------------------------------------------- #
# Selection + savings summary
# --------------------------------------------------------------------------- #
households = list_household_ids() or ["(no households loaded yet)"]
household_id = st.selectbox("Household", households)

# Generate recommendations
try:
    with st.spinner("Generating recommendations..."):
        recommendations = generate_recommendations(household_id)
except Exception as e:
    logger.error("Error generating recommendations for %s: %s", household_id, e)
    recommendations = []

# Compute savings summary
try:
    savings = summarise_savings(recommendations)
except Exception as e:
    logger.error("Error computing savings: %s", e)
    savings = {"total_kwh": 0.0, "total_cost_gbp": 0.0, "total_co2_kg": 0.0}

# --------------------------------------------------------------------------- #
# Potential Savings
# --------------------------------------------------------------------------- #
section_header("Potential Savings")
kpi_cols = st.columns(3)
with kpi_cols[0]:
    kpi_card(
        "Energy Saved",
        format_energy(savings["total_kwh"]),
        icon="⚡",
        help="Total estimated annual energy savings from all recommendations",
    )
with kpi_cols[1]:
    kpi_card(
        "Cost Saved",
        format_currency(savings["total_cost_gbp"]),
        icon="💷",
        help="Total estimated annual cost savings in GBP",
    )
with kpi_cols[2]:
    kpi_card(
        "CO₂ Reduced",
        f"{savings['total_co2_kg']:,.1f} kg",
        icon="🌱",
        help="Total estimated annual CO2 reduction",
    )

# --------------------------------------------------------------------------- #
# Recommendation cards
# --------------------------------------------------------------------------- #
section_header("Recommendations")
if not recommendations:
    # Check if we have data and models to give a more specific hint
    from utils.data_loader import load_processed_data
    from utils.prediction import load_model

    data_available = not load_processed_data().empty
    model_available = load_model() is not None

    if not data_available:
        hint = "No household data available. Please ensure the data pipeline has been run."
    elif not model_available:
        hint = "Recommendation engine requires trained models. Please train the models first."
    else:
        hint = "No recommendations generated for this household. Try a different household or check the data."

    empty_state(
        "No recommendations available",
        icon="💡",
        hint=hint,
    )
else:
    # Sort recommendations by priority (high > medium > low) for display
    priority_order = {"high": 3, "medium": 2, "low": 1}
    sorted_recommendations = sorted(
        recommendations,
        key=lambda r: priority_order.get(r.priority, 0),
        reverse=True,
    )

    for rec in sorted_recommendations:
        with st.container(border=True):
            st.subheader(rec.title)
            st.write(rec.detail)
            # Format the metrics
            energy_saved = format_energy(rec.estimated_kwh_saving)
            cost_saved = format_currency(rec.estimated_cost_saving)
            co2_saved = f"{rec.estimated_co2_reduction:,.1f} kg"
            st.caption(
                f"~{energy_saved} · {cost_saved} · {co2_saved} · priority: {rec.priority}"
            )