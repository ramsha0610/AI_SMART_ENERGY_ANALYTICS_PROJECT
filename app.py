"""AI Smart Energy Analytics System — application entry point.

Run with::

    streamlit run app.py

This module is deliberately thin: it configures Streamlit, bootstraps logging
and directories, injects the global theme, and wires up multi-page navigation
via :func:`st.navigation`. Each page lives in :mod:`pages` and is fully
navigable. The rich landing page is rendered by :func:`render_home`.
"""

from __future__ import annotations

import streamlit as st

import config
from utils.helpers import get_logger
from utils.visualization import (
    badges,
    feature_card,
    hero,
    inject_global_css,
    register_plotly_template,
    section_header,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# One-time application bootstrap
# --------------------------------------------------------------------------- #
def bootstrap() -> None:
    """Perform idempotent startup tasks and log the startup event.

    Ensures required directories exist, registers the Plotly theme, and injects
    global CSS. Guarded by ``st.session_state`` so it only logs once per session.
    """
    config.ensure_directories()
    register_plotly_template()
    inject_global_css()

    if not st.session_state.get("_booted", False):
        logger.info(
            "Starting %s v%s",
            config.APP_NAME,
            config.APP_VERSION,
        )
        st.session_state["_booted"] = True


# --------------------------------------------------------------------------- #
# Home / landing page
# --------------------------------------------------------------------------- #
def render_home() -> None:
    """Render the visually rich landing page."""
    hero(
        f"{config.APP_ICON} {config.APP_NAME}",
        config.APP_TAGLINE,
    )

    st.markdown(
        "An end-to-end analytics platform for the **London Smart Meter** dataset. "
        "It forecasts electricity demand, profiles household behaviour, clusters "
        "consumers, flags anomalous usage, and turns insight into **actionable, "
        "quantified energy-saving recommendations** — all inside a modern, "
        "responsive dashboard."
    )

    # ---- Quick navigation buttons ----------------------------------------- #
    nav_cols = st.columns(4)
    with nav_cols[0]:
        st.page_link("pages/Dashboard.py", label="Open Dashboard", icon="📊")
    with nav_cols[1]:
        st.page_link("pages/Forecast.py", label="Run a Forecast", icon="📈")
    with nav_cols[2]:
        st.page_link("pages/Clustering.py", label="Explore Clusters", icon="📉")
    with nav_cols[3]:
        st.page_link("pages/Recommendations.py", label="Get Recommendations", icon="💡")

    # ---- Objectives -------------------------------------------------------- #
    section_header("🎯 Objectives")
    obj_cols = st.columns(3)
    objectives = [
        ("📈", "Forecast demand", "Predict future half-hourly consumption per household."),
        ("🏡", "Understand usage", "Profile daily, weekly, and seasonal usage patterns."),
        ("💡", "Drive savings", "Recommend actions that cut cost and carbon."),
    ]
    for col, (icon, title, desc) in zip(obj_cols, objectives):
        with col:
            feature_card(icon, title, desc)

    # ---- Feature grid ------------------------------------------------------ #
    section_header("🧭 Capabilities")
    features = [
        ("📊", "Live Dashboard", "KPIs, trends, and weather impact at a glance."),
        ("📈", "Forecasting", "Time-series prediction with confidence intervals."),
        ("📉", "Clustering", "Segment households by consumption behaviour."),
        ("⚠️", "Anomaly Detection", "Surface abnormal usage and potential faults."),
        ("💡", "AI Recommendations", "Personalised, quantified saving tips."),
        ("🏡", "Household Analysis", "Deep-dive into any individual household."),
    ]
    grid = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with grid[i % 3]:
            feature_card(icon, title, desc)
            st.write("")

    # ---- Workflow ---------------------------------------------------------- #
    section_header("🔧 Workflow")
    st.markdown(
        """
        ```text
        Raw Smart-Meter Data
                │
                ▼
        Preprocessing  ──►  Feature Engineering
                │                    │
                ▼                    ▼
          Clean Dataset       Model-ready Features
                                     │
             ┌───────────────┬───────┴───────┬────────────────┐
             ▼               ▼               ▼                ▼
        Forecasting     Clustering     Anomaly Detect.   Recommendations
             └───────────────┴───────┬───────┴────────────────┘
                                     ▼
                        Interactive Streamlit Dashboard
        ```
        """
    )

    # ---- AI models --------------------------------------------------------- #
    section_header("🤖 AI Models")
    st.markdown(
        """
        - **Forecasting** — Random Forest & XGBoost on lagged, calendar, and weather features.
        - **Clustering** — KMeans on household consumption profiles.
        - **Anomaly detection** — Isolation Forest on consumption residuals.
        - **Recommendations** — cluster-baseline comparison with quantified savings.
        """
    )

    # ---- Tech stack -------------------------------------------------------- #
    section_header("🛠️ Technology Stack")
    badges(
        [
            "Python 3.11+",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Plotly",
            "scikit-learn",
            "XGBoost",
            "Joblib",
            "Matplotlib",
        ]
    )

    st.write("")
    st.info(
        "**v1.0 Release Candidate** — All ML models are trained and operational. "
        "Navigate the pages to explore forecasts, clustering, anomaly detection, and AI recommendations."
    )


# --------------------------------------------------------------------------- #
# App configuration & navigation
# --------------------------------------------------------------------------- #
def main() -> None:
    """Configure the page and launch multi-page navigation."""
    st.set_page_config(
        page_title=config.APP_NAME,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    bootstrap()

    home_page = st.Page(
        render_home,
        title=config.PAGES["home"]["title"],
        icon=config.PAGES["home"]["icon"],
        default=True,
    )
    dashboard = st.Page(
        "pages/Dashboard.py",
        title=config.PAGES["dashboard"]["title"],
        icon=config.PAGES["dashboard"]["icon"],
    )
    forecast = st.Page(
        "pages/Forecast.py",
        title=config.PAGES["forecast"]["title"],
        icon=config.PAGES["forecast"]["icon"],
    )
    household = st.Page(
        "pages/Household.py",
        title=config.PAGES["household"]["title"],
        icon=config.PAGES["household"]["icon"],
    )
    clustering = st.Page(
        "pages/Clustering.py",
        title=config.PAGES["clustering"]["title"],
        icon=config.PAGES["clustering"]["icon"],
    )
    anomaly = st.Page(
        "pages/Anomaly.py",
        title=config.PAGES["anomaly"]["title"],
        icon=config.PAGES["anomaly"]["icon"],
    )
    recommendations = st.Page(
        "pages/Recommendations.py",
        title=config.PAGES["recommendations"]["title"],
        icon=config.PAGES["recommendations"]["icon"],
    )
    about = st.Page(
        "pages/About.py",
        title=config.PAGES["about"]["title"],
        icon=config.PAGES["about"]["icon"],
    )

    nav = st.navigation(
        {
            "Overview": [home_page, dashboard],
            "Analytics": [forecast, household, clustering, anomaly],
            "Insights": [recommendations, about],
        }
    )

    with st.sidebar:
        st.caption(f"{config.APP_ICON} {config.APP_NAME}")
        st.caption(f"v{config.APP_VERSION} · Production")

    nav.run()


if __name__ == "__main__":
    main()
