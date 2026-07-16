"""Reusable visualization and UI-component helpers.

Unlike the ML-oriented utility modules, this file contains real (non-ML)
implementation because it provides the shared visual language of the app:

* A registered Plotly template matching the dark theme in :mod:`config`.
* Streamlit component helpers: KPI cards, section headers, and empty-state
  placeholders used across every page.

Centralising these keeps the UI consistent and DRY, and lets pages focus on
layout rather than styling details.
"""

from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import config
from utils.helpers import get_logger

logger = get_logger(__name__)

THEME = config.THEME


# --------------------------------------------------------------------------- #
# Plotly theming
# --------------------------------------------------------------------------- #
def register_plotly_template() -> None:
    """Register and activate the app's dark Plotly template.

    Idempotent — safe to call on every page load. Defines transparent
    backgrounds, themed fonts/gridlines, and the categorical colour cycle so
    all charts share a cohesive, commercial look.
    """
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family=THEME.font_family, color=THEME.text, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=list(THEME.categorical),
            xaxis=dict(gridcolor=THEME.border, zerolinecolor=THEME.border),
            yaxis=dict(gridcolor=THEME.border, zerolinecolor=THEME.border),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=20, t=50, b=40),
            hoverlabel=dict(font=dict(family=THEME.font_family)),
        )
    )
    pio.templates[config.PLOTLY_TEMPLATE] = template
    pio.templates.default = config.PLOTLY_TEMPLATE


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
def inject_global_css() -> None:
    """Inject the app-wide CSS (typography, KPI cards, spacing).

    Provides the modern, dark, commercial-dashboard look requested in the
    design brief. Call once near the top of every page.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: {THEME.font_family};
        }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; }}

        /* Hero */
        .se-hero {{
            background: linear-gradient(135deg, {THEME.surface} 0%, {THEME.surface_alt} 100%);
            border: 1px solid {THEME.border};
            border-radius: 18px;
            padding: 2.4rem 2.6rem;
            margin-bottom: 1.6rem;
        }}
        .se-hero h1 {{
            font-size: 2.5rem; font-weight: 800; margin: 0 0 .4rem 0;
            background: linear-gradient(90deg, {THEME.primary}, {THEME.secondary});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .se-hero p {{ color: {THEME.text_muted}; font-size: 1.1rem; margin: 0; }}

        /* KPI cards */
        .se-kpi {{
            background: {THEME.surface};
            border: 1px solid {THEME.border};
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            height: 100%;
            transition: transform .15s ease, border-color .15s ease;
        }}
        .se-kpi:hover {{ transform: translateY(-3px); border-color: {THEME.primary}; }}
        .se-kpi .label {{
            color: {THEME.text_muted}; font-size: .82rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: .04em;
        }}
        .se-kpi .value {{
            color: {THEME.text}; font-size: 1.9rem; font-weight: 800; margin-top: .25rem;
        }}
        .se-kpi .delta {{ font-size: .85rem; font-weight: 600; margin-top: .2rem; }}
        .se-kpi .delta.up {{ color: {THEME.success}; }}
        .se-kpi .delta.down {{ color: {THEME.danger}; }}
        .se-kpi .icon {{ font-size: 1.3rem; }}

        /* Feature cards */
        .se-feature {{
            background: {THEME.surface};
            border: 1px solid {THEME.border};
            border-radius: 16px;
            padding: 1.4rem;
            height: 100%;
        }}
        .se-feature h3 {{ margin: .3rem 0 .4rem 0; font-size: 1.1rem; }}
        .se-feature p {{ color: {THEME.text_muted}; font-size: .92rem; margin: 0; }}

        /* Section header */
        .se-section {{
            font-size: 1.35rem; font-weight: 700; margin: 1.6rem 0 .6rem 0;
            border-left: 4px solid {THEME.primary}; padding-left: .7rem;
        }}

        /* Badges */
        .se-badge {{
            display:inline-block; padding:.28rem .7rem; border-radius:999px;
            font-size:.78rem; font-weight:600; border:1px solid {THEME.border};
            background:{THEME.surface_alt}; color:{THEME.text_muted}; margin:.15rem;
        }}

        /* Empty state */
        .se-empty {{
            border: 1px dashed {THEME.border}; border-radius: 14px;
            padding: 2.4rem; text-align: center; color: {THEME.text_muted};
            background: {THEME.surface};
        }}
        .se-empty .big {{ font-size: 2rem; margin-bottom: .4rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Component helpers
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str) -> None:
    """Render the page hero banner with gradient title and subtitle."""
    st.markdown(
        f"""<div class="se-hero"><h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    """Render a consistent, accented section header."""
    st.markdown(f'<div class="se-section">{text}</div>', unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str,
    icon: str = "",
    delta: str | None = None,
    positive: bool = True,
    help: str | None = None,
) -> None:
    """Render a modern KPI card.

    Args:
        label: Small uppercase caption above the value.
        value: The headline metric (already formatted as a string).
        icon: Optional emoji/icon shown with the label.
        delta: Optional change indicator (e.g. ``"+3.2%"``).
        positive: Colours the delta green when ``True``, red otherwise.
        help: Optional tooltip text (currently unused in HTML rendering).
    """
    delta_html = ""
    if delta is not None:
        arrow = "▲" if positive else "▼"
        cls = "up" if positive else "down"
        delta_html = f'<div class="delta {cls}">{arrow} {delta}</div>'
    st.markdown(
        f"""
        <div class="se-kpi">
            <div class="label"><span class="icon">{icon}</span> {label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, description: str) -> None:
    """Render a feature highlight card (used on the Home page)."""
    st.markdown(
        f"""
        <div class="se-feature">
            <div style="font-size:1.8rem">{icon}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badges(items: Iterable[str]) -> None:
    """Render a row of pill-style badges (e.g. tech-stack chips)."""
    html = "".join(f'<span class="se-badge">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def empty_state(
    message: str = "Awaiting data & trained models",
    icon: str = "🧩",
    hint: str = "Ensure the data pipeline has run and models are trained to see analytics here.",
) -> None:
    """Render a friendly placeholder when analytics are not yet available.

    Used to communicate that the UI is ready and waiting for data or models.
    """
    st.markdown(
        f"""
        <div class="se-empty">
            <div class="big">{icon}</div>
            <div style="font-weight:600; color:{THEME.text}">{message}</div>
            <div style="margin-top:.3rem">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def placeholder_line_chart(title: str = "") -> go.Figure:
    """Return an empty, themed line-chart figure used as a placeholder.

    Renders axes and title so the layout looks intentional before real data
    is available. No synthetic/toy data is plotted.
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Consumption (kWh)",
        height=340,
    )
    fig.add_annotation(
        text="No data available",
        showarrow=False,
        font=dict(color=THEME.text_muted, size=14),
    )
    return fig
