"""Reusable visualization and UI-component helpers.

Provides the shared visual language of the app:

* A registered Plotly template matching the light enterprise theme.
* Streamlit component helpers: KPI cards, section headers, hero banners,
  feature cards, badges, and empty-state placeholders.
* Monochrome SVG icon system for consistent, professional iconography.

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
# Monochrome SVG icon library
# --------------------------------------------------------------------------- #
_ICON_PATHS: dict[str, str] = {
    "dashboard": (
        '<rect x="3" y="3" width="7" height="7" rx="1.5"/>'
        '<rect x="14" y="3" width="7" height="7" rx="1.5"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1.5"/>'
        '<rect x="14" y="14" width="7" height="7" rx="1.5"/>'
    ),
    "forecast": (
        '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>'
        '<polyline points="16 7 22 7 22 13"/>'
    ),
    "household": (
        '<path d="M3 12l9-8 9 8"/>'
        '<path d="M5 10v9a1 1 0 001 1h3v-6h6v6h3a1 1 0 001-1v-9"/>'
    ),
    "clustering": (
        '<circle cx="7" cy="7" r="3"/>'
        '<circle cx="17" cy="7" r="3"/>'
        '<circle cx="12" cy="17" r="3"/>'
    ),
    "anomaly": (
        '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3'
        'L13.71 3.86a2 2 0 00-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>'
    ),
    "recommendation": (
        '<path d="M9 18h6"/>'
        '<path d="M10 22h4"/>'
        '<path d="M12 2a7 7 0 015 11.9V16H7v-2.1A7 7 0 0112 2z"/>'
    ),
    "energy": (
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'
    ),
    "clock": (
        '<circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/>'
    ),
    "calendar": (
        '<rect x="3" y="4" width="18" height="18" rx="2"/>'
        '<line x1="16" y1="2" x2="16" y2="6"/>'
        '<line x1="8" y1="2" x2="8" y2="6"/>'
        '<line x1="3" y1="10" x2="21" y2="10"/>'
    ),
    "activity": (
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83'
        ' 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21'
        'a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33'
        'l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65'
        ' 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65'
        ' 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65'
        ' 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0'
        ' 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06'
        '.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h'
        '-.09a1.65 1.65 0 00-1.51 1z"/>'
    ),
    "shield": (
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/>'
    ),
    "trend": (
        '<line x1="2" y1="20" x2="22" y2="4"/>'
        '<polyline points="8 4 22 4 22 18"/>'
    ),
    "savings": (
        '<line x1="12" y1="1" x2="12" y2="23"/>'
        '<path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/>'
    ),
    "leaf": (
        '<path d="M17 8C8 10 5.9 16.17 3.82 21.34"/>'
        '<path d="M17 8c3 0 5-2 5-5-3 0-5 2-5 5z"/>'
    ),
    "flag": (
        '<path d="M4 15s1-1 4-1 5 2 8 2 4-1V3c-3 1-5-2-8-2s-4 3-4 3z"/>'
        '<line x1="4" y1="22" x2="4" y2="15"/>'
    ),
    "thermometer": (
        '<path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z"/>'
    ),
    "cpu": (
        '<rect x="4" y="4" width="16" height="16" rx="2"/>'
        '<rect x="9" y="9" width="6" height="6"/>'
        '<line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>'
        '<line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>'
        '<line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>'
        '<line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
    ),
    "layers": (
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/>'
    ),
}


def svg_icon(
    name: str,
    color: str | None = None,
    size: int = 20,
    opacity: float = 1.0,
) -> str:
    """Return an inline SVG icon string for use in ``st.markdown``.

    Args:
        name: Icon name (must be a key in ``_ICON_PATHS``).
        color: Stroke colour.  Defaults to ``THEME.text_muted``.
        size: Width/height in pixels.
        opacity: Opacity value (0–1).

    Returns:
        HTML ``<svg>`` string.  Falls back to a small circle if the name
        is not recognised.
    """
    c = color or THEME.text_muted
    paths = _ICON_PATHS.get(name)
    if paths is None:
        paths = '<circle cx="12" cy="12" r="4"/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{c}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;opacity:{opacity}">'
        f'{paths}</svg>'
    )


# --------------------------------------------------------------------------- #
# Plotly theming
# --------------------------------------------------------------------------- #
def register_plotly_template() -> None:
    """Register and activate the light enterprise Plotly template.

    Idempotent — safe to call on every page load.  Defines clean white
    backgrounds, professional fonts/gridlines, and the categorical colour
    cycle so all charts share a cohesive look.
    """
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family=THEME.font_family,
                color=THEME.text,
                size=13,
            ),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            colorway=list(THEME.categorical),
            xaxis=dict(
                gridcolor="#F3F4F6",
                zerolinecolor="#E5E7EB",
                linecolor="#E5E7EB",
                showline=True,
            ),
            yaxis=dict(
                gridcolor="#F3F4F6",
                zerolinecolor="#E5E7EB",
                linecolor="#E5E7EB",
                showline=True,
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=THEME.text_muted),
            ),
            margin=dict(l=48, r=24, t=52, b=48),
            hoverlabel=dict(
                font=dict(family=THEME.font_family, color=THEME.text),
                bgcolor="#FFFFFF",
                bordercolor="#E5E7EB",
            ),
            title=dict(
                font=dict(
                    color=THEME.text,
                    size=15,
                    family=THEME.font_family,
                ),
                x=0.02,
                xanchor="left",
            ),
        )
    )
    pio.templates[config.PLOTLY_TEMPLATE] = template
    pio.templates.default = config.PLOTLY_TEMPLATE


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, [class*="css"]  {{
    font-family: {font};
}}

/* App background */
.stApp {{
    background-color: #FFFFFF;
}}

/* Main content container */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: #F9FAFB;
    border-right: 1px solid #E5E7EB;
}}
section[data-testid="stSidebar"] .stSidebarNav {{
    padding-top: 0.5rem;
}}
section[data-testid="stSidebar"] .stSidebarNavSeparator {{
    border-color: #E5E7EB;
    margin: 0.4rem 0;
}}
section[data-testid="stSidebar"] p {{
    color: #6B7280;
    font-size: 0.82rem;
}}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 {{
    color: #111827;
    font-weight: 700;
    letter-spacing: -0.025em;
}}
h2, h3 {{
    color: #111827;
    font-weight: 600;
}}
p, span, label {{
    color: #374151;
}}

/* ── Streamlit metrics ──────────────────────────────────────────────────── */
div[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 0.9rem 1rem;
}}
div[data-testid="stMetric"] label {{
    color: #6B7280;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: #111827;
    font-weight: 700;
}}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {{
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    background: #FFFFFF;
    color: #374151;
    font-weight: 500;
    padding: 0.45rem 1.2rem;
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    border-color: #4F6BED;
    color: #4F6BED;
    background: #F8FAFF;
}}
.stButton > button[kind="primary"] {{
    background: #4F6BED;
    color: #FFFFFF;
    border: 1px solid #4F6BED;
}}
.stButton > button[kind="primary"]:hover {{
    background: #3D56D4;
    border-color: #3D56D4;
}}

/* ── Containers & expanders ─────────────────────────────────────────────── */
div[data-testid="stExpander"] {{
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}}
div[data-testid="stExpander"] summary {{
    font-weight: 500;
    color: #374151;
}}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    padding: 0.5rem 1rem;
    border-radius: 8px 8px 0 0;
    color: #6B7280;
    font-weight: 500;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    background: transparent;
    color: #4F6BED;
    border-bottom: 2px solid #4F6BED;
}}

/* ── Selectbox & inputs ─────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div > div {{
    border-radius: 8px;
    border: 1px solid #E5E7EB;
}}
div[data-testid="stNumberInput"] > div > div {{
    border-radius: 8px;
}}

/* ── Dataframes ─────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {{
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Alerts / info boxes ────────────────────────────────────────────────── */
div[data-testid="stAlert"] {{
    border-radius: 10px;
    border: 1px solid #E5E7EB;
}}

/* ── Hero banner ────────────────────────────────────────────────────────── */
.se-hero {{
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 3px solid #4F6BED;
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
}}
.se-hero h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    color: #111827;
    letter-spacing: -0.025em;
}}
.se-hero p {{
    color: #6B7280;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}}

/* ── KPI cards ──────────────────────────────────────────────────────────── */
.se-kpi {{
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    height: 100%;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.se-kpi:hover {{
    border-color: #D1D5DB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.se-kpi .label {{
    color: #6B7280;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.se-kpi .value {{
    color: #111827;
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 0.3rem;
    letter-spacing: -0.02em;
}}
.se-kpi .delta {{
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 0.25rem;
}}
.se-kpi .delta.up {{ color: #059669; }}
.se-kpi .delta.down {{ color: #DC2626; }}
.se-kpi .icon {{
    display: inline-flex;
    align-items: center;
    color: #6B7280;
}}

/* ── Feature cards ──────────────────────────────────────────────────────── */
.se-feature {{
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.3rem;
    height: 100%;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
.se-feature:hover {{
    border-color: #D1D5DB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.se-feature h3 {{
    margin: 0.5rem 0 0.3rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
}}
.se-feature p {{
    color: #6B7280;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.5;
}}

/* ── Section header ─────────────────────────────────────────────────────── */
.se-section {{
    font-size: 1.15rem;
    font-weight: 600;
    color: #111827;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #F3F4F6;
}}

/* ── Badges ─────────────────────────────────────────────────────────────── */
.se-badge {{
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 500;
    border: 1px solid #E5E7EB;
    background: #F9FAFB;
    color: #6B7280;
    margin: 0.15rem;
}}

/* ── Empty state ────────────────────────────────────────────────────────── */
.se-empty {{
    border: 1px dashed #E5E7EB;
    border-radius: 10px;
    padding: 2.4rem;
    text-align: center;
    color: #6B7280;
    background: #F9FAFB;
}}
.se-empty .big {{
    font-size: 1.5rem;
    margin-bottom: 0.4rem;
}}

/* ── Utility ────────────────────────────────────────────────────────────── */
.se-divider {{
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 1.5rem 0;
}}
</style>
"""


def inject_global_css() -> None:
    """Inject the app-wide CSS for the light enterprise theme.

    Call once near the top of every page.
    """
    st.markdown(_GLOBAL_CSS.format(font=THEME.font_family), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Component helpers
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str) -> None:
    """Render the page hero banner with clean typography and accent border."""
    st.markdown(
        f"""<div class="se-hero"><h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    """Render a consistent section header with subtle underline."""
    st.markdown(f'<div class="se-section">{text}</div>', unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str,
    icon: str = "",
    delta: str | None = None,
    positive: bool = True,
    help: str | None = None,
) -> None:
    """Render a clean KPI card with monochrome icon support.

    Args:
        label: Small uppercase caption above the value.
        value: The headline metric (already formatted as a string).
        icon: SVG icon name (string matching ``_ICON_PATHS`` key) or empty.
            Emoji icons are filtered out for a professional appearance.
        delta: Optional change indicator (e.g. ``"+3.2%"``).
        positive: Colours the delta green when ``True``, red otherwise.
        help: Optional tooltip text (currently unused in HTML rendering).
    """
    delta_html = ""
    if delta is not None:
        arrow = "▲" if positive else "▼"
        cls = "up" if positive else "down"
        delta_html = f'<div class="delta {cls}">{arrow} {delta}</div>'

    # Render icon as monochrome SVG if it matches a known icon name
    icon_html = ""
    if icon and icon in _ICON_PATHS:
        icon_html = svg_icon(icon, size=16)
    elif icon and len(icon) <= 2 and not icon.isascii():
        # Skip emoji icons silently
        icon_html = ""
    elif icon:
        # Plain text icon (first letter, styled)
        icon_html = (
            f'<span style="width:24px;height:24px;border-radius:6px;'
            f'background:#F3F4F6;display:inline-flex;align-items:center;'
            f'justify-content:center;font-size:0.75rem;font-weight:600;'
            f'color:#6B7280">{icon[0].upper()}</span>'
        )

    st.markdown(
        f"""
        <div class="se-kpi">
            <div class="label"><span class="icon">{icon_html}</span> {label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, description: str) -> None:
    """Render a feature highlight card (used on the Home page).

    Args:
        icon: SVG icon name or plain text initial.
        title: Card heading.
        description: Card body text.
    """
    # Render icon as a styled SVG or initial badge
    if icon in _ICON_PATHS:
        icon_html = (
            f'<div style="width:36px;height:36px;border-radius:8px;'
            f'background:#F3F4F6;display:flex;align-items:center;'
            f'justify-content:center;margin-bottom:0.2rem">'
            f'{svg_icon(icon, color="#6B7280", size=18)}</div>'
        )
    elif len(icon) <= 2:
        # Likely an emoji — render as a styled initial instead
        initial = icon if icon else "?"
        icon_html = (
            f'<div style="width:36px;height:36px;border-radius:8px;'
            f'background:#F3F4F6;display:flex;align-items:center;'
            f'justify-content:center;font-size:0.85rem;font-weight:600;'
            f'color:#6B7280;margin-bottom:0.2rem">{initial}</div>'
        )
    else:
        icon_html = ""

    st.markdown(
        f"""
        <div class="se-feature">
            {icon_html}
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
    icon: str = "",
    hint: str = "Ensure the data pipeline has run and models are trained to see analytics here.",
) -> None:
    """Render a clean placeholder when analytics are not yet available."""
    icon_html = ""
    if icon in _ICON_PATHS:
        icon_html = (
            f'<div class="big">'
            f'{svg_icon(icon, color="#9CA3AF", size=28)}</div>'
        )
    elif icon:
        icon_html = (
            f'<div class="big" style="color:#9CA3AF">'
            f'{svg_icon("info", color="#9CA3AF", size=28)}</div>'
        )
    st.markdown(
        f"""
        <div class="se-empty">
            {icon_html}
            <div style="font-weight:600; color:#374151">{message}</div>
            <div style="margin-top:.3rem">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def placeholder_line_chart(title: str = "") -> go.Figure:
    """Return an empty, themed line-chart figure used as a placeholder."""
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
