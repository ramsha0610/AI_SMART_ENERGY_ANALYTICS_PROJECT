"""General-purpose helper utilities and logging configuration.

This module intentionally contains only cross-cutting, reusable helpers so that
every other module can depend on it without creating circular imports. It hosts:

* A reusable, idempotent logging configuration (:func:`get_logger`).
* Small formatting helpers for currency, energy, and percentages.
* Lightweight file / model existence checks.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import config

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
_LOG_FILE: Path = config.LOGS_DIR / "app.log"


@lru_cache(maxsize=None)
def get_logger(name: str = config.APP_NAME) -> logging.Logger:
    """Return a configured, singleton logger for ``name``.

    The logger writes to both stdout and a log file under
    :data:`config.LOGS_DIR`. Results are cached so repeated calls with the same
    name reuse the same configured logger and never attach duplicate handlers.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ready-to-use :class:`logging.Logger`.
    """
    config.ensure_directories()

    logger = logging.getLogger(name)
    if logger.handlers:  # Already configured (defensive; lru_cache also guards).
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:  # pragma: no cover - never let logging break the app.
        logger.warning("Could not open log file at %s", _LOG_FILE)

    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_energy(kwh: float, decimals: int = 1) -> str:
    """Format a kWh value with a thousands separator and unit."""
    return f"{kwh:,.{decimals}f} kWh"


def format_currency(amount: float, symbol: str = "£", decimals: int = 2) -> str:
    """Format a monetary amount with a currency symbol."""
    return f"{symbol}{amount:,.{decimals}f}"


def format_percent(fraction: float, decimals: int = 1) -> str:
    """Format a 0-1 fraction (or 0-100 value) as a percentage string.

    Values greater than 1 are assumed to already be expressed on a 0-100 scale.
    """
    value = fraction * 100 if abs(fraction) <= 1 else fraction
    return f"{value:.{decimals}f}%"


def kwh_to_co2(kwh: float) -> float:
    """Convert energy (kWh) to estimated CO2 emissions (kg)."""
    return kwh * config.KG_CO2_PER_KWH


def kwh_to_cost(kwh: float) -> float:
    """Convert energy (kWh) to estimated cost in GBP."""
    return kwh * config.PRICE_PER_KWH_GBP


# --------------------------------------------------------------------------- #
# Filesystem / model helpers
# --------------------------------------------------------------------------- #
def file_exists(path: Path) -> bool:
    """Return ``True`` if ``path`` points to an existing file."""
    return Path(path).is_file()


def models_available() -> bool:
    """Return ``True`` if the core trained models are present on disk.

    Used by the UI to decide whether to show live results or "coming soon"
    placeholders.
    """
    # Check for regression model (keras or joblib)
    has_forecast = file_exists(config.FORECAST_MODEL_FILE) or file_exists(
        config.REGRESSION_MODEL_FILE
    )
    return all(
        [
            has_forecast,
            file_exists(config.CLUSTERING_MODEL_FILE),
            file_exists(config.ANOMALY_MODEL_FILE),
        ]
    )
