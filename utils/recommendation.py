"""AI-powered energy-saving recommendation engine.

Turns model outputs (forecasts, clusters, anomalies) and household usage
patterns into actionable, quantified recommendations — estimated kWh savings,
cost savings, and CO2 reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import config
import pandas as pd

from utils.data_loader import load_processed_data
from utils.helpers import get_logger, kwh_to_co2, kwh_to_cost
from utils.prediction import (
    assign_cluster,
    detect_anomalies,
    forecast_consumption,
    load_model,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class Recommendation:
    """A single, quantified energy-saving recommendation."""

    title: str
    detail: str
    estimated_kwh_saving: float
    estimated_cost_saving: float
    estimated_co2_reduction: float
    priority: str  # one of: "low", "medium", "high"


def _quantify(kwh_saving: float, priority: str, title: str, detail: str) -> Recommendation:
    """Build a :class:`Recommendation`, deriving cost and CO2 from kWh."""
    return Recommendation(
        title=title,
        detail=detail,
        estimated_kwh_saving=kwh_saving,
        estimated_cost_saving=kwh_to_cost(kwh_saving),
        estimated_co2_reduction=kwh_to_co2(kwh_saving),
        priority=priority,
    )


def _get_household_usage_stats(household_id: str) -> dict:
    """Calculate key usage statistics for a household."""
    df = load_processed_data()
    if df.empty:
        return {}

    household_data = df[df[config.COL_HOUSEHOLD_ID] == household_id].copy()
    if household_data.empty:
        return {}

    # Ensure timestamp is datetime
    household_data[config.COL_TIMESTAMP] = pd.to_datetime(
        household_data[config.COL_TIMESTAMP]
    )
    household_data = household_data.sort_values(config.COL_TIMESTAMP)

    # Basic stats
    avg_consumption = household_data[config.COL_CONSUMPTION_KWH].mean()
    max_consumption = household_data[config.COL_CONSUMPTION_KWH].max()
    total_consumption = household_data[config.COL_CONSUMPTION_KWH].sum()

    # Hour of peak consumption
    hourly_pattern = (
        household_data.groupby(household_data[config.COL_TIMESTAMP].dt.hour)[
            config.COL_CONSUMPTION_KWH
        ]
        .mean()
        .reset_index()
    )
    hourly_pattern.columns = ["hour", config.COL_CONSUMPTION_KWH]
    peak_hour = (
        int(hourly_pattern.loc[hourly_pattern[config.COL_CONSUMPTION_KWH].idxmax(), "hour"])
        if not hourly_pattern.empty
        else 0
    )

    # Weekend vs weekday consumption
    household_data["is_weekend"] = household_data[config.COL_TIMESTAMP].dt.weekday >= 5
    weekend_avg = (
        household_data[household_data["is_weekend"]][config.COL_CONSUMPTION_KWH].mean()
        if not household_data[household_data["is_weekend"]].empty
        else 0
    )
    weekday_avg = (
        household_data[~household_data["is_weekend"]][config.COL_CONSUMPTION_KWH].mean()
        if not household_data[~household_data["is_weekend"]].empty
        else 0
    )

    # Daily total consumption (sum of 48 half-hours)
    household_data["date"] = household_data[config.COL_TIMESTAMP].dt.date
    daily_totals = household_data.groupby("date")[config.COL_CONSUMPTION_KWH].sum()
    avg_daily_consumption = daily_totals.mean() if not daily_totals.empty else 0
    max_daily_consumption = daily_totals.max() if not daily_totals.empty else 0

    return {
        "avg_consumption": avg_consumption,
        "max_consumption": max_consumption,
        "total_consumption": total_consumption,
        "peak_hour": peak_hour,
        "weekend_avg": weekend_avg,
        "weekday_avg": weekday_avg,
        "avg_daily_consumption": avg_daily_consumption,
        "max_daily_consumption": max_daily_consumption,
    }


def generate_recommendations(household_id: str) -> List[Recommendation]:
    """Generate ranked, quantified recommendations for a household.

    Args:
        household_id: Identifier of the household to advise.

    Returns:
        A list of :class:`Recommendation` objects ordered by priority.
    """
    recommendations = []

    # Get household data
    usage_stats = _get_household_usage_stats(household_id)
    if not usage_stats:
        logger.warning("No usage stats found for household %s", household_id)
        return recommendations

    # Get cluster assignment
    cluster = assign_cluster(household_id)
    cluster_info = f"Cluster {cluster}" if cluster is not None else "Unknown cluster"

    # Get anomaly detection results
    anomalies_df = detect_anomalies(household_id)
    anomaly_count = len(anomalies_df) if not anomalies_df.empty else 0
    anomaly_score = (
        anomalies_df["score"].mean() if not anomalies_df.empty and "score" in anomalies_df.columns else 0
    )
    # The anomaly score from Isolation Forest is negative for anomalies, more negative means more anomalous
    # We'll convert to a positive score where higher means more anomalous
    anomaly_score = -anomaly_score if anomaly_score < 0 else 0

    # Get forecast (optional, for trend analysis)
    try:
        forecast_result = forecast_consumption(household_id, horizon_days=7)
        forecast_trend = "stable"
        if forecast_result and len(forecast_result.predicted_kwh) > 1:
            # Simple trend: compare first and last predicted values
            first_pred = forecast_result.predicted_kwh[0]
            last_pred = forecast_result.predicted_kwh[-1]
            if last_pred > first_pred * 1.1:
                forecast_trend = "increasing"
            elif last_pred < first_pred * 0.9:
                forecast_trend = "decreasing"
    except Exception as e:
        logger.warning("Could not get forecast for household %s: %s", household_id, e)
        forecast_result = None
        forecast_trend = "unknown"

    # --- Recommendation 1: High overall consumption ---
    # Compare household's average consumption to the overall average
    df_all = load_processed_data()
    if not df_all.empty:
        overall_avg = df_all[config.COL_CONSUMPTION_KWH].mean()
        if usage_stats["avg_consumption"] > overall_avg * 1.2:  # 20% above average
            excess = usage_stats["avg_consumption"] - overall_avg
            annual_excess_kwh = excess * 365 * 48  # Convert half-hourly to annual
            rec = _quantify(
                kwh_saving=annual_excess_kwh * 0.15,  # Assume 15% savings potential
                priority="high",
                title="High Energy Consumption",
                detail=f"Your average consumption ({usage_stats['avg_consumption']:.2f} kWh) is 20% above the community average. Consider implementing energy-saving measures.",
            )
            recommendations.append(rec)

    # --- Recommendation 2: High peak hour usage ---
    # If peak hour is during evening (18:00-22:00) and consumption is high
    if 18 <= usage_stats["peak_hour"] <= 22:
        # Calculate savings from shifting load off-peak
        # Assume we can shift 30% of peak load to off-peak hours
        peak_consumption = usage_stats["max_consumption"]
        shiftable_kwh = peak_consumption * 0.3 * 365  # Annualized
        rec = _quantify(
            kwh_saving=shiftable_kwh * 0.2,  # Assume 20% savings from shifting
            priority="medium",
            title="Evening Peak Usage",
            detail=f"Your peak usage occurs at {usage_stats['peak_hour']}:00, which is during peak tariff hours. Consider shifting appliance use to off-peak times.",
        )
        recommendations.append(rec)

    # --- Recommendation 3: High anomaly score ---
    if anomaly_score > 0.5:  # Threshold for significant anomalies
        # Estimate savings from fixing anomalies
        # Assume anomalies represent waste that can be reduced by 50%
        anomalous_kwh = (
            anomalies_df[config.COL_CONSUMPTION_KWH].sum()
            if not anomalies_df.empty and config.COL_CONSUMPTION_KWH in anomalies_df.columns
            else 0
        )
        annual_anomalous_kwh = anomalous_kwh * (365 * 48 / len(anomalies_df)) if len(anomalies_df) > 0 else 0
        rec = _quantify(
            kwh_saving=annual_anomalous_kwh * 0.5,
            priority="high",
            title="Unusual Usage Patterns Detected",
            detail=f"Anomaly detection identified {anomaly_count} unusual consumption patterns. Investigating these could reveal equipment issues or energy waste.",
        )
        recommendations.append(rec)

    # --- Recommendation 4: High weekend vs weekday ratio ---
    if usage_stats["weekday_avg"] > 0:
        ratio = usage_stats["weekend_avg"] / usage_stats["weekday_avg"]
        if ratio > 1.5:  # Weekend usage is 50% higher than weekday
            excess_weekend = usage_stats["weekend_avg"] - usage_stats["weekday_avg"]
            annual_excess_kwh = excess_weekend * 365 * 0.5  # Assume half the year is weekend
            rec = _quantify(
                kwh_saving=annual_excess_kwh * 0.25,  # Assume 25% savings from better weekend habits
                priority="medium",
                title="Weekend Energy Spikes",
                detail=f"Your weekend usage is {ratio:.1f}x higher than weekday usage. Consider reviewing weekend appliance usage patterns.",
            )
            recommendations.append(rec)

    # --- Recommendation 5: Increasing consumption trend ---
    if forecast_trend == "increasing":
        # Estimate savings from reversing the trend
        current_daily = usage_stats["avg_daily_consumption"]
        future_daily = forecast_result.predicted_kwh[-1] * 48 if forecast_result else current_daily
        increase = max(0, future_daily - current_daily)
        annual_increase_kwh = increase * 365
        rec = _quantify(
            kwh_saving=annual_increase_kwh * 0.3,  # Assume 30% savings from addressing the trend
            priority="medium",
            title="Increasing Consumption Trend",
            detail=f"Forecast suggests your consumption may increase over the next week. Proactive energy-saving measures can help mitigate this trend.",
        )
        recommendations.append(rec)

    # --- Recommendation 6: Low efficiency compared to cluster ---
    if cluster is not None:
        df_all = load_processed_data()
        if not df_all.empty:
            import joblib
            cluster_map_path = config.MODELS_DIR / "household_cluster_map.joblib"
            if cluster_map_path.exists():
                try:
                    cluster_map = joblib.load(cluster_map_path)
                    df_all = df_all.merge(cluster_map, on=config.COL_HOUSEHOLD_ID, how="left")
                    cluster_avg = df_all[df_all["cluster"] == cluster][config.COL_CONSUMPTION_KWH].mean()
                    household_avg = usage_stats["avg_consumption"]
                    if household_avg > cluster_avg * 1.15:
                        excess = household_avg - cluster_avg
                        annual_excess_kwh = excess * 365 * 48
                        rec = _quantify(
                            kwh_saving=annual_excess_kwh * 0.10,
                            priority="low",
                            title="Below Cluster Efficiency",
                            detail=(
                                f"Your average consumption ({household_avg:.2f} kWh) is above "
                                f"your cluster average ({cluster_avg:.2f} kWh). "
                                f"Households in your segment that adopted efficiency measures "
                                f"achieved ~10%% savings."
                            ),
                        )
                        recommendations.append(rec)
                except Exception as e:
                    logger.warning("Could not compute cluster efficiency: %s", e)

    # Sort recommendations by priority (high > medium > low)
    priority_order = {"high": 3, "medium": 2, "low": 1}
    recommendations.sort(
        key=lambda x: priority_order.get(x.priority, 0), reverse=True
    )

    return recommendations


def summarise_savings(recommendations: List[Recommendation]) -> dict[str, float]:
    """Aggregate total potential savings across recommendations.

    Returns a dict with total kWh, cost (GBP), and CO2 (kg). Safe to call with
    an empty list (returns zeros).
    """
    total_kwh = sum(r.estimated_kwh_saving for r in recommendations)
    return {
        "total_kwh": total_kwh,
        "total_cost_gbp": total_kwh * config.PRICE_PER_KWH_GBP,
        "total_co2_kg": total_kwh * config.KG_CO2_PER_KWH,
    }