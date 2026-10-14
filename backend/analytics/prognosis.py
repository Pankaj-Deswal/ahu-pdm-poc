"""Health tracking, failure risk bands, RUL estimate, and maintenance actions."""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"
RISK_CRITICAL = "Critical"

RECOMMENDATIONS = {
    RISK_LOW: "Continue normal monitoring.",
    RISK_MEDIUM: "Increase vibration inspection frequency; trend bearings closely.",
    RISK_HIGH: "Schedule bearing inspection within 2 weeks.",
    RISK_CRITICAL: "Plan bearing replacement before failure; reduce load if possible.",
}


def _health_from_score(smooth_score: np.ndarray) -> np.ndarray:
    """Map smoothed anomaly score into an estimated health value (1–100)."""
    health = 100.0 - 12.0 * smooth_score
    return np.clip(np.round(health), 1, 100).astype(int)


def _risk_from_health(health: np.ndarray) -> np.ndarray:
    """Map health scores into Low / Medium / High / Critical risk bands."""
    risk = np.full(health.shape, RISK_LOW, dtype=object)
    risk[health < 75] = RISK_MEDIUM
    risk[health < 55] = RISK_HIGH
    risk[health < 40] = RISK_CRITICAL
    return risk


def estimate_rul_days(health_series: pd.Series, threshold: int = 40) -> float | None:
    """Project remaining days until health reaches the maintenance threshold."""
    if len(health_series) < 3:
        return None
    y = health_series.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    # Need a clear declining trend
    if y[-1] >= y[0] - 1:
        if y[-1] > threshold:
            return None
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= -0.05:
        return None
    # Solve intercept + slope * t = threshold
    t = (threshold - intercept) / slope
    remaining = t - (len(y) - 1)
    if remaining < 0:
        return 0.0
    return float(remaining)


def track_degradation_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add estimated health, degradation index, risk level, RUL, and recommendation."""
    out = df.copy()
    score = out.get(
        "anomaly_score_smooth",
        out.get("anomaly_score", pd.Series(0.0, index=out.index)),
    ).to_numpy()

    health = _health_from_score(score)
    # Extra penalty when diagnosed as bearing wear
    if "likely_fault" in out.columns:
        bearing = out["likely_fault"].eq("bearing_wear").to_numpy()
        health = np.where(bearing, np.maximum(1, health - 5), health)

    risk = _risk_from_health(health)
    rec = np.array([RECOMMENDATIONS[r] for r in risk], dtype=object)

    out["estimated_health"] = health
    out["degradation_index"] = np.clip((100 - health) / 99.0, 0.0, 1.0)
    out["risk_level"] = risk
    out["recommendation"] = rec

    # Daily RUL forward-filled within day
    daily = (
        out.groupby("day", as_index=False)["estimated_health"]
        .mean()
        .sort_values("day")
    )
    # Use last up-to-7 daily points ending at each day
    rul_map: dict[int, float | None] = {}
    days = daily["day"].tolist()
    healths = daily["estimated_health"].tolist()
    for i, day in enumerate(days):
        start = max(0, i - 6)
        window = pd.Series(healths[start : i + 1])
        rul_map[int(day)] = estimate_rul_days(window)

    out["rul_days"] = out["day"].map(rul_map)
    return out


def latest_summary(df: pd.DataFrame) -> dict:
    """Build a compact status snapshot from the latest day of enriched data."""
    last = df.iloc[-1]
    last_day = int(last["day"])
    day_df = df[df["day"] == last_day]
    return {
        "asset_id": "AHU-FAN-001",
        "timestamp": str(last["timestamp"]),
        "day": last_day,
        "estimated_health": int(last["estimated_health"]),
        "risk_level": str(last["risk_level"]),
        "likely_fault": str(last["likely_fault"]),
        "fault_confidence": int(last["fault_confidence"]),
        "recommendation": str(last["recommendation"]),
        "rul_days": None if pd.isna(last.get("rul_days")) else float(last["rul_days"]),
        "anomaly_rate_today": float(day_df["is_anomaly"].mean()),
        "mean_anomaly_score_today": float(day_df["anomaly_score_smooth"].mean()),
    }
