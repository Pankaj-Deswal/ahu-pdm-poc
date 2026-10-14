"""End-to-end PdM pipeline: simulate → baseline → analytics → twin update → outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.anomaly import detect_anomalies
from analytics.diagnosis import diagnose_faults
from analytics.prognosis import latest_summary, track_degradation_and_risk
from baseline import compute_baseline, save_baseline
from digital_twin import FanDigitalTwin
from sensor_simulator import save_sensor_data, simulate


def update_twin_from_row(fan: FanDigitalTwin, row: Any) -> FanDigitalTwin:
    """Push one enriched reading into the digital twin operating/sensor/health state."""
    fan.update_operating(
        rpm=float(row["rpm"]),
        is_running=bool(row["is_running"]),
        load_fraction=float(row["load_fraction"]),
        ambient_temp_c=float(row["ambient_temp_c"]),
    )
    fan.update_sensors(
        vibration_rms_mms=float(row["vibration_rms_mms"]),
        acoustic_noise_dba=float(row["acoustic_noise_dba"]),
        bearing_temp_c=float(row["bearing_temp_c"]),
        motor_current_a=float(row["motor_current_a"]),
    )
    fan.update_health(bearings=int(row["estimated_health"]))
    return fan


def run_analytics(
    df: pd.DataFrame | None = None,
    fan: FanDigitalTwin | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], FanDigitalTwin]:
    """Run simulation (if needed), analytics, and update the digital twin."""
    if df is None:
        df = simulate()
    if fan is None:
        fan = FanDigitalTwin()

    baseline = compute_baseline(df, max_day=7)
    enriched = detect_anomalies(df, baseline)
    enriched = diagnose_faults(enriched)
    enriched = track_degradation_and_risk(enriched)

    update_twin_from_row(fan, enriched.iloc[-1])
    summary = latest_summary(enriched)
    summary["twin"] = fan.to_dict()
    return enriched, baseline, summary, fan


def daily_dashboard_series(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate enriched minute data into one summary row per day."""
    rows = []
    for day, g in df.groupby("day"):
        rows.append(
            {
                "day": int(day),
                "timestamp": str(g["timestamp"].iloc[-1]),
                "vibration_rms_mms": float(g["vibration_rms_mms"].mean()),
                "acoustic_noise_dba": float(g["acoustic_noise_dba"].mean()),
                "bearing_temp_c": float(g["bearing_temp_c"].mean()),
                "motor_current_a": float(g["motor_current_a"].mean()),
                "anomaly_score": float(g["anomaly_score_smooth"].mean()),
                "anomaly_rate": float(g["is_anomaly"].mean()),
                "estimated_health": float(g["estimated_health"].mean()),
                "true_degradation": float(g["true_degradation"].mean()),
                "likely_fault": str(g["likely_fault"].mode().iloc[0]),
                "risk_level": str(g["risk_level"].iloc[-1]),
                "recommendation": str(g["recommendation"].iloc[-1]),
            }
        )
    return rows


def save_pipeline_outputs(
    enriched: pd.DataFrame,
    baseline: pd.DataFrame,
    summary: dict[str, Any],
    fan: FanDigitalTwin,
    data_dir: str | Path,
) -> dict[str, Path]:
    """Persist CSV, dashboard JSON, and twin JSON outputs under data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "sensor": save_sensor_data(enriched, data_dir / "sensor_data.csv"),
        "baseline": save_baseline(baseline, data_dir / "healthy_baseline.csv"),
        "analytics": data_dir / "analytics_results.csv",
        "dashboard": data_dir / "dashboard.json",
        "twin": data_dir / "fan_twin_final.json",
    }
    cols = [
        c
        for c in [
            "timestamp",
            "day",
            "rpm",
            "load_fraction",
            "vibration_rms_mms",
            "acoustic_noise_dba",
            "bearing_temp_c",
            "motor_current_a",
            "anomaly_score",
            "anomaly_score_smooth",
            "is_anomaly",
            "likely_fault",
            "fault_confidence",
            "estimated_health",
            "degradation_index",
            "risk_level",
            "recommendation",
            "rul_days",
            "true_degradation",
            "fault_label",
        ]
        if c in enriched.columns
    ]
    enriched[cols].to_csv(paths["analytics"], index=False)

    payload = {
        "summary": {k: v for k, v in summary.items() if k != "twin"},
        "twin": fan.to_dict(),
        "daily": daily_dashboard_series(enriched),
    }
    paths["dashboard"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fan.export_json(str(paths["twin"]))
    return paths
