"""Tests for anomaly detection, diagnosis, prognosis, and pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from analytics.anomaly import detect_anomalies
from analytics.diagnosis import BEARING_FAULT, HEALTHY, diagnose_faults
from pipeline import daily_dashboard_series, run_analytics, save_pipeline_outputs
from analytics.prognosis import (
    RISK_CRITICAL,
    RISK_LOW,
    estimate_rul_days,
    latest_summary,
    track_degradation_and_risk,
)
from baseline import compute_baseline
from sensor_simulator import simulate


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    df = simulate()
    baseline = compute_baseline(df)
    out = detect_anomalies(df, baseline)
    out = diagnose_faults(out)
    out = track_degradation_and_risk(out)
    return out


def test_anomaly_columns(enriched: pd.DataFrame) -> None:
    for col in (
        "anomaly_score",
        "anomaly_score_smooth",
        "is_anomaly",
        "z_vibration_rms_mms",
    ):
        assert col in enriched.columns


def test_anomalies_rise_after_degradation_starts(enriched: pd.DataFrame) -> None:
    early = enriched[enriched["day"] <= 7]["is_anomaly"].mean()
    late = enriched[enriched["day"] >= 21]["is_anomaly"].mean()
    assert late > early
    assert late > 0.3


def test_diagnosis_bearing_on_severe_days(enriched: pd.DataFrame) -> None:
    severe = enriched[enriched["day"] >= 27]
    share = (severe["likely_fault"] == BEARING_FAULT).mean()
    assert share > 0.5
    healthy_days = enriched[enriched["day"] <= 7]
    assert (healthy_days["likely_fault"] == HEALTHY).mean() > 0.8


def test_health_declines_over_time(enriched: pd.DataFrame) -> None:
    h_early = enriched[enriched["day"] <= 7]["estimated_health"].mean()
    h_late = enriched[enriched["day"] >= 27]["estimated_health"].mean()
    assert h_early > h_late
    assert h_early > 85
    assert h_late < 70


def test_risk_escalates(enriched: pd.DataFrame) -> None:
    early_risk = set(enriched[enriched["day"] <= 7]["risk_level"].unique())
    late_risk = set(enriched[enriched["day"] >= 27]["risk_level"].unique())
    assert RISK_LOW in early_risk
    assert late_risk & {RISK_CRITICAL, "High", "Medium"}


def test_estimate_rul_days_declining() -> None:
    s = pd.Series([90, 80, 70, 60, 50])
    rul = estimate_rul_days(s, threshold=40)
    assert rul is not None
    assert rul > 0


def test_pipeline_and_summary(tmp_path: Path) -> None:
    enriched, baseline, summary, fan = run_analytics()
    assert len(enriched) == 43200
    assert not baseline.empty
    assert summary["asset_id"] == "AHU-FAN-001"
    assert 1 <= summary["estimated_health"] <= 100
    assert fan.health.bearings == summary["estimated_health"]
    assert fan.health.overall == summary["estimated_health"]

    paths = save_pipeline_outputs(enriched, baseline, summary, fan, tmp_path)
    assert paths["dashboard"].exists()
    assert paths["analytics"].exists()
    daily = daily_dashboard_series(enriched)
    assert len(daily) == 30
    assert "estimated_health" in daily[0]
