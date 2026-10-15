"""Tests for online warm-up analytics, diagnosis, prognosis, and pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from analytics.diagnosis import BEARING_FAULT, HEALTHY
from analytics.online_engine import OnlinePdMEngine
from analytics.prognosis import RISK_CRITICAL, RISK_LOW, estimate_rul_days
from fault_simulator import MINUTES_PER_DAY
from pipeline import daily_dashboard_series, run_analytics, save_pipeline_outputs
from sensor_simulator import simulate


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    enriched_df, _, _, _ = run_analytics(simulate())
    return enriched_df


def test_anomaly_columns(enriched: pd.DataFrame) -> None:
    for col in (
        "anomaly_score",
        "anomaly_score_smooth",
        "is_anomaly",
        "z_vibration_rms_mms",
        "phase",
    ):
        assert col in enriched.columns


def test_warmup_phase_is_healthy(enriched: pd.DataFrame) -> None:
    warm = enriched[enriched["day"] <= 7]
    assert (warm["phase"] == "warmup").all()
    assert warm["is_anomaly"].sum() == 0
    assert (warm["likely_fault"] == HEALTHY).all()
    assert (warm["estimated_health"] == 100).all()
    assert (warm["risk_level"] == RISK_LOW).all()


def test_live_phase_starts_day_8(enriched: pd.DataFrame) -> None:
    live = enriched[enriched["day"] >= 8]
    assert (live["phase"] == "live").all()


def test_anomalies_rise_after_degradation_starts(enriched: pd.DataFrame) -> None:
    early = enriched[enriched["day"] <= 7]["is_anomaly"].mean()
    late = enriched[enriched["day"] >= 21]["is_anomaly"].mean()
    assert early == 0.0
    assert late > early
    assert late > 0.3


def test_diagnosis_bearing_on_severe_days(enriched: pd.DataFrame) -> None:
    severe = enriched[enriched["day"] >= 27]
    share = (severe["likely_fault"] == BEARING_FAULT).mean()
    assert share > 0.5
    healthy_days = enriched[enriched["day"] <= 7]
    assert (healthy_days["likely_fault"] == HEALTHY).mean() == 1.0


def test_health_declines_over_time(enriched: pd.DataFrame) -> None:
    h_early = enriched[enriched["day"] <= 7]["estimated_health"].mean()
    h_late = enriched[enriched["day"] >= 27]["estimated_health"].mean()
    assert h_early > h_late
    assert h_early == 100
    assert h_late < 70


def test_risk_escalates(enriched: pd.DataFrame) -> None:
    early_risk = set(enriched[enriched["day"] <= 7]["risk_level"].unique())
    late_risk = set(enriched[enriched["day"] >= 27]["risk_level"].unique())
    assert early_risk == {RISK_LOW}
    assert late_risk & {RISK_CRITICAL, "High", "Medium"}


def test_estimate_rul_days_declining() -> None:
    s = pd.Series([90, 80, 70, 60, 50])
    rul = estimate_rul_days(s, threshold=40)
    assert rul is not None
    assert rul > 0


def test_online_engine_processes_one_row_at_a_time() -> None:
    df = simulate(n_minutes=8 * MINUTES_PER_DAY, seed=1)
    engine = OnlinePdMEngine()
    first = engine.process(df.iloc[0])
    assert first["phase"] == "warmup"
    assert first["estimated_health"] == 100

    engine2 = OnlinePdMEngine()
    out = engine2.process_frame(df)
    assert (out.loc[out["day"] <= 7, "phase"] == "warmup").all()
    assert (out.loc[out["day"] >= 8, "phase"] == "live").all()
    assert not engine2.baseline_frame().empty
    assert engine2.baseline.frozen is True


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
