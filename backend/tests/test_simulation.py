"""Validation tests for sensor simulation, fault schedule, and baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline import compute_baseline, save_baseline
from fault_simulator import MINUTES_PER_DAY, degradation_at_minute, degradation_series
from sensor_simulator import N_MINUTES, save_sensor_data, simulate


def test_stage_boundaries() -> None:
    s, lab = degradation_at_minute(0)
    assert float(s) == 0.0
    assert lab.item() == "healthy"

    _, lab = degradation_at_minute(12 * MINUTES_PER_DAY - 1)
    assert lab.item() == "healthy"

    s, lab = degradation_at_minute(12 * MINUTES_PER_DAY)
    assert lab.item() == "early"
    assert float(s) >= 0.0

    _, lab = degradation_at_minute(20 * MINUTES_PER_DAY)
    assert lab.item() == "moderate"
    _, lab = degradation_at_minute(26 * MINUTES_PER_DAY)
    assert lab.item() == "severe"


def test_severity_monotonic() -> None:
    sev, _ = degradation_series(N_MINUTES)
    assert np.all(np.diff(sev) >= -1e-12)
    assert float(sev[0]) == pytest.approx(0.0)
    assert float(sev[-1]) == pytest.approx(1.0, abs=1e-5)


@pytest.fixture(scope="module")
def sensor_df() -> pd.DataFrame:
    return simulate()


def test_row_count_and_reproducibility(sensor_df: pd.DataFrame) -> None:
    assert len(sensor_df) == N_MINUTES
    pd.testing.assert_frame_equal(sensor_df, simulate())


def test_required_columns(sensor_df: pd.DataFrame) -> None:
    required = {
        "timestamp",
        "rpm",
        "load_fraction",
        "ambient_temp_c",
        "vibration_rms_mms",
        "acoustic_noise_dba",
        "bearing_temp_c",
        "motor_current_a",
        "true_degradation",
        "fault_label",
    }
    assert required.issubset(sensor_df.columns)


def test_sensors_rise_with_degradation(sensor_df: pd.DataFrame) -> None:
    healthy = sensor_df[sensor_df["fault_label"] == "healthy"]
    severe = sensor_df[sensor_df["fault_label"] == "severe"]
    assert severe["vibration_rms_mms"].mean() > healthy["vibration_rms_mms"].mean()
    assert severe["acoustic_noise_dba"].mean() > healthy["acoustic_noise_dba"].mean()
    assert severe["bearing_temp_c"].mean() > healthy["bearing_temp_c"].mean()


def test_vibration_rises_with_rpm_when_healthy(sensor_df: pd.DataFrame) -> None:
    h = sensor_df[sensor_df["fault_label"] == "healthy"]
    means = h.groupby("rpm")["vibration_rms_mms"].mean().sort_index()
    assert means.iloc[-1] > means.iloc[0]


def test_save_and_baseline(sensor_df: pd.DataFrame, tmp_path: Path) -> None:
    path = save_sensor_data(sensor_df.head(1000), tmp_path / "s.csv")
    assert path.exists()

    baseline = compute_baseline(sensor_df, max_day=7)
    assert not baseline.empty

    bpath = save_baseline(baseline, tmp_path / "b.csv")
    assert bpath.exists()
    for col in [
        "vibration_rms_mms_mean",
        "vibration_rms_mms_std",
        "motor_current_a_mean",
    ]:
        assert col in baseline.columns
