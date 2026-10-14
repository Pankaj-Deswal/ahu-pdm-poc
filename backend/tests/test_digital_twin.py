"""Basic checks for FanDigitalTwin state updates and JSON export."""

import json
from pathlib import Path

import pytest

from digital_twin import FanDigitalTwin


def test_initial_health_is_full() -> None:
    fan = FanDigitalTwin()
    assert fan.specs.asset_id == "AHU-FAN-001"
    assert fan.operating.is_running is False
    assert fan.health.overall == 100
    assert fan.health.bearings == 100


def test_update_operating_and_sensors() -> None:
    fan = FanDigitalTwin()
    fan.update_operating(rpm=1500, load_fraction=0.8, ambient_temp_c=30.0)
    fan.update_sensors(
        vibration_rms_mms=2.1,
        acoustic_noise_dba=72.0,
        bearing_temp_c=45.0,
        motor_current_a=3.5,
    )

    assert fan.operating.rpm == 1500
    assert fan.operating.is_running is True
    assert fan.operating.load_fraction == 0.8
    assert fan.sensors.vibration_rms_mms == 2.1
    assert fan.sensors.motor_current_a == 3.5


def test_update_health_sets_overall_to_min() -> None:
    fan = FanDigitalTwin()
    fan.update_health(bearings=70, motor=90)
    assert fan.health.bearings == 70
    assert fan.health.motor == 90
    assert fan.health.overall == 70


def test_update_health_rejects_out_of_range() -> None:
    fan = FanDigitalTwin()
    with pytest.raises(ValueError):
        fan.update_health(bearings=0)
    with pytest.raises(ValueError):
        fan.update_health(motor=101)


def test_to_json_and_export(tmp_path: Path) -> None:
    fan = FanDigitalTwin()
    fan.update_operating(rpm=900, is_running=True)

    data = json.loads(fan.to_json())
    assert data["specs"]["asset_id"] == "AHU-FAN-001"
    assert data["operating"]["rpm"] == 900
    assert data["health"]["overall"] == 100
    assert "bearings" in data["components"]

    path = tmp_path / "twin.json"
    fan.export_json(str(path))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["operating"]["rpm"] == 900
