"""Digital twin for a direct-drive centrifugal AHU fan with VFD."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Specs:
    """Static machine nameplate and design specifications."""

    asset_id: str = "AHU-FAN-001"
    motor_power_kw: float = 2.2
    motor_voltage_v: float = 400.0
    motor_phases: int = 3
    rated_speed_rpm: int = 1500
    operating_speeds_rpm: list[int] = field(
        default_factory=lambda: [900, 1200, 1500]
    )
    drive_type: str = "direct-drive centrifugal with VFD"


@dataclass
class OperatingState:
    """Live operating conditions: speed, run status, load, and ambient temperature."""

    rpm: float = 0.0
    is_running: bool = False
    load_fraction: float = 0.0  # 0.0–1.0
    ambient_temp_c: float = 25.0


@dataclass
class SensorValues:
    """Latest readings for the four condition-monitoring sensors."""

    vibration_rms_mms: float = 0.0
    acoustic_noise_dba: float = 0.0
    bearing_temp_c: float = 25.0
    motor_current_a: float = 0.0


@dataclass
class Health:
    """Estimated component and overall health scores (integers 1–100)."""

    motor: int = 100
    bearings: int = 100
    impeller: int = 100
    vfd: int = 100
    housing: int = 100
    overall: int = 100


class FanDigitalTwin:
    """Stores AHU fan specifications, live state, sensors, and health."""

    COMPONENTS = ("motor", "bearings", "impeller", "vfd", "housing")

    def __init__(self) -> None:
        """Create a healthy twin with default specs and empty live state."""
        self.specs = Specs()
        self.operating = OperatingState()
        self.sensors = SensorValues()
        self.health = Health()

    def update_operating(
        self,
        rpm: float | None = None,
        is_running: bool | None = None,
        load_fraction: float | None = None,
        ambient_temp_c: float | None = None,
    ) -> None:
        """Update RPM, running flag, load, and/or ambient temperature."""
        if rpm is not None:
            self.operating.rpm = float(rpm)
        if is_running is not None:
            self.operating.is_running = bool(is_running)
        if load_fraction is not None:
            self.operating.load_fraction = float(load_fraction)
        if ambient_temp_c is not None:
            self.operating.ambient_temp_c = float(ambient_temp_c)

        # Keep running flag consistent with RPM when not set explicitly.
        if is_running is None and rpm is not None:
            self.operating.is_running = self.operating.rpm > 0

    def update_sensors(
        self,
        vibration_rms_mms: float | None = None,
        acoustic_noise_dba: float | None = None,
        bearing_temp_c: float | None = None,
        motor_current_a: float | None = None,
    ) -> None:
        """Update any of the four sensor readings on the twin."""
        if vibration_rms_mms is not None:
            self.sensors.vibration_rms_mms = float(vibration_rms_mms)
        if acoustic_noise_dba is not None:
            self.sensors.acoustic_noise_dba = float(acoustic_noise_dba)
        if bearing_temp_c is not None:
            self.sensors.bearing_temp_c = float(bearing_temp_c)
        if motor_current_a is not None:
            self.sensors.motor_current_a = float(motor_current_a)

    def update_health(self, **component_scores: int) -> None:
        """Set component health (1–100) and recompute overall as the minimum."""
        for name, score in component_scores.items():
            if name not in self.COMPONENTS:
                raise ValueError(f"Unknown component: {name}")
            value = int(round(score))
            if value < 1 or value > 100:
                raise ValueError(f"Health for {name} must be 1–100, got {score}")
            setattr(self.health, name, value)

        scores = [getattr(self.health, c) for c in self.COMPONENTS]
        self.health.overall = min(scores)

    def to_dict(self) -> dict[str, Any]:
        """Return the twin state as a plain dictionary."""
        return {
            "specs": asdict(self.specs),
            "components": list(self.COMPONENTS),
            "operating": asdict(self.operating),
            "sensors": asdict(self.sensors),
            "health": asdict(self.health),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the twin state to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def export_json(self, path: str) -> None:
        """Write the twin JSON representation to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
            f.write("\n")

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the current twin state dictionary."""
        return deepcopy(self.to_dict())
