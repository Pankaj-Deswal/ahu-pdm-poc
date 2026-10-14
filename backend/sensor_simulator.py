"""AHU fan sensor time-series simulation.

Assumptions (healthy machine)
-----------------------------
- Discrete operating speeds: 900 / 1200 / 1500 RPM (VFD setpoints).
- Load fraction ~0.4–0.95 while running; short low-load periods.
- Ambient temp: daily sinusoid ~18–32 °C plus slow drift.
- Vibration RMS (mm/s): rises with RPM and load (aerodynamic / imbalance).
- Acoustic (dBA): rises mainly with RPM, mildly with load.
- Motor current (A): ~ proportional to load × (rpm/rated)^1.1; ~4.5 A full load.
- Bearing temperature (°C): equilibrium = ambient + speed/load heating;
  first-order thermal lag (inertia) toward equilibrium.
- Measurement noise: independent Gaussian on each channel.

Degradation effects (severity s in [0, 1])
-----------------------------------------
- Vibration += ~4.5 * s  (dominant fault signature)
- Acoustic  += ~12 * s
- Bearing ΔT += ~18 * s
- Current   += ~0.25 * s  (minor friction drag)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fault_simulator import MINUTES_PER_DAY, degradation_series

SEED = 42
N_DAYS = 30
N_MINUTES = N_DAYS * MINUTES_PER_DAY
OPERATING_SPEEDS = np.array([900, 1200, 1500], dtype=float)
RATED_RPM = 1500.0
FULL_LOAD_CURRENT_A = 4.5
THERMAL_ALPHA = 0.04  # ~25 min time constant at 1-min sampling


def _operating_schedule(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
    """Build piecewise RPM, load, and ambient-temperature schedules."""
    rpm = np.empty(n)
    load = np.empty(n)
    ambient = np.empty(n)

    i = 0
    while i < n:
        # Hold a setpoint for 2–8 hours
        hold = int(rng.integers(2 * 60, 8 * 60 + 1))
        j = min(i + hold, n)
        speed = float(rng.choice(OPERATING_SPEEDS))
        base_load = float(rng.uniform(0.45, 0.95))
        rpm[i:j] = speed
        # Mild load wander within the hold
        load[i:j] = np.clip(
            base_load + rng.normal(0.0, 0.03, size=j - i), 0.3, 1.0
        )
        i = j

    t_min = np.arange(n)
    # Daily ambient cycle + slow seasonal-like drift
    hour = (t_min % MINUTES_PER_DAY) / 60.0
    day = t_min / MINUTES_PER_DAY
    ambient[:] = (
        25.0
        + 5.0 * np.sin(2 * np.pi * (hour - 6) / 24.0)
        + 1.5 * np.sin(2 * np.pi * day / 30.0)
        + rng.normal(0.0, 0.2, size=n)
    )
    return rpm, load, ambient


def _healthy_targets(
    rpm: np.ndarray, load: np.ndarray, ambient: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Compute healthy vibration, acoustic, current, and bearing equilibrium targets."""
    r = rpm / RATED_RPM
    vib = 0.6 + 1.8 * r**1.5 + 0.9 * load * r
    aco = 55.0 + 18.0 * r + 4.0 * load
    current = FULL_LOAD_CURRENT_A * load * (r**1.1)
    # Steady-state bearing excess over ambient
    bearing_eq = ambient + 8.0 + 12.0 * r + 6.0 * load
    return vib, aco, current, bearing_eq


def simulate(n_minutes: int = N_MINUTES, seed: int = SEED) -> pd.DataFrame:
    """Generate sensor time series with operating conditions and bearing degradation."""
    rng = np.random.default_rng(seed)
    rpm, load, ambient = _operating_schedule(rng, n_minutes)
    severity, fault_label = degradation_series(n_minutes)

    vib0, aco0, cur0, bearing_eq = _healthy_targets(rpm, load, ambient)

    # Fault effects on targets
    vib_t = vib0 + 4.5 * severity
    aco_t = aco0 + 12.0 * severity
    cur_t = cur0 + 0.25 * severity
    bearing_eq = bearing_eq + 18.0 * severity

    # Thermal inertia (first-order filter)
    bearing = np.empty(n_minutes)
    bearing[0] = bearing_eq[0]
    for t in range(1, n_minutes):
        bearing[t] = bearing[t - 1] + THERMAL_ALPHA * (bearing_eq[t] - bearing[t - 1])

    vibration = vib_t + rng.normal(0.0, 0.08, size=n_minutes)
    acoustic = aco_t + rng.normal(0.0, 0.6, size=n_minutes)
    current = cur_t + rng.normal(0.0, 0.05, size=n_minutes)
    bearing = bearing + rng.normal(0.0, 0.15, size=n_minutes)

    # Optional frequency-domain features for future diagnosis
    shaft_hz = rpm / 60.0
    bpfo_hz = 3.2 * shaft_hz  # placeholder outer-race order
    vib_bpfo_amp = 0.05 + 2.5 * severity + rng.normal(0.0, 0.02, size=n_minutes)

    start = pd.Timestamp("2025-01-01 00:00:00")
    timestamps = start + pd.to_timedelta(np.arange(n_minutes), unit="m")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "day": (np.arange(n_minutes) // MINUTES_PER_DAY) + 1,
            "rpm": rpm,
            "load_fraction": load,
            "ambient_temp_c": ambient,
            "is_running": True,
            "vibration_rms_mms": np.clip(vibration, 0.0, None),
            "acoustic_noise_dba": acoustic,
            "bearing_temp_c": bearing,
            "motor_current_a": np.clip(current, 0.0, None),
            "shaft_hz": shaft_hz,
            "bpfo_hz": bpfo_hz,
            "vib_bpfo_amp": np.clip(vib_bpfo_amp, 0.0, None),
            "true_degradation": severity,
            "fault_label": fault_label,
        }
    )
    return df


def save_sensor_data(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the sensor dataframe to a CSV file and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
