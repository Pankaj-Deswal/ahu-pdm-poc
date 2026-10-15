"""One-reading-at-a-time PdM engine with Days 1–7 baseline warm-up.

Flow per sample
---------------
- Days 1–7: update running healthy stats only; stay healthy (no fault alarms).
- Day 8+: freeze baseline, then anomaly → diagnosis → prognosis on each reading.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from analytics.diagnosis import BEARING_FAULT, HEALTHY, UNKNOWN
from analytics.prognosis import (
    RECOMMENDATIONS,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    estimate_rul_days,
)
from baseline import SENSOR_COLS

WARMUP_DAYS = 7
DEFAULT_THRESHOLD = 3.0
PERSIST_MINUTES = 15

SENSOR_WEIGHTS = {
    "vibration_rms_mms": 1.0,
    "acoustic_noise_dba": 0.8,
    "bearing_temp_c": 0.9,
    "motor_current_a": 0.3,
}


def load_bin_value(load: float) -> str:
    """Map a single load fraction to low / mid / high."""
    if load <= 0.55:
        return "low"
    if load <= 0.75:
        return "mid"
    return "high"


@dataclass
class _BinStats:
    """Running mean/variance for one (rpm, load_bin) operating cell."""

    n: int = 0
    mean: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in SENSOR_COLS})
    m2: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in SENSOR_COLS})

    def update(self, row: pd.Series | dict[str, Any]) -> None:
        """Fold one healthy reading into this bin (Welford online stats)."""
        self.n += 1
        for col in SENSOR_COLS:
            x = float(row[col])
            delta = x - self.mean[col]
            self.mean[col] += delta / self.n
            delta2 = x - self.mean[col]
            self.m2[col] += delta * delta2

    def std(self, col: str) -> float:
        """Sample standard deviation for a sensor (tiny floor if unknown)."""
        if self.n < 2:
            return 1e-6
        return max((self.m2[col] / (self.n - 1)) ** 0.5, 1e-6)


class OnlineBaseline:
    """Accumulate and freeze healthy baseline stats during warm-up."""

    def __init__(self) -> None:
        self._bins: dict[tuple[int, str], _BinStats] = defaultdict(_BinStats)
        self.frozen = False

    def update(self, row: pd.Series | dict[str, Any]) -> None:
        """Update running stats for this reading's operating bin."""
        if self.frozen:
            return
        key = (int(row["rpm"]), load_bin_value(float(row["load_fraction"])))
        self._bins[key].update(row)

    def freeze(self) -> None:
        """Stop learning; later days use these fixed normals."""
        self.frozen = True

    def lookup(self, rpm: int, load_bin: str) -> _BinStats | None:
        """Return stats for an operating bin, if seen in warm-up."""
        return self._bins.get((int(rpm), load_bin))

    def to_dataframe(self) -> pd.DataFrame:
        """Export frozen/current baseline as the usual mean/std table."""
        rows = []
        for (rpm, bin_label), stats in sorted(self._bins.items()):
            row: dict[str, Any] = {
                "rpm": rpm,
                "load_bin": bin_label,
                "n_samples": stats.n,
            }
            for col in SENSOR_COLS:
                row[f"{col}_mean"] = float(stats.mean[col])
                row[f"{col}_std"] = float(stats.std(col))
            rows.append(row)
        return pd.DataFrame(rows)


class OnlinePdMEngine:
    """Process one sensor reading at a time through warm-up then live PdM."""

    def __init__(
        self,
        warmup_days: int = WARMUP_DAYS,
        threshold: float = DEFAULT_THRESHOLD,
        persist_minutes: int = PERSIST_MINUTES,
    ) -> None:
        self.warmup_days = warmup_days
        self.threshold = threshold
        self.persist_minutes = persist_minutes
        self.baseline = OnlineBaseline()
        self._score_buf: deque[float] = deque(maxlen=persist_minutes)
        self._daily_health: dict[int, list[float]] = defaultdict(list)
        self._completed_day_health: list[tuple[int, float]] = []
        self._current_day: int | None = None

    def process(self, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        """Ingest one reading; return enriched analytics fields for that reading."""
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        day = int(data["day"])
        self._roll_day(day)

        # Freeze baseline the first time we leave warm-up.
        if day > self.warmup_days and not self.baseline.frozen:
            self.baseline.freeze()

        if day <= self.warmup_days:
            self.baseline.update(data)
            result = self._warmup_result(data)
        else:
            result = self._live_result(data)

        self._daily_health[day].append(float(result["estimated_health"]))
        return result

    def process_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the online engine over a full dataframe (row by row)."""
        rows = [self.process(row) for _, row in df.iterrows()]
        # Flush last day into completed history for RUL consistency at end.
        self._roll_day(None)
        return pd.DataFrame(rows)

    def baseline_frame(self) -> pd.DataFrame:
        """Return baseline table learned during warm-up."""
        return self.baseline.to_dataframe()

    def _roll_day(self, new_day: int | None) -> None:
        """When the calendar day changes, store yesterday's mean health for RUL."""
        if self._current_day is None:
            self._current_day = new_day
            return
        if new_day != self._current_day:
            vals = self._daily_health.get(self._current_day, [])
            if vals:
                self._completed_day_health.append(
                    (self._current_day, float(np.mean(vals)))
                )
            self._current_day = new_day

    def _warmup_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Warm-up output: learn baseline, keep twin healthy, no alarms."""
        out = dict(data)
        out["load_bin"] = load_bin_value(float(data["load_fraction"]))
        for col in SENSOR_COLS:
            out[f"z_{col}"] = 0.0
        out["anomaly_score"] = 0.0
        out["anomaly_score_smooth"] = 0.0
        out["is_anomaly"] = False
        out["likely_fault"] = HEALTHY
        out["fault_confidence"] = 95
        out["estimated_health"] = 100
        out["degradation_index"] = 0.0
        out["risk_level"] = RISK_LOW
        out["recommendation"] = RECOMMENDATIONS[RISK_LOW]
        out["rul_days"] = None
        out["phase"] = "warmup"
        self._score_buf.clear()
        return out

    def _live_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Live PdM: anomaly → diagnosis → health/risk/recommendation/RUL."""
        out = dict(data)
        load_bin = load_bin_value(float(data["load_fraction"]))
        out["load_bin"] = load_bin
        stats = self.baseline.lookup(int(data["rpm"]), load_bin)

        z: dict[str, float] = {}
        weighted_sq = 0.0
        weight_sum = 0.0
        for col in SENSOR_COLS:
            if stats is None:
                z[col] = 0.0
            else:
                z[col] = (float(data[col]) - stats.mean[col]) / stats.std(col)
            out[f"z_{col}"] = z[col]
            w = SENSOR_WEIGHTS[col]
            weighted_sq += w * z[col] ** 2
            weight_sum += w

        score = float(np.sqrt(weighted_sq / weight_sum)) if weight_sum else 0.0
        self._score_buf.append(score)
        smooth = float(np.mean(self._score_buf))
        is_anomaly = smooth >= self.threshold

        out["anomaly_score"] = score
        out["anomaly_score_smooth"] = smooth
        out["is_anomaly"] = bool(is_anomaly)

        # Diagnosis
        z_v, z_a, z_t, z_c = (
            z["vibration_rms_mms"],
            z["acoustic_noise_dba"],
            z["bearing_temp_c"],
            z["motor_current_a"],
        )
        bearing = (
            is_anomaly
            and z_v > 2.0
            and z_a > 1.5
            and z_t > 1.5
            and abs(z_c) < 4.0
        )
        bpfo = float(data.get("vib_bpfo_amp", 0.0) or 0.0)
        if is_anomaly and bpfo > 0.4 and z_v > 1.5:
            bearing = True

        if bearing:
            fault = BEARING_FAULT
            strength = (
                min(max(z_v / 6.0, 0.0), 1.0) * 0.4
                + min(max(z_a / 5.0, 0.0), 1.0) * 0.25
                + min(max(z_t / 5.0, 0.0), 1.0) * 0.25
                + min(max(1.0 - abs(z_c) / 4.0, 0.0), 1.0) * 0.1
            )
            confidence = int(np.clip(round(40 + 60 * strength), 1, 100))
        elif is_anomaly:
            fault = UNKNOWN
            confidence = 40
        else:
            fault = HEALTHY
            confidence = 95

        out["likely_fault"] = fault
        out["fault_confidence"] = confidence

        # Prognosis
        health = int(np.clip(round(100.0 - 12.0 * smooth), 1, 100))
        if fault == BEARING_FAULT:
            health = max(1, health - 5)

        if health >= 75:
            risk = RISK_LOW
        elif health >= 55:
            risk = RISK_MEDIUM
        elif health >= 40:
            risk = RISK_HIGH
        else:
            risk = RISK_CRITICAL

        out["estimated_health"] = health
        out["degradation_index"] = float(np.clip((100 - health) / 99.0, 0.0, 1.0))
        out["risk_level"] = risk
        out["recommendation"] = RECOMMENDATIONS[risk]
        out["rul_days"] = self._rul_for_today(day=int(data["day"]), health=health)
        out["phase"] = "live"
        return out

    def _rul_for_today(self, day: int, health: int) -> float | None:
        """Estimate RUL from completed daily health means plus today's running mean."""
        today_vals = list(self._daily_health.get(day, [])) + [float(health)]
        series = [h for _, h in self._completed_day_health[-6:]]
        series.append(float(np.mean(today_vals)))
        return estimate_rul_days(pd.Series(series), threshold=40)
