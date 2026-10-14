"""True bearing degradation schedule (ground truth for later evaluation).

Assumptions
-----------
- Days 1–12: healthy (severity = 0)
- Days 13–20: early wear (severity ramps 0 → 0.35)
- Days 21–26: moderate wear (0.35 → 0.70)
- Days 27–30: severe wear (0.70 → 1.0)
- Severity is a hidden 0–1 variable; estimated twin health is separate.
"""

from __future__ import annotations

import numpy as np

MINUTES_PER_DAY = 24 * 60


def degradation_at_minute(minute_index: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return true degradation severity (0–1) and fault stage label for each minute."""
    idx = np.asarray(minute_index, dtype=float)
    day = idx / MINUTES_PER_DAY  # day 0.0 … 30.0

    severity = np.zeros_like(day)
    label = np.full(day.shape, "healthy", dtype=object)

    # Early: days 12–20 (i.e. calendar days 13–20 inclusive → day index 12..20)
    early = (day >= 12) & (day < 20)
    severity[early] = 0.35 * (day[early] - 12) / 8.0
    label[early] = "early"

    # Moderate: days 20–26
    moderate = (day >= 20) & (day < 26)
    severity[moderate] = 0.35 + 0.35 * (day[moderate] - 20) / 6.0
    label[moderate] = "moderate"

    # Severe: days 26–30 (calendar days 27–30); hits 1.0 on the final minute
    severe = day >= 26
    denom = 4.0 - 1.0 / MINUTES_PER_DAY
    severity[severe] = 0.70 + 0.30 * np.clip((day[severe] - 26) / denom, 0.0, 1.0)
    label[severe] = "severe"

    return severity, label


def degradation_series(n_minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the full ground-truth degradation profile for n_minutes of simulation."""
    return degradation_at_minute(np.arange(n_minutes))
