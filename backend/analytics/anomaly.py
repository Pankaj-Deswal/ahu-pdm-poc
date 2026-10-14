"""Baseline z-score anomaly detection across vibration, acoustic, temp, current."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baseline import SENSOR_COLS, load_bin

# Current is weakly affected by bearing wear — down-weight it.
SENSOR_WEIGHTS = {
    "vibration_rms_mms": 1.0,
    "acoustic_noise_dba": 0.8,
    "bearing_temp_c": 0.9,
    "motor_current_a": 0.3,
}

DEFAULT_THRESHOLD = 3.0
PERSIST_MINUTES = 15  # require sustained elevation to flag anomaly


def detect_anomalies(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    persist_minutes: int = PERSIST_MINUTES,
) -> pd.DataFrame:
    """Add per-sensor z-scores, combined anomaly score, and anomaly flags."""
    out = df.copy()
    out["rpm_key"] = out["rpm"].astype(int)
    out["load_bin"] = load_bin(out["load_fraction"])

    base = baseline.copy()
    base["rpm"] = base["rpm"].astype(int)

    merged = out.merge(
        base,
        left_on=["rpm_key", "load_bin"],
        right_on=["rpm", "load_bin"],
        how="left",
        suffixes=("", "_base"),
    )

    z_cols = []
    weighted_sq = np.zeros(len(merged))
    weight_sum = 0.0

    for col in SENSOR_COLS:
        mean_c = f"{col}_mean"
        std_c = f"{col}_std"
        z_c = f"z_{col}"
        std = merged[std_c].replace(0, np.nan).fillna(1e-6)
        z = (merged[col] - merged[mean_c]) / std
        merged[z_c] = z.fillna(0.0)
        z_cols.append(z_c)
        w = SENSOR_WEIGHTS[col]
        weighted_sq += w * merged[z_c].to_numpy() ** 2
        weight_sum += w

    merged["anomaly_score"] = np.sqrt(weighted_sq / weight_sum)
    # Smooth to reduce chatter, then require sustained breach
    smooth = (
        pd.Series(merged["anomaly_score"])
        .rolling(persist_minutes, min_periods=1)
        .mean()
        .to_numpy()
    )
    merged["anomaly_score_smooth"] = smooth
    merged["is_anomaly"] = smooth >= threshold

    keep_extra = z_cols + [
        "load_bin",
        "anomaly_score",
        "anomaly_score_smooth",
        "is_anomaly",
    ]
    for col in keep_extra:
        out[col] = merged[col].to_numpy()

    out = out.drop(columns=["rpm_key"])
    return out
