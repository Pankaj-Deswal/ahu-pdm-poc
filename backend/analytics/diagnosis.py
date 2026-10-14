"""Rule-based multi-signal fault diagnosis (bearing-wear fingerprint)."""

from __future__ import annotations

import numpy as np
import pandas as pd

BEARING_FAULT = "bearing_wear"
HEALTHY = "healthy"
UNKNOWN = "unknown"


def diagnose_faults(df: pd.DataFrame) -> pd.DataFrame:
    """Assign likely_fault and fault_confidence from multi-sensor z-score patterns."""
    out = df.copy()
    z_v = out.get("z_vibration_rms_mms", pd.Series(0.0, index=out.index))
    z_a = out.get("z_acoustic_noise_dba", pd.Series(0.0, index=out.index))
    z_t = out.get("z_bearing_temp_c", pd.Series(0.0, index=out.index))
    z_c = out.get("z_motor_current_a", pd.Series(0.0, index=out.index))
    is_anom = out.get("is_anomaly", pd.Series(False, index=out.index))

    bearing = (
        is_anom
        & (z_v > 2.0)
        & (z_a > 1.5)
        & (z_t > 1.5)
        & (z_c.abs() < 4.0)
    )
    # BPFO feature strengthens bearing confidence when available
    if "vib_bpfo_amp" in out.columns:
        bearing = bearing | (is_anom & (out["vib_bpfo_amp"] > 0.4) & (z_v > 1.5))

    fault = np.where(
        bearing,
        BEARING_FAULT,
        np.where(is_anom, UNKNOWN, HEALTHY),
    )

    # Confidence 1–100 from how strongly the bearing pattern matches
    strength = (
        np.clip(z_v / 6.0, 0, 1) * 0.4
        + np.clip(z_a / 5.0, 0, 1) * 0.25
        + np.clip(z_t / 5.0, 0, 1) * 0.25
        + np.clip(1.0 - z_c.abs() / 4.0, 0, 1) * 0.1
    )
    confidence = np.where(
        fault == BEARING_FAULT,
        np.clip(np.round(40 + 60 * strength), 1, 100).astype(int),
        np.where(fault == UNKNOWN, 40, 95),
    )

    out["likely_fault"] = fault
    out["fault_confidence"] = confidence
    return out
