"""Healthy baseline statistics from early operating data.

Uses Days 1–7, grouped by RPM setpoint and load bin, so later anomaly
detection can compare like-for-like operating conditions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SENSOR_COLS = [
    "vibration_rms_mms",
    "acoustic_noise_dba",
    "bearing_temp_c",
    "motor_current_a",
]


def load_bin(load: pd.Series) -> pd.Series:
    """Bucket load fraction into low / mid / high operating bins."""
    return pd.cut(
        load,
        bins=[0.0, 0.55, 0.75, 1.01],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )


def compute_baseline(df: pd.DataFrame, max_day: int = 7) -> pd.DataFrame:
    """Compute per-sensor mean and std by RPM and load bin for healthy days."""
    healthy = df[df["day"] <= max_day].copy()
    healthy["load_bin"] = load_bin(healthy["load_fraction"])
    healthy["rpm"] = healthy["rpm"].astype(int)

    rows = []
    grouped = healthy.groupby(["rpm", "load_bin"], observed=True)
    for (rpm, bin_label), g in grouped:
        row = {"rpm": int(rpm), "load_bin": str(bin_label), "n_samples": len(g)}
        for col in SENSOR_COLS:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["rpm", "load_bin"]).reset_index(drop=True)


def save_baseline(baseline: pd.DataFrame, path: str | Path) -> Path:
    """Write healthy baseline statistics to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(path, index=False)
    return path


def plot_sensor_trends(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Save daily-mean sensor and degradation trend plots to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = (
        df.groupby("day", as_index=False)
        .agg(
            vibration_rms_mms=("vibration_rms_mms", "mean"),
            acoustic_noise_dba=("acoustic_noise_dba", "mean"),
            bearing_temp_c=("bearing_temp_c", "mean"),
            motor_current_a=("motor_current_a", "mean"),
            true_degradation=("true_degradation", "mean"),
        )
    )

    written: list[Path] = []

    # Combined 4-panel trend
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        (axes[0, 0], "vibration_rms_mms", "Vibration RMS (mm/s)"),
        (axes[0, 1], "acoustic_noise_dba", "Acoustic noise (dBA)"),
        (axes[1, 0], "bearing_temp_c", "Bearing temperature (°C)"),
        (axes[1, 1], "motor_current_a", "Motor current (A)"),
    ]
    for ax, col, title in panels:
        ax.plot(daily["day"], daily[col], color="#1f4e79", lw=1.8)
        ax.axvspan(1, 12.5, color="#c6efce", alpha=0.45, label="healthy")
        ax.axvspan(12.5, 20.5, color="#ffe699", alpha=0.45, label="early")
        ax.axvspan(20.5, 26.5, color="#f4b183", alpha=0.45, label="moderate")
        ax.axvspan(26.5, 30.5, color="#f4a6a6", alpha=0.45, label="severe")
        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("AHU-FAN-001 — daily mean sensors vs bearing degradation", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "sensor_trends.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # Degradation ground truth
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(daily["day"], daily["true_degradation"], color="#8b0000", lw=2)
    ax.set_xlabel("Day")
    ax.set_ylabel("True degradation (0–1)")
    ax.set_title("Hidden bearing degradation severity")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "degradation_severity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    return written
