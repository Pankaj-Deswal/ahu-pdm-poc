"""CLI entry point: run the full PdM pipeline and write files/plots."""

from __future__ import annotations

from pathlib import Path

from baseline import plot_sensor_trends
from pipeline import run_analytics, save_pipeline_outputs

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"


def main() -> None:
    """Execute simulation, analytics, file export, and trend plotting."""
    print("Running AHU PdM pipeline (simulate → analytics → dashboard)...")
    enriched, baseline, summary, fan = run_analytics()
    paths = save_pipeline_outputs(enriched, baseline, summary, fan, DATA_DIR)

    for name, path in paths.items():
        print(f"  wrote {name}: {path}")

    for p in plot_sensor_trends(enriched, PLOTS_DIR):
        print(f"  wrote plot: {p}")

    print("\nSummary:")
    for key in (
        "estimated_health",
        "risk_level",
        "likely_fault",
        "fault_confidence",
        "rul_days",
        "recommendation",
    ):
        print(f"  {key}: {summary[key]}")

    print("\nDigital twin health:")
    print(fan.to_json())


if __name__ == "__main__":
    main()
