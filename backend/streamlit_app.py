"""Live Streamlit PdM dashboard — in-memory day/hour streaming (no database).

Run from the backend/ directory:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics.prognosis import latest_summary
from baseline import compute_baseline
from digital_twin import FanDigitalTwin
from fault_simulator import MINUTES_PER_DAY
from pipeline import (
    daily_dashboard_series,
    run_analytics,
    save_pipeline_outputs,
    update_twin_from_row,
)
from sensor_simulator import SEED as DEFAULT_SEED
from sensor_simulator import simulate

DATA_DIR = Path(__file__).resolve().parent / "data"
RISK_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}

st.set_page_config(page_title="AHU Fan PdM Live", page_icon="🔧", layout="wide")

st.title("AHU-FAN-001 — Live Predictive Maintenance")
st.caption(
    "Streams analytics in memory day-by-day (or hour-by-hour) "
    "so charts and metrics grow like a live plant feed."
)


def _prepare_run(n_days: int, seed: int) -> pd.DataFrame:
    """Simulate and run full analytics once so playback has correct scores."""
    df = simulate(n_minutes=n_days * MINUTES_PER_DAY, seed=seed)
    enriched, _, _, _ = run_analytics(df=df)
    return enriched


def _chunk_frames(enriched: pd.DataFrame, mode: str):
    """Yield (label, dataframe chunk) for live playback."""
    if mode == "Day":
        for day, g in enriched.groupby("day"):
            yield f"Day {int(day)}", g
        return

    work = enriched.reset_index(drop=True).copy()
    work["_hour"] = work.index // 60
    for hour, g in work.groupby("_hour"):
        day = int(g["day"].iloc[-1])
        yield f"Day {day} · hour {int(hour) % 24 + 1}", g.drop(columns=["_hour"])


def _daily_so_far(seen: pd.DataFrame) -> pd.DataFrame:
    """Build daily chart rows from the in-memory buffer streamed so far."""
    return pd.DataFrame(daily_dashboard_series(seen))


def _render_live_panel(
    latest: pd.Series,
    daily: pd.DataFrame,
    twin: dict,
    rows_seen: int,
    status_label: str,
) -> None:
    """Draw current metrics/charts from the in-memory buffer."""
    st.markdown(f"**Live clock:** `{latest.get('timestamp', '—')}` · {status_label}")
    st.caption(f"In-memory readings streamed: **{rows_seen:,}**")

    risk = str(latest.get("risk_level", "Low"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asset health", f"{int(latest.get('estimated_health', 0))} / 100")
    c2.metric("Risk", f"{RISK_COLORS.get(risk, '')} {risk}")
    c3.metric("Likely fault", str(latest.get("likely_fault", "—")).replace("_", " "))
    rul = latest.get("rul_days")
    c4.metric("RUL (days)", "n/a" if rul is None or pd.isna(rul) else f"{float(rul):.1f}")

    st.info(f"**Recommended action:** {latest.get('recommendation', '—')}")

    if daily.empty:
        return

    left, right = st.columns(2)
    chart = daily.set_index("day").copy()
    if "true_degradation" in chart.columns:
        chart["true_degradation_pct"] = chart["true_degradation"] * 100.0

    with left:
        st.subheader("Health vs degradation (so far)")
        cols = [c for c in ["estimated_health", "true_degradation_pct"] if c in chart.columns]
        st.line_chart(
            chart[cols].rename(
                columns={
                    "estimated_health": "Estimated health",
                    "true_degradation_pct": "True degradation (×100)",
                }
            )
        )

    with right:
        st.subheader("Anomaly score & rate (so far)")
        cols = [c for c in ["anomaly_score", "anomaly_rate"] if c in chart.columns]
        st.line_chart(
            chart[cols].rename(
                columns={
                    "anomaly_score": "Anomaly score",
                    "anomaly_rate": "Anomaly rate",
                }
            )
        )

    st.subheader("Sensors (daily mean so far)")
    sensor_cols = [
        c
        for c in [
            "vibration_rms_mms",
            "bearing_temp_c",
            "acoustic_noise_dba",
            "motor_current_a",
        ]
        if c in chart.columns
    ]
    st.line_chart(
        chart[sensor_cols].rename(
            columns={
                "vibration_rms_mms": "Vibration (mm/s)",
                "bearing_temp_c": "Bearing temp (°C)",
                "acoustic_noise_dba": "Acoustic (dBA)",
                "motor_current_a": "Motor current (A)",
            }
        )
    )

    st.subheader("Digital twin (live)")
    t1, t2 = st.columns(2)
    with t1:
        st.json(
            {
                "operating": twin.get("operating"),
                "sensors": twin.get("sensors"),
                "health": twin.get("health"),
            }
        )
    with t2:
        health = twin.get("health", {})
        st.bar_chart(
            pd.DataFrame(
                {
                    "component": list(health.keys()),
                    "health": list(health.values()),
                }
            ).set_index("component")
        )

    st.subheader("Run history (daily)")
    history_cols = [
        c
        for c in [
            "day",
            "timestamp",
            "estimated_health",
            "likely_fault",
            "risk_level",
            "recommendation",
        ]
        if c in daily.columns
    ]
    with st.expander("Daily analytics table", expanded=True):
        st.dataframe(
            daily[history_cols].rename(
                columns={
                    "day": "Day",
                    "timestamp": "Timestamp",
                    "estimated_health": "Estimated health",
                    "likely_fault": "Likely fault",
                    "risk_level": "Risk level",
                    "recommendation": "Recommendation",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


with st.sidebar:
    st.header("Live run")
    n_days = st.slider("Days to simulate", 14, 30, 30, 1)
    seed = st.number_input("Random seed", 0, 10_000, DEFAULT_SEED)
    step_mode = st.selectbox("Stream step", ["Day", "Hour"], index=0)
    speed = st.select_slider(
        "Playback delay",
        options=["Fast", "Normal", "Slow"],
        value="Normal",
    )
    delay = {"Fast": 0.05, "Normal": 0.25, "Slow": 0.7}[speed]
    save_files = st.checkbox("Also save CSV/JSON under data/", value=True)
    start = st.button("▶ Start live stream", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "Each step appends to an **in-memory buffer**. "
        "Charts grow over time; optional CSV/JSON export is separate."
    )

progress = st.progress(0.0, text="Idle")
live_area = st.empty()

if start:
    with st.spinner("Preparing simulation + analytics…"):
        enriched = _prepare_run(int(n_days), int(seed))
        if save_files:
            fan_full = FanDigitalTwin()
            update_twin_from_row(fan_full, enriched.iloc[-1])
            summary = latest_summary(enriched)
            baseline = compute_baseline(enriched)
            save_pipeline_outputs(enriched, baseline, summary, fan_full, DATA_DIR)

    fan = FanDigitalTwin()
    seen_parts: list[pd.DataFrame] = []
    chunks = list(_chunk_frames(enriched, step_mode))
    total = len(chunks)

    for i, (label, chunk) in enumerate(chunks, start=1):
        seen_parts.append(chunk)
        seen = pd.concat(seen_parts, ignore_index=True)
        last_row = chunk.iloc[-1]
        update_twin_from_row(fan, last_row)

        progress.progress(i / total, text=f"Streaming {label} ({i}/{total})")
        with live_area.container():
            _render_live_panel(
                last_row,
                _daily_so_far(seen),
                fan.to_dict(),
                len(seen),
                label,
            )
        time.sleep(delay)

    progress.progress(1.0, text="Live stream complete")
    daily_full = _daily_so_far(enriched)
    twin_final = fan.to_dict()
    st.session_state["last_result"] = {
        "latest": enriched.iloc[-1].to_dict(),
        "daily": daily_full,
        "twin": twin_final,
        "rows": len(enriched),
    }
    with live_area.container():
        _render_live_panel(
            enriched.iloc[-1],
            daily_full,
            twin_final,
            len(enriched),
            "Run complete — full history",
        )
    st.success(f"Finished. Streamed {len(enriched):,} readings in memory.")

elif "last_result" in st.session_state:
    r = st.session_state["last_result"]
    with live_area.container():
        _render_live_panel(
            pd.Series(r["latest"]),
            r["daily"],
            r["twin"],
            r["rows"],
            "Last completed run — full history",
        )
else:
    st.info("Configure the sidebar, then click **Start live stream**.")
