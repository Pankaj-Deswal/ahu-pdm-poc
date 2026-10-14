# AHU Fan Predictive Maintenance POC

End-to-end demo:

**Machine → Sensor simulation → Digital twin → Baseline → Anomaly → Diagnosis → Prognosis → Live dashboard**

One AHU fan (`AHU-FAN-001`), one fault (bearing wear), fully simulated data — no real hardware.

```text
ahu-pdm-poc/
└── backend/
    ├── digital_twin.py       # live machine state
    ├── sensor_simulator.py   # how the fan and sensors behave
    ├── fault_simulator.py    # how bearing wear progresses over time
    ├── baseline.py           # what “normal” looks like
    ├── analytics/
    │   ├── anomaly.py        # is something wrong?
    │   ├── diagnosis.py      # what kind of fault?
    │   └── prognosis.py      # how bad, how urgent, what to do?
    ├── pipeline.py           # runs the full chain
    ├── streamlit_app.py      # live dashboard
    ├── main.py               # batch run + file/plot export
    ├── data/                 # CSV / JSON outputs
    └── tests/
```

---

## Quick start

```bash
cd backend
pip install -r requirements.txt

# Live dashboard (recommended)
streamlit run streamlit_app.py

# Or batch run + tests
python main.py
pytest -q
```

In Streamlit, click **Start live stream**. The screen advances day by day (or hour by hour) and charts grow as if the plant were live. You can optionally save CSV/JSON under `data/`. At the end, a short history lists Day, Timestamp, Estimated health, Likely fault, Risk level, and Recommendation.

---

## How the system runs (end to end)

1. **Simulate** the fan running for up to 30 days, one reading per minute.
2. Learn a **healthy baseline** from the first week.
3. **Detect anomalies** when sensors look unusual for the current speed and load.
4. **Diagnose** whether the pattern looks like bearing wear.
5. **Estimate health, risk, and maintenance advice**, and how soon action may be needed.
6. **Update the digital twin** and show results in the UI (or write files via `main.py`).

---

## Digital twin

The twin is a software copy of the fan. It stores:

- **Who it is** — asset ID, motor size, rated speed, allowed VFD speeds
- **What it is made of** — motor, bearings, impeller, VFD, housing
- **How it is running now** — speed, on/off, load, ambient temperature
- **What sensors say** — vibration, noise, bearing temperature, motor current
- **How healthy we think it is** — score from 1 to 100 for each part; overall health is the weakest part

You can update operating conditions, sensors, and health, and export the twin as JSON.

Important: the twin’s estimated health is our **best guess from analytics**. Separately, the simulator keeps a hidden true wear level so we can check later whether detection worked.

---

## Sensor simulation (creating the data)

The simulator invents a realistic operating story:

- The fan runs for about **30 days**, with a reading every minute.
- Speed steps between **900, 1200, and 1500 RPM**, held for a few hours at a time (like a VFD setpoint).
- Load and room temperature change gently through the day.
- When the machine is healthy, sensors behave sensibly: faster speed usually means more vibration and noise; current follows electrical load; bearing temperature rises with speed/load but **changes gradually** (metal does not heat instantly).
- A little sensor noise is added so readings are not perfectly smooth.
- The same random seed always reproduces the same dataset.

### Bearing wear over time (ground truth)

Wear is introduced on a fixed schedule so we know the real answer:

| Period | Condition |
|---|---|
| Days 1–12 | Healthy |
| Days 13–20 | Early wear |
| Days 21–26 | Moderate wear |
| Days 27–30 | Severe wear |

As wear grows, vibration, noise, and bearing temperature climb. Motor current barely moves. That pattern is what diagnosis looks for later.

---

## Baseline (what “normal” means)

From **days 1–7** (still healthy), the system learns normal behaviour for each operating situation:

- Same speed (900 / 1200 / 1500)
- Similar load band (low / mid / high)

For each situation it remembers a typical sensor level and how much natural scatter to expect.

Why this matters: a high vibration reading at 1500 RPM may be normal, but the same reading at 900 RPM may not. We always compare against the right “neighbourhood,” not one global average.

---

## Anomaly detection (is something wrong?)

For every minute:

1. Find the matching healthy baseline for current speed and load.
2. Ask how far each sensor sits from that healthy typical value.
3. Blend the four sensors into one overall “how unusual?” score. Vibration matters most; current matters least (bearing wear barely changes current).
4. Average that score over the last **15 minutes** so a single noisy blip does not raise an alarm.
5. If the smoothed score stays high enough, mark the minute as **anomalous**.

So anomaly detection answers: **“This does not look like normal operation for these conditions.”** It does not yet say *why*.

---

## Fault diagnosis (what kind of problem?)

If the minute is anomalous, check whether the sensor pattern matches **bearing wear**:

- Vibration clearly high
- Noise clearly high
- Bearing temperature clearly high
- Motor current still roughly normal

If that fingerprint fits → label **bearing wear** (with a confidence score).  
If something is wrong but the pattern does not fit → **unknown**.  
If nothing is flagged anomalous → **healthy**.

An optional bearing-frequency clue can strengthen the bearing call when present.

---

## Prognosis (how bad, how urgent, what next?)

Using the anomaly score (and the fault label):

1. **Estimated health (1–100)** — the more unusual the sensors, the lower the health. A confirmed bearing diagnosis makes health a bit worse.
2. **Risk level** from health:
   - High health → **Low** risk (keep monitoring)
   - Falling health → **Medium** (watch vibration more closely)
   - Lower still → **High** (schedule a bearing inspection soon)
   - Very low → **Critical** (plan replacement before failure; ease load if you can)
3. **Recommendation** — plain-language maintenance action tied to that risk band.
4. **Time left (RUL)** — look at recent daily health trend and roughly estimate how many days until health hits the critical zone. If the trend is unclear, this may show as not available.

These estimates update the twin’s bearing/overall health for the dashboard.

---

## Orchestration and outputs

| Piece | What it does |
|---|---|
| `pipeline.py` | Ties simulation → baseline → anomaly → diagnosis → prognosis → twin update, and can write files |
| `main.py` | One-shot batch run; also writes trend plots under `plots/` |
| `streamlit_app.py` | Live in-memory playback and end-of-run history table |

### Files under `data/`
- `sensor_data.csv` — full simulated history including true wear labels
- `healthy_baseline.csv` — normal operating stats from week 1
- `analytics_results.csv` — anomaly / fault / health / risk columns
- `dashboard.json` — summary for dashboards
- `fan_twin_final.json` — final twin snapshot

---

## Design choices

- Clear engineering rules on top of physically plausible simulated sensors — not a black-box ML demo.
- Twin health is estimated; true wear is kept separately so you can judge detection quality.
- Built as a credible POC for one machine and one fault, not a production plant system.
