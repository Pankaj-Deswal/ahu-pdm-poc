"""Multi-signal analytics for predictive maintenance."""

from analytics.anomaly import detect_anomalies
from analytics.diagnosis import diagnose_faults
from analytics.prognosis import latest_summary, track_degradation_and_risk

__all__ = [
    "detect_anomalies",
    "diagnose_faults",
    "latest_summary",
    "track_degradation_and_risk",
]
