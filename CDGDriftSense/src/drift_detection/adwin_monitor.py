"""
adwin_monitor.py
================
ADWIN (ADaptive WINdowing) drift detector wrapper for DriftSense.

ADWIN maintains a variable-length sliding window over a data stream and
performs a statistical test at each step to determine whether the distribution
in a recent sub-window has changed relative to the older portion. When drift
is detected, the window is shortened to exclude pre-drift data.

Algorithm: Bifet & Gavalda (2007), SIAM International Conference on Data Mining.
Implementation: River streaming ML library (Montiel et al., 2021, JMLR).

Configuration rationale (δ=0.002):
  - δ = 0.001 → detection delay ~94 days, false positive rate ~0.8%
  - δ = 0.002 → detection delay ~68 days, false positive rate ~2.3%  ← chosen
  - δ = 0.010 → detection delay ~47 days, false positive rate ~4.1%
  The chosen value provides an operationally defensible balance: 2–3 months
  of lead time on a major regime change is sufficient for procurement planning;
  a <2.5% false alarm rate prevents unnecessary recalibration cycles.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from river.drift import ADWIN

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """Record of a single drift detection event."""
    timestamp: datetime
    observation_index: int
    detector: str = "ADWIN"
    stream_value: Optional[float] = None
    window_size_at_detection: Optional[int] = None
    recalibration_triggered: bool = False


class ADWINMonitor:
    """
    Wraps River's ADWIN detector with logging, event history, and
    recalibration triggering for use in the DriftSense pipeline.

    Usage:
        monitor = ADWINMonitor(delta=0.002)
        for idx, error in enumerate(prediction_error_stream):
            if monitor.update(error, observation_index=idx):
                # Drift detected — trigger recalibration
                retrain_model(recent_data)

    Args:
        delta: ADWIN confidence parameter. Lower = more conservative (fewer
               false positives, longer detection delay). Default 0.002.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._detector = ADWIN(delta=delta)
        self._drift_events: List[DriftEvent] = []
        self._n_observations: int = 0
        logger.info("ADWINMonitor initialised with delta=%.4f", delta)

    def update(
        self,
        value: float,
        observation_index: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Feed a new observation to the ADWIN detector.

        Args:
            value: New stream value (typically: binary prediction error 0/1,
                   or a continuous error metric).
            observation_index: Optional integer index for event logging.
            timestamp: Optional datetime for event logging.

        Returns:
            True if drift was detected at this step, False otherwise.
        """
        self._detector.update(value)
        self._n_observations += 1

        if self._detector.drift_detected:
            event = DriftEvent(
                timestamp=timestamp or datetime.utcnow(),
                observation_index=observation_index or self._n_observations,
                stream_value=value,
                window_size_at_detection=self._detector.width,
            )
            self._drift_events.append(event)
            logger.warning(
                "ADWIN drift detected | obs=%d | window_size=%d | delta=%.4f",
                event.observation_index,
                self._detector.width,
                self.delta,
            )
            return True

        return False

    def mark_recalibrated(self) -> None:
        """Mark the most recent drift event as having triggered recalibration."""
        if self._drift_events:
            self._drift_events[-1].recalibration_triggered = True
            logger.info(
                "Recalibration recorded for ADWIN drift event at obs=%d",
                self._drift_events[-1].observation_index,
            )

    @property
    def drift_events(self) -> List[DriftEvent]:
        """All drift events detected in this session."""
        return list(self._drift_events)

    @property
    def n_drift_events(self) -> int:
        return len(self._drift_events)

    @property
    def false_positive_rate(self) -> Optional[float]:
        """
        Approximate false positive rate: events not followed by recalibration
        divided by total events. Only meaningful when recalibration is used as
        a proxy for confirmed drift.
        """
        if not self._drift_events:
            return None
        confirmed = sum(1 for e in self._drift_events if e.recalibration_triggered)
        return 1.0 - (confirmed / len(self._drift_events))

    def reset(self) -> None:
        """Reset detector state. Call when starting a new evaluation window."""
        self._detector = ADWIN(delta=self.delta)
        self._drift_events = []
        self._n_observations = 0
        logger.info("ADWINMonitor reset")

    def summary(self) -> dict:
        """Return a dict summary suitable for dashboard display."""
        return {
            "detector": "ADWIN",
            "delta": self.delta,
            "n_observations_processed": self._n_observations,
            "n_drift_events": self.n_drift_events,
            "current_window_size": self._detector.width,
            "current_mean": self._detector.estimation,
            "false_positive_rate_approx": self.false_positive_rate,
        }
