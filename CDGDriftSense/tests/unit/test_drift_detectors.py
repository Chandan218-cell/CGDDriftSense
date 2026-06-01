"""
test_drift_detectors.py
=======================
Unit tests for ADWIN and Page-Hinkley drift monitor wrappers.

Tests cover:
  - No false positive on stationary stream
  - Drift detection on a stream with injected mean shift
  - Correct event logging (timestamp, index, window size)
  - Recalibration flag propagation
  - Reset clears state
  - Summary dict contains expected keys
"""

import pytest
import numpy as np
from datetime import datetime
from src.drift_detection.adwin_monitor import ADWINMonitor, DriftEvent
from src.drift_detection.page_hinkley_monitor import PageHinkleyMonitor


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def stationary_stream():
    """Gaussian stream with no drift (mean=0.1, std=0.05, n=500)."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(loc=0.1, scale=0.05, size=500)


@pytest.fixture
def drifting_stream():
    """
    Stream with abrupt mean shift at observation 300.
    Pre-drift: mean=0.1; post-drift: mean=0.9.
    Both detectors should fire before observation 400.
    """
    rng = np.random.default_rng(seed=42)
    pre = rng.normal(loc=0.1, scale=0.05, size=300)
    post = rng.normal(loc=0.9, scale=0.05, size=300)
    return np.concatenate([pre, post])


# ─── ADWIN tests ──────────────────────────────────────────────────────────────

class TestADWINMonitor:

    def test_no_drift_on_stationary_stream(self, stationary_stream):
        """ADWIN should not fire on a stationary Gaussian stream."""
        monitor = ADWINMonitor(delta=0.002)
        drift_fired = any(
            monitor.update(v, observation_index=i)
            for i, v in enumerate(stationary_stream)
        )
        assert not drift_fired, (
            f"ADWIN fired on stationary stream — check delta or stream properties. "
            f"Events: {monitor.drift_events}"
        )

    def test_detects_mean_shift(self, drifting_stream):
        """ADWIN must detect the mean shift before observation 400."""
        monitor = ADWINMonitor(delta=0.002)
        detected_at = None
        for i, v in enumerate(drifting_stream):
            if monitor.update(v, observation_index=i):
                detected_at = i
                break

        assert detected_at is not None, "ADWIN failed to detect injected mean shift"
        assert detected_at < 400, (
            f"ADWIN detected drift too late (obs {detected_at}); expected < 400"
        )

    def test_drift_event_logged(self, drifting_stream):
        """DriftEvent record should contain expected fields after detection."""
        monitor = ADWINMonitor(delta=0.002)
        ts = datetime(2023, 6, 1)
        for i, v in enumerate(drifting_stream):
            monitor.update(v, observation_index=i, timestamp=ts)
            if monitor.n_drift_events > 0:
                break

        assert monitor.n_drift_events >= 1
        event = monitor.drift_events[0]
        assert isinstance(event, DriftEvent)
        assert event.timestamp == ts
        assert event.detector == "ADWIN"
        assert event.window_size_at_detection is not None

    def test_recalibration_flag(self, drifting_stream):
        monitor = ADWINMonitor(delta=0.002)
        for i, v in enumerate(drifting_stream):
            if monitor.update(v, observation_index=i):
                monitor.mark_recalibrated()
                break

        assert monitor.drift_events[-1].recalibration_triggered is True

    def test_reset_clears_state(self, drifting_stream):
        monitor = ADWINMonitor(delta=0.002)
        for i, v in enumerate(drifting_stream):
            monitor.update(v, observation_index=i)

        monitor.reset()
        assert monitor.n_drift_events == 0
        assert monitor._n_observations == 0

    def test_summary_keys(self):
        monitor = ADWINMonitor()
        monitor.update(0.1)
        summary = monitor.summary()
        for key in ["detector", "delta", "n_observations_processed", "n_drift_events",
                    "current_window_size", "current_mean"]:
            assert key in summary, f"Missing key '{key}' in summary dict"


# ─── Page-Hinkley tests ───────────────────────────────────────────────────────

class TestPageHinkleyMonitor:

    def test_no_drift_on_stationary_stream(self, stationary_stream):
        """Page-Hinkley should not fire on a stationary Gaussian stream."""
        monitor = PageHinkleyMonitor(min_instances=30, delta=0.005, threshold=50)
        drift_fired = any(
            monitor.update(v, observation_index=i)
            for i, v in enumerate(stationary_stream)
        )
        assert not drift_fired, (
            f"PH fired on stationary stream. Events: {monitor.drift_events}"
        )

    def test_detects_abrupt_shift(self, drifting_stream):
        """Page-Hinkley must detect abrupt mean shift faster than ADWIN."""
        ph = PageHinkleyMonitor(min_instances=30, delta=0.005, threshold=50)
        adwin = ADWINMonitor(delta=0.002)

        ph_detected_at = None
        adwin_detected_at = None

        for i, v in enumerate(drifting_stream):
            if ph_detected_at is None and ph.update(v, observation_index=i):
                ph_detected_at = i
            if adwin_detected_at is None and adwin.update(v, observation_index=i):
                adwin_detected_at = i

        assert ph_detected_at is not None, "PH failed to detect injected shift"
        assert adwin_detected_at is not None, "ADWIN failed to detect injected shift"
        # PH should be faster on abrupt shifts
        assert ph_detected_at <= adwin_detected_at, (
            f"Expected PH ({ph_detected_at}) to be faster than ADWIN ({adwin_detected_at}) "
            f"on abrupt mean shift — check hyperparameter settings"
        )

    def test_reset(self, drifting_stream):
        monitor = PageHinkleyMonitor()
        for i, v in enumerate(drifting_stream):
            monitor.update(v, observation_index=i)

        monitor.reset()
        assert monitor.n_drift_events == 0
