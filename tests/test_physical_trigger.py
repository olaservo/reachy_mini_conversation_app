"""Tests for the antenna hold-to-talk detector."""

import math

from reachy_mini_conversation_app.physical_trigger import AntennaHoldDetector


def _feed(detector: AntennaHoldDetector, samples: list[list[float]]) -> list[bool]:
    return [detector.update(s) for s in samples]


def test_first_sample_sets_the_baseline_and_is_never_a_press() -> None:
    """The first reading is the baseline, whatever it is."""
    detector = AntennaHoldDetector()
    assert detector.update([0.5, -0.5]) is False
    assert detector.held is False


def test_press_needs_consecutive_samples_then_releases_with_hysteresis() -> None:
    """Press after press_samples readings; release below the lower threshold."""
    detector = AntennaHoldDetector(press_delta_rad=0.25, release_delta_rad=0.10, press_samples=3)
    detector.update([0.0, 0.0])
    # Two over-threshold samples are not a press yet; the third is.
    assert _feed(detector, [[0.3, 0.0], [0.3, 0.0]]) == [False, False]
    assert detector.update([0.3, 0.0]) is True
    # Between release and press thresholds the hold persists.
    assert detector.update([0.15, 0.0]) is True
    assert detector.update([0.05, 0.0]) is False


def test_a_single_sample_spike_does_not_press() -> None:
    """Alternating spikes never reach the sample count."""
    detector = AntennaHoldDetector(press_samples=3)
    detector.update([0.0, 0.0])
    assert _feed(detector, [[0.4, 0.0], [0.0, 0.0], [0.4, 0.0], [0.0, 0.0]]) == [False] * 4


def test_baseline_tracks_slow_drift_only_while_released() -> None:
    """Slow drift is absorbed; a hold does not move the baseline."""
    detector = AntennaHoldDetector(press_delta_rad=0.25, release_delta_rad=0.10, baseline_alpha=0.5, press_samples=1)
    detector.update([0.0, 0.0])
    # Slow drift under the press threshold is absorbed into the baseline.
    for _ in range(20):
        assert detector.update([0.2, 0.2]) is False
    # A hold does not move the baseline, so releasing back to rest reads as released.
    assert detector.update([0.6, 0.2]) is True
    for _ in range(5):
        assert detector.update([0.6, 0.2]) is True
    assert detector.update([0.2, 0.2]) is False


def test_residual_of_a_commanded_sway_never_presses() -> None:
    """Fed measured minus commanded, the robot's own antenna moves read as noise."""
    detector = AntennaHoldDetector(press_samples=3)
    # A 15 degree sway at 0.5 Hz, sampled at 30 Hz, with the servo lagging one sample.
    amplitude, hz, rate = math.radians(15), 0.5, 30
    commanded = [amplitude * math.sin(2 * math.pi * hz * i / rate) for i in range(rate * 4)]
    measured = [0.0] + commanded[:-1]
    for c, m in zip(commanded, measured):
        assert detector.update([m - c, -(m - c)]) is False


def test_reset_forgets_baseline_and_hold() -> None:
    """reset() starts over from the next reading."""
    detector = AntennaHoldDetector(press_samples=1)
    detector.update([0.0, 0.0])
    assert detector.update([0.5, 0.0]) is True
    detector.reset()
    assert detector.held is False
    assert detector.update([0.5, 0.0]) is False  # new baseline, not a press
