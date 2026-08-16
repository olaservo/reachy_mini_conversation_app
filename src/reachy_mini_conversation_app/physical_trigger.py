"""Physical push-to-talk trigger: hold an antenna to talk.

A hysteresis detector that reports whether an antenna is currently deflected past
a threshold ("held") relative to a slowly-tracked resting baseline, so the console
poll loop can drive a :class:`~reachy_mini_conversation_app.listen_gate.ListenGate`
hold state: deflect an antenna to open the mic (hold-to-talk), release to send.

This mirrors the daemon's one-shot ``AntennaTouchDetector`` (same calibrated
deltas) but reports a *level* rather than a single edge, which is what hold-to-talk
needs. The baseline drifts toward the resting pose only while released, so it
absorbs slow drift and adopts a new rest pose after an emotion without ever
chasing an active hold.
"""

from __future__ import annotations


# Match the daemon's calibrated antenna-touch thresholds
# (reachy_mini.daemon.app.startup_app.AntennaTouchDetector).
DEFAULT_PRESS_DELTA_RAD = 0.25  # deflection (rad) that starts a hold
DEFAULT_RELEASE_DELTA_RAD = 0.10  # deflection (rad) at/below which the hold releases
DEFAULT_BASELINE_ALPHA = 0.02  # resting-baseline tracking per update, while released


class AntennaHoldDetector:
    """Level (held True/False) detector on antenna deflection from a resting baseline."""

    def __init__(
        self,
        press_delta_rad: float = DEFAULT_PRESS_DELTA_RAD,
        release_delta_rad: float = DEFAULT_RELEASE_DELTA_RAD,
        baseline_alpha: float = DEFAULT_BASELINE_ALPHA,
    ) -> None:
        """Create a detector with the given hysteresis thresholds (radians)."""
        self.press_delta_rad = press_delta_rad
        self.release_delta_rad = release_delta_rad
        self.baseline_alpha = baseline_alpha
        self._baseline: list[float] | None = None
        self._held = False

    def reset(self) -> None:
        """Forget the baseline and release. Call whenever detection is paused."""
        self._baseline = None
        self._held = False

    @property
    def held(self) -> bool:
        """Whether an antenna is currently deflected past the press threshold."""
        return self._held

    def update(self, present: list[float]) -> bool:
        """Feed the latest present antenna positions (rad); return the held state."""
        if not present:
            return self._held
        if self._baseline is None:
            self._baseline = list(present)
            self._held = False
            return False

        deflection = max(abs(p - b) for p, b in zip(present, self._baseline))
        if not self._held:
            if deflection >= self.press_delta_rad:
                self._held = True
            else:
                # Track the resting baseline only while released — never chase a hold.
                self._baseline = [
                    b + (p - b) * self.baseline_alpha for p, b in zip(present, self._baseline)
                ]
        elif deflection <= self.release_delta_rad:
            self._held = False
        return self._held
