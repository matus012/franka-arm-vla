"""Camera timing shared by data collection, the student runtime and the dataset (one source of truth).

A camera frame is captured on every `period`-th control step from episode start (the chosen rate, Stage 0.3:
period 3 = 16.7 Hz) and reaches the policy `latency` frames later (latency 1). At control step t the policy sees the
frame captured at step frame_step(t). Before the first delayed frame exists, the first frame (step 0) is used.
"""

from __future__ import annotations

PERIOD = 3  # Stage 0.3 result: 16.7 Hz
LATENCY = 1


def n_frames(T: int, period: int = PERIOD) -> int:
    return (T + period - 1) // period


def frame_index(t: int, period: int = PERIOD, latency: int = LATENCY) -> int:
    """Index (into the per-episode list of captured frames) of the frame the policy sees at control step t."""
    return max(t // period - latency, 0)


def frame_step(t: int, period: int = PERIOD, latency: int = LATENCY) -> int:
    return frame_index(t, period, latency) * period


class DelayedFrames:
    """Runtime buffer: capture() on tick steps with the freshly rendered frame, current() every step."""

    def __init__(self, period: int = PERIOD, latency: int = LATENCY):
        self.period, self.latency = period, latency
        self.reset()

    def reset(self) -> None:
        self.buf: list[tuple[int, object]] = []  # (frame index, frame), newest last

    def is_tick(self, t: int) -> bool:
        return t % self.period == 0

    def capture(self, t: int, frame) -> None:
        assert self.is_tick(t)
        self.buf = (self.buf + [(t // self.period, frame)])[-(self.latency + 1):]

    def current(self) -> tuple[int, object]:
        return self.buf[0]
