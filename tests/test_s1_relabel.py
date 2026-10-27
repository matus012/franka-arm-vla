"""Stage 1 fix diagnosis D2: scripts/s1_relabel.relabel on a synthetic deadbeat expert (the 113 expert's label rule
a_t = (setpoint_t - cmd_t) / STEP with cmd_{t+1} = cmd_t + x_t * STEP, x = clip(a + noise)). The relabel must remove
the -1 reaction to the previous executed noise and, executed open loop, reproduce the setpoint path exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from s1_relabel import label_stats, relabel  # noqa: E402


def demo(T: int = 400, sigma: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    sp = np.cumsum(0.4 * np.sin(t / 40.0 + np.arange(4)) , 0)  # smooth setpoint path in action units (|step| <= 0.4)
    cmd = np.zeros(4)
    a, x = np.zeros((T, 5)), np.zeros((T, 5))
    for k in range(T):
        a[k, :4] = np.clip(sp[k] - cmd, -1, 1)
        a[k, 4] = float(k > T // 2)
        x[k] = a[k]
        x[k, :4] = np.clip(a[k, :4] + g.normal(0, sigma, 4), -1, 1)
        cmd = cmd + x[k, :4]
    return a, x, sp


def test_relabel_removes_deadbeat_and_follows_the_setpoint():
    a, x, sp = demo()
    r = relabel(a, x)
    assert np.array_equal(r[:, 4], a[:, 4]) and np.array_equal(r[0], a[0])
    np.testing.assert_allclose(np.cumsum(r[:, :4], 0), sp, atol=1e-9)  # open loop at sigma 0 = the setpoint path
    o, n = label_stats([(a, a, x)]), label_stats([(r, a, x)])
    assert all(s < -0.9 for s in o["slope_diff_on_prev_noise"])
    assert all(abs(s) < 0.1 for s in n["slope_diff_on_prev_noise"])
    assert max(n["std_label_diff"]) < 0.05 < min(o["std_label_diff"])


def test_relabel_is_identity_without_noise_and_stays_in_range():
    a, x, _ = demo(sigma=0.0)
    assert np.array_equal(relabel(a, x), a)
    a, x, _ = demo(sigma=2.0, seed=3)  # heavy clipping
    r = relabel(a, x)
    assert np.abs(r[:, :4]).max() <= 1.0
