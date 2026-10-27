"""Camera-rate / latency bookkeeping of the teacher's cube estimate (Stage 0.3), no simulator."""

import torch

from vision_lift.cube_estimate import CubeEstimate


def _run(period, latency, steps=12):
    ce = CubeEstimate(1, "cpu", period=period, latency=latency)
    used = []
    for t in range(steps):
        pos = torch.full((1, 3), float(t))  # the true cube "position" encodes the step index
        p, _ = ce.step(pos, torch.zeros(1))
        used.append(int(p[0, 0]))
    return used


def test_period1_latency1_is_one_step_old():
    assert _run(1, 1) == [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_period3_latency1_holds_previous_frame():
    # ticks at 0,3,6,9: frame k is used from tick k+1 on; before that the first capture
    assert _run(3, 1) == [0, 0, 0, 0, 0, 0, 3, 3, 3, 6, 6, 6]


def test_period5_latency0_is_sample_and_hold():
    assert _run(5, 0) == [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 10, 10]


def test_noise_is_iid_per_frame_and_held():
    ce = CubeEstimate(1000, "cpu", period=2, latency=0, pos_sigma=0.01, seed=1)
    z = torch.zeros(1000, 3)
    a, _ = ce.step(z, torch.zeros(1000))
    b, _ = ce.step(z, torch.zeros(1000))
    c, _ = ce.step(z, torch.zeros(1000))
    assert torch.equal(a, b) and not torch.equal(b, c)
    assert abs(float(a.std()) - 0.01) < 0.001
