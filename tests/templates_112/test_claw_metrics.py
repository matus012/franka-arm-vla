"""Detector tests for the new claw metrics (early close, drop before HOLD, jam/drop phase), no simulator."""

import torch

from lift_rl.phases import CARRY, CLOSE, DESCEND, HOLD, HOVER, LIFT
from vision_lift import claw_metrics as CM


def test_early_close():
    cmd = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    band = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    # env0: closes at t=2 in band -> fine; env1: closes at t=1 out of band -> early; env2: closed at t=0 in band;
    # env3: never closes
    assert CM.early_close(cmd, band).tolist() == [False, True, False, False]


def test_drop_before_hold():
    ph = torch.tensor([[HOVER, LIFT, CARRY, HOLD, HOLD], [HOVER, LIFT, CARRY, HOLD, HOLD], [DESCEND, CLOSE, LIFT, HOVER, HOVER]])
    drop = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], dtype=torch.bool)
    assert CM.drop_before_hold(drop, ph).tolist() == [True, False, True]


def test_jam_onset_and_phase():
    js = torch.zeros(2, 30, dtype=torch.bool)
    js[0, 5:20] = True  # 15-step run -> onset at t = 14 (10th step)
    js[1, 0:9] = True  # 9 steps: no jam
    on = CM.jam_onset(js, 10)
    assert on[0].nonzero().flatten().tolist() == [14] and not on[1].any()
    ph = torch.full((2, 30), LIFT)
    ph[0, 14] = CARRY
    assert CM.first_event_phase(on, ph).tolist() == [CARRY, -1]
    assert CM.by_phase(torch.tensor([CARRY, -1, HOLD]))["carry"] == 1
