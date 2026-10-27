"""Phase re-sync relabeler (proposal after 3.1 attempt 1): stateless phase from the true state, no simulator."""

import math

import torch

from lift_rl.phases import CARRY, CLOSE, CUBE_HALF, DESCEND, HOLD, HOVER, LIFT
from vision_lift.relabel import resync_teacher_obs, stateless_phase

CUBE = torch.tensor([[0.5, 0.0, CUBE_HALF]])
TGT = torch.tensor([[0.5, 0.15, 0.35]])
Z = torch.zeros(1)


def ph(tcp, speed=0.0, lf=0.0, rf=0.0, palm=0.0, cube=CUBE, tilt=0.0, yaw=0.0):
    return stateless_phase(torch.tensor([tcp]), torch.tensor([speed]), torch.tensor([tilt]), torch.tensor([yaw]),
                           cube, TGT, torch.tensor([lf]), torch.tensor([rf]), torch.tensor([palm]))


def test_phases():
    assert ph([0.5, 0.0, 0.40])[0].item() == HOVER  # far above the pre-grasp point
    assert ph([0.60, 0.0, 0.10])[0].item() == HOVER  # off the column
    assert ph([0.5, 0.005, 0.08])[0].item() == DESCEND  # in the column, below pre-grasp height
    assert ph([0.5, 0.005, 0.08], tilt=math.radians(20))[0].item() == HOVER  # tilted
    assert ph([0.5, 0.005, CUBE_HALF + 0.005], speed=0.05)[0].item() == CLOSE  # sweet band, slow
    assert ph([0.5, 0.005, CUBE_HALF + 0.005], speed=0.30)[0].item() == DESCEND  # in band but fast
    lifted = torch.tensor([[0.5, 0.0, CUBE_HALF + 0.03]])
    assert ph([0.5, 0.0, 0.05], lf=6, rf=6, cube=lifted)[0].item() == LIFT
    high = torch.tensor([[0.5, 0.05, CUBE_HALF + 0.15]])
    assert ph([0.5, 0.05, 0.17], lf=6, rf=6, cube=high)[0].item() == CARRY
    assert ph([0.5, 0.15, 0.35], lf=6, rf=6, cube=TGT)[0].item() == HOLD
    on_table_3n = ph([0.5, 0.0, CUBE_HALF], lf=3, rf=3)  # closing on the table, pads 3 N: not yet confirmed
    assert not bool(on_table_3n[1])
    raised_3n = ph([0.5, 0.0, 0.05], lf=3, rf=3, cube=lifted)  # cube lifted, pads 3 N: still held (keep threshold)
    assert bool(raised_3n[1]) and raised_3n[0].item() == LIFT
    p, g = ph([0.5, 0.15, 0.35], lf=6, rf=6, palm=3, cube=TGT)  # palm contact: not a grasp
    assert not bool(g) and p.item() != HOLD


def test_obs_replacement_touches_only_the_phase_slice():
    o = torch.randn(3, 48)
    r = resync_teacher_obs(o, torch.tensor([0, 2, 5]), torch.tensor([False, False, True]))
    assert torch.equal(r[:, :41], o[:, :41])
    assert r[0, 41:47].tolist() == [1, 0, 0, 0, 0, 0] and r[2, 46].item() == 1 and r[2, 47].item() == 1


def _step(cm, tcp, lf=0.0, rf=0.0, cube=CUBE, speed=0.0):
    cm.step(torch.tensor([tcp]), torch.tensor([speed]), Z, Z, cube, TGT, torch.tensor([lf]), torch.tensor([rf]), Z, Z)


def test_catchup_r2_descend_without_the_waypoint():
    from vision_lift.relabel import CatchUpMachine

    cm = CatchUpMachine(1, "cpu")
    _step(cm, [0.5, 0.012, 0.09])  # in the column, 1.2 cm off-axis, below pre-grasp height, never at the waypoint
    assert cm.phase.item() == DESCEND and int(cm.r2) == 1


def test_catchup_r1_grasp_without_close():
    from vision_lift.relabel import CatchUpMachine

    cm = CatchUpMachine(1, "cpu")
    lifted = torch.tensor([[0.5, 0.0, CUBE_HALF + 0.03]])
    _step(cm, [0.5, 0.0, 0.05], lf=6, rf=6, cube=lifted)
    assert cm.phase.item() == LIFT and bool(cm.grasp) and int(cm.r1) == 1
    _step(cm, [0.5, 0.0, 0.05], lf=0.0, rf=6, cube=lifted)  # one pad drops out for a step: the latch holds
    assert cm.phase.item() == LIFT and bool(cm.grasp)


def test_catchup_matches_stateful_on_the_teacher_path():
    """Waypoint -> descend -> close -> confirm: R1/R2 never fire, phases follow the 111 machine."""
    from vision_lift.relabel import CatchUpMachine

    cm = CatchUpMachine(1, "cpu")
    _step(cm, [0.5, 0.0, CUBE_HALF * 2 + 0.10])  # at the pre-grasp waypoint -> DESCEND (native)
    assert cm.phase.item() == DESCEND and int(cm.r2) == 0
    for _ in range(3):
        _step(cm, [0.5, 0.0, CUBE_HALF], speed=0.05)  # sweet spot, slow -> CLOSE after 2 steps (native)
    assert cm.phase.item() == CLOSE
    _step(cm, [0.5, 0.0, CUBE_HALF], lf=6, rf=6)  # confirm -> LIFT (native)
    assert cm.phase.item() == LIFT and int(cm.r1) == 0
