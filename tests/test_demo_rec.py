"""Task 011 demo recording / composing helpers (CPU): viewer projection, verdict words, the declared episode choice."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vla_tabletop.demo_rec import project  # noqa: E402


def test_project_identity_camera():
    K = torch.tensor([[[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]]])
    pos = torch.zeros(1, 3)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # xyzw identity: camera axes = world axes (ROS: z forward)
    uvz = project(K, pos, q, torch.tensor([[[1.0, 2.0, 10.0]]]))
    assert torch.allclose(uvz[0, 0], torch.tensor([60.0, 60.0, 10.0]))


def test_project_rotated_and_translated_camera():
    K = torch.tensor([[[200.0, 0, 0], [0, 200.0, 0], [0, 0, 1]]])
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    s = 2 ** -0.5
    q = torch.tensor([[0.0, s, 0.0, s]])  # +90 deg about world y: camera z (forward) -> world +x
    uvz = project(K, pos, q, torch.tensor([[[3.0, 0.0, 0.0]]]))  # 2 m straight ahead
    assert torch.allclose(uvz[0, 0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-5)


def _rec(i, success, **flags):
    return {"index": i, "task": "pick", "success": success, **flags}


def test_reason_priority_and_timeouts():
    from demo_compose import reason

    assert reason(_rec(0, True, early_close=True)) == "success"
    assert reason(_rec(0, False, early_close=True, wrong_object=True)) == "wrong cube"
    assert reason(_rec(0, False, drop=True, slam=True)) == "dropped"
    assert reason(_rec(0, False)) == "not lifted in time"


class _E:
    def __init__(self, i, ok):
        self.sig = {"index": i, "success": ok}


def test_first_shown_follows_the_declared_rule():
    from demo_compose import first_shown

    eps = [_E(0, True), _E(1, False), _E(2, False)]
    assert first_shown(eps, "PASS") is eps[0]
    assert first_shown(eps, "FAIL") is eps[1]
    assert first_shown([_E(0, True)], "FAIL").sig["index"] == 0
