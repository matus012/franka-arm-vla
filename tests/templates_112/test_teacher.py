"""Teacher checksum + plain-module loader (brief section 4). The rsl_rl-equivalence check runs in the sim
(scripts/s01_teacher_recheck.py, >= 1,000 recorded observations, max abs diff <= 1e-6)."""

import torch

from vision_lift.teacher import TEACHER_SHA256, load_teacher, sha256


def test_teacher_checksum():
    assert sha256() == TEACHER_SHA256


def test_teacher_loads_and_is_deterministic():
    a = load_teacher("cpu")
    x = torch.randn(64, 48)
    assert a(x).shape == (64, 4)
    assert torch.equal(a(x), a(x))
    assert not any(p.requires_grad for p in a.parameters())
