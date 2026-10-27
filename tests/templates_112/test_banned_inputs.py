"""Stage 0.4 (2/3): banned-input test (brief section 5).

Banned: cube pose in any form, contact forces, any phase label or sweet-spot signal computed from privileged state.
Checked three ways (the sim-level check — student obs group terms + cube-teleport invariance in the running env —
lives in tests/test_sim_112.py):
  1. the student's non-image observation group is exactly the whitelist, term by term, function by function;
  2. the only batch -> model path (`model_inputs`) passes exactly the whitelisted input keys;
  3. poisoning every label / privileged field with NaN and garbage leaves the student's output bit-identical, and the
     student's own source mentions no privileged quantity.
"""

from __future__ import annotations

import inspect

import torch

import lift_rl.mdp as mdp
import lift_rl.phased_cfg as P
from vision_lift import envs as E
from vision_lift.dataset import INPUT_KEYS, model_inputs
from vision_lift.student import StudentPolicy, _Backbone

BANNED_WORDS = ("cube", "object", "phase", "contact", "force", "pad", "palm", "sweet", "band", "grasp", "priv",
                "teacher", "label")


def test_student_obs_group_is_the_whitelist():
    g = E.StudentObsCfg()
    terms = {k: v for k, v in vars(g).items() if hasattr(v, "func")}
    assert tuple(terms) == E.STUDENT_TERMS == ("joint_pos", "joint_vel", "gripper_width", "target", "prev_action")
    allowed = {"joint_pos": mdp.joint_pos_rel, "joint_vel": mdp.joint_vel_rel, "gripper_width": E.student_gripper_width,
               "target": P.obs_target, "prev_action": mdp.last_action}
    for k, term in terms.items():
        assert term.func is allowed[k], k
        assert not term.params, (k, term.params)  # last_action without action_name = full previous action (arm + claw)
    src = inspect.getsource(E.student_gripper_width)
    assert "panda_finger" in src and "contact" not in src and "object" not in src
    assert E.STUDENT_PROPRIO_DIM == 27


def test_model_inputs_whitelist():
    assert INPUT_KEYS == ("side_rgb", "wrist_rgb", "side_depth", "wrist_depth", "proprio")
    batch = {k: 1 for k in INPUT_KEYS} | {"teacher_arm": 2, "claw_label": 3, "cube_pos": 4, "phase": 5, "_where": 6}
    assert set(model_inputs(batch)) == set(INPUT_KEYS)


def _batch(B=4, poison=False):
    g = torch.Generator().manual_seed(0)
    b = {"side_rgb": torch.randint(0, 256, (B, 3, 128, 128), generator=g, dtype=torch.uint8),
         "wrist_rgb": torch.randint(0, 256, (B, 3, 128, 128), generator=g, dtype=torch.uint8),
         "side_depth": torch.rand(B, 1, 128, 128, generator=g), "wrist_depth": torch.rand(B, 1, 128, 128, generator=g),
         "proprio": torch.randn(B, 27, generator=g)}
    fill = float("nan") if poison else 0.0
    b |= {"teacher_arm": torch.full((B, 4), fill), "claw_label": torch.full((B,), fill),
          "cube_pos": torch.full((B, 3), fill if poison else 0.5), "cube_yaw": torch.full((B,), fill),
          "phase": torch.full((B,), 3 if poison else 0), "in_band": torch.full((B,), poison),
          "pad_force": torch.full((B, 2), 1e6 if poison else 0.0), "_where": torch.zeros(B, 4)}
    return b


def test_poisoned_privileged_fields_do_not_change_the_student():
    for depth in (False, True):
        torch.manual_seed(1)
        m = StudentPolicy(depth=depth).eval()
        a0, l0 = m(model_inputs(_batch(poison=False)))
        a1, l1 = m(model_inputs(_batch(poison=True)))
        assert torch.isfinite(a1).all() and torch.isfinite(l1).all()
        assert torch.equal(a0, a1) and torch.equal(l0, l1)


def test_student_source_mentions_no_privileged_quantity():
    for obj in (StudentPolicy, _Backbone):
        src = inspect.getsource(obj).lower()
        for w in BANNED_WORDS:
            assert w not in src, (obj.__name__, w)


def test_aux_target_never_reaches_the_model_and_is_unused_when_acting():
    b = _batch(poison=False)
    b["aux_rel_yaw_deg"] = torch.full((4,), 30.0)
    assert "aux_rel_yaw_deg" not in model_inputs(b)
    torch.manual_seed(2)
    m = StudentPolicy(depth=False).eval()
    a0, l0 = m(model_inputs(b))
    b["aux_rel_yaw_deg"] = torch.full((4,), float("nan"))
    a1, l1 = m(model_inputs(b))
    assert torch.equal(a0, a1) and torch.equal(l0, l1)
    act_src = inspect.getsource(StudentPolicy.forward) + inspect.getsource(StudentPolicy.act)
    assert "aux" not in act_src  # acting never touches the auxiliary head
