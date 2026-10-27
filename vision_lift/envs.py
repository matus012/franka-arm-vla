"""112 environments: the 111 phase worlds (copied, frozen) + student cameras, and the student variant.

Teacher env  (`Vision-Teacher-<W>-v0`): the 111 task-space env of world W with its claw reflex, unchanged, + cameras.
Student env  (`Vision-Student-<W>-v0`): same world, but
  - action = 4 arm deltas (same IK, hand held vertical) + 1 claw command (1 = close, 0 = open), executed directly:
    finger target 0 (close) or 0.04 m (open); no reflex;
  - finger actuator effort limit = CLAW_EFFORT_N (calibrated so the steady per-pad force matches the reflex hold,
    brief section 5; value recorded in GATES.md);
  - the IK's cube-weight feed-forward `held` = the claw command of the current step, never the phase machine;
  - observation group "student" = the only non-image inputs the student gets (whitelist, tests/test_banned_inputs.py);
    group "policy" stays the 48-dim teacher observation (labels for DAgger), never fed to the student.
The phase machine (true state) keeps running in both envs for labels, metrics and videos.
Worlds W: Fixed (Stage 2), P3 cube position (3.1), P4 + yaw (3.2), P5 + target (3.3), P6 + mass/friction (3.4, 1, 5).
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

from lift_rl import mdp
from lift_rl import phased_cfg as P
from lift_rl.task_space import STEP_POS, STEP_YAW, LEAD_MAX, TaskSpaceIKAction, TaskSpaceIKActionCfg

from .cameras import add_student_cameras

# Finger actuator effort limit [N] on the student claw. Set by the Stage-2-prep calibration (GATES.md); None = the
# 111 default (200 N) — only used by the calibration script itself.
CLAW_EFFORT_N: float | None = 6.1  # calibrated 2026-09-30, GATES.md "Claw calibration"
CLAW_OPEN, CLAW_CLOSED = 0.04, 0.0
STUDENT_TERMS = ("joint_pos", "joint_vel", "gripper_width", "target", "prev_action")  # the whitelist
STUDENT_PROPRIO_DIM = 9 + 9 + 1 + 3 + 5


def _t(x):
    return x.torch if hasattr(x, "torch") else x


# --------------------------------------------------------------------------- student actions
class StudentIKAction(TaskSpaceIKAction):
    """111 task-space IK with one change: `held` (cube-weight feed-forward) = this step's claw command (closed)."""

    def process_actions(self, actions: torch.Tensor):
        self._lazy()
        if self._need_init.any():
            ids = self._need_init.nonzero(as_tuple=False)[:, 0]
            self._init_ids(ids)
            self._need_init[ids] = False
        a = actions.clamp(-1.0, 1.0)
        self._raw[:] = a
        ik = self.ik
        p = ik._tcp()
        self.target = torch.maximum(torch.minimum(self.target + a[:, :3] * STEP_POS, self.hi), self.lo)
        lead = self.target - p
        n = torch.linalg.norm(lead, dim=-1, keepdim=True)
        self.target = p + lead * torch.clamp(LEAD_MAX / n.clamp_min(1e-9), max=1.0)
        self.yaw = self.yaw + a[:, 3] * STEP_YAW
        r = ik.robot
        qh = _t(r.data.body_link_quat_w)[:, ik.hand]
        R = matrix_from_quat(qh)
        z0 = torch.zeros_like(self.yaw)
        q_des = quat_from_euler_xyz(torch.full_like(self.yaw, math.pi), z0, self.yaw)
        held = self._env.action_manager.action[:, 4] > 0.5  # the student's own claw command (current step)
        vmax = torch.full((self.num_envs,), ik.V_TRACK, device=self.device)
        dn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        a_arm = ik._track(self.target, vmax, dn, q_des, p, qh, R, held)
        self._q_cmd = a_arm * ik.a_scale + ik.a_offset


@configclass
class StudentIKActionCfg(TaskSpaceIKActionCfg):
    class_type: type = StudentIKAction


class StudentClawAction(BinaryJointPositionAction):
    """Binary claw executed directly: command > 0.5 -> close (target 0), else open (0.04 m)."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        close = actions > 0.5
        self._processed_actions = torch.where(close, self._close_command, self._open_command)


@configclass
class StudentClawActionCfg(BinaryJointPositionActionCfg):
    class_type: type = StudentClawAction


# --------------------------------------------------------------------------- student observation terms (whitelist)
def student_gripper_width(env):
    r = env.scene["robot"]
    ids = r.find_joints("panda_finger.*")[0]
    return _t(r.data.joint_pos)[:, ids].sum(1, keepdim=True)


@configclass
class StudentObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 9: 7 arm + 2 finger, relative to default
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 9
    gripper_width = ObsTerm(func=student_gripper_width)  # 1: finger width [m]
    target = ObsTerm(func=P.obs_target)  # 3: target position, robot base frame (= env frame, no rotation)
    prev_action = ObsTerm(func=mdp.last_action)  # 5: previous 4 arm deltas + claw command

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


# --------------------------------------------------------------------------- configs
_WORLDS = {"Fixed": P.LiftPhasedFixedTSEnvCfg, "P3": P.LiftPhasedP3TSEnvCfg, "P4": P.LiftPhasedP4TSEnvCfg,
           "P5": P.LiftPhasedP5TSEnvCfg, "P6": P.LiftPhasedP6TSEnvCfg}


def _teacher_cfg(base):
    @configclass
    class TeacherCfg(base):
        def __post_init__(self):
            super().__post_init__()
            add_student_cameras(self)

    TeacherCfg.__name__ = f"Teacher{base.__name__}"
    return TeacherCfg


def _student_cfg(base):
    @configclass
    class StudentCfg(base):
        def __post_init__(self):
            super().__post_init__()
            add_student_cameras(self)
            self.actions.arm_action = StudentIKActionCfg(asset_name="robot")
            self.actions.gripper_action = StudentClawActionCfg(
                asset_name="robot", joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": CLAW_OPEN}, close_command_expr={"panda_finger_.*": CLAW_CLOSED})
            if CLAW_EFFORT_N is not None:
                self.scene.robot.actuators["panda_hand"].effort_limit_sim = CLAW_EFFORT_N
            self.observations.student = StudentObsCfg()

    StudentCfg.__name__ = f"Student{base.__name__}"
    return StudentCfg


TEACHER_CFGS = {w: _teacher_cfg(c) for w, c in _WORLDS.items()}
STUDENT_CFGS = {w: _student_cfg(c) for w, c in _WORLDS.items()}


def set_claw_effort(cfg, effort_n: float) -> None:
    """Override the finger effort limit on a student cfg instance (calibration runs)."""
    cfg.scene.robot.actuators["panda_hand"].effort_limit_sim = effort_n


def teacher_id(world: str) -> str:
    return f"Vision-Teacher-{world}-v0"


def student_id(world: str) -> str:
    return f"Vision-Student-{world}-v0"


for _w in _WORLDS:
    for _id, _cfgs in ((teacher_id(_w), TEACHER_CFGS), (student_id(_w), STUDENT_CFGS)):
        gym.register(id=_id, entry_point="lift_rl.guarded_env:GuardedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": f"{__name__}:{_cfgs[_w].__name__}",
                             "rsl_rl_cfg_entry_point": "lift_rl.agents:LiftPPORunnerCfg"})
        globals()[_cfgs[_w].__name__] = _cfgs[_w]
