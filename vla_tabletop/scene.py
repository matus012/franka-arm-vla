"""113 tabletop scene: env cfg + gym registration (brief sections 3 and 6, CONTRACT "World catalog" / "Sim observation").

- Franka + SeattleLab table from 111 (`lift_rl.env_cfg.LiftSceneCfg`), 111 physics fix 2 (4 substeps, stiff contacts,
  finger armature 0.1), 50 Hz control (dt 0.01, decimation 2).
- Action = the 112 student action (CONTRACT): 4 task-space deltas through the 111 DLS IK (hand vertical) + binary claw
  executed directly (finger effort 6.1 N, `vision_lift.envs.CLAW_EFFORT_N`). One 113 change inside the IK term: the cube
  weight feed-forward (`held` = this step's claw command, as in 112) uses the mass of the episode's manipulated cube
  (named by the instruction; small 0.216 / large 0.42 kg) instead of the single 111 cube's mass.
- Observation group "policy" = the 19-dim state: joint_pos_rel 9, joint_vel_rel 9, gripper_width 1. Nothing else.
- All 8 cubes (`cube_<color>_<size>`, primitive cuboids, density `catalog.DENSITY`, friction 1.0 = the 111 cube) and
  4 pads (`pad_<color>`, `catalog.PAD_EDGE_M` squares, `PAD_THICKNESS_M` thick) in every env. Pads are dynamic
  bodies WITH collision (a placed cube rests on the pad top, not sunk into it), PAD_MASS = 0.5 kg mats resting on the
  table (friction 1.0). Kinematic pads were tried first: with kinematic bodies in the model the effort-limited claw
  dropped every lifted cube (0 / 32 holds vs 30 / 32 with dynamic pads, dev runs 2026-10-01; cause inside
  Newton / MuJoCo-Warp not isolated).
- Objects not used in an episode are parked on the ground far outside the reach box and both camera frusta
  (`layout.park_pose`, tests/test_sim_113.py).
- Per-episode layout: `vla_tabletop/layout.py` (pure numpy, seeded), written by the reset event from env._layout.
- Cameras side + wrist from `vision_lift/cameras.py`, resolution a parameter (aim the side camera with `aim_side`).
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.sensors import ContactSensorCfg

from lift_rl import mdp
from lift_rl.env_cfg import LiftEnvCfg
from lift_rl.guarded_env import GuardedRLEnv
from lift_rl.instrument import TABLE_SHAPE
from lift_rl.scripted import ScriptedController
from lift_rl.task_space import _IK
from vision_lift import envs as VE
from vision_lift.cameras import add_student_cameras
from vla_common.catalog import (ALL_CUBES, COLORS, CUBE_NAMES, DENSITY, EDGE_M, NOMINAL_RGB, PAD_EDGE_M, PAD_NAMES,
                                PAD_THICKNESS_M, CubeId, cube_name, pad_name)

from .layout import OBJ_NAMES, layout_array, park_pose

TASK_ID = "VLA-Tabletop-v0"
CUBE_MASS = {s: DENSITY * EDGE_M[s] ** 3 for s in EDGE_M}
PAD_MASS = 0.5


# --------------------------------------------------------------------------- reset event
def _t(x):
    return x.torch if hasattr(x, "torch") else x


def apply_layout(env, env_ids: torch.Tensor) -> None:
    """Reset event: write every object's pose from env._layout (N, 12, 5) (all parked if unset); zero velocities."""
    if not hasattr(env, "_layout"):
        env._layout = torch.as_tensor(layout_array([{}] * env.num_envs), device=env.device)
    ids = env_ids.to(env.device).long()
    lay = env._layout[ids]
    o = env.scene.env_origins[ids]
    z = torch.zeros_like(lay[:, 0, 3])
    for j, name in enumerate(OBJ_NAMES):
        from isaaclab.utils.math import quat_from_euler_xyz

        q = quat_from_euler_xyz(z, z, lay[:, j, 3])
        pose = torch.cat([lay[:, j, :3] + o, q], 1)
        a = env.scene[name]
        a.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        a.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


# --------------------------------------------------------------------------- action: 112 student IK, per-episode mass
class _TabletopIK(_IK):
    """111 DLS IK tracking state (task_space._IK) without the 111 single 'object': the cube-weight feed-forward mass is
    env._ff_mass (N,), the mass of the episode's manipulated cube (set by the rollout from the TaskSpec)."""

    def __init__(self, env):  # ScriptedController.__init__ minus the 111 'object' lines
        from lift_rl.metrics import Thresholds

        self.u = u = env.unwrapped if hasattr(env, "unwrapped") else env
        self.reflex, self.debug_env, self.th = True, -1, Thresholds()
        self.robot = r = u.scene["robot"]
        self.N, self.dev = u.num_envs, u.device
        self.hand = r.body_names.index("panda_hand")
        self.jac_body = self.hand - 1 if r.is_fixed_base else self.hand
        self.arm_ids = r.find_joints("panda_joint.*")[0]
        nb = getattr(r, "num_base_dofs", 0)
        self.jac_cols = [j + nb for j in self.arm_ids]
        self.q_default = _t(r.data.default_joint_pos)[:, self.arm_ids].clone()
        lim = _t(r.data.soft_joint_pos_limits)[:, self.arm_ids]
        self.q_lo, self.q_hi = lim[..., 0], lim[..., 1]
        self.kp = _t(r.data.joint_stiffness)[:, self.arm_ids].clamp_min(1.0)
        self.kd = _t(r.data.joint_damping)[:, self.arm_ids]
        self.damp = torch.tensor(ScriptedController.DAMP, device=self.dev)
        term = u.action_manager.get_term("arm_action")
        self.a_scale, self.a_offset = term._scale, term._offset
        m = _t(r.data.body_mass)
        self.link_mass = m[:, 1:] if r.is_fixed_base else m
        self.started = False

    @property
    def cube_mass(self) -> torch.Tensor:
        if not hasattr(self.u, "_ff_mass"):
            self.u._ff_mass = torch.full((self.N,), CUBE_MASS["small"], device=self.dev)
        return self.u._ff_mass


class TabletopIKAction(VE.StudentIKAction):
    def _lazy(self):
        if self.ik is None:
            self.ik = _TabletopIK(self._env)
            self.ik._reset()
            self.arm_ids = self.ik.arm_ids


@configclass
class TabletopIKActionCfg(VE.StudentIKActionCfg):
    class_type: type = TabletopIKAction


@configclass
class ActionsCfg:
    arm_action = TabletopIKActionCfg(asset_name="robot")
    gripper_action = VE.StudentClawActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": VE.CLAW_OPEN}, close_command_expr={"panda_finger_.*": VE.CLAW_CLOSED})


# --------------------------------------------------------------------------- observation: the 19-dim state only
POLICY_TERMS = ("joint_pos", "joint_vel", "gripper_width")
STATE_DIM = 19


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 9: 7 arm + 2 finger, relative to default
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 9
        gripper_width = ObsTerm(func=VE.student_gripper_width)  # 1: finger width [m]

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    layout = EventTerm(func=apply_layout, mode="reset")


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# --------------------------------------------------------------------------- objects + sensors
# contact gap 1 mm = the 111 DexCube collision box (Newton shape_gap 0.001, results of scripts dev check); the
# CuboidCfg default would give Newton's default gap 0.01
COLLISION = sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
def _cube_cfg(c: CubeId) -> RigidObjectCfg:
    e = EDGE_M[c[1]]
    x, y, z = park_pose(cube_name(c))
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + cube_name(c),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(e, e, e),
            rigid_props=RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                               max_angular_velocity=1000.0, max_linear_velocity=1000.0,
                                               max_depenetration_velocity=5.0, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS[c[1]]),
            collision_props=COLLISION,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=NOMINAL_RGB[c[0]]),
        ),
    )


def _pad_cfg(color: str) -> RigidObjectCfg:
    x, y, z = park_pose(pad_name(color))
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + pad_name(color),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(PAD_EDGE_M, PAD_EDGE_M, PAD_THICKNESS_M),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=PAD_MASS),
            collision_props=COLLISION,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=NOMINAL_RGB[color]),
        ),
    )


ROBOT_BODIES = ("panda_hand", "panda_leftfinger", "panda_rightfinger")


def add_sensors(cfg) -> None:
    """Contact sensors (observe only): robot bodies vs table (slam, 111 rule); cubes vs robot bodies / other cubes / pads
    (grasp, jam, stacking); cubes vs table."""
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.contact_hand_table = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_*",
                                                    filter_shape_prim_expr=[TABLE_SHAPE], history_length=0)
    cfg.scene.contact_cubes = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/cube_.*",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in ROBOT_BODIES] + ["{ENV_REGEX_NS}/Robot/panda_link*",
                                                                                       "{ENV_REGEX_NS}/cube_.*",
                                                                                       "{ENV_REGEX_NS}/pad_.*"],
        history_length=0)
    cfg.scene.contact_cubes_table = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/cube_.*",
                                                     filter_shape_prim_expr=[TABLE_SHAPE], history_length=0)
    cfg.scene.contact_hand_pads = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/pad_.*",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in ROBOT_BODIES], history_length=0)


@configclass
class TabletopEnvCfg(LiftEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()  # 111 scene, physics fix 2, finger armature, decimation 2, dt 0.01
        self.commands = None
        self.curriculum = None
        self.episode_length_s = 1000.0  # the rollout decides the episode length; no timeout inside it
        self.scene.object = None
        self.scene.env_spacing = 2.5
        for c in ALL_CUBES:
            setattr(self.scene, cube_name(c), _cube_cfg(c))
        for color in COLORS:
            setattr(self.scene, pad_name(color), _pad_cfg(color))
        add_sensors(self)
        self.scene.robot.actuators["panda_hand"].effort_limit_sim = VE.CLAW_EFFORT_N


def make_cfg(num_envs: int, seed: int, res: int | None = None, seg: bool = False) -> TabletopEnvCfg:
    """Env cfg for num_envs envs on cuda:0; res = camera resolution (None: no cameras); seg adds instance segmentation."""
    cfg = TabletopEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = "cuda:0"
    if res is not None:
        add_student_cameras(cfg, res=res, data_types=("rgb", "instance_segmentation_fast") if seg else ("rgb",))
    return cfg


def shape_ids(u, name: str, visible_only: bool = True) -> torch.Tensor:
    """(num_envs, k) Newton shape indices of scene object `name` (prim /World/envs/env_<i>/<name>/...) per env;
    visible_only keeps shapes with the VISIBLE flag (what the renderer and instance segmentation use)."""
    from isaaclab_newton.physics import NewtonManager

    m = NewtonManager.get_model()
    flags = m.shape_flags.numpy()
    out = []
    for i in range(u.num_envs):
        key = f"/World/envs/env_{i}/{name}/"
        out.append([k for k, lab in enumerate(m.shape_label) if lab.startswith(key) and (flags[k] & 1 or not visible_only)])
    assert all(len(r) == len(out[0]) and r for r in out), (name, out[:2])
    return torch.tensor(out, device=u.device)


def apply_nominal_colors(u) -> None:
    """Nominal visuals: cube / pad colours = catalog.NOMINAL_RGB, table = the 112 default tint (visual_rand.DEFAULT_TABLE).
    Written to the Newton model's shape_color (read by the renderer); the USD preview material of a primitive cuboid is
    not picked up by the Newton renderer (all cubes rendered grey otherwise) and the table's palette colour depends on
    the shape count (green here, which would read as a green object)."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from .visual_rand import DEFAULT_TABLE

    colors = wp.to_torch(NewtonManager.get_model().shape_color)
    for c in ALL_CUBES:
        colors[shape_ids(u, cube_name(c)).flatten()] = torch.tensor(NOMINAL_RGB[c[0]], device=colors.device)
    for color in COLORS:
        colors[shape_ids(u, pad_name(color)).flatten()] = torch.tensor(NOMINAL_RGB[color], device=colors.device)
    colors[shape_ids(u, "Table").flatten()] = torch.tensor(DEFAULT_TABLE, device=colors.device)


class TabletopEnv(GuardedRLEnv):
    """GuardedRLEnv whose NaN detection checks the robot and every cube (there is no 111 'object'); nominal colours
    are written at construction."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        apply_nominal_colors(self)

    def _detect_nonfinite(self) -> torch.Tensor:
        import warp as wp

        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        robot = self.scene["robot"]
        ts = [robot.data.joint_pos, robot.data.joint_vel]
        for name in CUBE_NAMES:
            ts += [self.scene[name].data.root_pos_w, self.scene[name].data.root_lin_vel_w]
        for t in ts:
            t = _t(t)
            bad |= ~torch.isfinite(t.reshape(self.num_envs, -1)).all(dim=1)
        solver, _ = self._solver()
        data = getattr(solver, "mjw_data", None)
        if data is not None:
            for name in ("qpos", "qvel", "qacc"):
                arr = wp.to_torch(getattr(data, name))
                if arr.shape[0] == self.num_envs:
                    bad |= ~torch.isfinite(arr.reshape(self.num_envs, -1)).all(dim=1)
        return bad


gym.register(id=TASK_ID, entry_point=f"{__name__}:TabletopEnv", disable_env_checker=True,
             kwargs={"env_cfg_entry_point": f"{__name__}:TabletopEnvCfg"})


# --------------------------------------------------------------------------- sim-state readers (env frame)
def object_poses(u) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 12, 3) positions and (N, 12) yaws of OBJ_NAMES, env frame."""
    from isaaclab.utils.math import euler_xyz_from_quat

    o = u.scene.env_origins
    pos = torch.stack([_t(u.scene[n].data.root_pos_w) - o for n in OBJ_NAMES], 1)
    yaw = torch.stack([euler_xyz_from_quat(_t(u.scene[n].data.root_quat_w))[2] for n in OBJ_NAMES], 1)
    return pos, yaw


def object_quats(u) -> torch.Tensor:
    return torch.stack([_t(u.scene[n].data.root_quat_w) for n in OBJ_NAMES], 1)


def cube_lin_vel(u) -> torch.Tensor:
    return torch.stack([_t(u.scene[n].data.root_lin_vel_w) for n in CUBE_NAMES], 1)


def cube_contacts(u) -> dict[str, torch.Tensor]:
    """Contact-force magnitudes [N], per cube (rows in CUBE_NAMES order): vs left / right finger, palm (panda_hand),
    arm links (max), every other cube (N, 8, 8), every pad (N, 8, 4), table (N, 8); hand bodies vs table, vs pads, max of
    both (F_HT, GATES conventions); any robot body vs table."""
    if not hasattr(u, "_cc_idx"):
        s = u.scene["contact_cubes"]
        rows = [next(i for i, n in enumerate(s.sensor_names) if n == name) for name in CUBE_NAMES]
        f = {n: i for i, n in enumerate(s.filter_object_names)}
        st = u.scene["contact_cubes_table"]
        rows_t = [next(i for i, n in enumerate(st.sensor_names) if n == name) for name in CUBE_NAMES]
        ht = u.scene["contact_hand_table"]
        u._cc_idx = {
            "rows": torch.tensor(rows, device=u.device), "rows_t": torch.tensor(rows_t, device=u.device),
            "lf": f["panda_leftfinger"], "rf": f["panda_rightfinger"], "palm": f["panda_hand"],
            "links": torch.tensor([i for n, i in f.items() if n.startswith("panda_link")], device=u.device),
            "cubes": torch.tensor([f[n] for n in CUBE_NAMES], device=u.device),
            "pads": torch.tensor([f[n] for n in PAD_NAMES], device=u.device),
            "hand_rows": torch.tensor([i for i, n in enumerate(ht.sensor_names) if n in ROBOT_BODIES], device=u.device),
        }
    ix = u._cc_idx
    n = u.num_envs
    fm = _t(u.scene["contact_cubes"].data.force_matrix_w)
    fm = fm.reshape(n, fm.shape[1], -1, 3).norm(dim=-1)[:, ix["rows"]]  # (N, 8, F)
    ft = _t(u.scene["contact_cubes_table"].data.force_matrix_w)
    ft = ft.reshape(n, ft.shape[1], -1, 3).norm(dim=-1)[:, ix["rows_t"]].amax(-1)  # (N, 8)
    rt = _t(u.scene["contact_hand_table"].data.force_matrix_w)
    rt = rt.reshape(n, rt.shape[1], -1, 3).norm(dim=-1).amax(-1)  # (N, robot bodies)
    hp = _t(u.scene["contact_hand_pads"].data.force_matrix_w)
    hp = hp.reshape(n, -1, 3).norm(dim=-1).amax(-1)  # (N,) any hand / finger body vs any pad
    ht = rt[:, ix["hand_rows"]].amax(-1)
    return {"lf": fm[..., ix["lf"]], "rf": fm[..., ix["rf"]], "palm": fm[..., ix["palm"]],
            "arm": fm[..., ix["links"]].amax(-1) if len(ix["links"]) else torch.zeros_like(fm[..., 0]),
            "cube": fm[..., ix["cubes"]], "pad": fm[..., ix["pads"]], "table": ft,
            "hand_table": ht, "hand_pad": hp, "hand_table_or_pad": torch.maximum(ht, hp), "robot_table": rt.amax(-1)}


def tcp_pose(u) -> tuple[torch.Tensor, torch.Tensor]:
    """Fingertip centre (env frame) and hand rotation matrix (from the hand link pose)."""
    from lift_rl.phased_cfg import tcp_and_rot

    return tcp_and_rot(u)


def gripper_width(u) -> torch.Tensor:
    return VE.student_gripper_width(u)[:, 0]
