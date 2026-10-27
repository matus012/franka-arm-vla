"""Batched episode runner (GATES conventions "Episode stream"): env + controller + detectors (+ cameras / recording).

Episode stream: episode i of seed S draws its task spec, layout and instruction from np.random.default_rng([S, i]) only,
so outcomes never depend on the number of parallel envs (a partial last batch is padded with copies and the copies are
dropped). One batch = up to num_envs episodes run in lockstep from a reset for T control steps (no early termination).
Per step: the controller acts on the state before the step, the env steps, the detectors read the state after it.
Cameras (optional): a frame is captured at every PERIOD-th step (before that step's action) and the policy-side frame
index follows vision_lift/frames.py with vla_common.timing values passed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from vla_common.catalog import ALL_CUBES, COLORS, EDGE_M, PAD_EDGE_M, PAD_THICKNESS_M
from vla_common.tasks import (HELD_OUT_PLACE_PAIRS, HELD_OUT_STACK_PAIRS, HELD_OUT_TARGETS, SEEN_CUBES, STACK_ALLOWED,
                              TaskSpec, instruction, matches, matches_dest, sample_spec)

from .detectors import TASK_IDS, Detectors, Geometry, StepState
from .layout import TABLE_Z, layout_array, sample_layout


@dataclass
class Episode:
    index: int
    spec: TaskSpec
    instruction: str
    layout: dict
    layout_info: dict = field(default_factory=dict)
    layout_id: int | None = None  # paired streams only: group number g (same scene for every member)
    member: int | None = None  # paired streams only: position of this episode's target among the group's valid targets
    relation: str | None = None  # paired s3a / s3b only: RELATIONS type vs member 0 ("anchor" for member 0)


def make_stream(seed: int, start: int, n: int, make_spec, split: str = "train",
                cube_yaw_max_deg: float = 180.0) -> list[Episode]:
    """Episodes start .. start+n-1 of seed's stream. make_spec(rng) -> TaskSpec, or (TaskSpec, layout dict).
    cube_yaw_max_deg: layout.sample_layout (180 = U(-pi, pi), the default stream; specs / positions / instructions do
    not depend on it)."""
    out = []
    for i in range(start, start + n):
        rng = np.random.default_rng([seed, i])
        r = make_spec(rng)
        if isinstance(r, tuple):
            spec, lay, info = r[0], r[1], {}
        else:
            spec = r
            lay, info = sample_layout(spec, rng, cube_yaw_max_deg=cube_yaw_max_deg)
        out.append(Episode(i, spec, instruction(spec, rng, split), lay, info))
    return out


# --------------------------------------------------------------------------- paired (counterfactual) stream
PAIR_STAGES = {"s3a": "place", "s3b": "stack", "s3a1": "place", "s3b1": "stack"}  # two-noun paired streams: a member
# target is a (cube, destination) pair; s3a1 / s3b1 (task 009b) = the one-size-per-scene samplers, same member rule
PAIR_SELECT_KEY = 1000  # s3a / s3b member selection: rng([seed, g, PAIR_SELECT_KEY]) (instructions: rng([seed, g, m]))
MAX_PAIR_MEMBERS = 3
RELATIONS = ("same_target", "same_dest", "both_differ")  # s3a: same cube / other pad, other cube / same pad, both differ


def valid_targets(spec: TaskSpec, split: str) -> list:
    """Cubes of the scene that may be a pick target in `split` (the set vla_common.tasks._pick draws from), catalog order."""
    ok = HELD_OUT_TARGETS if split == "heldout_comp" else SEEN_CUBES
    return [c for c in ALL_CUBES if c in ok and c in [tuple(x) for x in spec.cubes]]


def valid_pairs(spec: TaskSpec) -> list[tuple]:
    """s3a / s3b, split train: every (cube, pad) / (top, bottom) of this table the stage sampler can draw for split train
    (tasks._sample_place / _sample_stack: manipulated cube in SEEN_CUBES, held-out pairs excluded; stack: top != bottom,
    STACK_ALLOWED, the bottom may be a held-out cube). Catalog order (cube, then pad colour / bottom cube)."""
    on = [c for c in ALL_CUBES if c in [tuple(x) for x in spec.cubes]]
    if spec.task == "place":
        return [(c, p) for c in on if c in SEEN_CUBES for p in COLORS
                if p in spec.pads and (c, p) not in HELD_OUT_PLACE_PAIRS]
    assert spec.task == "stack", spec.task
    return [(t, b) for t in on if t in SEEN_CUBES for b in on
            if b != t and (t, b) not in HELD_OUT_STACK_PAIRS and STACK_ALLOWED(t, b)]


def relation(a: tuple, b: tuple) -> str:
    """RELATIONS type of two distinct (target, destination) members."""
    assert a != b, a
    return "same_target" if a[0] == b[0] else "same_dest" if a[1] == b[1] else "both_differ"


def pair_members(seed: int, g: int, spec: TaskSpec) -> list[tuple]:
    """s3a / s3b members of group g: member 0 = the sampler's own (target, dest) (valid for split train by construction);
    then one uniform draw from the valid pairs with the same target and another destination, then one from those with
    another target and the same destination (a 3-member group then holds one pair of each RELATIONS type: m0-m1
    same_target, m0-m2 same_dest, m1-m2 both_differ); an empty class is skipped and the group is filled up to
    MAX_PAIR_MEMBERS by uniform draws from the remaining valid pairs (any type). Draws from rng([seed, g, PAIR_SELECT_KEY])
    in this order."""
    rng = np.random.default_rng([seed, g, PAIR_SELECT_KEY])
    m0 = (tuple(spec.target), tuple(spec.dest) if spec.task == "stack" else spec.dest)
    valid = valid_pairs(spec)
    assert m0 in valid, (m0, spec)
    out = [m0]
    for cls in ([x for x in valid if x[0] == m0[0] and x[1] != m0[1]], [x for x in valid if x[0] != m0[0] and x[1] == m0[1]]):
        if cls:
            out.append(cls[int(rng.integers(len(cls)))])
    rest = [x for x in valid if x not in out]
    while len(out) < MAX_PAIR_MEMBERS and rest:
        out.append(rest.pop(int(rng.integers(len(rest)))))
    return out


def _group_spec(seed: int, g: int, stage: str, split: str):
    rng = np.random.default_rng([seed, g])
    return sample_spec(stage, split, rng), rng


def _members(seed: int, g: int, stage: str, split: str, spec: TaskSpec) -> list:
    """Group g's member targets: s2a / s2 the valid pick targets (cubes); s3a / s3b pair_members ((target, dest))."""
    if stage in PAIR_STAGES:
        assert split == "train", f"paired {stage}: split train only, got {split}"
        return pair_members(seed, g, spec)
    return valid_targets(spec, split)


def make_paired_stream(seed: int, start_group: int, n_groups: int, stage: str = "s2a", split: str = "train",
                       cube_yaw_max_deg: float = 180.0) -> tuple[list[Episode], list[dict]]:
    """Counterfactual paired scenes: group g draws its pick spec and layout ONCE from rng([seed, g]) (the same draws as
    episode g of make_stream); one episode per valid target on the table (valid_targets, catalog order), member m with
    instruction(.., rng([seed, g, m])). Same cubes / positions / yaws / pads, so the t=0 images are identical.
    Groups with < 2 valid targets are skipped (returned as {"group", "n_valid"}). Episode.index = running count of the
    episodes of all earlier non-skipped groups from group 0 (specs only, no layout needed), so any batch of groups
    reproduces the indices of the full stream. Returns (episodes of groups start_group .. start_group+n_groups-1, skipped).
    stage s2a (one size, distinct colours) or s2 (2b amended: the s2 pick sampler, 2-5 cubes, colours AND sizes): a target
    is a (colour, size) cube, so two sizes of one colour are two members; held-out cubes on the table are never members.
    vla_common.tasks.instruction states the size whenever the colour is on the table twice; every member's instruction is
    asserted to name exactly its target among the table cubes (tasks.matches).
    stage s3a (place) / s3b (stack), split train (task 007): members = pair_members (2-3 distinct (cube, pad) / (top,
    bottom) pairs; held-out pairs and held-out manipulated cubes never members); Episode.relation = the member's RELATIONS
    type vs member 0 ("anchor" for member 0); every instruction is asserted to name exactly its target AND its
    destination (tasks.matches / matches_dest). Stage s3a1 / s3b1 (task 009b, one cube size per scene): exactly as s3a / s3b
    (valid_pairs on a one-size scene = the pairs the s3a1 / s3b1 sampler can draw for that scene)."""
    base = 0
    for g in range(start_group):
        n = len(_members(seed, g, stage, split, _group_spec(seed, g, stage, split)[0]))
        base += n if n >= 2 else 0
    out, skipped = [], []
    for g in range(start_group, start_group + n_groups):
        spec, rng = _group_spec(seed, g, stage, split)
        assert spec.task == PAIR_STAGES.get(stage, "pick"), spec.task
        tg = _members(seed, g, stage, split, spec)
        if len(tg) < 2:
            skipped.append({"group": g, "n_valid": len(tg)})
            continue
        lay, info = sample_layout(spec, rng, cube_yaw_max_deg=cube_yaw_max_deg)
        for m, t in enumerate(tg):
            if stage in PAIR_STAGES:
                ms = TaskSpec(spec.task, t[0], t[1], list(spec.cubes), list(spec.pads))
                ins = instruction(ms, np.random.default_rng([seed, g, m]), split)
                assert matches(ms, ins) == [t[0]] and matches_dest(ms, ins) == [t[1]], (g, m, ins, ms.cubes)
                out.append(Episode(base + len(out), ms, ins, dict(lay), dict(info), layout_id=g, member=m,
                                   relation="anchor" if m == 0 else relation(tg[0], t)))
                continue
            ms = TaskSpec("pick", t, None, list(spec.cubes), list(spec.pads))
            ins = instruction(ms, np.random.default_rng([seed, g, m]), split)
            assert matches(ms, ins) == [t], (g, m, ins, ms.cubes)
            out.append(Episode(base + len(out), ms, ins, dict(lay), dict(info), layout_id=g, member=m))
    return out, skipped


def geometry(z_rest: dict[str, float] | None = None, h_pad: float = PAD_THICKNESS_M) -> Geometry:
    """Detector geometry; z_rest per size (measured in 0A.1) or the nominal table top + half edge."""
    zr = z_rest or {s: TABLE_Z + EDGE_M[s] / 2 for s in EDGE_M}
    return Geometry(edge=torch.tensor([EDGE_M[s] for _, s in ALL_CUBES]),
                    z_rest=torch.tensor([zr[s] for _, s in ALL_CUBES]), h_pad=h_pad, l_pad=PAD_EDGE_M)


def spec_indices(specs: list[TaskSpec], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task = torch.tensor([TASK_IDS[s.task] for s in specs], device=device)
    target = torch.tensor([ALL_CUBES.index(tuple(s.target)) for s in specs], device=device)
    dest = torch.tensor([COLORS.index(s.dest) if s.task == "place" else ALL_CUBES.index(tuple(s.dest))
                         if s.task == "stack" else 0 for s in specs], device=device)
    return task, target, dest


# --------------------------------------------------------------------------- sim state
def _t(x):
    return x.torch if hasattr(x, "torch") else x


def read_state(u) -> dict[str, torch.Tensor]:
    """Everything the detectors and the expert read, env frame, one control step."""
    from .scene import CUBE_NAMES, PAD_NAMES, cube_contacts, gripper_width, tcp_pose

    o = u.scene.env_origins
    cp = torch.stack([_t(u.scene[n].data.root_pos_w) - o for n in CUBE_NAMES], 1)
    cq = torch.stack([_t(u.scene[n].data.root_quat_w) for n in CUBE_NAMES], 1)
    cv = torch.stack([_t(u.scene[n].data.root_lin_vel_w) for n in CUBE_NAMES], 1)
    pp = torch.stack([_t(u.scene[n].data.root_pos_w) - o for n in PAD_NAMES], 1)
    pq = torch.stack([_t(u.scene[n].data.root_quat_w) for n in PAD_NAMES], 1)
    c = cube_contacts(u)
    tcp, R = tcp_pose(u)
    term = u.action_manager.get_term("arm_action")
    fresh = term._need_init  # before the first action after a reset the term initialises its target from the arm
    cmd_t = torch.where(fresh[:, None], tcp, term.target).clone()
    cmd_y = torch.where(fresh, torch.atan2(R[:, 1, 0], R[:, 0, 0]), term.yaw).clone()
    return {"cube_pos": cp, "cube_quat": cq, "cube_vel": cv, "pad_pos": pp, "pad_quat": pq, "f_l": c["lf"],
            "f_r": c["rf"], "f_p": c["palm"], "f_a": c["arm"], "f_ht": c["hand_table_or_pad"], "f_ct": c["table"],
            "f_robot_table": c["robot_table"], "tcp": tcp, "hand_rot": R, "width": gripper_width(u),
            "finger_vel": _t(u.scene["robot"].data.joint_vel)[:, u.scene["robot"].find_joints("panda_finger.*")[0]],
            "invalid": u._nan_flag.clone(), "cmd_target": cmd_t, "cmd_yaw": cmd_y}


def to_step_state(s: dict, claw: torch.Tensor) -> StepState:
    keys = ("cube_pos", "cube_quat", "cube_vel", "pad_pos", "pad_quat", "f_l", "f_r", "f_p", "f_a", "f_ht", "f_ct",
            "f_robot_table", "tcp", "hand_rot", "invalid")
    return StepState(**{k: s[k] for k in keys}, claw=claw)


def quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rad, (-pi, pi]) of (x, y, z, w) quaternions (the sim convention here)."""
    return torch.atan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]), 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))


def fold_yaw_deg(y: torch.Tensor) -> torch.Tensor:
    """Cube yaw (deg) folded mod 90 into [-45, 45): the cube is 4-fold symmetric. +-45 and 135 fold to -45."""
    return torch.remainder(y + 45.0, 90.0) - 45.0


def to_snapshot(s: dict):
    from .expert import Snapshot

    q = s["cube_quat"]
    yaw = quat_yaw(q)
    R = s["hand_rot"]
    return Snapshot(cube_pos=s["cube_pos"], cube_yaw=yaw, pad_pos=s["pad_pos"], lf=s["f_l"], rf=s["f_r"],
                    tcp=s["tcp"], hand_yaw=torch.atan2(R[:, 1, 0], R[:, 0, 0]), width=s["width"],
                    finger_vel=s["finger_vel"], cmd_target=s["cmd_target"], cmd_yaw=s["cmd_yaw"])


# --------------------------------------------------------------------------- runner
class Runner:
    """Runs batches of stream episodes with the expert. Cameras: env built with them; capture=True stores frames."""

    def __init__(self, env, geo: Geometry, grasp_dz: torch.Tensor | None = None):
        self.env, self.u = env, env.unwrapped
        self.geo, self.grasp_dz = geo, grasp_dz
        self.aimed = False

    def run(self, episodes: list[Episode], T: int, expert_kw: dict | None = None, on_step=None,
            controller=None, stop_when_complete: bool = False) -> dict:
        """All episodes (in batches of num_envs). Returns per-episode outcome tensors (CPU) + 'index'.
        controller(u, batch_episodes, state0) -> object with act(t, snap_fn) -> (label, exec); default: the expert."""
        outs = []
        B = self.u.num_envs
        for b0 in range(0, len(episodes), B):
            batch = episodes[b0:b0 + B]
            pad = batch + [batch[-1]] * (B - len(batch))
            o = self.run_batch(pad, T, expert_kw or {}, on_step, controller, stop_when_complete)
            outs.append({k: v[: len(batch)].cpu() for k, v in o.items()})
        res = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
        res["index"] = torch.tensor([e.index for e in episodes])
        return res

    @torch.inference_mode()
    def run_batch(self, batch: list[Episode], T: int, expert_kw: dict, on_step=None, controller=None,
                  stop_when_complete: bool = False) -> dict:
        """stop_when_complete (timing passes only): stop once every episode has a completion step."""
        from vision_lift.cameras import aim_side

        from .expert import Expert
        from .scene import CUBE_MASS

        u, dev = self.u, self.u.device
        specs = [e.spec for e in batch]
        lay = torch.as_tensor(layout_array([e.layout for e in batch]), device=dev)
        u._layout = lay
        task, target, dest = spec_indices(specs, dev)
        u._ff_mass = torch.tensor([CUBE_MASS[s.target[1]] for s in specs], device=dev)
        self.env.reset()
        if "side_cam" in u.scene.keys() and not self.aimed:
            aim_side(u)
            self.aimed = True
        s = read_state(u)
        ar = torch.arange(len(batch), device=dev)
        spawn = torch.rad2deg(quat_yaw(s["cube_quat"][ar, target]))  # target cube, true state, before the first action
        det = Detectors(task, target, dest, lay[:, :8, 4] > 0.5, self.geo, grasp_dz=self.grasp_dz)
        if controller is None:
            ctrl = Expert(task, target, dest, self.geo.edge, TABLE_Z, self.geo.h_pad, u.step_dt, **expert_kw)
            ctrl.reset(to_snapshot(s))
        else:
            ctrl = controller(u, batch, s)
        self.ctrl = ctrl
        for t in range(T):
            label, exe = ctrl.act(t, lambda: to_snapshot(s))
            if on_step is not None:
                on_step(t, u, s, label, exe, ctrl, det)
            self.env.step(exe)
            s = read_state(u)
            det.update(to_step_state(s, exe[:, 4]))
            if stop_when_complete and t % 50 == 49 and bool((det.completion_step >= 0).all()):
                break
        out = det.outcome()
        out["final_cube_pos"] = s["cube_pos"]
        out["spawn_yaw_deg"], out["spawn_yaw_fold_deg"] = spawn, fold_yaw_deg(spawn)
        return out
