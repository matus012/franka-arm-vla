"""113 success / failure detectors, exactly as pre-registered in GATES.md "Conventions / Success and failure detectors".

Pure torch on per-step tensors (no simulator; unit-tested in tests/test_detectors.py with crafted states). They read only
the true sim state (object poses, velocities, contact forces, TCP, claw command), never an expert's phase machine; the
only latches are on raw contact forces (grasp confirmation). Usable on any controller (expert or policy).

Usage: d = Detectors(task, target, dest, on_table, geo, grasp_dz); d.update(step) once per control step after the env
step (step = StepState); d.outcome() at the end -> per-episode tensors. `d.success_now` is the task's success condition
evaluated as if the current step were the last one (completion step = first step where it is true).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from lift_rl.metrics import Thresholds

DETECTOR_VERSION = "113-detectors-v1 (GATES.md conventions 2026-10-01)"
PICK, PLACE, STACK = 0, 1, 2
TASK_IDS = {"pick": PICK, "place": PLACE, "stack": STACK}


@dataclass
class Consts113:
    """GATES.md 113 constants (pre-registered). The 111 ones come from lift_rl.metrics.Thresholds + PhaseParams."""

    eps_z: float = 0.005  # rest tolerance
    flat_deg: float = 10.0
    still_v: float = 0.02  # m/s
    k_place: int = 10
    k_stack: int = 50
    clear_dz: float = 0.05
    release_zone_dz: float = 0.03
    wrong_grasp_steps: int = 5
    pick_dz: float = 0.100  # pick: z_target - z_rest >= 10 cm
    off_support_dz: float = 0.02  # 111 off_table_dz
    grasp_confirm_n: float = 5.0  # 111 PhaseParams.pad_confirm_n
    drop_steps: int = 5  # 111 PhaseParams.drop_steps (drop = not held for > 5 steps)
    disturbed_m: float = 0.02
    band_xy: float = 0.015  # 111 PhaseParams sweet_xy / sweet_dz / tilt_ok / yaw_ok (early-close band)
    band_dz: float = 0.015
    band_tilt_deg: float = 10.0
    band_yaw_deg: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Geometry:
    """Per-cube geometry (rows = CUBE_NAMES order) + pads, measured in 0A.1."""

    edge: torch.Tensor  # (8,) e_c
    z_rest: torch.Tensor  # (8,) resting centre height on the table, env frame
    h_pad: float  # pad top above the table top
    l_pad: float  # pad edge


@dataclass
class StepState:
    """One control step, all in the env frame. N envs, 8 cubes (CUBE_NAMES order), 4 pads (PAD_NAMES order)."""

    cube_pos: torch.Tensor  # (N, 8, 3)
    cube_quat: torch.Tensor  # (N, 8, 4) xyzw
    cube_vel: torch.Tensor  # (N, 8, 3) linear velocity
    pad_pos: torch.Tensor  # (N, 4, 3)
    pad_quat: torch.Tensor  # (N, 4, 4) xyzw
    f_l: torch.Tensor  # (N, 8) left finger pad vs cube
    f_r: torch.Tensor  # (N, 8)
    f_p: torch.Tensor  # (N, 8) palm (panda_hand body)
    f_a: torch.Tensor  # (N, 8) max over arm links
    f_ht: torch.Tensor  # (N,) max hand / finger force vs the table or any pad
    f_ct: torch.Tensor  # (N, 8) cube vs table (reported only; never used for slam)
    f_robot_table: torch.Tensor  # (N,) max force of any robot body vs the table (clean grasp c1)
    tcp: torch.Tensor  # (N, 3) fingertip centre
    hand_rot: torch.Tensor  # (N, 3, 3) hand rotation matrix (columns = hand x, y, z in the env frame)
    claw: torch.Tensor  # (N,) executed claw command (> 0.5 = close)
    invalid: torch.Tensor | None = None  # (N,) NaN-guard flag this step


# --------------------------------------------------------------------------- geometry helpers
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion(s) (..., 4) -> rotation matrices (..., 3, 3)."""
    x, y, z, w = q.unbind(-1)
    n = (x * x + y * y + z * z + w * w).clamp_min(1e-12)
    s = 2.0 / n
    return torch.stack([
        1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
        s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
        s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def to_frame(p: torch.Tensor, origin: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Point(s) p (..., 3) expressed in the frame (origin (..., 3), rot (..., 3, 3) columns = frame axes)."""
    return (rot.transpose(-1, -2) @ (p - origin)[..., None])[..., 0]


def tilt_deg(quat: torch.Tensor) -> torch.Tensor:
    """Angle between world z and the cube body axis closest to it [deg]."""
    R = quat_to_mat(quat)
    return torch.rad2deg(torch.arccos(R[..., 2, :].abs().amax(-1).clamp(max=1.0)))


def in_pad(c_pos: torch.Tensor, pad_pos: torch.Tensor, pad_quat: torch.Tensor, l_pad: float) -> torch.Tensor:
    """(N, 8, 4): cube centre xy in pad frame within +-L_pad/2."""
    local = to_frame(c_pos[:, :, None, :], pad_pos[:, None, :, :], quat_to_mat(pad_quat)[:, None])
    return (local[..., 0].abs() <= l_pad / 2) & (local[..., 1].abs() <= l_pad / 2)


def on_top(c_pos: torch.Tensor, c_quat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    """(N, 8c, 8b): c's centre xy in b's body frame within +-e_b/2 and z_c > z_b; never c == b."""
    local = to_frame(c_pos[:, :, None, :], c_pos[:, None, :, :], quat_to_mat(c_quat)[:, None])
    h = edge[None, None, :] / 2
    out = (local[..., 0].abs() <= h) & (local[..., 1].abs() <= h) & (c_pos[:, :, None, 2] > c_pos[:, None, :, 2])
    return out & ~torch.eye(c_pos.shape[1], dtype=torch.bool, device=c_pos.device)[None]


def yaw_err_mod90_deg(hand_rot: torch.Tensor, cube_quat: torch.Tensor) -> torch.Tensor:
    """(N,) angle between the hand closing axis (hand y) and the nearest cube face normal, in the xy plane, mod 90 [deg]."""
    Rc = quat_to_mat(cube_quat)
    a = torch.atan2(hand_rot[:, 1, 1], hand_rot[:, 0, 1])
    b = torch.atan2(Rc[:, 1, 0], Rc[:, 0, 0])
    d = torch.remainder(a - b + math.pi / 4, math.pi / 2) - math.pi / 4
    return torch.rad2deg(d.abs())


def hand_tilt_deg(hand_rot: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.arccos((-hand_rot[:, 2, 2]).clamp(-1.0, 1.0)))


# --------------------------------------------------------------------------- per-step predicates
class Predicates:
    """All GATES.md per-step predicates for one step, (N, 8) unless noted."""

    def __init__(self, s: StepState, geo: Geometry, th: Thresholds, k: Consts113):
        C = th.contact_n
        e, zr = geo.edge.to(s.cube_pos.device), geo.z_rest.to(s.cube_pos.device)
        z = s.cube_pos[..., 2]
        self.held = (s.f_l > C) & (s.f_r > C)
        self.released = (s.f_l <= C) & (s.f_r <= C) & (s.f_p <= C) & (s.f_a <= C)
        self.robot_touch = (s.f_l > C) | (s.f_r > C) | (s.f_p > C) | (s.f_a > C)
        tilt = tilt_deg(s.cube_quat)
        speed = torch.linalg.norm(s.cube_vel, dim=-1)
        flat_still = (tilt < k.flat_deg) & (speed < k.still_v)
        self.in_pad = in_pad(s.cube_pos, s.pad_pos, s.pad_quat, geo.l_pad)  # (N, 8, 4)
        self.s_tab = zr[None].expand_as(z)
        self.s_pad = self.s_tab + geo.h_pad
        self.rests_on_pad = self.in_pad & ((z - self.s_pad).abs() < k.eps_z)[..., None] & flat_still[..., None]  # (N,8,4)
        self.rests_on_table = (((z - self.s_tab).abs() < k.eps_z) | self.rests_on_pad.any(-1)) & flat_still
        self.on_top = on_top(s.cube_pos, s.cube_quat, e)  # (N, 8c, 8b)
        self.s_b = s.cube_pos[:, None, :, 2] + e[None, None, :] / 2 + e[None, :, None] / 2  # (N, 8c, 8b)
        still_b = speed < k.still_v
        self.rests_on_cube = (self.on_top & ((z[:, :, None] - self.s_b).abs() < k.eps_z) & flat_still[:, :, None]
                              & still_b[:, None, :] & self.rests_on_table[:, None, :])  # (N, 8c, 8b)
        self.clear = s.tcp[:, None, 2] >= z + e[None] / 2 + k.clear_dz
        sup = torch.maximum(self.s_tab, torch.where(self.in_pad.any(-1), self.s_pad, self.s_tab))
        sup = torch.maximum(sup, torch.where(self.on_top, self.s_b, torch.full_like(self.s_b, -1e9)).amax(-1))
        self.support = sup
        self.elevated = z > sup + k.off_support_dz
        self.confirm = (s.f_l > k.grasp_confirm_n) & (s.f_r > k.grasp_confirm_n) & (s.f_p <= C)
        self.z = z


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (N, K, ...) -> x[n, idx[n]]."""
    return x[torch.arange(x.shape[0], device=x.device), idx]


class Detectors:
    """Streaming per-episode detectors for N envs. task (N,) in {0 pick, 1 place, 2 stack}; target (N,) cube index;
    dest (N,) pad index (place) / bottom cube index (stack) / anything (pick); on_table (N, 8) cubes on the table."""

    def __init__(self, task: torch.Tensor, target: torch.Tensor, dest: torch.Tensor, on_table: torch.Tensor, geo: Geometry,
                 grasp_dz: torch.Tensor | None = None, th: Thresholds | None = None, k: Consts113 | None = None):
        self.th, self.k, self.geo = th or Thresholds(), k or Consts113(), geo
        dev = task.device
        self.N = n = task.shape[0]
        self.task, self.target, self.dest, self.on_table = task.long(), target.long(), dest.long(), on_table.bool()
        self.is_pick, self.is_place, self.is_stack = self.task == PICK, self.task == PLACE, self.task == STACK
        self.grasp_dz = grasp_dz.to(dev) if grasp_dz is not None else torch.zeros(8, device=dev)  # per cube (size)
        z = lambda dt=torch.bool: torch.zeros(n, dtype=dt, device=dev)  # noqa: E731
        self.t = 0
        self.pick_run, self.place_run, self.stack_run = z(torch.long), z(torch.long), z(torch.long)
        self.wrong_place_run = torch.zeros(n, 4, dtype=torch.long, device=dev)
        self.wrong_stack_run = torch.zeros(n, 8, dtype=torch.long, device=dev)
        self.held_run = torch.zeros(n, 8, dtype=torch.long, device=dev)
        self.wrong_grasp = z()
        self.wrong_grasp_cube = torch.full((n,), -1, dtype=torch.long, device=dev)  # first triggering non-target cube
        self.rest_pad_run = torch.zeros(n, 8, 4, dtype=torch.long, device=dev)
        self.rest_stack_run = torch.zeros(n, 8, 8, dtype=torch.long, device=dev)
        self.rest_cube = torch.full((n,), -1, dtype=torch.long, device=dev)  # place / stack: latest settled (cube, destination)
        self.rest_dest = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.slam = z()
        self.slam_sens = {f: z() for f in self.th.slam_n_sensitivity}
        self.max_ht = z(torch.float32)
        self.jam_run, self.max_jam_run, self.jam = z(torch.long), z(torch.long), z()
        self.reached_dest_held = z()
        self.release_rest_seen = z()  # stack: released while rests_on_cube(target, bottom)
        self.rest_run_since = z(torch.long)  # consecutive rests_on_cube(target, bottom) steps since that moment
        self.stable_after_release = z()
        self.confirmed = z()
        self.confirm_dz = torch.full((n,), float("nan"), device=dev)
        self.loss_run, self.drop = z(torch.long), z()
        self.prev_claw = z()
        self.early_close = z()
        self.xy0: torch.Tensor | None = None
        self.disturbed = z()
        self.invalid = z()
        self.success_now = z()
        self.completion_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        # clean grasp (111 c1-c4, per-size rest height)
        self.max_robot_table = z(torch.float32)
        self.tilt_violation = z()
        self.liftoff, self.liftoff_both = z(), z()
        self.palm_ever = z()
        self.last: dict[str, torch.Tensor] = {}

    def update(self, s: StepState) -> None:
        k, th, N = self.k, self.th, self.N
        P = Predicates(s, self.geo, th, k)
        ar = torch.arange(N, device=s.tcp.device)
        tg, ds = self.target, self.dest
        dpad = ds.clamp(0, 3)
        dcube = ds.clamp(0, 7)
        if s.invalid is not None:
            self.invalid |= s.invalid.bool()
        if self.xy0 is None:
            self.xy0 = s.cube_pos[..., :2].clone()

        held_t = _gather(P.held, tg)
        rel_t = _gather(P.released, tg)
        z_t = _gather(P.z, tg)
        zr_t = self.geo.z_rest.to(z_t.device)[tg]
        # ---- success conditions (as if this step were the last)
        pick_ok = held_t & (z_t - zr_t >= k.pick_dz)
        self.pick_run = torch.where(pick_ok, self.pick_run + 1, torch.zeros_like(self.pick_run))
        rop = P.rests_on_pad[ar, tg]  # (N, 4)
        place_ok = rop[ar, dpad] & rel_t
        self.place_run = torch.where(place_ok, self.place_run + 1, torch.zeros_like(self.place_run))
        roc = P.rests_on_cube[ar, tg]  # (N, 8 b)
        rel_b = P.released  # (N, 8)
        stack_ok = roc[ar, dcube] & rel_t & rel_b[ar, dcube]
        self.stack_run = torch.where(stack_ok, self.stack_run + 1, torch.zeros_like(self.stack_run))
        clear_t = _gather(P.clear, tg)
        succ = torch.where(self.is_pick, pick_ok,
                           torch.where(self.is_place, (self.place_run >= k.k_place) & clear_t,
                                       (self.stack_run >= k.k_stack) & clear_t))
        self.success_now = succ & ~self.invalid
        self.completion_step = torch.where((self.completion_step < 0) & self.success_now,
                                           torch.full_like(self.completion_step, self.t), self.completion_step)
        self.clear_now = clear_t
        self.rest_dest_now = torch.where(self.is_place, rop[ar, dpad], roc[ar, dcube])
        self.held_t_now = held_t
        # ---- wrong object
        nontarget = self.on_table.clone()
        nontarget[ar, tg] = False
        self.held_run = torch.where(P.held, self.held_run + 1, torch.zeros_like(self.held_run))
        wg = ((self.held_run >= k.wrong_grasp_steps) | (P.elevated & P.robot_touch)) & nontarget
        self.wrong_grasp |= wg.any(-1)
        self.wrong_grasp_cube = torch.where((self.wrong_grasp_cube < 0) & wg.any(-1), wg.long().argmax(-1),
                                            self.wrong_grasp_cube)  # lowest cube index among a step's first triggers
        other_pad = torch.ones(N, 4, dtype=torch.bool, device=ar.device)
        other_pad[ar, dpad] = False
        self.wrong_place_run = torch.where(rop & rel_t[:, None], self.wrong_place_run + 1, torch.zeros_like(self.wrong_place_run))
        other_b = self.on_table.clone()
        other_b[ar, dcube] = False
        other_b[ar, tg] = False
        self.wrong_stack_run = torch.where(roc & rel_t[:, None] & rel_b, self.wrong_stack_run + 1,
                                           torch.zeros_like(self.wrong_stack_run))
        self.wrong_dest_now = torch.where(
            self.is_place, ((self.wrong_place_run >= k.k_place) & other_pad).any(-1) & clear_t,
            torch.where(self.is_stack, ((self.wrong_stack_run >= k.k_stack) & other_b).any(-1),
                        torch.zeros_like(clear_t)))
        # ---- where a released, moved cube settled (any cube, any pad / bottom cube; latest wins; probe s3a / s3b)
        mv = torch.linalg.norm(s.cube_pos[..., :2] - self.xy0, dim=-1) > k.disturbed_m  # (N, 8)
        self.rest_pad_run = torch.where(P.rests_on_pad & (rel_b & mv)[..., None], self.rest_pad_run + 1,
                                        torch.zeros_like(self.rest_pad_run))
        self.rest_stack_run = torch.where(P.rests_on_cube & (rel_b & mv)[:, :, None] & rel_b[:, None, :],
                                          self.rest_stack_run + 1, torch.zeros_like(self.rest_stack_run))
        hp = (self.rest_pad_run >= k.k_place).flatten(1)  # (N, 32) = cube * 4 + pad
        hs = (self.rest_stack_run >= k.k_stack).flatten(1)  # (N, 64) = cube * 8 + bottom
        fp, fs = hp.long().argmax(-1), hs.long().argmax(-1)
        upd = (self.is_place & hp.any(-1)) | (self.is_stack & hs.any(-1))
        self.rest_cube = torch.where(upd, torch.where(self.is_place, fp // 4, fs // 8), self.rest_cube)
        self.rest_dest = torch.where(upd, torch.where(self.is_place, fp % 4, fs % 8), self.rest_dest)
        # ---- slam
        self.max_ht = torch.maximum(self.max_ht, s.f_ht)
        self.slam |= s.f_ht > th.slam_n
        for f in self.slam_sens:
            self.slam_sens[f] |= s.f_ht > f
        # ---- jam
        jam_step = ((s.f_p > th.contact_n) | (P.elevated & P.robot_touch & ~P.held)).any(-1)
        self.jam_run = torch.where(jam_step, self.jam_run + 1, torch.zeros_like(self.jam_run))
        self.max_jam_run = torch.maximum(self.max_jam_run, self.jam_run)
        self.jam |= self.jam_run >= th.jam_steps
        # ---- destination zone, release failure, stack stability
        in_pad_t = P.in_pad[ar, tg, dpad]
        dz_place = in_pad_t & (z_t <= _gather(P.s_pad, tg) + k.release_zone_dz)
        sb_t = P.s_b[ar, tg, dcube]
        dz_stack = P.on_top[ar, tg, dcube] & (z_t <= sb_t + k.release_zone_dz)
        dest_zone = torch.where(self.is_place, dz_place, torch.where(self.is_stack, dz_stack, torch.zeros_like(dz_place)))
        self.reached_dest_held |= held_t & dest_zone & ~self.is_pick
        roc_tb = roc[ar, dcube]
        start = self.is_stack & ~self.release_rest_seen & rel_t & roc_tb
        self.release_rest_seen |= start
        self.rest_run_since = torch.where(self.release_rest_seen & roc_tb, self.rest_run_since + 1,
                                          torch.zeros_like(self.rest_run_since))
        self.stable_after_release |= self.rest_run_since >= k.k_stack
        # ---- early close (claw 0 -> 1, or 1 at step 0, outside the target's grasp band)
        closed = s.claw > 0.5
        rise = closed & ~self.prev_claw
        cq = s.cube_quat[ar, tg]
        cp = s.cube_pos[ar, tg]
        band = ((hand_tilt_deg(s.hand_rot) < k.band_tilt_deg) & (yaw_err_mod90_deg(s.hand_rot, cq) < k.band_yaw_deg)
                & (torch.linalg.norm(s.tcp[:, :2] - cp[:, :2], dim=-1) < k.band_xy)
                & ((s.tcp[:, 2] - (cp[:, 2] + self.grasp_dz[tg])).abs() < k.band_dz))
        self.early_close |= rise & ~band
        self.prev_claw = closed
        # ---- grasp confirmation (latched on raw forces) and drop
        conf_t = _gather(P.confirm, tg)
        newc = conf_t & ~self.confirmed
        self.confirm_dz = torch.where(newc, s.tcp[:, 2] - z_t, self.confirm_dz)
        self.confirmed |= conf_t
        lost = self.confirmed & ~held_t & ~dest_zone
        self.loss_run = torch.where(lost, self.loss_run + 1, torch.zeros_like(self.loss_run))
        self.drop |= self.loss_run > k.drop_steps
        # ---- distractor disturbed (non-target, non-destination cubes on the table)
        moved = torch.linalg.norm(s.cube_pos[..., :2] - self.xy0, dim=-1) > k.disturbed_m
        dist_set = nontarget.clone()
        dist_set[ar, dcube] &= ~self.is_stack
        self.disturbed |= (moved & dist_set).any(-1)
        # ---- clean grasp c1-c4 (111 definitions, per-size rest height)
        self.max_robot_table = torch.maximum(self.max_robot_table, s.f_robot_table)
        self.palm_ever |= _gather(s.f_p, tg) > th.contact_n
        new_lift = ~self.liftoff & (z_t > zr_t + th.liftoff_dz)
        both1 = (_gather(s.f_l, tg) > th.contact_n) & (_gather(s.f_r, tg) > th.contact_n)
        self.liftoff_both |= new_lift & both1
        self.liftoff |= new_lift
        in_zone = ~self.liftoff & (s.tcp[:, 2] < zr_t + th.clean_descent_zone)
        self.tilt_violation |= in_zone & (hand_tilt_deg(s.hand_rot) > th.clean_tilt_deg)
        self.t += 1

    def outcome(self) -> dict[str, torch.Tensor]:
        k = self.k
        ok = ~self.invalid
        succ = self.success_now & ok
        hold50 = self.pick_run >= k.k_stack  # 0A.1 "holds": pick condition on every one of the last 50 steps
        notpick = ~self.is_pick
        rel_fail = self.reached_dest_held & ~succ & notpick
        return {
            "success": succ,
            "pick_hold_last50": hold50 & ok & self.is_pick,
            "completion_step": self.completion_step,
            "wrong_grasp": self.wrong_grasp,
            "wrong_grasp_cube": self.wrong_grasp_cube,  # ALL_CUBES index, -1 = none (confusion type, Stage 2 report)
            "rest_cube": self.rest_cube,  # place / stack: ALL_CUBES index of the cube that last settled released on a pad /
            "rest_dest": self.rest_dest,  # bottom cube, and that pad (COLORS index) / bottom cube (ALL_CUBES index); -1 = none
            "wrong_dest": self.wrong_dest_now & notpick,
            "wrong_object": self.wrong_grasp | (self.wrong_dest_now & notpick),
            "slam": self.slam,
            **{f"slam_{int(f)}N": v for f, v in self.slam_sens.items()},
            "max_hand_table_or_pad_force": self.max_ht,
            "jam": self.jam,
            "max_jam_run": self.max_jam_run,
            "release_failure": rel_fail,
            "release_fail_not_released": rel_fail & self.held_t_now,
            "release_fail_off_destination": rel_fail & ~self.rest_dest_now,
            "release_fail_not_clear": rel_fail & ~self.clear_now,
            "release_fail_unstable": rel_fail & self.is_stack & self.release_rest_seen & (self.stack_run < k.k_stack),
            "stack_release_rest_seen": self.release_rest_seen & self.is_stack,
            "stack_stable_after_release": self.stable_after_release & self.release_rest_seen & self.is_stack,
            "early_close": self.early_close,
            "grasp_confirmed": self.confirmed,
            "confirm_tcp_minus_cube_z": self.confirm_dz,
            "drop": self.drop,
            "distractor_disturbed": self.disturbed,
            "clean": ((self.max_robot_table <= self.th.clean_table_n) & ~self.tilt_violation & self.liftoff
                      & self.liftoff_both & ~self.palm_ever & ok),
            "clean_c1_no_table_hit": self.max_robot_table <= self.th.clean_table_n,
            "clean_c2_top_down": ~self.tilt_violation,
            "clean_c3_both_pads_at_liftoff": self.liftoff & self.liftoff_both,
            "clean_c4_no_palm": ~self.palm_ever,
            "invalid": self.invalid,
        }
