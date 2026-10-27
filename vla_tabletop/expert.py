"""113 expert (brief section 5, GATES 0A.2 / 0A.3): the 111 scripted controller extended to pick among distractors,
place on a pad and stack, with per-cube sizes, acting ONLY through the contract action (5,) = 4 task-space deltas (the
112 StudentIKAction: 1 cm / 3 deg per step, hand vertical) + binary claw, like `lift_rl.task_space.scripted_delta_policy`.

Kept from 111 (`lift_rl.scripted.ScriptedController`): the phase machine on true positions (hover 10 cm above the cube
top -> vertical descent to the cube centre -> close until both pads touch -> lift), its gates (hover xy < 5 mm, yaw <
5 deg for 5 steps; pads > 1 N for 3 steps; 0.5 s close timeout; 0.3 s settle), and its acceleration-limited setpoint
profile (speed caps per phase, A_MAX) which becomes per-step deltas of the action term's target. New in 113:
- per-cube half edge / rest height (never the 111 constant 0.0207); target chosen by the TaskSpec;
- lift 15 cm; pick: hold there with the claw closed;
- place / stack: carry at the lift height above the destination (stack: hand yaw re-aligned to the bottom cube's faces),
  lower until the cube bottom is RELEASE_DZ above the support (pad top, or bottom-cube top = its centre + half edge),
  open the claw (RELEASE: until the fingers are open or 0.6 s), retreat to 10 cm above the cube top, wait.

Observation model (GATES 0A.3 "honest emulation"): the expert only ever reads a SNAPSHOT of the true state (objects:
poses, contacts; own arm: TCP, hand yaw, finger width). fresh=True = the unchunked expert (0A.2): a fresh snapshot every
step. Chunked (fresh=False): snapshots are stored at camera ticks (t % period == 0); at query steps t = 0, N, 2N, ... it
takes the snapshot of step period*max(t//period - latency, 0) (the frame the policy sees at t; vla_common.timing PERIOD /
LATENCY passed explicitly; N = 1, period 1, latency 0 reproduces fresh) and emits the next N actions open loop:
the snapshot also holds the action term's commanded target / yaw at the frame step (GATES 0A.3 attempt 1): the expert
re-anchors its target estimate on it at every query and rolls it forward with its own executed deltas and the workspace
box clamp only (no lead-clamp model between queries); predicted TCP = snapshot TCP + its own commanded target change
since the frame step, predicted hand yaw = snapshot yaw + its own commanded yaw change,
objects are static except the target moving with the predicted TCP while predicted held, and contact-gated transitions
(close -> lift) fire only at query steps. The env's IK still tracks every delta at 50 Hz on the true arm.

The commanded target / yaw are dead-reckoned from the expert's own EXECUTED actions with the action term's own update
rule (box clamp, then target at most 5 cm ahead of the TCP, using the expert's TCP estimate; initialised from the arm
at step 0) - with a fresh snapshot this reproduces the term's target exactly. Action noise: executed = clip(label + N(0, sigma) on the 4 arm dims), claw not
noised; the label = the clean action clipped to [-1, 1] (contract v2 a). sigma = 0 in every gate run.
"""

from __future__ import annotations

import math

import torch

from lift_rl.scripted import ScriptedController as SC
from lift_rl.task_space import BOX_HI, BOX_LO, LEAD_MAX, STEP_POS, STEP_YAW
from vision_lift.frames import frame_step

from .detectors import PICK, PLACE, STACK

EXPERT_VERSION = "113-expert-v1"
HOVER, DESCEND, CLOSE, LIFT, CARRY, LOWER, RELEASE, RETREAT, DONE = range(9)
PHASES = ("hover", "descend", "close", "lift", "carry", "lower", "release", "retreat", "done")
LIFT_DZ = 0.15  # lift the cube 15 cm (GATES 0A.1); also the carry height for place / stack
RELEASE_DZ = 0.005  # cube bottom this far above its support when the claw opens
RETREAT_DZ = 0.10  # final TCP height above the released cube's top (clear needs 0.05)
RELEASE_MAX_STEPS = 30
OPEN_MARGIN = 0.01  # fingers count as open when the width > cube edge + 1 cm
CARRY_YAW_TOL = math.radians(3.0)
FINGER_STILL = 0.005  # m/s, both finger joints
STILL_STEPS = 5
CLOSE_MAX = 50  # 1 s (111: 0.5 s; the settle wait needs longer)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class Snapshot(dict):
    """keys: cube_pos (N,8,3), cube_yaw (N,8), pad_pos (N,4,3), lf / rf (N,8), tcp (N,3), hand_yaw (N,), width (N,),
    finger_vel (N,2) finger joint velocities [m/s], cmd_target (N,3) / cmd_yaw (N,) the action term's commanded target."""


class Expert:
    """Batched expert for N envs. task / target / dest (N,) as in detectors (dest = pad index or bottom cube index);
    edge (8,) cube edges; table_z, h_pad = support heights. Call reset(snap0) at step 0, then act(t, snap_fn) every step
    -> (label (N,5), executed (N,5)). snap_fn() returns the CURRENT true snapshot; the expert decides when to read it."""

    def __init__(self, task: torch.Tensor, target: torch.Tensor, dest: torch.Tensor, edge: torch.Tensor, table_z: float,
                 h_pad: float, dt: float, fresh: bool = True, n_query: int = 1, period: int = 3, latency: int = 1,
                 sigma: float = 0.0, noise_seed: int = 0, lift_trigger: str = "width"):
        self.N, self.dev = task.shape[0], task.device
        self.task, self.target, self.dest = task.long(), target.long(), dest.long()
        self.edge = edge.to(self.dev)
        self.table_z, self.h_pad, self.dt = table_z, h_pad, dt
        self.n_query, self.period, self.latency = n_query, period, latency
        self.chunked = not fresh
        self.sigma = sigma
        # "width" (default since 0A.2 attempt 1, expert fix 1 of 2, GATES 18a2524): lift when both pads touch for
        # PAD_STEPS and the gripper-width rate (sum of the two finger velocities, visible to a policy through joint_vel)
        # is < FINGER_STILL for STILL_STEPS, else at the CLOSE_MAX timeout; the sideways shuttle of cube + fingers leaves
        # the width unchanged (results/expert_close_diag.json). "settle" (0A.2 attempt 0 / 0A.3 attempt 1 expert,
        # explicit flag only): each finger joint's speed < FINGER_STILL instead.
        assert lift_trigger in ("settle", "width"), lift_trigger
        self.lift_trigger = lift_trigger
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(noise_seed)
        self.lo = torch.tensor(BOX_LO, device=self.dev)
        self.hi = torch.tensor(BOX_HI, device=self.dev)
        self.ar = torch.arange(self.N, device=self.dev)

    # ------------------------------------------------------------------ state
    def reset(self, snap0: Snapshot) -> None:
        N, dev = self.N, self.dev
        z = lambda dt=torch.long: torch.zeros(N, dtype=dt, device=dev)  # noqa: E731
        self.phase, self.phase_t = z(), z()
        self.hover_run, self.pad_run, self.at_goal_t, self.low_t, self.still_run = z(), z(), z(), z(), z()
        self.sp = snap0["tcp"].clone()
        self.v = torch.zeros(N, 3, device=dev)
        self.target_cmd = snap0["tcp"].clone()  # dead-reckoned action-term target (= term.target at reset)
        self.yaw_cmd = snap0["hand_yaw"].clone()  # = term.yaw at reset
        self.grasp_z, self.goal_z = z(torch.float32), z(torch.float32)
        self.lift_goal = torch.zeros(N, 3, device=dev)
        self.offset = torch.zeros(N, 3, device=dev)
        self.psi_hold = snap0["hand_yaw"].clone()
        self.exe_hist: dict[int, torch.Tensor] = {}
        self.snaps: dict[int, Snapshot] = {}
        self.snap: Snapshot | None = None
        self.snap_step = 0
        self.close_exit = z()
        self.entered = torch.full((N, len(PHASES)), -1, dtype=torch.long, device=dev)
        self.entered[:, HOVER] = 0
        self.t = 0

    def _advance(self, mask: torch.Tensor, to: int, frm: int | None = None) -> torch.Tensor:
        m = mask & (self.phase == (to - 1 if frm is None else frm))
        self.phase = torch.where(m, torch.full_like(self.phase, to), self.phase)
        self.phase_t = torch.where(m, torch.zeros_like(self.phase_t), self.phase_t)
        self.entered[:, to] = torch.where(m & (self.entered[:, to] < 0), torch.full_like(self.phase, self.t),
                                          self.entered[:, to])
        return m

    # ------------------------------------------------------------------ observation model
    def _observe(self, t: int, snap_fn) -> tuple[Snapshot, bool]:
        """(snapshot used at step t, is_query). Unchunked: fresh every step. Chunked: camera-tick snapshots stored,
        the one of frame_step(t) taken at query steps and held until the next query."""
        if not self.chunked:
            self.snap, self.snap_step = snap_fn(), t
            return self.snap, True
        if t % self.period == 0:
            self.snaps[t] = snap_fn()  # the camera tick's true state (what a frame captured now would show)
        query = t % self.n_query == 0
        if query:
            fs = frame_step(t, self.period, self.latency)
            self.snap, self.snap_step = self.snaps[fs], fs
            cmd = self.snaps[fs]  # the action term's commanded target / yaw at the frame step (GATES 0A.3 attempt 1)
            self.snaps = {k: v for k, v in self.snaps.items() if k >= fs}
            tc, yc = cmd["cmd_target"].clone(), cmd["cmd_yaw"].clone()  # re-anchor, roll forward with own deltas + box
            for k in range(fs, t):
                e = self.exe_hist[k]
                tc = torch.maximum(torch.minimum(tc + e[:, :3] * STEP_POS, self.hi), self.lo)
                yc = yc + e[:, 3] * STEP_YAW
            self.anchor = (cmd["cmd_target"], cmd["cmd_yaw"])
            self.target_cmd, self.yaw_cmd = tc, yc
        return self.snap, query

    # ------------------------------------------------------------------ control
    @torch.inference_mode()
    def act(self, t: int, snap_fn) -> tuple[torch.Tensor, torch.Tensor]:
        assert t == self.t
        snap, query = self._observe(t, snap_fn)
        if not self.chunked:
            p, psi_now = snap["tcp"], snap["hand_yaw"]
        else:  # snapshot TCP / yaw + its own commanded change since the frame step (no sim reads between queries)
            p = snap["tcp"] + (self.target_cmd - self.anchor[0])
            psi_now = snap["hand_yaw"] + (self.yaw_cmd - self.anchor[1])
        ar, tg = self.ar, self.target
        e_t = self.edge[tg]
        half_t = e_t / 2
        cube = snap["cube_pos"][ar, tg].clone()
        held_pred = (self.phase >= LIFT) & (self.phase <= LOWER)
        cube = torch.where(held_pred[:, None], cube + (p - snap["tcp"]), cube)
        cube_yaw = snap["cube_yaw"][ar, tg]
        is_pick = self.task == PICK
        # destination support top and xy (place: pad; stack: bottom cube), from the snapshot (objects static)
        pad = snap["pad_pos"][ar, self.dest.clamp(0, 3)]
        bot = snap["cube_pos"][ar, self.dest.clamp(0, 7)]
        bot_yaw = snap["cube_yaw"][ar, self.dest.clamp(0, 7)]
        dest_xy = torch.where((self.task == STACK)[:, None], bot[:, :2], pad[:, :2])
        sup_top = torch.where(self.task == STACK, bot[:, 2] + self.edge[self.dest.clamp(0, 7)] / 2,
                              torch.full_like(bot[:, 2], self.table_z + self.h_pad))

        # hand yaw goal: closing axis on the nearest pair of target faces (111 rule) until lifted
        psi_face = cube_yaw + torch.round((psi_now - cube_yaw) / (math.pi / 2)) * (math.pi / 2)
        psi_bot = bot_yaw + torch.round((psi_now - bot_yaw) / (math.pi / 2)) * (math.pi / 2)
        yaw_err = _wrap(psi_now - psi_face).abs()

        ph = self.phase
        # ---- HOVER -> DESCEND
        hover_goal = torch.stack([cube[:, 0], cube[:, 1], cube[:, 2] + half_t + SC.HOVER_DZ], 1)
        xy_err = torch.linalg.norm(p[:, :2] - cube[:, :2], dim=-1)
        ok = (torch.linalg.norm(self.sp - hover_goal, dim=-1) < 2e-3) & (xy_err < SC.XY_TOL) & (yaw_err < SC.YAW_TOL)
        self.hover_run = torch.where((ph == HOVER) & ok, self.hover_run + 1, torch.zeros_like(self.hover_run))
        m = self._advance((ph == HOVER) & (self.hover_run >= SC.HOVER_STEPS), DESCEND)
        self.grasp_z = torch.where(m, cube[:, 2].clamp_min(SC.GRASP_MIN_Z), self.grasp_z)
        # ---- DESCEND -> CLOSE
        sp_down = (ph == DESCEND) & ((self.sp[:, 2] - self.grasp_z).abs() < 1e-4)
        self.at_goal_t = torch.where(sp_down, self.at_goal_t + 1, torch.zeros_like(self.at_goal_t))
        reached = (p[:, 2] - self.grasp_z).abs() < 0.004
        m = self._advance(sp_down & (reached | (self.at_goal_t > SC.SETTLE_STEPS)), CLOSE)
        # yaw goal frozen from the close on: a goal re-derived from the held cube's own yaw chases itself (the cube
        # turns with the hand) and spun the hand during the hold (dev run, 2026-10-01)
        self.psi_hold = torch.where(m, psi_face, self.psi_hold)
        # ---- CLOSE -> LIFT (contact-gated: evaluated at query steps only)
        both = (snap["lf"][ar, tg] > 1.0) & (snap["rf"][ar, tg] > 1.0)
        self.pad_run = torch.where((self.phase == CLOSE) & both, self.pad_run + 1, torch.zeros_like(self.pad_run))
        # 113: also wait until both fingers are still (the effort-limited claw has no damping once saturated, so any
        # sideways momentum of cube + fingers left from the closing would shuttle them between the finger limits for
        # the whole hold; on the table, friction stops it - dev runs 2026-10-01)
        if self.lift_trigger == "width":
            still = snap["finger_vel"].sum(-1).abs() < FINGER_STILL
        else:
            still = snap["finger_vel"].abs().amax(-1) < FINGER_STILL
        self.still_run = torch.where((self.phase == CLOSE) & still, self.still_run + 1, torch.zeros_like(self.still_run))
        gate = (self.pad_run >= SC.PAD_STEPS) & (self.still_run >= STILL_STEPS) & query
        go = gate | (self.phase_t >= CLOSE_MAX)
        m = self._advance((ph == CLOSE) & go, LIFT)
        self.close_exit = torch.where(m, torch.where(gate, 1, 2), self.close_exit)  # 1 contact/still gate, 2 timeout
        self.lift_goal = torch.where(m[:, None], torch.cat([p[:, :2], (self.grasp_z + LIFT_DZ)[:, None]], 1),
                                     self.lift_goal)
        # ---- LIFT -> HOLD (pick: stays in LIFT) / CARRY
        lifted = (ph == LIFT) & (torch.linalg.norm(self.sp - self.lift_goal, dim=-1) < 1e-4) & \
                 (torch.linalg.norm(p - self.lift_goal, dim=-1) < 0.01) & ~is_pick
        m = self._advance(lifted, CARRY)
        self.offset = torch.where(m[:, None], cube - p, self.offset)
        # ---- CARRY -> LOWER
        carry_goal = torch.cat([dest_xy - self.offset[:, :2], self.lift_goal[:, 2:]], 1)
        psi_carry = torch.where(self.task == STACK, psi_bot, self.psi_hold)
        arrived = (ph == CARRY) & (torch.linalg.norm(self.sp - carry_goal, dim=-1) < 1e-4) & \
                  (torch.linalg.norm(p - carry_goal, dim=-1) < 0.01) & (_wrap(psi_now - psi_carry).abs() < CARRY_YAW_TOL)
        m = self._advance(arrived, LOWER)
        self.psi_hold = torch.where(m, psi_carry, self.psi_hold)
        self.goal_z = torch.where(m, sup_top + RELEASE_DZ + half_t - self.offset[:, 2], self.goal_z)
        # ---- LOWER -> RELEASE
        sp_low = (ph == LOWER) & ((self.sp[:, 2] - self.goal_z).abs() < 1e-4)
        self.low_t = torch.where(sp_low, self.low_t + 1, torch.zeros_like(self.low_t))
        self._advance(sp_low & (((p[:, 2] - self.goal_z).abs() < 0.003) | (self.low_t > SC.SETTLE_STEPS)), RELEASE)
        # ---- RELEASE -> RETREAT (fingers open, or timeout)
        opened = snap["width"] > e_t + OPEN_MARGIN
        m = self._advance((ph == RELEASE) & (opened | (self.phase_t >= RELEASE_MAX_STEPS)), RETREAT)
        self.goal_z = torch.where(m, sup_top + e_t + RETREAT_DZ, self.goal_z)
        # ---- RETREAT -> DONE
        self._advance((ph == RETREAT) & (torch.linalg.norm(self.sp[:, 2:] - self.goal_z[:, None], dim=-1) < 1e-4), DONE)

        # ---- setpoint for the (possibly new) phase
        ph = self.phase
        grasp_pt = torch.stack([cube[:, 0], cube[:, 1], self.grasp_z], 1)
        here = torch.cat([self.sp[:, :2], self.goal_z[:, None]], 1)
        goal = torch.where((ph == HOVER)[:, None], hover_goal, grasp_pt)
        goal = torch.where((ph == CLOSE)[:, None], torch.cat([self.sp[:, :2], self.grasp_z[:, None]], 1), goal)
        goal = torch.where((ph == LIFT)[:, None], self.lift_goal, goal)
        goal = torch.where((ph == CARRY)[:, None], carry_goal, goal)
        goal = torch.where(((ph >= LOWER) & (ph != DONE))[:, None], here, goal)
        goal = torch.where((ph == DONE)[:, None], self.sp, goal)
        vmax = torch.tensor([SC.V_APPROACH, SC.V_DESCEND, SC.V_DESCEND, SC.V_LIFT, SC.V_CARRY, SC.V_DESCEND,
                             SC.V_DESCEND, SC.V_LIFT, SC.V_LIFT], device=self.dev)[ph]
        dn = (ph == DESCEND) | (ph == LOWER)
        self._profile(goal, vmax, dn)
        psi = torch.where(ph <= DESCEND, psi_face, self.psi_hold)  # approach: the target's faces; then frozen
        psi = torch.where(ph == CARRY, psi_carry, psi)
        claw = ((ph >= CLOSE) & (ph <= LOWER)).float()

        # ---- action through the 4-D task-space interface (scripted_delta_policy) + claw
        dpos = (self.sp - self.target_cmd) / STEP_POS
        dyaw = _wrap(psi - self.yaw_cmd) / STEP_YAW
        label = torch.cat([dpos, dyaw[:, None], claw[:, None]], 1)
        label[:, :4] = label[:, :4].clamp(-1.0, 1.0)
        exe = label.clone()
        if self.sigma > 0:
            noise = torch.randn(self.N, 4, generator=self.gen, device=self.dev) * self.sigma
            exe[:, :4] = (exe[:, :4] + noise).clamp(-1.0, 1.0)
        # the action term's target update: fresh = replayed exactly (box clamp, then <= 5 cm lead on the true TCP);
        # chunked = own deltas + box clamp only (the lead clamp depends on the TCP it cannot see; re-anchored per query)
        tc = torch.maximum(torch.minimum(self.target_cmd + exe[:, :3] * STEP_POS, self.hi), self.lo)
        if self.chunked:
            self.target_cmd = tc
        else:
            lead = tc - p
            n = torch.linalg.norm(lead, dim=-1, keepdim=True)
            self.target_cmd = p + lead * torch.clamp(LEAD_MAX / n.clamp_min(1e-9), max=1.0)
        self.exe_hist[t] = exe
        self.yaw_cmd = self.yaw_cmd + exe[:, 3] * STEP_YAW
        self.phase_t += 1
        self.t += 1
        return label, exe

    def _profile(self, goal: torch.Tensor, vmax: torch.Tensor, dn: torch.Tensor) -> None:
        """111 setpoint profile (ScriptedController._track, position part): speed cap vmax, accelerate / brake at
        <= A_MAX, stop on the goal; dn rows keep xy on the goal (vertical descent)."""
        dt = self.dt
        self.sp[:, :2] = torch.where(dn[:, None], goal[:, :2], self.sp[:, :2])
        d = goal - self.sp
        d[:, :2] = torch.where(dn[:, None], 0.0, d[:, :2])
        n = torch.linalg.norm(d, dim=-1, keepdim=True)
        speed = torch.minimum(vmax[:, None], torch.sqrt(2 * SC.A_MAX * n))
        v_des = d / n.clamp_min(1e-9) * speed
        dv = v_des - self.v
        self.v = self.v + dv * torch.clamp(SC.A_MAX * dt / torch.linalg.norm(dv, dim=-1, keepdim=True).clamp_min(1e-9),
                                           max=1.0)
        step = self.v * dt
        snap = torch.linalg.norm(step, dim=-1, keepdim=True) >= n
        self.sp = torch.where(snap, goal, self.sp + step)
        self.sp[:, :2] = torch.where(dn[:, None], goal[:, :2], self.sp[:, :2])
        self.v = torch.where(snap, torch.zeros_like(self.v), self.v)
