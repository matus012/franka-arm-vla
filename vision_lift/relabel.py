"""Phase re-sync relabeler for DAgger labels (proposed after the Stage 3.1 attempt-1 finding; NOT in use until an
owner-approved gate says so).

Problem: the 111 phase machine has memory (HOVER -> DESCEND needs the 1.5 cm pre-grasp waypoint; the grasp is a
latch). On student-driven states its phase, the teacher's phase one-hot and the claw label depend on the path, not on
what the student can see: 69.8 % of lifted-cube steps in the 3.1 round-1 student episodes were labelled "open".

Re-sync: for DAgger labels only, the labeler's phase is recomputed every step from the TRUE current state, no memory:
  grasped (palm <= contact_n and both pads > 5 N, or both pads > 1 N with the cube raised > 5 mm above rest)
                                                     -> HOLD if cube within carry_tol of the target,
                                                        else CARRY if the cube is >= lift_dz above rest, else LIFT
  else in_band and TCP speed < sweet_speed           -> CLOSE
  else aligned and TCP inside the pre-grasp column    -> DESCEND   (column: xy within sweet_xy of the cube centre,
       (below the pre-grasp height + hover_tol)                    height below pre-grasp point + hover_tol)
  else                                                -> HOVER
The grasp flag = the current (stateless) grasp. Thresholds are the 111 PhaseParams. The teacher's weights are not
touched; teacher-driven collection keeps the original stateful machine.
"""

from __future__ import annotations

import torch

from lift_rl.phases import CARRY, CLOSE, CUBE_HALF, DESCEND, HOLD, HOVER, LIFT, PhaseParams, phase_one_hot

CONFIRM_N = 5.0  # the stateful machine's grasp-confirm threshold (CLOSE -> LIFT on the table)
KEEP_N = 1.0  # its keep threshold (the latch drops only after > 5 steps with < 2 pads > 1 N)
RAISED_M = 0.005  # cube raised above rest: the keep threshold applies once the cube is off the table


def stateless_phase(tcp, speed, tilt, yaw_err, cube, target, lf, rf, palm_f,
                    p: PhaseParams | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """All (N,) / (N, 3), env frame, same units as PhaseMachine.step. Returns (phase, grasp)."""
    p = p or PhaseParams()
    aligned = (tilt < p.tilt_ok) & (yaw_err < p.yaw_ok)
    xy_err = torch.linalg.norm(tcp[:, :2] - cube[:, :2], dim=-1)
    dz = tcp[:, 2] - (cube[:, 2] + p.grasp_dz)
    in_band = aligned & (xy_err < p.sweet_xy) & (dz.abs() < p.sweet_dz)
    pre_z = cube[:, 2] + CUBE_HALF + p.pregrasp_dz
    in_column = aligned & (xy_err < p.sweet_xy) & (tcp[:, 2] < pre_z + p.hover_tol)
    cube_h = cube[:, 2] - CUBE_HALF
    confirm = (lf > CONFIRM_N) & (rf > CONFIRM_N)
    keep = (lf > KEEP_N) & (rf > KEEP_N) & (cube_h > RAISED_M)
    grasp = (confirm | keep) & (palm_f <= p.contact_n)
    dist = torch.linalg.norm(cube - target, dim=-1)
    ph = torch.full_like(xy_err, HOVER, dtype=torch.long)
    ph = torch.where(in_column, torch.full_like(ph, DESCEND), ph)
    ph = torch.where(in_band & (speed < p.sweet_speed), torch.full_like(ph, CLOSE), ph)
    grasped_ph = torch.where(dist < p.carry_tol, torch.full_like(ph, HOLD),
                             torch.where(cube_h >= p.lift_dz, torch.full_like(ph, CARRY), torch.full_like(ph, LIFT)))
    ph = torch.where(grasp, grasped_ph, ph)
    return ph, grasp


def resync_teacher_obs(obs_policy: torch.Tensor, phase: torch.Tensor, grasp: torch.Tensor) -> torch.Tensor:
    """Teacher observation with the phase one-hot + grasp flag replaced by the re-synced values."""
    from .teacher import OBS_SLICES

    o = obs_policy.clone()
    o[:, OBS_SLICES["phase"]] = torch.cat([phase_one_hot(phase), grasp.float()[:, None]], 1)
    return o


class CatchUpMachine:
    """Relabeler v2 (after the stateless proof failed): the 111 stateful PhaseMachine (memory, grasp latch, reopen and
    drop rules unchanged) plus two forward catch-up transitions that only fire in states the teacher's own path never
    produces, so student-driven states cannot leave the labeler stuck behind the true situation:
      R1 phase in {HOVER, DESCEND} and both pads > 5 N and palm <= 1 N -> grasp latched, phase = HOLD / CARRY / LIFT by
         geometry (the teacher's claw is open in HOVER/DESCEND, so its own grasps always go through CLOSE);
      R2 phase == HOVER and aligned and TCP inside the pre-grasp column (xy < 1.5 cm of the cube centre, below the
         pre-grasp height + 1.5 cm) -> DESCEND (the teacher reaches this column through the pre-grasp waypoint).
    Driven with the SAME inputs as the env's machine, stepped once per control step from step 1 on (step 0 = reset),
    so on teacher episodes it matches the env's machine except where R1/R2 fire."""

    def __init__(self, num_envs: int, device, p: PhaseParams | None = None):
        from lift_rl.phases import PhaseMachine

        self.p = p or PhaseParams()
        self.m = PhaseMachine(num_envs, device, self.p)
        self.r1 = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.r2 = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self) -> None:
        self.m.reset()
        self.r1.zero_()
        self.r2.zero_()

    @property
    def phase(self) -> torch.Tensor:
        return self.m.phase

    @property
    def grasp(self) -> torch.Tensor:
        return self.m.grasp

    def step(self, tcp, speed, tilt, yaw_err, cube, target, lf, rf, palm_f, table_f) -> None:
        p, m = self.p, self.m
        m.step(tcp, speed, tilt, yaw_err, cube, target, lf, rf, palm_f, table_f)
        aligned = (tilt < p.tilt_ok) & (yaw_err < p.yaw_ok)
        xy_err = torch.linalg.norm(tcp[:, :2] - cube[:, :2], dim=-1)
        pre_z = cube[:, 2] + CUBE_HALF + p.pregrasp_dz
        cube_h = cube[:, 2] - CUBE_HALF
        dist = torch.linalg.norm(cube - target, dim=-1)
        r1 = (m.phase <= DESCEND) & (lf > CONFIRM_N) & (rf > CONFIRM_N) & (palm_f <= p.contact_n)
        geo = torch.where(dist < p.carry_tol, torch.full_like(m.phase, HOLD),
                          torch.where(cube_h >= p.lift_dz, torch.full_like(m.phase, CARRY), torch.full_like(m.phase, LIFT)))
        r2 = ~r1 & (m.phase == HOVER) & aligned & (xy_err < p.sweet_xy) & (tcp[:, 2] < pre_z + p.hover_tol)
        m.phase = torch.where(r1, geo, torch.where(r2, torch.full_like(m.phase, DESCEND), m.phase))
        m.grasp = m.grasp | r1
        m.ever_grasped |= r1
        m.lost_run = torch.where(r1, torch.zeros_like(m.lost_run), m.lost_run)
        idx = torch.arange(m.n, device=m.dev)
        m.entered[idx, m.phase] = True
        self.r1 += r1.long()
        self.r2 += r2.long()
