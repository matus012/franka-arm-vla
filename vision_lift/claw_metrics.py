"""New claw metrics (brief section 7, operational definitions by the consultant) + per-phase breakdowns.

- early close: the claw command switches to close (0 -> 1, or closed at step 0) while the hand is outside the
  sweet-spot band (`in_band` false: aligned, xy < 1.5 cm, height +-1.5 cm), 111 phase machine on the true state.
- drop: the phase machine's drop event (grasp lost for > 5 steps) before the first entry into HOLD.
- jam phase / drop phase: the phase-machine phase at the step the jam run reaches JAM_STEPS / at the drop event.
All inputs are per-step arrays (E, T) recorded during the rollout; pure torch, unit-tested without a simulator.
"""

from __future__ import annotations

import torch

from lift_rl.phases import HOLD, PHASES


def early_close(cmd: torch.Tensor, in_band: torch.Tensor) -> torch.Tensor:
    closed = cmd > 0.5
    prev = torch.cat([torch.zeros_like(closed[:, :1]), closed[:, :-1]], 1)
    return (closed & ~prev & ~in_band.bool()).any(1)


def drop_before_hold(drop: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    hold_seen = torch.cummax((phase == HOLD).int(), dim=1).values.bool()
    before = torch.cat([torch.zeros_like(hold_seen[:, :1]), hold_seen[:, :-1]], 1)  # HOLD entered strictly earlier
    return (drop.bool() & ~before).any(1)


def first_event_phase(event: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """(E,) phase at the first step where `event` is true, -1 if never."""
    any_ev = event.bool().any(1)
    idx = event.int().argmax(1)
    ph = phase.gather(1, idx[:, None])[:, 0]
    return torch.where(any_ev, ph, torch.full_like(ph, -1))


def jam_onset(jam_step: torch.Tensor, jam_steps: int = 10) -> torch.Tensor:
    """(E, T) true on the step where a run of consecutive jam steps reaches `jam_steps` (111 metrics.py rule)."""
    run = torch.zeros(jam_step.shape[0], dtype=torch.int32, device=jam_step.device)
    out = torch.zeros_like(jam_step, dtype=torch.bool)
    for t in range(jam_step.shape[1]):
        run = torch.where(jam_step[:, t].bool(), run + 1, torch.zeros_like(run))
        out[:, t] = run == jam_steps
    return out


def by_phase(phase_at_event: torch.Tensor) -> dict[str, int]:
    return {name: int((phase_at_event == k).sum()) for k, name in enumerate(PHASES)}
