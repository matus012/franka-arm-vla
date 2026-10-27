"""Camera-rate cube estimate for the privileged teacher (Stage 0.3; reused by the Stage 5 modular baseline).

The teacher's cube-derived observation terms (TCP->cube, cube->target, and the sin/cos cube-yaw part of the
orientation term) are recomputed from a cube estimate that
  - refreshes only on camera ticks (every `period` control steps, counted from the episode start),
  - arrives with `latency` camera frames of delay (latency 1: the frame captured on tick k is used from tick k+1 until
    tick k+2, so its age is period .. 2*period-1 control steps),
  - carries i.i.d. Gaussian position noise (sigma per axis, drawn once per camera frame).
Before the first delayed frame exists (episode start) the first capture is used. Proprioception, the TCP, the target,
the hand tilt, pad forces and the phase one-hot stay fresh every step (brief 0.3); the phase machine and reflex keep
the true state.
"""

from __future__ import annotations

import torch

from lift_rl.phased_cfg import cube_state, tcp_and_rot
from vision_lift.teacher import OBS_SLICES


class CubeEstimate:
    def __init__(self, num_envs: int, device, period: int, latency: int = 1, pos_sigma: float = 0.0,
                 seed: int = 0):
        assert period >= 1 and latency >= 0
        self.n, self.dev, self.period, self.latency, self.sigma = num_envs, device, period, latency, pos_sigma
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.frames: list[tuple[torch.Tensor, torch.Tensor]] = []  # captured (pos, yaw), newest last
        self.t = 0

    def reset(self) -> None:
        """All envs start a new episode together (lift_rl.evaluation.rollout resets every env at once)."""
        self.frames, self.t = [], 0

    def capture(self, pos: torch.Tensor, yaw: torch.Tensor) -> None:
        noise = torch.randn(pos.shape, generator=self.gen, device=self.dev) * self.sigma if self.sigma > 0 else 0.0
        self.frames = (self.frames + [(pos + noise, yaw.clone())])[-(self.latency + 1):]

    def step(self, pos_true: torch.Tensor, yaw_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Call once per control step (before the policy acts) with the true cube pose; returns the estimate in use."""
        if self.t % self.period == 0:
            self.capture(pos_true, yaw_true)
        self.t += 1
        return self.frames[0]  # oldest kept frame = `latency` frames old (or the first capture)


def replace_cube_terms(obs: torch.Tensor, u, cube_pos: torch.Tensor, cube_yaw: torch.Tensor) -> torch.Tensor:
    """Teacher observation with every cube-derived term recomputed from (cube_pos, cube_yaw), env frame."""
    o = obs.clone()
    tcp, R = tcp_and_rot(u)
    target = obs[:, OBS_SLICES["target"]]
    o[:, OBS_SLICES["tcp_to_cube"]] = cube_pos - tcp
    o[:, OBS_SLICES["cube_to_target"]] = target - cube_pos
    psi = torch.atan2(R[:, 1, 1], R[:, 0, 1])
    s = OBS_SLICES["orientation"]
    o[:, s.start + 1] = torch.sin(4 * (psi - cube_yaw))
    o[:, s.start + 2] = torch.cos(4 * (psi - cube_yaw))
    return o


def true_cube(u) -> tuple[torch.Tensor, torch.Tensor]:
    return cube_state(u)
