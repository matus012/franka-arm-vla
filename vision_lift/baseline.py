"""Stage 5 modular baseline: the Stage 1 cube estimator's pose fed to the frozen teacher in place of the true pose.

Consultant clarifications (brief section 6, Stage 5), all implemented here:
  - the estimate replaces the true cube pose EVERYWHERE the teacher uses it: the observation terms (TCP->cube,
    cube->target, cube-yaw part of the orientation term) AND the env's phase machine, which drives the reflex claw and
    the phase one-hot (`EstimatedPhaseState` below);
  - the estimator runs at the Stage 0.3 camera rate (every 3 steps) with the same 1-frame latency (vision_lift.frames);
  - pad forces and the palm flag stay (tactile sensing) — disclosed: the baseline has touch, the student does not;
  - the evaluator keeps its own phase machine on the TRUE state (`TruePhaseMachine`) for funnel / claw metrics; success,
    slam and jam come from the 111 EpisodeTracker on the true state as always.
Everything else about the teacher env (reflex, IK with pm.grasp feed-forward) is the 111 env unchanged.
"""

from __future__ import annotations

import torch

from lift_rl import mdp
from lift_rl.phased_cfg import PhaseState, contacts, cube_state, tcp_and_rot
from lift_rl.phases import PhaseMachine, PhaseParams
from lift_rl.staged import yaw_mod90_error

from .cube_estimate import replace_cube_terms


def machine_inputs(env, cube: torch.Tensor, yaw: torch.Tensor, prev_tcp: torch.Tensor):
    """The PhaseMachine.step inputs with a given cube pose (same code path as lift_rl PhaseState.get)."""
    tcp, R = tcp_and_rot(env)
    fresh = env.episode_length_buf <= 1
    speed = torch.linalg.norm(tcp - prev_tcp, dim=-1) / env.step_dt
    speed = torch.where(fresh, torch.zeros_like(speed), speed)
    tilt = torch.arccos((-R[:, 2, 2]).clamp(-1.0, 1.0))
    target = mdp.target_position_w(env) - env.scene.env_origins
    lf, rf, palm, table = contacts(env)
    return tcp, (tcp, speed, tilt, yaw_mod90_error(R[:, :, 1], yaw), cube, target, lf, rf, palm, table)


class EstimatedPhaseState(PhaseState):
    """The env's phase machine, fed with the ESTIMATED cube pose (env._cube_est = (pos, yaw), set by the policy)."""

    def get(self) -> dict[str, torch.Tensor]:
        env = self.env
        if self.step_id == env.common_step_counter:
            return self.m.out
        est = getattr(env, "_cube_est", None)
        cube, yaw = est if est is not None else cube_state(env)
        tcp, inp = machine_inputs(env, cube, yaw, self.prev_tcp)
        self.speed = inp[1]
        self.prev_tcp = tcp.clone()
        out = self.m.step(*inp)
        out["tcp"] = tcp
        self.step_id = env.common_step_counter
        return out


def install_estimated_phase_state(env) -> None:
    """Call right after gym.make, before the first reset."""
    u = env.unwrapped
    u._phase_state = EstimatedPhaseState(u, PhaseParams())
    u._phase_machine = u._phase_state.m


class TruePhaseMachine:
    """Evaluator-side phase machine on the true state (for the funnel and claw metrics of the baseline)."""

    def __init__(self, u):
        self.u = u
        self.m = PhaseMachine(u.num_envs, u.device, PhaseParams())
        self.prev_tcp = tcp_and_rot(u)[0].clone()

    def reset(self):
        self.m.reset()
        self.prev_tcp = tcp_and_rot(self.u)[0].clone()

    def step(self):
        cube, yaw = cube_state(self.u)
        tcp, inp = machine_inputs(self.u, cube, yaw, self.prev_tcp)
        self.prev_tcp = tcp.clone()
        return self.m.step(*inp)


def estimate_to_pose(pos: torch.Tensor, sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimator output -> (cube position, yaw mod 90 deg as an angle in (-pi/4, pi/4])."""
    return pos, torch.atan2(sc[:, 0], sc[:, 1]) / 4


def teacher_obs_with_estimate(obs_policy: torch.Tensor, u, pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return replace_cube_terms(obs_policy, u, pos, yaw)

