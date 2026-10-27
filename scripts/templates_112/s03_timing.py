"""Stage 0.3: teacher with camera-rate, 1-frame-latency, noisy cube observations plus action noise.

Every cube-derived teacher term (TCP->cube, cube->target, cube-yaw part of the orientation term) comes from a cube
estimate refreshed every --period control steps with --latency camera frames of delay and N(0, --pos_sigma) position
noise per axis per frame (vision_lift/cube_estimate.py). Executed action = clip(teacher + N(0, --action_sigma), -1, 1).
Proprioception fresh every step; phase machine / reflex / evaluator on the true state.
--chunk: fallback mode, the teacher action computed on each camera frame is held for the whole frame (K = period).

Usage: python scripts/s03_timing.py --period 5 --out results/s03_timing_p5.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--period", type=int, required=True, help="control steps per camera frame (50 Hz / rate)")
p.add_argument("--latency", type=int, default=1)
p.add_argument("--pos_sigma", type=float, default=0.01)
p.add_argument("--action_sigma", type=float, default=0.1)
p.add_argument("--chunk", action="store_true")
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=1000)
p.add_argument("--num_episodes", type=int, default=1000)
p.add_argument("--seed", type=int, default=2030)
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.evaluation import make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import Thresholds, summarize  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift.cube_estimate import CubeEstimate, replace_cube_terms, true_cube  # noqa: E402
from vision_lift.teacher import TEACHER_TASK, load_teacher  # noqa: E402

cfg = make_eval_cfg(load_cfg_from_registry(TEACHER_TASK, "env_cfg_entry_point"), args.num_envs, args.seed)
assert_run_preconditions(cfg)
th = Thresholds()

with launch_simulation(cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(TEACHER_TASK, cfg=cfg)
    u = env.unwrapped
    actor = load_teacher(u.device)
    assert_on_cuda(env, actor)
    ce = CubeEstimate(u.num_envs, u.device, period=args.period, latency=args.latency, pos_sigma=args.pos_sigma,
                      seed=args.seed + 1)
    gen = torch.Generator(device=u.device)
    gen.manual_seed(args.seed + 2)
    steps = cfg.eval_steps
    st = {"k": 0, "held": None, "err_sum": 0.0, "err_n": 0}

    def policy(obs):
        if st["k"] % steps == 0:
            ce.reset()
        tick = st["k"] % args.period == 0
        pos, yaw = true_cube(u)
        est_pos, est_yaw = ce.step(pos, yaw)
        st["err_sum"] += float(torch.linalg.norm(est_pos - pos, dim=-1).mean())
        st["err_n"] += 1
        st["k"] += 1
        if args.chunk and not tick and st["held"] is not None:
            return st["held"]
        a = actor(replace_cube_terms(obs["policy"], u, est_pos, est_yaw))
        a = (a + torch.randn(a.shape, generator=gen, device=u.device) * args.action_sigma).clamp(-1.0, 1.0)
        st["held"] = a
        return a

    batches = []
    while sum(b["success"].numel() for b in batches) < args.num_episodes:
        batches.append(rollout(env, policy, th))
    outcomes = {k: torch.cat([b[k] for b in batches])[: args.num_episodes] for k in batches[0]}
    summary = summarize(outcomes)
    res = {"stage": "0.3", "period_steps": args.period, "camera_hz": round(50.0 / args.period, 1),
           "latency_frames": args.latency, "cube_pos_sigma_m": args.pos_sigma, "action_sigma": args.action_sigma,
           "chunk": args.chunk, "seed": args.seed, "num_envs": args.num_envs,
           "mean_cube_estimate_error_m": st["err_sum"] / max(1, st["err_n"]),
           "gate_pass": summary["success"]["rate"] >= 0.95, "nan_guard_events": u.nan_guard_events,
           "wall_s": round(time.time() - t0, 1), **summary}
    print(json.dumps(res, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    env.close()
