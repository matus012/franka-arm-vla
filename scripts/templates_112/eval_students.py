"""Closed-loop evaluation of one or more student checkpoints (and optionally the binary-claw teacher = "claw ceiling")
in the student env with cameras, in ONE sim process. Deterministic student actions, no recording.

Usage:
  python scripts/eval_students.py --world Fixed --ckpts a.pt,b.pt --episodes 128 --num_envs 128 --seed 7101 \
      --json results/s2_val_bc.json [--claw_ceiling]
Per checkpoint: 111 metrics + early close, drop before HOLD, jam/drop by phase, drops/success by mass/friction bin.
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
p.add_argument("--world", required=True, choices=("Fixed", "P3", "P4", "P5", "P6"))
p.add_argument("--ckpts", default="")
p.add_argument("--claw_ceiling", action="store_true", help="also run the teacher arm + teacher claw label at 6.1 N")
p.add_argument("--episodes", type=int, required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--json", required=True)
p.add_argument("--visual", default="none", choices=("none", "nominal", "train", "heldout"))
p.add_argument("--visual_seed", type=int, default=0)
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
import vision_lift.envs as E  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.evaluation import make_eval_cfg  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift.collect import Runner, summarize_112  # noqa: E402
from vision_lift.teacher import load_teacher  # noqa: E402
from vision_lift.train import load_student  # noqa: E402

task = E.student_id(args.world)
cfg = make_eval_cfg(load_cfg_from_registry(task, "env_cfg_entry_point"), args.num_envs, args.seed)
assert_run_preconditions(cfg)
runs = [c for c in args.ckpts.split(",") if c] + (["claw_ceiling"] if args.claw_ceiling else [])

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    teacher = load_teacher(u.device)
    assert_on_cuda(env, teacher)
    visual = None
    if args.visual != "none":
        from vision_lift.visual_rand import VisualRandomizer
    results = {}
    for run in runs:
        if args.visual != "none":  # same visual draws for every run (re-seeded like the env)
            visual = VisualRandomizer(u, args.visual, args.visual_seed)
        t0 = time.time()
        # every run starts from the same env seed: re-seed the sim RNGs used by the reset events
        u.seed(args.seed)
        if run == "claw_ceiling":
            runner = Runner(env, teacher, None, False, "teacher", 1.0, 0.0, seed=args.seed, visual=visual)
        else:
            student, depth = load_student(ROOT / run, u.device)
            assert next(student.parameters()).device.type == "cuda"
            runner = Runner(env, teacher, student, depth, "student", 0.0, 0.0, seed=args.seed, visual=visual)
        batches = []
        while sum(b["success"].numel() for b in batches) < args.episodes:
            batches.append(runner.run_batch())
        outcomes = {k: torch.cat([b[k] for b in batches])[: args.episodes] for k in batches[0]}
        s = summarize_112(outcomes)
        s["wall_s"] = round(time.time() - t0, 1)
        results[run] = s
        print(json.dumps({"run": run, "success": s["success"]["rate"], "slam": s["slam"]["rate"],
                          "jam": s["jam"]["rate"], "early_close": s["early_close"]["rate"],
                          "drop_before_hold": s["drop_before_hold"]["rate"], "wall_s": s["wall_s"]}), flush=True)
    out = {"world": args.world, "seed": args.seed, "visual": args.visual, "visual_seed": args.visual_seed, "episodes": args.episodes, "num_envs": args.num_envs,
           "claw_effort_N": E.CLAW_EFFORT_N, "nan_guard_events": u.nan_guard_events, "results": results}
    o = ROOT / args.json
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
