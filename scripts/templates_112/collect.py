"""Roll out N-env batches in the student env with cameras: record teacher data, evaluate a student, or run a DAgger
(mixed) collection. Writes one HDF5 file per batch (unless --no_record) and a JSON summary.

Usage:
  python scripts/collect.py --world P6 --driver teacher --episodes 64 --num_envs 32 --seed 1000 \
      --out_dir datasets/s1_p6 --summary results/collect_s1.json [--action_sigma 0.1]
  python scripts/collect.py --world Fixed --driver student --student ckpt.pt --episodes 200 --no_record ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--world", required=True, choices=("Fixed", "P3", "P4", "P5", "P6"))
p.add_argument("--driver", default="teacher", choices=("teacher", "student", "mixed"))
p.add_argument("--beta", type=float, default=1.0)
p.add_argument("--student", default=None)
p.add_argument("--episodes", type=int, required=True)
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--action_sigma", type=float, default=0.0)
p.add_argument("--out_dir", default=None)
p.add_argument("--no_record", action="store_true")
p.add_argument("--summary", required=True)
p.add_argument("--tag", default="")
p.add_argument("--relabel", default="stateful", choices=("stateful", "resync", "catchup"))
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

task = E.student_id(args.world)
cfg = make_eval_cfg(load_cfg_from_registry(task, "env_cfg_entry_point"), args.num_envs, args.seed)
assert_run_preconditions(cfg)
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    t0 = time.time()
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    teacher = load_teacher(u.device)
    student, depth = None, False
    if args.student:
        from vision_lift.train import load_student

        student, depth = load_student(args.student, u.device)
    assert_on_cuda(env, teacher)
    visual = None
    if args.visual != "none":
        from vision_lift.visual_rand import VisualRandomizer

        visual = VisualRandomizer(u, args.visual, args.visual_seed)
    runner = Runner(env, teacher, student, depth, args.driver, args.beta, args.action_sigma, seed=args.seed,
                    relabel=args.relabel, visual=visual)
    batches, files = [], []
    nb = (args.episodes + args.num_envs - 1) // args.num_envs
    peak_vram = 0
    for b in range(nb):
        path = None
        if not args.no_record:
            path = ROOT / args.out_dir / f"{args.tag}w{args.world}_s{args.seed}_b{b:03d}.h5"
            files.append(str(path.relative_to(ROOT)))
        attrs = {"world": args.world, "driver": args.driver, "beta": args.beta, "action_sigma": args.action_sigma,
                 "seed": args.seed, "batch": b, "relabel": args.relabel, "visual": args.visual, "visual_seed": args.visual_seed, "student": args.student or "", "git": sha, "claw_effort_N": E.CLAW_EFFORT_N}
        batches.append(runner.run_batch(path, attrs))
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())
        print(f"[collect] batch {b + 1}/{nb} done, {time.time() - t0:.0f} s, "
              f"success so far {float(torch.cat([x['success'] for x in batches]).float().mean()):.3f}", flush=True)
    outcomes = {k: torch.cat([x[k] for x in batches])[: args.episodes] for k in batches[0]}
    s = summarize_112(outcomes)
    wall = time.time() - t0
    disk = sum((ROOT / f).stat().st_size for f in files)
    res = {"task": task, "world": args.world, "driver": args.driver, "beta": args.beta, "student": args.student,
           "action_sigma": args.action_sigma, "seed": args.seed, "num_envs": args.num_envs, "git": sha,
           "claw_effort_N": E.CLAW_EFFORT_N, "files": files, "disk_bytes": disk,
           "disk_mb_per_episode": disk / 1e6 / max(1, args.episodes) if files else None,
           "wall_s": round(wall, 1), "episodes_per_hour": round(args.episodes / wall * 3600, 1),
           "torch_peak_vram_mb": round(peak_vram / 1e6), "nan_guard_events": u.nan_guard_events, **s}
    print(json.dumps(res, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    out = ROOT / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    env.close()
