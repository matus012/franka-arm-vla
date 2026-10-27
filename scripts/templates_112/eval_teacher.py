"""The frozen teacher with its 111 reflex claw on a given world (the Stage 2/3/5 reference "teacher on the same
setting and seeds"). 111 task-space env of that world, no cameras, deterministic actions.

Usage: python scripts/eval_teacher.py --world P3 --episodes 1000 --num_envs 1000 --seed 12345 --json results/x.json
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
p.add_argument("--episodes", type=int, required=True)
p.add_argument("--num_envs", type=int, default=1000)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--json", required=True)
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.evaluation import make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import Thresholds, summarize  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift.teacher import load_teacher  # noqa: E402

TASK = {"Fixed": "LiftRL-Phased-Fixed-TS-v0", "P3": "LiftRL-Phased-P3-TS-v0", "P4": "LiftRL-Phased-P4-TS-v0",
        "P5": "LiftRL-Phased-P5-TS-v0", "P6": "LiftRL-Phased-P6-TS-v0"}[args.world]
cfg = make_eval_cfg(load_cfg_from_registry(TASK, "env_cfg_entry_point"), args.num_envs, args.seed)
assert_run_preconditions(cfg)
with launch_simulation(cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    actor = load_teacher(u.device)
    assert_on_cuda(env, actor)
    batches = []
    while sum(b["success"].numel() for b in batches) < args.episodes:
        batches.append(rollout(env, lambda o: actor(o["policy"]), Thresholds()))
    outcomes = {k: torch.cat([b[k] for b in batches])[: args.episodes] for k in batches[0]}
    s = summarize(outcomes)
    res = {"task": TASK, "world": args.world, "seed": args.seed, "policy": "teacher + 111 reflex claw",
           "nan_guard_events": u.nan_guard_events, "wall_s": round(time.time() - t0, 1), **s}
    print(json.dumps({k: res[k]["rate"] for k in ("success", "slam", "jam")}))
    o = ROOT / args.json
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(res, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
