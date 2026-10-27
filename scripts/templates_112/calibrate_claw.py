"""Stage-2 prep (setup, not a gate): calibrate the student claw's finger effort limit so that its steady per-pad force
matches the reflex hold measured in Stage 0.1 (pad-force NORM, median 6.43 N, on steps with phase == HOLD).

Teacher drives the STUDENT env (phase-6 world, seed 12345 as in 0.1); the claw command is the teacher's claw label
(phase machine >= CLOSE) executed directly under effort limit --effort; no reflex. Same statistic as 0.1: pad-force
norm samples (left + right) on steps with phase == HOLD and the claw closed.

Usage: python scripts/calibrate_claw.py --effort 6.0 --out results/claw_calib/effort_6.0.json
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
p.add_argument("--effort", type=float, required=True)
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=1000)
p.add_argument("--seed", type=int, default=12345)
p.add_argument("--world", default="P6")
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
import vision_lift.envs as E  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.evaluation import contact_readings, make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import Thresholds, summarize  # noqa: E402
from lift_rl.phases import CLOSE, HOLD  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift.teacher import load_teacher  # noqa: E402

task = E.student_id(args.world)
cfg = make_eval_cfg(load_cfg_from_registry(task, "env_cfg_entry_point"), args.num_envs, args.seed)
cfg.scene.side_cam = None  # no rendering needed
cfg.scene.wrist_cam = None
E.set_claw_effort(cfg, args.effort)
assert_run_preconditions(cfg)
th = Thresholds()

with launch_simulation(cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    actor = load_teacher(u.device)
    assert_on_cuda(env, actor)
    lim = u.scene["robot"].data.joint_effort_limits
    lim = lim.torch if hasattr(lim, "torch") else lim
    fid = u.scene["robot"].find_joints("panda_finger.*")[0]
    samples = []

    def policy(obs):
        claw = (u._phase_machine.phase >= CLOSE).float()[:, None]
        return torch.cat([actor(obs["policy"]), claw], 1)

    def on_step(t, uu):
        pm = uu._phase_machine
        c = contact_readings(uu)
        m = (pm.phase == HOLD) & (uu.action_manager.action[:, 4] > 0.5)
        if bool(m.any()):
            samples.append(torch.stack([c["lf"], c["rf"]], 1)[m].flatten().cpu())

    res = rollout(env, policy, th, on_step=on_step)
    s = summarize(res)
    hs = torch.cat(samples) if samples else torch.zeros(1)
    q = lambda k: float(torch.quantile(hs, k))  # noqa: E731
    out = {"effort_limit_N": args.effort, "applied_finger_effort_limit_N": [float(v) for v in lim[0, fid]],
           "world": args.world, "seed": args.seed, "episodes": args.num_envs,
           "hold_pad_force_norm_N": {"n": int(hs.numel()), "median": q(0.5), "p10": q(0.1), "p90": q(0.9)},
           "teacher_in_student_env": {k: s[k]["rate"] for k in ("success", "success_5cm", "slam", "jam", "reached_close",
                                                                  "reached_lift", "reached_hold")},
           "episodes_with_drop": s.get("episodes_with_drop"), "nan_guard_events": u.nan_guard_events,
           "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1))
    env.close()
