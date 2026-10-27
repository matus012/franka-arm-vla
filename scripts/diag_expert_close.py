"""Diagnostic (not a gate): why the expert's CLOSE -> LIFT contact gate does not fire (0B.3 finding: lifts at the
CLOSE_MAX timeout). Runs the unchunked expert on a seed's stream and logs, on every step the expert is in CLOSE, the
gate inputs it reads: both target pads > 1 N (PAD_STEPS run), per-finger speed vs FINGER_STILL (the "settle"
stillness test, STILL_STEPS run) and the gripper-width rate (the "width" candidate). Per episode: exit reason
(1 = gate, 2 = timeout), close duration, fractions of CLOSE steps meeting each condition, longest runs, success.

Usage: python scripts/diag_expert_close.py --stage s1 --seed 31600 --n 10 --trigger settle --tag overfit
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
p.add_argument("--stage", required=True, choices=["s1", "s2", "s3a", "s3b"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--n", type=int, default=300)
p.add_argument("--start", type=int, default=0)
p.add_argument("--trigger", default="settle", choices=["settle", "width"])
p.add_argument("--num_envs", type=int, default=300)
p.add_argument("--tag", default="")
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vla_common.catalog import ALL_CUBES  # noqa: E402
from vla_common.tasks import sample_spec  # noqa: E402
from vla_tabletop.expert import CLOSE, FINGER_STILL  # noqa: E402
from vla_tabletop.rollout import Runner, geometry, make_stream  # noqa: E402

RES = ROOT / "results"
TASK = {"s1": "pick", "s2": "pick", "s3a": "place", "s3b": "stack"}[args.stage]
meas = json.loads((RES / "0a1_measure.json").read_text())
tj = json.loads((RES / f"0a2_{TASK}_timing.json").read_text())
T = tj["T_steps"]
geo = geometry(meas["z_rest_m"], h_pad=float(np.mean(list(meas["h_pad_m"].values()))))
grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES])
N = min(args.num_envs, args.n)
cfg = S.make_cfg(N, args.seed)
assert_run_preconditions(cfg)

with launch_simulation(cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    eps = make_stream(args.seed, args.start, args.n, lambda rng: sample_spec(args.stage, "train", rng))
    run = Runner(env, geo, grasp_dz=grasp_dz.to(u.device))
    ar = torch.arange(N, device=u.device)
    acc: dict = {}
    rows: list[dict] = []

    def on_step(t, uu, s, lab, exe, ctrl, det):
        if t == 0:
            acc.clear()
            z = lambda: torch.zeros(N, device=uu.device)  # noqa: E731
            acc.update(steps=z(), both=z(), still_settle=z(), still_width=z(), max_pad_run=z(), max_still_run=z(),
                       vmax_sum=z(), wrate_sum=z())
        inc = (ctrl.phase == CLOSE).float()
        tg = ctrl.target
        both = (s["f_l"][ar, tg] > 1.0) & (s["f_r"][ar, tg] > 1.0)
        fv = s["finger_vel"]
        acc["steps"] += inc
        acc["both"] += inc * both.float()
        acc["still_settle"] += inc * (fv.abs().amax(-1) < FINGER_STILL).float()
        acc["still_width"] += inc * (fv.sum(-1).abs() < FINGER_STILL).float()
        acc["max_pad_run"] = torch.maximum(acc["max_pad_run"], inc * ctrl.pad_run.float())
        acc["max_still_run"] = torch.maximum(acc["max_still_run"], inc * ctrl.still_run.float())
        acc["vmax_sum"] += inc * fv.abs().amax(-1)
        acc["wrate_sum"] += inc * fv.sum(-1).abs()
        acc["ctrl"] = ctrl

    for b0 in range(0, len(eps), N):
        batch = eps[b0:b0 + N]
        padded = batch + [batch[-1]] * (N - len(batch))
        o = run.run_batch(padded, T, {"fresh": True, "lift_trigger": args.trigger}, on_step=on_step)
        c = acc["ctrl"]
        for i, e in enumerate(batch):
            st = max(float(acc["steps"][i]), 1.0)
            rows.append({"index": e.index, "size": e.spec.target[1], "exit": int(c.close_exit[i]),
                         "close_steps": int(acc["steps"][i]), "frac_both_pads": float(acc["both"][i]) / st,
                         "frac_still_settle": float(acc["still_settle"][i]) / st,
                         "frac_still_width": float(acc["still_width"][i]) / st,
                         "max_pad_run": int(acc["max_pad_run"][i]), "max_still_run": int(acc["max_still_run"][i]),
                         "mean_max_finger_speed": float(acc["vmax_sum"][i]) / st,
                         "mean_width_rate": float(acc["wrate_sum"][i]) / st,
                         "success": bool(o["success"][i]), "drop": bool(o["drop"][i])})
    out = {"diag": "expert close timeout", "stage": args.stage, "task": TASK, "seed": args.seed, "start": args.start,
           "n": args.n, "T": T, "trigger": args.trigger, "num_envs": N,
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "wall_s": round(time.time() - t0, 1), "episodes": rows}
    d = RES / "expert_close_diag"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{args.stage}_s{args.seed}_{args.trigger}{args.tag}.json"
    f.write_text(json.dumps(out, indent=1))
    ex = np.array([r["exit"] for r in rows])
    print(f"{args.stage} {args.trigger}: timeout {np.mean(ex == 2):.3f} gate {np.mean(ex == 1):.3f} "
          f"success {np.mean([r['success'] for r in rows]):.3f} wall {out['wall_s']}", flush=True)
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
