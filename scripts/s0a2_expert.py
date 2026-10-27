"""Stage 0A.2 (GATES.md): the 113 expert on pick / place / stack among distractors (also used by 0A.3 with --chunk).

Modes (one simulator per process; per-episode outcomes saved so runs can be merged / extended on the same stream):
  timing  : validation seed, 300 episodes, horizon cap 30 s -> completion-time p99 -> T_task = ceil(p99 + 2 s) whole
            seconds; grasp_dz per size (median TCP - cube centre z at grasp confirmation)
            -> results/0a2_<task>_timing.json
  run     : episodes [start, start+n) of a seed's stream at T_task (from the timing JSON) with the given expert
            settings -> results/<prefix>_<task>_s<seed>_<start>.json
  summary : merges the run files of one (prefix, task, seed) and applies the smart length rule.

Usage:
  python scripts/s0a2_expert.py --mode timing --task pick
  python scripts/s0a2_expert.py --mode run --task pick --seed 11200 --n 300 --start 0
  python scripts/s0a2_expert.py --mode run --task pick --seed 11200 --n 300 --sigma 0.1 --prefix 0a2noise
  python scripts/s0a2_expert.py --mode run --task place --seed 11310 --chunk 10 --prefix 0a3_N10
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--mode", required=True, choices=["timing", "run", "summary"])
p.add_argument("--task", required=True, choices=["pick", "place", "stack"])
p.add_argument("--seed", type=int, default=None)
p.add_argument("--n", type=int, default=300)
p.add_argument("--start", type=int, default=0)
p.add_argument("--split", default="train")
p.add_argument("--sigma", type=float, default=0.0)
p.add_argument("--chunk", type=int, default=0, help="0 = fresh (unchunked) expert; N = chunked emulation (0A.3)")
p.add_argument("--prefix", default="0a2")
p.add_argument("--num_envs", type=int, default=300)
p.add_argument("--T", type=int, default=0, help="override T (dev only)")
p.add_argument("--cube-yaw-max-deg", type=float, default=180.0,
               help="cube spawn yaw U(-m, m) deg (vla_tabletop.layout; 180 = U(-pi, pi), unchanged)")
args = p.parse_args()
# timing writes the shared results/0a2_<task>_timing.json (T_task for every stage): only on the default stream
assert args.mode != "timing" or args.cube_yaw_max_deg == 180.0, "--mode timing only with --cube-yaw-max-deg 180"

import numpy as np  # noqa: E402

from vla_common import timing as TM  # noqa: E402
from vla_common.tasks import sample_spec  # noqa: E402

RES = ROOT / "results"
STAGE = {"pick": "s2", "place": "s3a", "stack": "s3b"}[args.task]
VAL_SEED = {"pick": 21200, "place": 21210, "stack": 21220}[args.task]
CAP_STEPS = 30 * TM.CONTROL_HZ
OUTCOME_KEYS = ("index", "success", "completion_step", "wrong_grasp", "wrong_dest", "wrong_object", "slam", "slam_10N",
                "slam_50N", "jam", "drop", "early_close", "release_failure", "release_fail_not_released",
                "release_fail_off_destination", "release_fail_not_clear", "release_fail_unstable",
                "stack_release_rest_seen", "stack_stable_after_release", "distractor_disturbed", "clean",
                "clean_c1_no_table_hit", "clean_c2_top_down", "clean_c3_both_pads_at_liftoff", "clean_c4_no_palm",
                "grasp_confirmed", "confirm_tcp_minus_cube_z", "invalid")


def sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def rate(x) -> dict:
    from lift_rl.metrics import wilson_ci

    x = np.asarray(x, bool)
    k, n = int(x.sum()), int(x.size)
    return {"count": k, "n": n, "rate": k / n if n else float("nan"), "ci95": list(wilson_ci(k, n))}


def timing_file() -> Path:
    return RES / f"0a2_{args.task}_timing.json"


BARS = {"pick": 0.95, "place": 0.90, "stack": 0.85}

# --------------------------------------------------------------------------- summary (pure python)
if args.mode == "summary":
    files = sorted(RES.glob(f"{args.prefix}_{args.task}_s{args.seed}_*.json"))
    runs = sorted([json.loads(f.read_text()) for f in files], key=lambda r: r["start"])
    ep = {k: sum((r["episodes"][k] for r in runs), []) for k in runs[0]["episodes"]}
    assert ep["index"] == list(range(len(ep["index"]))), "stream must be contiguous from 0"
    yaws = {r.get("cube_yaw_max_deg", 180.0) for r in runs}  # absent = written before the flag existed = 180
    assert len(yaws) == 1, f"runs with different --cube-yaw-max-deg: {yaws}"
    n = len(ep["index"])
    succ = rate(ep["success"])
    s300 = rate(ep["success"][:300])
    bar = BARS[args.task]
    d = 100 * (s300["rate"] - bar)
    if abs(d) > 5:
        decision, gate = "decided at 300", s300["rate"] >= bar
    elif n >= 1000:
        decision, gate = "decided at 1000", succ["rate"] >= bar
    else:
        decision, gate = "extend to 1000", None
    cs = np.asarray(ep["completion_step"])
    done = cs[cs >= 0]
    out = {"stage": "0A.2" if args.prefix == "0a2" else args.prefix, "task": args.task, "seed": args.seed,
           "episodes": n, "bar": bar, "success": succ, "success_first300": s300, "smart_length": decision,
           "gate_pass": gate, "cube_yaw_max_deg": yaws.pop(), "T": runs[0]["T"], "git_sha_runs": sorted({r["git_sha"] for r in runs}),
           "expert": runs[0]["expert"], "detector_version": runs[0]["detector_version"],
           "completion_time_s": {"p50": float(np.percentile(done, 50)) / TM.CONTROL_HZ if done.size else None,
                                 "p99": float(np.percentile(done, 99)) / TM.CONTROL_HZ if done.size else None},
           **{k: rate(ep[k]) for k in OUTCOME_KEYS if k not in ("index", "success", "completion_step",
                                                                 "confirm_tcp_minus_cube_z")},
           "wall_s": sum(r["wall_s"] for r in runs), "files": [f.name for f in files],
           "failing_indices": [i for i, s in zip(ep["index"], ep["success"]) if not s]}
    if args.task == "stack":
        rs = np.asarray(ep["stack_release_rest_seen"], bool)
        out["stack_stability"] = rate(np.asarray(ep["stack_stable_after_release"], bool)[rs])
    f = RES / f"{args.prefix}_{args.task}.json" if args.seed is None else RES / f"{args.prefix}_{args.task}_s{args.seed}.json"
    f.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("failing_indices",)}, indent=1))
    sys.exit(0)

# --------------------------------------------------------------------------- simulator
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.metrics import Thresholds  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vla_common.catalog import ALL_CUBES  # noqa: E402
from vla_tabletop.detectors import DETECTOR_VERSION, Consts113  # noqa: E402
from vla_tabletop.expert import EXPERT_VERSION  # noqa: E402
from vla_tabletop.rollout import Runner, geometry, make_stream  # noqa: E402

seed = args.seed if args.seed is not None else VAL_SEED
cfg = S.make_cfg(args.num_envs, seed)
assert_run_preconditions(cfg)
meas = json.loads((RES / "0a1_measure.json").read_text())
geo = geometry(meas["z_rest_m"], h_pad=float(np.mean(list(meas["h_pad_m"].values()))))
if args.mode == "timing":
    T = args.T or CAP_STEPS
    grasp_dz = None
else:
    tj = json.loads(timing_file().read_text())
    T = args.T or tj["T_steps"]
    grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES])
expert_kw = {"fresh": True} if args.chunk == 0 else {"fresh": False, "n_query": args.chunk, "period": TM.PERIOD,
                                                       "latency": TM.LATENCY}
expert_kw.update({"sigma": args.sigma, "noise_seed": seed * 1000 + args.start})

with launch_simulation(cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    print(f"[0a2] device={u.device} envs={u.num_envs} task={args.task} mode={args.mode} seed={seed} T={T}", flush=True)
    run = Runner(env, geo, grasp_dz=grasp_dz.to(u.device) if grasp_dz is not None else None)
    n = 300 if args.mode == "timing" else args.n
    eps = make_stream(seed, args.start if args.mode == "run" else 0, n, lambda rng: sample_spec(STAGE, args.split, rng),
                      split=args.split, cube_yaw_max_deg=args.cube_yaw_max_deg)
    res = run.run(eps, T, expert_kw=expert_kw, stop_when_complete=args.mode == "timing")
    sizes = [e.spec.target[1] for e in eps]
    if args.mode == "timing":
        cs = res["completion_step"].numpy()
        done = cs[cs >= 0]
        p99 = float(np.percentile(done, 99)) / TM.CONTROL_HZ
        T_s = math.ceil(p99 + 2.0)
        cz = res["confirm_tcp_minus_cube_z"].numpy()
        gdz = {s: float(np.nanmedian(cz[[k for k, z in enumerate(sizes) if z == s]])) for s in ("small", "large")}
        out = {"stage": "0A.2 timing", "task": args.task, "seed": seed, "episodes": n, "cap_steps": CAP_STEPS,
               "completion_fraction": float((cs >= 0).mean()), "completion_time_s_p50": float(np.percentile(done, 50)) / 50,
               "completion_time_s_p99": p99, "T_s": T_s, "T_steps": T_s * TM.CONTROL_HZ, "grasp_dz_m": gdz,
               "success_at_cap": float(res["success"].float().mean()), "git_sha": sha(), "expert_version": EXPERT_VERSION,
               "expert": expert_kw, "detector_version": DETECTOR_VERSION, "wall_s": round(time.time() - t0, 1),
               "nan_guard_events": u.nan_guard_events, "cube_yaw_max_deg": args.cube_yaw_max_deg}
        timing_file().write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    else:
        epd = {k: res[k].tolist() for k in OUTCOME_KEYS}
        epd["target_size"] = sizes
        epd["specs"] = [e.spec.to_json() for e in eps]
        out = {"stage": args.prefix, "task": args.task, "split": args.split, "seed": seed, "start": args.start, "n": n,
               "T": T, "num_envs": args.num_envs, "git_sha": sha(), "expert_version": EXPERT_VERSION,
               "expert": expert_kw, "detector_version": DETECTOR_VERSION, "thresholds": Thresholds().to_dict(),
               "consts113": Consts113().to_dict(), "timing": {"period": TM.PERIOD, "latency": TM.LATENCY},
               "grasp_dz_m": grasp_dz.tolist() if grasp_dz is not None else None,
               "geometry": {"edge": geo.edge.tolist(), "z_rest": geo.z_rest.tolist(), "h_pad": geo.h_pad, "l_pad": geo.l_pad},
               "success_rate": float(np.mean(epd["success"])), "nan_guard_events": u.nan_guard_events,
               "wall_s": round(time.time() - t0, 1), "cube_yaw_max_deg": args.cube_yaw_max_deg, "episodes": epd}
        f = RES / f"{args.prefix}_{args.task}_s{seed}_{args.start}.json"
        f.write_text(json.dumps(out))
        print(json.dumps({k: v for k, v in out.items() if k not in ("episodes", "thresholds", "consts113")}, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
