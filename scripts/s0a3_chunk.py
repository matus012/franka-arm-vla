"""Stage 0A.3 (GATES.md) summary: chunked expert (N = 5, 10, 20; fallback 3) vs the unchunked reference on the same
episodes (paired). The runs themselves are scripts/s0a2_expert.py --mode run with --prefix 0a3_ref (fresh expert) or
--prefix 0a3_N<k> --chunk <k>, on the 0A.3 test seeds. This script merges them, applies the smart length rule to the
paired difference (decided at 300 if success(N) - success(ref) + 3 points is beyond +-5 points, else both extended to
1,000) and picks the largest passing N. Pure python.

Usage: python scripts/s0a3_chunk.py [--tasks pick,place,stack] [--candidates 5,10,20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lift_rl.metrics import wilson_ci  # noqa: E402
from vision_lift.frames import frame_step  # noqa: E402
from vla_common import timing as TM  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tasks", default="pick,place,stack")
p.add_argument("--candidates", default="5,10,20")
p.add_argument("--prefix", default="0a3", help="run prefix: 0a3 (attempt 1) or 0a3a2 (attempt 2)")
p.add_argument("--attempt", type=int, default=0, help="test seeds = base + attempt")
args = p.parse_args()
RES = ROOT / "results"
SEEDS = {"pick": 11300 + args.attempt, "place": 11310 + args.attempt, "stack": 11320 + args.attempt}
PF = args.prefix
KEYS = ("success", "slam", "jam", "drop", "early_close", "release_failure", "wrong_object")


def load(prefix: str, task: str) -> dict | None:
    files = sorted(RES.glob(f"{prefix}_{task}_s{SEEDS[task]}_*.json"))
    if not files:
        return None
    runs = sorted([json.loads(f.read_text()) for f in files], key=lambda r: r["start"])
    ep = {k: sum((r["episodes"][k] for r in runs), []) for k in runs[0]["episodes"]}
    assert ep["index"] == list(range(len(ep["index"]))), (prefix, task)
    ep["_T"], ep["_files"], ep["_wall"] = runs[0]["T"], [f.name for f in files], sum(r["wall_s"] for r in runs)
    return ep


def rate(x) -> dict:
    x = np.asarray(x, bool)
    k, n = int(x.sum()), int(x.size)
    return {"count": k, "n": n, "rate": k / n if n else None, "ci95": list(wilson_ci(k, n))}


def staleness(n: int, T: int) -> dict:
    s = [t - frame_step(t, TM.PERIOD, TM.LATENCY) for t in range(0, T, n) if t >= 2 * TM.PERIOD]
    return {"min": min(s), "mean": float(np.mean(s)), "max": max(s)}


sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
tasks = args.tasks.split(",")
ref = {t: load(f"{PF}_ref", t) for t in tasks}
(RES / f"{PF}_reference.json").write_text(json.dumps(
    {t: {"episodes": len(r["index"]), **{k: rate(r[k]) for k in KEYS}, "T": r["_T"], "files": r["_files"],
         "wall_s": r["_wall"]} for t, r in ref.items() if r}, indent=1))
decision = {"git_sha": sha, "period": TM.PERIOD, "latency": TM.LATENCY, "candidates": {}}
for n in [int(c) for c in args.candidates.split(",")]:
    out = {"N": n, "tasks": {}}
    ok_all, pending = True, False
    for t in tasks:
        c, r = load(f"{PF}_N{n}", t), ref[t]
        if c is None or r is None:
            ok_all, pending = False, True
            continue
        m = min(len(c["index"]), len(r["index"]))
        d300 = 100 * (np.mean(c["success"][:300]) - np.mean(r["success"][:300]))
        margin300 = d300 + 3.0
        if abs(margin300) > 5:
            dec, ok = "decided at 300", bool(margin300 >= 0)
        elif m >= 1000:
            dn = 100 * (np.mean(c["success"][:1000]) - np.mean(r["success"][:1000]))
            dec, ok = "decided at 1000", bool(dn + 3.0 >= 0)
        else:
            dec, ok, pending = "extend both to 1000", None, True
        ok_all &= bool(ok)
        out["tasks"][t] = {"episodes": int(m), "chunked": {k: rate(c[k][:m]) for k in KEYS},
                           "reference": {k: rate(r[k][:m]) for k in KEYS},
                           "diff_points": float(100 * (np.mean(c["success"][:m]) - np.mean(r["success"][:m]))),
                           "diff_points_first300": float(d300), "smart_length": dec, "pass": ok,
                           "mean_completion_s": float(np.mean([x for x in c["completion_step"][:m] if x >= 0])) / 50,
                           "staleness_steps": staleness(n, c["_T"]), "T": c["_T"], "files": c["_files"],
                           "wall_s": c["_wall"]}
    out["pass_all_tasks"] = None if pending else ok_all
    (RES / f"{PF}_chunk_N{n}.json").write_text(json.dumps(out, indent=1))
    decision["candidates"][n] = out["pass_all_tasks"]
passing = [n for n, v in decision["candidates"].items() if v]
decision["chosen_N"] = max(passing) if passing and all(v is not None for v in decision["candidates"].values()) else None
(RES / f"{PF}_chunk.json").write_text(json.dumps(decision, indent=1))
print(json.dumps(decision, indent=1))
for n in decision["candidates"]:
    o = json.loads((RES / f"{PF}_chunk_N{n}.json").read_text())
    for t, v in o["tasks"].items():
        print(n, t, v["smart_length"], v["pass"], round(v["diff_points"], 2), v["chunked"]["success"]["rate"],
              v["reference"]["success"]["rate"], v["staleness_steps"])
