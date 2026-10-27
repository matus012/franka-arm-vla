"""GATES "Stage 0A.4 — decision rule after the expert fix": compare the fixed expert (lift_trigger "width") with the
attempt-0 expert ("settle") on the same validation-seed episodes (same env count / batch layout): per episode the TCP path
up to the close (every step before the first CLOSE entry of either run) and the CLOSE entry step. Material = median
path difference > 1 cm or median |entry-step change| > 5 steps.

Usage: python scripts/diag_0a4_rule.py --task pick --seed 21200 --n 100   (writes results/0a4_rule/<task>_s<seed>_<trigger>.pt)
       python scripts/diag_0a4_rule.py --summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["pick", "place", "stack"])
p.add_argument("--seed", type=int)
p.add_argument("--n", type=int, default=100)
p.add_argument("--T", type=int, default=250)
p.add_argument("--summary", action="store_true")
args = p.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "results" / "0a4_rule"
if args.summary:
    res = {"rule": "material if median TCP path difference before the close > 1 cm or median |close-entry step change| > 5",
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "tasks": {}}
    for f in sorted(OUT.glob("*_settle.pt")):
        a, b = torch.load(f), torch.load(str(f).replace("_settle.pt", "_width.pt"))
        ea, eb = a["close_entry"], b["close_entry"]
        diffs = []
        for i in range(len(ea)):
            k = int(min(ea[i], eb[i]))
            diffs.append(float(torch.linalg.norm(a["tcp"][:k, i] - b["tcp"][:k, i], dim=-1).mean()) if k > 0 else 0.0)
        d = np.array(diffs)
        de = (eb - ea).abs().float().numpy()
        res["tasks"][f.stem.replace("_settle", "")] = {"n": len(d), "path_diff_median_m": float(np.median(d)),
                                                        "path_diff_max_m": float(d.max()),
                                                        "close_entry_change_median_steps": float(np.median(de)),
                                                        "close_entry_change_max_steps": float(de.max())}
    allm = [v["path_diff_median_m"] for v in res["tasks"].values()]
    alle = [v["close_entry_change_median_steps"] for v in res["tasks"].values()]
    res["material"] = bool(max(allm) > 0.01 or max(alle) > 5)
    (ROOT / "results" / "0a4_rule.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0)

import gymnasium as gym  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vla_common.catalog import ALL_CUBES  # noqa: E402
from vla_common.tasks import sample_spec  # noqa: E402
from vla_tabletop.expert import CLOSE  # noqa: E402
from vla_tabletop.rollout import Runner, geometry, make_stream  # noqa: E402

STAGE = {"pick": "s2", "place": "s3a", "stack": "s3b"}[args.task]
meas = json.loads((ROOT / "results" / "0a1_measure.json").read_text())
tj = json.loads((ROOT / "results" / f"0a2_{args.task}_timing.json").read_text())
geo = geometry(meas["z_rest_m"], h_pad=float(np.mean(list(meas["h_pad_m"].values()))))
grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES])
cfg = S.make_cfg(args.n, args.seed)
assert_run_preconditions(cfg)
OUT.mkdir(parents=True, exist_ok=True)
with launch_simulation(cfg, {"headless": True}):
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    run = Runner(env, geo, grasp_dz=grasp_dz.to(u.device))
    eps = make_stream(args.seed, 0, args.n, lambda rng: sample_spec(STAGE, "train", rng))
    for trig in ("settle", "width"):
        log = {"tcp": [], "entry": None}

        def on_step(t, uu, s, lab, exe, ctrl, det):
            log["tcp"].append(s["tcp"].cpu())
            log["ctrl"] = ctrl

        run.run_batch(eps, args.T, {"fresh": True, "lift_trigger": trig}, on_step=on_step)
        entry = log["ctrl"].entered[:, CLOSE].cpu()
        entry = torch.where(entry < 0, torch.full_like(entry, args.T), entry)
        torch.save({"tcp": torch.stack(log["tcp"]), "close_entry": entry}, OUT / f"{args.task}_s{args.seed}_{trig}.pt")
        print(trig, "close entry median", float(entry.float().median()), flush=True)
    env.close()
