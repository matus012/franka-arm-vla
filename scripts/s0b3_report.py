"""Assemble a 0B.3 result JSON + loss curve from the pulled PERUN artifacts (laptop, no GPU).

  python scripts/s0b3_report.py --aux results/0b3_r1_aux --out results/0b3_overfit_r1.json \
      --frames results/frames/0b3_r1 --attempt "0 (re-run)" --jobs 95798=collect ...

aux dir: train_summary.json, steps.jsonl, offline.json (s0b3_prep offline), prereq_b.json, replay.json, eval.json
(s0b3_overfit eval), lift_delay.json (s0b3_lift_delay), tests.txt (pytest summary line of prerequisite c).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--aux", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--frames", type=Path, required=True)
p.add_argument("--attempt", required=True)
p.add_argument("--jobs", nargs="+", default=[])
a = p.parse_args()
J = lambda n: json.loads((a.aux / n).read_text())  # noqa: E731
tr, off, pb, rp, ev, ld = (J(n) for n in ("train_summary.json", "offline.json", "prereq_b.json", "replay.json",
                                          "eval.json", "lift_delay.json"))
steps = [json.loads(x) for x in (a.aux / "steps.jsonl").read_text().splitlines()]
loss = np.array([s["loss"] for s in steps])
a.frames.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7, 4))
plt.semilogy(loss, lw=0.4, alpha=0.5, label="per step")
plt.semilogy(np.arange(49, len(loss)), np.convolve(loss, np.ones(50) / 50, "valid"), label="mean of 50")
plt.xlabel("step"), plt.ylabel("flow-matching loss"), plt.legend(), plt.title(f"0B.3 attempt {a.attempt}")
plt.savefig(a.frames / "loss_curve.png", dpi=120), plt.close()
eps = [e for e in ev["episodes"] if e["noise_seed"] < 10]
held = [e for e in eps if e["grasp_confirmed"] and not (e["success"] or e["early_close"] or e["drop"])]
out = {"gate": "0B.3", "attempt": a.attempt, "result": "PASS" if ev["pass"] else "FAIL", "success": ev["success"],
       "n": ev["n_scored"], "ci95": ev["ci95"], "bar": ev["bar"], "per_state_success": ev["per_state_success"],
       **{k: ev[f"n_{k}"] for k in ("slam", "jam", "early_close", "drop", "invalid")},
       "held_not_lifted": len(held), "held_not_lifted_states": sorted({e["state"] for e in held}),
       "demo_lift_by_timeout": {"n": ld["n_timeout"], "of": ld["n"], "close_max": ld["close_max"],
                                "delay_per_demo": {k: v["delay"] for k, v in ld["per_demo"].items()}},
       "offline": {k: off[k] for k in ("mae_arm_norm", "claw_acc", "mae_arm_norm_per_query", "ckpt_sha")},
       "training": {k: tr.get(k) for k in ("git_sha", "argv", "final_step", "total_steps", "samples_seen_this_run",
                                           "samples_per_epoch", "steps_per_epoch", "epochs_done", "wall_s",
                                           "steady_state", "peak_mem_allocated_gib", "flags", "overfit_abort_check",
                                           "loss_last50_mean", "final_ckpt", "dtype", "scheduler")},
       "lr_at_final_step": steps[-1]["lr"],
       "loss_curve_mean_per_100": {str(i + 100): float(loss[i:i + 100].mean()) for i in range(0, len(loss), 100)},
       "prerequisites": {"a_replay_ceiling": {"success": rp["success"], "n": rp["n_scored"], "layout": rp["layout"],
                                              "pass": rp["pass"]},
                         "b_norm_roundtrip": {"max_abs_err": pb["max_abs_roundtrip_err"], "tol": pb["tol"],
                                              "claw_flips": pb["claw_threshold_flips"], "norm_report": pb["norm_report"],
                                              "pass": pb["pass"]},
                         "c_tests": (a.aux / "tests.txt").read_text().strip()},
       "eval": {k: ev[k] for k in ("B", "N", "H", "T_steps", "layout", "ckpt", "ckpt_sha", "stats_sha", "server_s_p50",
                                   "server_s_p95", "wall_s", "gpu_h", "git_sha", "nan_guard_events")},
       "jobs": dict(j.split("=", 1) for j in a.jobs), "frames": [str(a.frames)], "eval_episodes": ev["episodes"]}
a.out.write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("result", "success", "per_state_success", "slam", "jam", "early_close", "drop",
                                      "held_not_lifted", "demo_lift_by_timeout", "lr_at_final_step")}))
print("offline per query", np.round(off["mae_arm_norm_per_query"], 3).tolist())
