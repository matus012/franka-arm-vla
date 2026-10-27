"""GATES 0C summary: results/0c/0c_check_*.json (scripts/s0c_check.py on PERUN) + results/0c_audit.txt -> results/0c_demos.json.

  python scripts/s0c_report.py --gpu-h <collect GPU-h> --jobs <job ids>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("0c_s1_pick", "0c_s2_pick", "0c_s3a_place", "0c_s3b_stack")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-h", type=float, required=True)
    p.add_argument("--jobs", required=True)
    a = p.parse_args()
    audit = (ROOT / "results" / "0c_audit.txt").read_text()
    viol = {Path(d).name: int(v) for d, v in re.findall(r'"dataset": "([^"]+)",\s*"violations": (\d+)', audit)}
    sets = {}
    for n in NAMES:
        c = json.loads((ROOT / "results" / "0c" / f"0c_check_{n}.json").read_text())
        ex, es = c["export"], c["expert_success"]
        sets[n] = {"task": c["task"], "ceiling_pct": c["ceiling_pct"],
                   "dataset": ex["root"].replace("/data", "/mnt/project/<project-id>/113_vla_tabletop/data"),
                   "episodes": ex["episodes"], "rows": ex["rows"], "train": ex["train_episodes"], "val": ex["val_episodes"],
                   "disk_gb": ex["disk_bytes"] / 1e9, "pooled": es["pooled"],
                   "batches": [{k: b[k] for k in ("start", "episodes", "kept", "attempted", "rate_pct", "ci95_pct",
                                                  "nan_guard_events", "episodes_per_h", "wall_s")} | {"failures": b["per_task"]}
                               for b in es["batches"]],
                   "per_colour": {k[7:]: v for k, v in es["per_target"].items() if k.startswith("colour:")},
                   "per_size": {k[5:]: v for k, v in es["per_target"].items() if k.startswith("size:")},
                   "per_cube": {k[5:]: v for k, v in es["per_target"].items() if k.startswith("cube:")},
                   "noise_std": c["columns_noise"]["noise_std_per_arm_dim"],
                   "state_dims_std_lt_1e-6": c["columns_noise"]["state_dims_std_lt_1e-6"], "video": c["video"],
                   "audit_violations": viol.get(n), "items": c["items"] | {"2_audit_clean": viol.get(n) == 0},
                   "sheet": f"results/frames/0c/{n}.png", "sheet_episodes": c["sheet"]["episodes"]}
    out = {"gate": "0C", "datasets": sets, "s4r": "0c_s4r_mix DEFERRED (GATES '0C note', not generated)",
           "ceilings_source": "GATES 0A.2 attempt 1 RESULT (9e0d0af): pick 100, place 100, stack 97.3 %",
           "pass_items_1_to_4": all(all(s["items"].values()) for s in sets.values()),
           "gpu_h_collect": a.gpu_h, "jobs": a.jobs, "audit_mutants": "results/0c_audit_mutants.txt"}
    (ROOT / "results" / "0c_demos.json").write_text(json.dumps(out, indent=1))
    for n, s in sets.items():
        pc = " ".join(f"{k} {v['rate_pct']:.1f}" for k, v in s["per_colour"].items())
        print(n, s["pooled"]["kept"], s["pooled"]["attempted"], f"{s['pooled']['rate_pct']:.2f}",
              [f"{b['rate_pct']:.1f}" for b in s["batches"]], "|", pc, "|", s["episodes"], s["rows"], s["train"], s["val"],
              f"{s['disk_gb']:.2f}GB", s["audit_violations"], s["state_dims_std_lt_1e-6"], f"psnr {s['video']['psnr_min']:.2f}")
    print("PASS 1-4:", out["pass_items_1_to_4"])


if __name__ == "__main__":
    main()
