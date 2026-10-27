"""0B.3 report (not gated): which demos lift at the expert's CLOSE_MAX timeout, read from the exported `action` column
(actions.npz of `s0b3_prep.py prereq`). Close = first step with claw > 0.5; lift = first step after it with the z
delta (arm dim 2) > 0.05. The expert holds dz = 0 through CLOSE and ramps it up on LIFT, so lift - close = CLOSE_MAX
means the timeout fired (checked on the attempt-0 demos: 50 for states 0, 2, 3, 4, 7, 8, 9, 9-10 for 1, 5, 6 =
the GATES attempt-1 entry's split).

  python scripts/s0b3_lift_delay.py results/0b3_r1_aux/actions.npz --out results/0b3_r1_aux/lift_delay.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLOSE_MAX = int(re.search(r"^CLOSE_MAX = (\d+)", (ROOT / "vla_tabletop" / "expert.py").read_text(), re.M).group(1))

p = argparse.ArgumentParser()
p.add_argument("actions", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
d = np.load(a.actions)
per = {}
for k in sorted(d.files, key=int):
    x = d[k]
    close = int(np.flatnonzero(x[:, 4] > 0.5)[0])
    lift = int(close + np.flatnonzero(x[close:, 2] > 0.05)[0])
    per[k] = {"close_step": close, "lift_step": lift, "delay": lift - close, "timeout": lift - close >= CLOSE_MAX}
res = {"actions": str(a.actions), "close_max": CLOSE_MAX, "per_demo": per,
       "n_timeout": sum(v["timeout"] for v in per.values()), "n": len(per)}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: (v["delay"], v["timeout"]) for k, v in per.items()}), "n_timeout", res["n_timeout"])
