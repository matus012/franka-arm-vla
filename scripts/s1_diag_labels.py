"""GATES Stage 1 diagnosis (data only, no policy): how much of the stored clean label is a reaction to the previous
step's executed demo noise. Per arm dim over every episode of the given collect dirs:
  d_t = a_t - a_{t-1} (label first difference), n_{t-1} = action_exec_{t-1} - a_{t-1} (executed noise)
  -> std(a), std(d), least-squares slope of d_t on n_{t-1}, R^2 (fraction of the label's step-to-step change explained
     by the last noise draw), and the std of a's high-frequency part = std(d) / sqrt(2).
  python scripts/s1_diag_labels.py --src /data/0c/s1/b000 /data/c3r1_demos --out /runs/d1_labels/labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def episodes(src: Path):
    for f in sorted(src.glob("*.h5")):
        with h5py.File(f, "r") as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Group) and "action" in obj and "action_exec" in obj:
                    out.append((obj["action"][()], obj["action_exec"][()]))
            out: list = []
            h.visititems(visit)
            yield from out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, nargs="+")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    res = {}
    for src in a.src:
        A, D, N = [], [], []
        for act, exe in episodes(src):
            A.append(act[1:, :4])
            D.append(act[1:, :4] - act[:-1, :4])
            N.append(exe[:-1, :4] - act[:-1, :4])
        A, D, N = (np.concatenate(x).astype(np.float64) for x in (A, D, N))
        slope = (D * N).sum(0) / np.maximum((N * N).sum(0), 1e-12)
        r2 = 1 - ((D - slope * N) ** 2).sum(0) / np.maximum(((D - D.mean(0)) ** 2).sum(0), 1e-12)
        res[str(src)] = {"episodes": len(A) // 349 if len(A) else 0, "rows": len(A), "std_label": A.std(0).tolist(),
                         "std_label_diff": D.std(0).tolist(), "std_noise": N.std(0).tolist(),
                         "slope_diff_on_prev_noise": slope.tolist(), "r2": r2.tolist(),
                         "hf_std_over_label_std": (D.std(0) / np.sqrt(2) / A.std(0)).tolist()}
        print(src, json.dumps(res[str(src)]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
