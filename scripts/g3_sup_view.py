"""Task 003 supplementary step 2 view (GATES 003 "Supplementary step 2 view"; reported only, NOT a reading), CPU.

From <stem>_divchunks.npz (scripts/g3_probe.py divonly: per layout k instruction chunks with noise q = 0, then member
0's instruction with noise q = 1; normalized `norm` and un-normalized `raw`, (rows, 50, 5)):
  - instruction / noise effect per arm dim (normalized), over dims 0-2 only, over 0-3 in un-normalized units;
  - heading of the summed xy motion over the chunk (raw dims 0, 1): |angle(q0) - angle(q1)| vs the max pairwise angle
    difference between instructions; "mode switch" = the noise chunk heads > 45 deg away from the q0 chunk.

  python scripts/g3_sup_view.py --chunks results/g3_sup/c034440_divchunks.npz --out results/g3_sup/view_034440.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_probe import divergence  # noqa: E402


def split(chunks: np.ndarray, layout: np.ndarray) -> list[np.ndarray]:
    """Rows of each layout, in order (k instruction chunks + 1 noise chunk)."""
    lids = list(dict.fromkeys(layout.tolist()))
    return [chunks[layout == l] for l in lids]


def effects(groups: list[np.ndarray], dims: list[int]) -> dict:
    ci = [g[:-1][..., dims] for g in groups]
    npair = [(g[0][..., dims], g[-1][..., dims]) for g in groups]
    return divergence(ci, npair)


def heading(c: np.ndarray) -> float:
    s = c[:, :2].sum(0)
    return float(np.degrees(np.arctan2(s[1], s[0])))


def adiff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    D = np.load(a.chunks)
    gn, gr = split(D["norm"], D["layout_id"]), split(D["raw"], D["layout_id"])
    per_dim = {str(d): effects(gn, [d]) for d in range(4)}
    noise_ang, instr_ang = [], []
    for g in gr:
        h = [heading(c) for c in g]
        noise_ang.append(adiff(h[0], h[-1]))
        ins = h[:-1]
        instr_ang.append(max(adiff(ins[i], ins[j]) for i in range(len(ins)) for j in range(i + 1, len(ins))))
    noise_ang, instr_ang = np.array(noise_ang), np.array(instr_ang)
    res = {"chunks": str(a.chunks), "n_layouts": len(gn),
           "normalized_dims_0_3": effects(gn, [0, 1, 2, 3]), "normalized_dims_0_2": effects(gn, [0, 1, 2]),
           "normalized_per_dim": per_dim, "raw_dims_0_3": effects(gr, [0, 1, 2, 3]), "raw_dims_0_2": effects(gr, [0, 1, 2]),
           "heading_deg": {"noise_median": float(np.median(noise_ang)), "instr_max_pair_median": float(np.median(instr_ang)),
                           "noise_gt45_frac": float((noise_ang > 45).mean()), "instr_gt45_frac": float((instr_ang > 45).mean())},
           "raw_summed_xy_norm_median": float(np.median([np.linalg.norm(c[:, :2].sum(0)) for g in gr for c in g]))}
    a.out.write_text(json.dumps(res, indent=1))
    print("G3_SUP=" + json.dumps({k: (v["ratio"] if isinstance(v, dict) and "ratio" in v else v) for k, v in res.items()
                                  if k not in ("chunks", "normalized_per_dim")}
                                 | {"per_dim_ratio": {d: round(v["ratio"], 3) for d, v in per_dim.items()}}))


if __name__ == "__main__":
    main()
