"""GATES Stage 2 normalization table (policy venv, CPU): parent (Stage 1 checkpoint) vs Stage 2 MEAN_STD stats.

  policy/.venv/Scripts/python scripts/s2_norm.py --dataset <exported s2 set> --init <parent pretrained_model dir> \
      --out results/s2_norm.json [--root <datasets root>] [--video-backend pyav]

Stage 2 stats = vla_policy.train_smolvla.norm_stats on the train split (same STD floor), exactly as the trainer.
Parent stats = the init dir's preprocessor normalizer (state + action) and postprocessor unnormalizer (action).
Per dim: parent / s2 mean + std, ratio, near-constant flags (std < 1e-6), ratio outside [0.5, 2] listed with the
normalized range (x - mean_s2) / std_s2 over the s2 train rows and the fraction of rows beyond +-10.
instrument_finding = any dim with > 0.1 % of rows beyond +-10 -> exit code 1.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla_policy.train_smolvla import STD_FLOOR, norm_stats, open_dataset, resolve_dataset, split_episodes  # noqa: E402

RATIO_BAND, BEYOND, BEYOND_FRAC = (0.5, 2.0), 10.0, 0.001


def analyze(x: np.ndarray, parent: dict, s2: dict) -> list[dict]:
    """x (rows, D) s2 train rows; parent / s2: {"mean","std"} arrays as stored in the checkpoints (std already floored
    to 1 where near-constant). Pure function."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    raw_std = x.std(0)
    out = []
    for d in range(x.shape[1]):
        pm, ps, sm, ss = (float(a[k][d]) for a in (parent, s2) for k in ("mean", "std"))
        ratio = ss / ps if ps > 0 else float("inf")
        z = (x[:, d] - sm) / ss
        beyond = float((np.abs(z) > BEYOND).mean())
        row = {"dim": d, "parent_mean": pm, "parent_std": ps, "s2_mean": sm, "s2_std": ss, "std_ratio": ratio,
               "near_constant_parent": ps < STD_FLOOR, "near_constant_s2": bool(raw_std[d] < STD_FLOOR),
               "ratio_outside_band": not (RATIO_BAND[0] <= ratio <= RATIO_BAND[1]),
               "norm_min": float(z.min()), "norm_max": float(z.max()), "frac_beyond_10": beyond,
               "instrument_finding": beyond > BEYOND_FRAC}
        out.append(row)
    return out


def load_parent(init: Path) -> dict:
    from safetensors.numpy import load_file
    f = lambda pat: Path(sorted(glob.glob(str(init / pat)))[0])  # noqa: E731
    pre = load_file(str(f("policy_preprocessor_step_*_normalizer_processor.safetensors")))
    post = load_file(str(f("policy_postprocessor_step_*_unnormalizer_processor.safetensors")))
    get = lambda d, k: {s: d[f"{k}.{s}"].astype(np.float64) for s in ("mean", "std")}  # noqa: E731
    return {"observation.state": get(pre, "observation.state"), "action": get(post, "action"),
            "pre_action": get(pre, "action")}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--video-backend", default="pyav")
    a = p.parse_args(argv)
    ds = open_dataset(resolve_dataset(a.dataset, a.root), a.video_backend)
    is_val, _, _ = split_episodes(ds)
    stats, _ = norm_stats(ds, ~is_val)
    parent = load_parent(a.init)
    from vla_policy.train_smolvla import column
    res = {"dataset": a.dataset, "init": str(a.init), "train_rows": int((~is_val).sum()), "std_floor": STD_FLOOR,
           "ratio_band": list(RATIO_BAND), "beyond": BEYOND, "beyond_frac_threshold": BEYOND_FRAC,
           "parent_pre_post_action_equal": all(np.array_equal(parent["action"][s], parent["pre_action"][s])
                                               for s in ("mean", "std"))}
    for k in ("observation.state", "action"):
        x = column(ds, k).reshape(len(ds), -1)[~is_val]
        res[k] = analyze(x, parent[k], {s: stats[k][s].astype(np.float64) for s in ("mean", "std")})
    res["instrument_finding"] = any(r["instrument_finding"] for k in ("observation.state", "action") for r in res[k])
    res["ratio_outside_band"] = {k: [r["dim"] for r in res[k] if r["ratio_outside_band"]]
                                 for k in ("observation.state", "action")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(f"[s2_norm] instrument_finding={res['instrument_finding']} ratio_outside_band={res['ratio_outside_band']} -> {a.out}")
    return 1 if res["instrument_finding"] else 0


if __name__ == "__main__":
    sys.exit(main())
