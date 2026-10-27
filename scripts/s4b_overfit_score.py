"""Overfit capacity check: score an eval_policy closed-loop JSON on the trained-on vs the other episodes (CPU, any venv).

  python scripts/s4b_overfit_score.py --eval /runs/s4b_overfit/eval_s32500.json \
      --summary /runs/s4b_overfit/train_summary.json --out /runs/s4b_overfit/overfit_score.json

`--eval` = eval_policy on the stream the training episodes came from (e.g. --stage s2 --split train --seed 32500
--episodes 128 with the collect flags); `--summary` = train_baseline's train_summary.json of a `--max-train-episodes K`
run, whose `kept_train_episodes` lists (seed, episode_index) = the stream seed and episode index of each trained episode.
A kept episode maps to the eval record with the same `index` when the seed equals the eval config's seed; kept episodes of
another seed, or whose index has no eval record, are listed (not scored). Counts and rates (95 % Wilson) of success /
early_close / grasp_confirmed / wrong_object on the trained-on records and, separately, on the records not trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("success", "early_close", "grasp_confirmed", "wrong_object")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rates(recs: list[dict]) -> dict:
    n = len(recs)
    return {"n": n, **{m: {"count": (k := sum(bool(r[m]) for r in recs)), "n": n, "rate": k / n if n else None,
                           "ci95": wilson(k, n)} for m in METRICS}}


def score(ev: dict, summary: dict) -> dict:
    kept = summary["kept_train_episodes"]
    assert kept is not None and all("seed" in k for k in kept), "train_summary has no (seed, episode_index) of kept episodes"
    seed = ev["config"]["seed"]
    by_idx = {r["index"]: r for r in ev["episodes"]}
    want = {k["episode_index"] for k in kept if k["seed"] == seed}
    trained = [by_idx[i] for i in sorted(want) if i in by_idx]
    rest = [r for i, r in sorted(by_idx.items()) if i not in want]
    return {"eval_seed": seed, "n_kept": len(kept), "n_kept_same_seed": len(want),
            "kept_other_seed": [k for k in kept if k["seed"] != seed],
            "kept_without_eval_record": sorted(i for i in want if i not in by_idx),
            "trained_on": rates(trained), "not_trained_on": rates(rest), "n_eval_records": len(by_idx)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = score(json.loads(a.eval.read_text()), json.loads(a.summary.read_text())) | {"eval": str(a.eval), "summary": str(a.summary)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1))
    for g in ("trained_on", "not_trained_on"):
        print(f"[overfit] {g} n={r[g]['n']} " + " ".join(f"{m}={r[g][m]['count']} ({r[g][m]['rate']})" for m in METRICS), flush=True)
    print(f"[overfit] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
