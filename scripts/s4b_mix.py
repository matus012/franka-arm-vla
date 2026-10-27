"""Instrument B-I4 (GATES Stage 4 re-plan): baseline mix / sampler check from a run's steps.jsonl + the dataset. CPU.

  python scripts/s4b_mix.py --dataset /data/datasets/0c_s4_all_rl224_y15g --run /runs/s4_baseline_seed0 \
      --out results/s4b/seed0/mix.json [--mutant drop-stream|seed1]

Per epoch (epoch of step s = (s - 1) // steps_per_epoch, steps_per_epoch = S // batch because the trainer uses
drop_last): every logged row is a `train` row (no `val` row); no window is drawn twice; the rows of every step equal
WindowSampler(seed, epoch).order(epoch)[k*bs:(k+1)*bs] (the trainer's sampler, train rows only); each stream's
(stage, seed) draws equal its train window count MINUS the windows of the epoch's dropped tail (the last S mod bs
windows of the shuffled order -- the sampler check above proves they are the tail; total deficit < bs, none negative).
Streams = the (file_attrs.stage, seed) of meta/episodes_113.jsonl. Mutants: `drop-stream` (the draws of the last stream
are removed before counting) and `seed1` (the sampler read with seed + 1) must FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_policy.train_smolvla import WindowSampler  # noqa: E402


def read_cols(d: Path) -> dict:
    import pyarrow.parquet as pq

    t = [pq.read_table(f, columns=["index", "episode_index", "frame_idx", "split"]) for f in sorted((d / "data").rglob("*.parquet"))]
    c = {k: np.concatenate([np.asarray(x.column(k).to_pylist()) for x in t]).reshape(-1) for k in ("index", "episode_index", "frame_idx", "split")}
    o = np.argsort(c["index"], kind="stable")
    assert np.array_equal(c["index"][o], np.arange(len(o))), "dataset `index` is not 0..n-1"
    return {k: v[o] for k, v in c.items()}


def check(steps: list[dict], ep: np.ndarray, fi: np.ndarray, is_val: np.ndarray, stream: np.ndarray, seed: int,
          bs: int, epochs: int, mutant: str | None = None) -> dict:
    """Pure. stream: int per row (stream id). steps: steps.jsonl rows (step, index[, epoch])."""
    sam = WindowSampler(ep, fi, seed + (1 if mutant == "seed1" else 0), keep=~is_val)
    S = len(sam)
    spe = S // bs
    key = ep.astype(np.int64) * (int(fi.max()) + 1) + fi.astype(np.int64)
    ids = sorted(set(stream.tolist()))
    win = {s: len(np.unique(key[(~is_val) & (stream == s)])) for s in ids}
    assert sum(win.values()) == S
    drop = ids[-1] if mutant == "drop-stream" else None
    per, bad_steps, epoch_field_bad = {}, [], 0
    by_epoch: dict[int, list[dict]] = {e: [] for e in range(epochs)}
    for r in steps:
        e = (r["step"] - 1) // spe
        epoch_field_bad += "epoch" in r and r["epoch"] != e
        if e in by_epoch:
            by_epoch[e].append(r)
    for e in range(epochs):
        order = sam.order(e)
        rs = by_epoch[e]
        rows = np.concatenate([np.asarray(r["index"], np.int64) for r in rs]) if rs else np.zeros(0, np.int64)
        for r in rs:  # indices == the sampler's, on every step
            k = (r["step"] - 1) % spe
            if r["index"] != order[k * bs:(k + 1) * bs].tolist():
                bad_steps.append(r["step"])
        if drop is not None:
            rows = rows[stream[rows] != drop]
        w = key[rows]
        draws = {s: int((stream[rows] == s).sum()) for s in ids}
        dropped_tail = len(order) - spe * bs
        deficit = {s: win[s] - draws[s] for s in ids}
        per[e] = {"steps": len(rs), "rows": int(len(rows)), "val_rows": int(is_val[rows].sum()),
                  "duplicate_windows": int(len(w) - len(np.unique(w))), "draws_per_stream": draws,
                  "train_windows_per_stream": win, "deficit_per_stream": deficit, "dropped_tail": dropped_tail}
        per[e]["pass"] = bool(len(rs) == spe and per[e]["val_rows"] == 0 and per[e]["duplicate_windows"] == 0
                              and all(0 <= v for v in deficit.values()) and sum(deficit.values()) == dropped_tail
                              and dropped_tail < bs)
    res = {"seed": seed, "batch_size": bs, "epochs": epochs, "S": S, "steps_per_epoch": spe, "steps_logged": len(steps),
           "streams": {str(s): win[s] for s in ids}, "steps_with_sampler_mismatch": len(bad_steps),
           "first_mismatch": bad_steps[:5], "epoch_field_mismatch": epoch_field_bad, "per_epoch": per, "mutant": mutant}
    res["pass"] = bool(all(v["pass"] for v in per.values()) and not bad_steps and epoch_field_bad == 0
                       and len(steps) == spe * epochs)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mutant", choices=("drop-stream", "seed1"), default=None)
    a = ap.parse_args()
    summ = json.loads((a.run / "train_summary.json").read_text())
    c = read_cols(a.dataset)
    side = sorted((json.loads(x) for x in (a.dataset / "meta" / "episodes_113.jsonl").read_text().splitlines()),
                  key=lambda s: s["lerobot_episode_index"])
    keys = sorted({(s["file_attrs"]["stage"], int(s["seed"])) for s in side})
    sid = {s["lerobot_episode_index"]: keys.index((s["file_attrs"]["stage"], int(s["seed"]))) for s in side}
    ep = c["episode_index"].astype(np.int64)
    stream = np.array([sid[int(e)] for e in ep])
    steps = [json.loads(x) for x in (a.run / "steps.jsonl").read_text().splitlines()]
    res = check(steps, ep, c["frame_idx"], c["split"].astype(str) == "val", stream, int(summ["argv"]["seed"]),
                int(summ["argv"]["batch_size"]), int(summ["epochs"]), a.mutant)
    res |= {"dataset": str(a.dataset), "run": str(a.run), "stream_keys": [list(k) for k in keys]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("B-I4=" + json.dumps({k: res[k] for k in ("pass", "mutant", "S", "steps_per_epoch", "steps_with_sampler_mismatch")}))
    if not res["pass"] and a.mutant is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
