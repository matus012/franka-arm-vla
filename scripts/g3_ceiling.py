"""Task 003 step 2 ceiling (GATES 003 "Step 2 ceiling"; reported only, NOT a reading; lead review F3), policy venv, CPU.

The training data's own instruction effect: in the exported paired set, the first 100 layout groups (sidecar
meta/episodes_113.jsonl order), each member's first 50 `action` rows (the trained-on label r), arm dims 0-3,
normalized with the checkpoint's action stats (results/s2a_pair_norm.json `parent_mean` / `parent_std`, = 034440's
pre / post stats), mean pairwise Frobenius distance between the members (same layout, different targets) — the
quantity step 2 calls the instruction effect, measured on the demonstrations. Also the first 10 rows.

  python scripts/g3_ceiling.py --dataset /data/datasets/0c_s2a_pair_rl224_y15 --norm results/s2a_pair_norm.json \
      --out /runs/g3_ceil/ceiling.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_LAYOUTS, T = 100, 50


def pairwise_effect(chunks: list[np.ndarray], steps: int | None = None) -> float:
    """chunks[l] (k_l, T, 4): mean over layouts of the mean pairwise Frobenius distance (as g3_probe.divergence)."""
    out = []
    for c in chunks:
        c = c[:, :steps]
        out.append(np.mean([np.linalg.norm(c[i] - c[j]) for i in range(len(c)) for j in range(i + 1, len(c))]))
    return float(np.mean(out))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import pyarrow.parquet as pq

    side = [json.loads(l) for l in (a.dataset / "meta" / "episodes_113.jsonl").read_text().splitlines() if l.strip()]
    lids = list(dict.fromkeys(int(s["layout_id"]) for s in side))[:N_LAYOUTS]
    want = {int(s["lerobot_episode_index"]): (int(s["layout_id"]), int(s["member"])) for s in side
            if int(s["layout_id"]) in set(lids)}
    nm = json.loads(a.norm.read_text())["action"]
    mu = np.array([r["parent_mean"] for r in nm[:4]])
    sd = np.array([r["parent_std"] for r in nm[:4]])
    rows: dict[int, list] = {}
    for f in sorted((a.dataset / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "action"]).to_pydict()
        for e, fi, act in zip(t["episode_index"], t["frame_index"], t["action"]):
            if e in want and fi < T:
                rows.setdefault(e, []).append((fi, act[:4]))
    chunks, raw, info = [], [], []
    for lid in lids:
        eps = sorted((m, e) for e, (l, m) in want.items() if l == lid)
        c = []
        for _, e in eps:
            r = sorted(rows[e])
            assert [x[0] for x in r] == list(range(T)), (e, len(r))
            c.append(np.array([x[1] for x in r], np.float64))
        c = np.stack(c)
        raw.append(c)
        chunks.append((c - mu) / sd)
        info.append({"layout_id": lid, "lerobot_episodes": [e for _, e in eps]})
    res = {"dataset": str(a.dataset), "n_layouts": len(chunks), "members_mean": float(np.mean([len(c) for c in chunks])),
           "data_instruction_effect": {"full_chunk": pairwise_effect(chunks), "first_10": pairwise_effect(chunks, 10)},
           "data_instruction_effect_raw": {"full_chunk": pairwise_effect(raw), "first_10": pairwise_effect(raw, 10)},
           "layouts": info}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("G3_CEIL=" + json.dumps({k: v for k, v in res.items() if k != "layouts"}))


if __name__ == "__main__":
    main()
