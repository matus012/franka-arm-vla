"""Underfit vs time-memorisation diagnostic for BaselinePolicy checkpoints (policy venv, CPU or --device cuda).

  python scripts/s4b_underfit.py --dataset /data/datasets/0c_s4_all_rl224_y15g --split val \
      --ckpt /runs/s4_baseline_seed2/ckpt_028638 /runs/s4_baseline_seed2/ckpt_057276 --out underfit_val.json [--max-rows 4096]

Rows: a seeded random subset (`--max-rows`) of the rows of the split (`val` / `train` episodes, the dataset's `split`
column) PLUS every t=0 row (frame_index 0) of that split's episodes. Images / state / labels come from the LeRobot dataset
exactly as the trainer builds them (train_baseline.make_ds + collate, no augmentation); the action chunk and its pad mask
are rebuilt from the action column (row j of an episode of length L: positions i < N -> action[min(j+i, L-1)], padded
when j+i >= L; asserted equal to the dataset item on the first batch). Predictions are BaselinePolicy.predict, i.e. what
the server executes (arm clipped to [-1, 1], claw = sigmoid > 0.5).
Metrics per group (all tasks / pick / place / stack, task = labels[0]) and subset (all sampled rows / t=0 rows / rows within
+-10 steps of a claw label change; the "all" subset therefore includes the forced t=0 rows): per-arm-dim MAE in raw units over
the unpadded chunk positions, and claw accuracy over the same positions.
References, built from TRAIN-split rows only and scored on the same rows:
  M = global mean predictor (train mean of the action column per arm dim; claw = train majority, a tie counts as 0);
  T = time-indexed mean predictor: per (task, frame_index) and chunk position, the mean unpadded train action (claw by
      majority); a (task, frame_index, position) with no train data falls back to M (count in `T_fallback_positions`).
`ratio_vs_M` / `ratio_vs_T` = model MAE / reference MAE per arm dim (None where the reference MAE is 0). ~1 vs M = underfit;
~1 vs T without beating it = learned the time schedule, not the scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_policy import train_baseline as TB  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402
from vla_policy.baseline import BaselinePolicy  # noqa: E402

TASKS = ("pick", "place", "stack")
SUBSETS = ("all", "t0", "claw_change")
NEAR = 10  # +- steps around a claw label change


def layout(ep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row index within its episode, episode length) per row; episodes must be contiguous and ordered."""
    _, first, inv, cnt = np.unique(ep, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    assert (np.diff(inv) >= 0).all() and (np.diff(first) > 0).all(), "episode rows not contiguous / ordered"
    return np.arange(len(ep)) - first[inv], cnt[inv]


def chunks(action: np.ndarray, j: np.ndarray, L: np.ndarray, rows: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(R,n,5) action chunk and (R,n) pad mask of `rows` (episode-end clamp, as LeRobot's delta_timestamps)."""
    i = np.arange(n)[None]
    src = rows[:, None] + np.minimum(i, L[rows, None] - 1 - j[rows, None])
    return action[src], (j[rows, None] + i) >= L[rows, None]


def claw_change_near(action: np.ndarray, ep: np.ndarray) -> np.ndarray:
    """bool per row: within +-NEAR steps (same episode) of a row whose claw label differs from the previous row's."""
    claw = action[:, 4] > 0.5
    chg = np.r_[False, (claw[1:] != claw[:-1]) & (ep[1:] == ep[:-1])]
    idx, near = np.arange(len(ep)), np.zeros(len(ep), bool)
    for o in range(-NEAR, NEAR + 1):
        s = np.clip(idx + o, 0, len(ep) - 1)
        near |= chg[s] & (ep[s] == ep)
    return near


def reference_tables(action: np.ndarray, task: np.ndarray, j: np.ndarray, L: np.ndarray, train: np.ndarray, n: int) -> dict:
    """Reference predictors from the rows `train` (bool per row) only. -> M: arm (4,), claw int; T: arm (K,n,4), claw
    (K,n) bool, count (K,n) unpadded train samples, key = task * (max j + 1) + j."""
    tr = np.flatnonzero(train)
    m_arm = action[tr, :4].astype(np.float64).mean(0)
    m_claw = bool(action[tr, 4].astype(np.float64).mean() > 0.5)
    w = int(j.max()) + 1
    k = 3 * w
    key = task[tr] * w + j[tr]
    cnt, s_arm, s_claw = np.zeros((k, n)), np.zeros((k, n, 4)), np.zeros((k, n))
    for i in range(n):
        v = (j[tr] + i) < L[tr]
        src, kv = tr[v] + i, key[v]
        cnt[:, i] = np.bincount(kv, minlength=k)
        s_claw[:, i] = np.bincount(kv, weights=action[src, 4].astype(np.float64), minlength=k)
        for d in range(4):
            s_arm[:, i, d] = np.bincount(kv, weights=action[src, d].astype(np.float64), minlength=k)
    ok = cnt > 0
    t_arm = np.where(ok[..., None], s_arm / np.maximum(cnt, 1)[..., None], m_arm)
    t_claw = np.where(ok, s_claw / np.maximum(cnt, 1) > 0.5, m_claw)
    return {"M": {"arm": m_arm, "claw": m_claw}, "T": {"arm": t_arm, "claw": t_claw, "count": cnt, "w": w}}


def reference_preds(ref: dict, task: np.ndarray, j: np.ndarray, rows: np.ndarray, n: int,
                    pad: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(M preds (R,n,5), T preds (R,n,5), number of unpadded (row, position) entries where T fell back to M)."""
    m = np.empty((len(rows), n, 5))
    m[..., :4], m[..., 4] = ref["M"]["arm"], ref["M"]["claw"]
    t = ref["T"]
    key = task[rows] * t["w"] + j[rows]
    out = np.concatenate([t["arm"][key], t["claw"][key][..., None]], -1).astype(np.float64)
    miss = t["count"][key] == 0
    out[miss] = m[miss]
    return m, out, int((miss & ~pad).sum())


def score(pred: np.ndarray, tgt: np.ndarray, pad: np.ndarray, sel: np.ndarray) -> dict:
    k = ~pad[sel]
    if not k.any():
        return {"n_rows": int(sel.sum()), "arm_mae": None, "arm_mae_mean": None, "claw_acc": None}
    mae = (np.abs(pred[sel, :, :4] - tgt[sel, :, :4]) * k[..., None]).sum((0, 1)) / k.sum()
    acc = float(((pred[sel, :, 4] > 0.5) == (tgt[sel, :, 4] > 0.5))[k].mean())
    return {"n_rows": int(sel.sum()), "arm_mae": mae.tolist(), "arm_mae_mean": float(mae.mean()), "claw_acc": acc}


def evaluate(pred: np.ndarray, tgt: np.ndarray, pad: np.ndarray, task: np.ndarray, t0: np.ndarray, near: np.ndarray) -> dict:
    """metrics[group][subset] for per-row arrays of the sampled rows."""
    subsets = {"all": np.ones(len(task), bool), "t0": t0, "claw_change": near}
    return {g: {s: score(pred, tgt, pad, m & ((task == TASKS.index(g)) if g in TASKS else True))
                for s, m in subsets.items()} for g in ("overall",) + TASKS}


def ratio(num: dict, den: dict) -> dict:
    """model metrics / reference metrics per group, subset: arm MAE per dim and mean."""
    def r(a, b):
        return None if a is None or not b else a / b
    return {g: {s: {"arm_mae": None if num[g][s]["arm_mae"] is None else [r(a, b) for a, b in zip(num[g][s]["arm_mae"], den[g][s]["arm_mae"])],
                    "arm_mae_mean": r(num[g][s]["arm_mae_mean"], den[g][s]["arm_mae_mean"])} for s in num[g]} for g in num}


def sample_rows(frame_in_ep: np.ndarray, in_split: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Seeded random subset of the split's rows plus all its t=0 rows (sorted)."""
    pool = np.flatnonzero(in_split)
    pick = np.random.default_rng(seed).choice(pool, size=min(max_rows, len(pool)), replace=False)
    return np.union1d(pick, pool[frame_in_ep[pool] == 0])


@torch.no_grad()
def run(ds, ckpts: list[Path], split: str, max_rows: int, seed: int = 0, bs: int = 128, workers: int = 0,
        device: str = "cpu") -> dict:
    models = {str(c): BaselinePolicy.load(c, device) for c in ckpts}
    n = {m.cfg.n_action_steps for m in models.values()}
    assert len(n) == 1, f"ckpts differ in n_action_steps: {n}"
    n = n.pop()
    ep = T.column(ds, "episode_index").reshape(-1)
    action = T.column(ds, "action").astype(np.float32).reshape(len(ep), -1)
    task = T.column(ds, "labels").reshape(len(ep), 4)[:, 0].astype(np.int64)
    j, L = layout(ep)
    fi = T.column(ds, "frame_index").reshape(-1)
    assert np.array_equal(fi, j), "frame_index column != row index within the episode"
    is_val, _, _ = T.split_episodes(ds)
    ref = reference_tables(action, task, j, L, ~is_val, n)
    rows = sample_rows(j, is_val if split == "val" else ~is_val, max_rows, seed)
    tgt, pad = chunks(action, j, L, rows, n)
    preds = {k: np.zeros((len(rows), n, 5), np.float32) for k in models}
    dl = DataLoader(ds, batch_size=bs, sampler=rows.tolist(), num_workers=workers, collate_fn=lambda it: TB.collate(it, False))
    o = 0
    for b in dl:
        m = len(b["index"])
        if o == 0:
            assert np.array_equal(b["index"].numpy(), rows[:m])
            assert np.array_equal(b["action"].numpy(), tgt[:m]) and np.array_equal(b["pad"].numpy(), pad[:m]), \
                "rebuilt action chunk != dataset item"
        for k, mod in models.items():
            preds[k][o:o + m] = mod.predict(b["side"].numpy(), b["wrist"].numpy(), b["state"].numpy(), b["labels"].numpy())
        o += m
    assert o == len(rows)
    t0, near = j[rows] == 0, claw_change_near(action, ep)[rows]
    mp, tp, fb = reference_preds(ref, task, j, rows, n, pad)
    ev = lambda p: evaluate(p, tgt, pad, task[rows], t0, near)  # noqa: E731
    mm, tm = ev(mp), ev(tp)
    return {"split": split, "seed": seed, "max_rows": max_rows, "N": n, "n_rows_sampled": len(rows), "n_t0_rows": int(t0.sum()),
            "n_near_claw_change_rows": int(near.sum()), "train_rows_for_references": int((~is_val).sum()),
            "references": {"M": mm, "T": tm, "M_arm_mean": ref["M"]["arm"].tolist(), "M_claw": ref["M"]["claw"],
                           "T_fallback_positions": fb, "T_positions_total": int((~pad).sum())},
            "ckpts": {k: {"metrics": (e := ev(p)), "ratio_vs_M": ratio(e, mm), "ratio_vs_T": ratio(e, tm)}
                      for k, p in preds.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--split", choices=("val", "train"), required=True)
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        assert torch.cuda.is_available(), "--device cuda but CUDA unavailable. STOP."
    n = BaselinePolicy.load(a.ckpt[0], "cpu").cfg.n_action_steps
    meta = TB.make_ds(a.dataset, None, n, a.video_backend)
    ds = TB.make_ds(a.dataset, None, n, a.video_backend, meta.fps)
    r = run(ds, a.ckpt, a.split, a.max_rows, a.seed, a.batch_size, a.workers, a.device) | {
        "dataset": str(a.dataset), "git_sha": T.git_sha(), "device": a.device}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1))
    for k, c in r["ckpts"].items():
        for g in ("overall",) + TASKS:
            for s in SUBSETS:
                m, rm, rt = c["metrics"][g][s], c["ratio_vs_M"][g][s], c["ratio_vs_T"][g][s]
                if m["n_rows"]:
                    print(f"[underfit] {Path(k).name} {a.split} {g:7s} {s:11s} n={m['n_rows']:5d} arm_mae={m['arm_mae_mean']:.4f} "
                          f"claw_acc={m['claw_acc']:.3f} /M={rm['arm_mae_mean']:.3f} /T={rt['arm_mae_mean']:.3f}", flush=True)
    print(f"[underfit] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
