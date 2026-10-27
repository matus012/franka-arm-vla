"""GATES 0C gate items 1, 3, 4, 5 on an EXPORTED 0C dataset (PERUN, policy venv, cpu_short; item 2 = tests/test_data_audit.py).

  python scripts/s0c_check.py --dataset /data/datasets/0c_s2_pick --out /runs/k0c

Per dataset (sources = meta/export_113.json "sources", the collect batch dirs):
 1. counts / integrity: episodes = --kept (meta), rows per episode = T; every non-image column of the data parquet equals
    the HDF5 source exactly (state, action = label, frame_idx, frame_index 0..T-1, labels, split, task = instruction);
    PSNR >= 35 dB + frame identity (decoded row image closer to its own captured frame than to any other different one)
    on 10 episodes decoded with torchcodec.
 3. expert success per batch (collect_summary.json) and pooled vs the 0A.2 attempt-1 ceilings (pick 100, place 100,
    stack 97.3 %): pooled >= ceiling - 5 points, every batch >= ceiling - 10, every batch n >= 200; per target colour /
    size from the sidecars (attempted indices, failing indices) with the stream regenerated from (seed, i, sampler).
 4. noise: per arm dim over the exported episodes' rows with |label| <= 0.7: std(action_exec - action) in
    [0.095, 0.105]; claw columns identical; parquet action == HDF5 action exactly; mutant: parquet action vs HDF5
    action_exec must mismatch (the check would flag an exporter writing action_exec).
 5. decoded-frame sheet: 3 episodes (first, middle, last) decoded from the dataset: side | wrist at start, grasp, carry,
    release / hold, end, with instruction, labels, label and executed action.
Writes <out>/0c_check_<name>.json and <out>/<name>.png.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_policy.roundtrip import psnr  # noqa: E402

CEILING = {"pick": 100.0, "place": 100.0, "stack": 97.3}  # GATES 0A.2 attempt 1 RESULT (9e0d0af), percent
POOLED_TOL, BATCH_TOL, BATCH_MIN_N = 5.0, 10.0, 200  # GATES 0C item 3
PSNR_BAR, PSNR_EPISODES = 35.0, 10  # GATES 0C item 1 (0B.1 check 2)
NOISE_BAND, LABEL_ABS_MAX = (0.095, 0.105), 0.7  # GATES 0C item 4
CAMS = ("side", "wrist")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def rate(k: int, n: int) -> dict:
    return {"kept": k, "attempted": n, "rate_pct": 100 * k / n, "ci95_pct": [100 * x for x in wilson(k, n)]}


def expert_success(srcs: list[Path]) -> dict:
    """Per batch (collect dir) and pooled, plus per target colour / size, from the collector sidecars."""
    from vla_common.tasks import sample_spec

    batches, per = [], {}
    for d in srcs:
        s = json.loads((d / "collect_summary.json").read_text())
        cfg = s["config"]
        w = tuple(float(x) for x in cfg["mix"].split(","))
        batches.append({"dir": str(d), "start": cfg["start"], "episodes": cfg["episodes"], "envs": cfg["envs"],
                        **rate(s["kept"], s["attempted"]), "per_task": s["per_task"],
                        "nan_guard_events": s["nan_guard_events"], "cmd_resync_max_m": s["cmd_resync_max_m"],
                        "episodes_per_h": s["this_invocation"]["episodes_per_h"],
                        "wall_s": s["this_invocation"]["wall_s"], "disk_bytes": s["disk_bytes"]})
        for sc in sorted(d.glob(f"{s['prefix']}_*_e*.json")):
            c = json.loads(sc.read_text())
            fail = set(c["failing_indices"])
            for i in c["indices"]:
                spec = sample_spec(cfg["stage"], cfg["split"], np.random.default_rng([cfg["seed"], i]), w)
                for key in (f"colour:{spec.target[0]}", f"size:{spec.target[1]}", f"cube:{'_'.join(spec.target)}"):
                    k, n = per.get(key, (0, 0))
                    per[key] = (k + (i not in fail), n + 1)
    k, n = sum(b["kept"] for b in batches), sum(b["attempted"] for b in batches)
    return {"batches": batches, "pooled": rate(k, n), "per_target": {key: rate(*v) for key, v in sorted(per.items())}}


def read_parquet(root: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    cols = ["episode_index", "frame_index", "observation.state", "action", "frame_idx", "labels", "split", "task_index"]
    tabs = [pq.read_table(f, columns=cols) for f in sorted((root / "data").rglob("*.parquet"))]
    out = {}
    for c in cols:
        vals = [x for t in tabs for x in t.column(c).to_pylist()]
        out[c] = np.asarray([x[0] if isinstance(x, list) and c == "split" else x for x in vals],
                            dtype=object if c == "split" else None)
    tt = pq.read_table(root / "meta" / "tasks.parquet")
    tcol = "task" if "task" in tt.column_names else "__index_level_0__"
    out["tasks"] = dict(zip(tt.column("task_index").to_pylist(), tt.column(tcol).to_pylist()))
    return out


def columns_and_noise(root: Path, side: list[dict], expect=lambda act, exe: act) -> dict:
    """`expect(action, action_exec)` = the action column the export must hold (0C attempt 1: s1_relabel.relabel)."""
    pq = read_parquet(root)
    ep = pq["episode_index"].reshape(-1)
    order = np.argsort(ep, kind="stable")
    bounds = np.searchsorted(ep[order], np.arange(len(side) + 1))
    mism, rows_bad, mutant_flagged, diffs, labs, claw_bad = [], [], 0, [], [], 0
    state_all = pq["observation.state"]
    for s in side:
        e = s["lerobot_episode_index"]
        r = order[bounds[e]:bounds[e + 1]]
        with h5py.File(s["source"], "r") as f:
            g = f[s["group"]]
            st, act, exe, fi = g["state"][()], g["action"][()], g["action_exec"][()], g["frame_idx"][()]
        T = st.shape[0]
        if len(r) != T or T != int(s["T"]):
            rows_bad.append((e, len(r), T))
            continue
        lab = json.loads(s["labels"]) if isinstance(s["labels"], str) else s["labels"]
        ok = (np.array_equal(pq["frame_index"][r].reshape(-1), np.arange(T))
              and np.array_equal(state_all[r].astype(np.float32), st)
              and np.array_equal(pq["action"][r].astype(np.float32), expect(act, exe))
              and np.array_equal(pq["frame_idx"][r].reshape(-1), fi)
              and all(list(x) == lab for x in pq["labels"][r])
              and set(pq["split"][r]) == {s["split"]}
              and {pq["tasks"][int(k)] for k in pq["task_index"][r]} == {s["instruction"]})
        if not ok:
            mism.append(e)
        mutant_flagged += not np.array_equal(pq["action"][r].astype(np.float32), exe)
        diffs.append(exe[:, :4] - act[:, :4])
        labs.append(act[:, :4])
        claw_bad += int((exe[:, 4] != act[:, 4]).sum())
    d, lb = np.concatenate(diffs), np.concatenate(labs)
    std = [float(d[np.abs(lb[:, j]) <= LABEL_ABS_MAX, j].std()) for j in range(4)]
    nrows = [int((np.abs(lb[:, j]) <= LABEL_ABS_MAX).sum()) for j in range(4)]
    sstd = state_all.std(0)
    return {"column_mismatch_episodes": mism, "rows_per_episode_mismatch": rows_bad,
            "noise_std_per_arm_dim": std, "noise_rows_per_dim": nrows, "claw_exec_ne_label_rows": claw_bad,
            "mutant_action_exec_flagged_episodes": mutant_flagged, "episodes_checked": len(side),
            "state_dims_std_lt_1e-6": [int(j) for j in np.flatnonzero(sstd < 1e-6)],
            "state_std": sstd.tolist()}


def open_ds(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=f"local/{root.name}", root=root, video_backend="torchcodec", return_uint8=True)


def decoded(ds, lo: int, t: int) -> dict[str, np.ndarray]:
    r = ds[lo + t]
    return {c: r[f"observation.images.{c}"].permute(1, 2, 0).numpy() for c in CAMS}


def video_check(ds, side: list[dict]) -> dict:
    eps = ds.meta.episodes
    pick = sorted({int(round(x)) for x in np.linspace(0, len(side) - 1, PSNR_EPISODES)})
    ps, ident_fail, rows = [], 0, 0
    for e in pick:
        s = side[e]
        lo = int(eps[e]["dataset_from_index"])
        with h5py.File(s["source"], "r") as f:
            src = {c: f[s["group"]][f"{c}_rgb"][()] for c in CAMS}
            fi = f[s["group"]]["frame_idx"][()]
        for t in range(len(fi)):
            im = decoded(ds, lo, t)
            k = int(fi[t])
            for c in CAMS:
                ps.append(psnr(im[c], src[c][k]))
                mse = ((src[c].astype(np.float32) - im[c].astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
                other = np.array([not np.array_equal(src[c][j], src[c][k]) for j in range(src[c].shape[0])])
                ident_fail += bool(other.any() and mse[k] >= mse[other].min())
            rows += 1
    fin = [p for p in ps if np.isfinite(p)]
    return {"episodes": pick, "rows": rows, "psnr_min": float(min(ps)), "psnr_mean": float(np.mean(fin)),
            "identity_fail": ident_fail, "backend": "torchcodec"}


def fmt(a: np.ndarray) -> str:
    return " ".join(f"{float(x):+.2f}" for x in a)


def key_steps(act: np.ndarray, task: str) -> list[tuple[str, int]]:
    T = act.shape[0]
    closed = np.flatnonzero(act[:, 4] > 0.5)
    tg = int(closed[0]) if len(closed) else T // 3
    opened = np.flatnonzero(act[tg:, 4] <= 0.5)
    tr = tg + int(opened[0]) if task != "pick" and len(opened) else None
    tc = (tg + tr) // 2 if tr is not None else tg + (T - 1 - tg) // 3
    last = ("release", tr) if tr is not None else ("hold", (tc + T - 1) // 2)
    return [("start", 0), ("grasp", tg), ("carry", tc), last, ("end", T - 1)]


def sheet(ds, side: list[dict], path: Path, relabel=None, chosen: list[int] | None = None, note=None) -> list[dict]:
    """`relabel(action, action_exec)`: also print r (0C attempt 1 sheets: original label, r, executed). `chosen`: the
    sidecar indices to draw (default first, middle, last); `note(sidecar row) -> str` is appended to each episode's title
    (Stage 4: table rgb / light elevation)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = ds.meta.episodes
    chosen = chosen or [0, len(side) // 2, len(side) - 1]
    fig, ax = plt.subplots(2 * len(chosen), 5, figsize=(13, 6.4 * len(chosen)), squeeze=False)
    info = []
    for i, e in enumerate(chosen):
        s = side[e]
        lo = int(eps[e]["dataset_from_index"])
        with h5py.File(s["source"], "r") as f:
            act, exe = f[s["group"]]["action"][()], f[s["group"]]["action_exec"][()]
        rl = relabel(act.astype(np.float32), exe.astype(np.float32)) if relabel else None
        steps = key_steps(act, s["task"])
        for j, (name, t) in enumerate(steps):
            im = decoded(ds, lo, t)
            for r, c in enumerate(CAMS):
                a = ax[2 * i + r, j]
                a.imshow(im[c], interpolation="nearest")
                a.axis("off")
                if r == 0:
                    a.set_title(f"{name} t={t}\nlabel {fmt(act[t])}\n"
                                + (f"r     {fmt(rl[t])}\n" if rl is not None else "") + f"exec  {fmt(exe[t])}",
                                fontsize=6)
        ax[2 * i, 0].text(0, 1.45, f"ep {e} (seed {s['seed']} i {s['episode_index']}, {s['split']}): "
                          f"\"{s['instruction']}\"  labels {s['labels']}" + (f"  {note(s)}" if note else "")
                          + "  rows: side / wrist", fontsize=7,
                          transform=ax[2 * i, 0].transAxes)
        info.append({"lerobot_episode": e, "episode_index": s["episode_index"], "instruction": s["instruction"],
                     "spec": s["spec_json"], "steps": steps})
    fig.suptitle(f"{path.stem}: decoded from the LeRobot dataset (torchcodec)", fontsize=9)
    fig.subplots_adjust(hspace=0.55, top=0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return info


def check(root: Path, out: Path) -> dict:
    t0 = time.time()
    exp = json.loads((root / "meta" / "export_113.json").read_text())
    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    tasks = sorted({s["task"] for s in side})
    assert len(tasks) == 1, tasks
    task = tasks[0]
    srcs = [Path(x) for x in exp["sources"]]
    es = expert_success(srcs)
    ceil = CEILING[task]
    ds = open_ds(root)
    cn = columns_and_noise(root, side)
    vc = video_check(ds, side)
    sh = sheet(ds, side, out / f"{root.name}.png")
    T = {int(s["T"]) for s in side}
    items = {
        "1_count_eq_plan": exp["episodes"] == int(json.loads((root / "meta" / "split_113.json").read_text())
                                                 ["kept_per_stream"]) == len(side),
        "1_rows_per_episode_eq_T": not cn["rows_per_episode_mismatch"] and exp["rows"] == len(side) * max(T)
        and len(T) == 1,
        "1_columns_exact": not cn["column_mismatch_episodes"],
        "1_psnr_min_ge_35": vc["psnr_min"] >= PSNR_BAR, "1_frame_identity": vc["identity_fail"] == 0,
        "3_pooled_ge_ceiling_minus_5": es["pooled"]["rate_pct"] >= ceil - POOLED_TOL,
        "3_every_batch_ge_ceiling_minus_10": all(b["rate_pct"] >= ceil - BATCH_TOL for b in es["batches"]),
        "3_every_batch_n_ge_200": all(b["attempted"] >= BATCH_MIN_N for b in es["batches"]),
        "4_noise_std_in_band": all(NOISE_BAND[0] <= x <= NOISE_BAND[1] for x in cn["noise_std_per_arm_dim"]),
        "4_claw_identical": cn["claw_exec_ne_label_rows"] == 0,
        "4_mutant_action_exec_flagged": cn["mutant_action_exec_flagged_episodes"] == len(side),
    }
    res = {"dataset": str(root), "name": root.name, "task": task, "ceiling_pct": ceil, "export": exp,
           "expert_success": es, "columns_noise": cn, "video": vc, "sheet": {"png": f"{root.name}.png", "episodes": sh},
           "items": items, "pass_items_1_3_4": all(items.values()), "wall_s": round(time.time() - t0, 1)}
    (out / f"0c_check_{root.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"name": root.name, "items": items, "pooled": es["pooled"],
                      "noise_std": cn["noise_std_per_arm_dim"], "psnr_min": vc["psnr_min"]}), flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.dataset:
        check(d, a.out)


if __name__ == "__main__":
    main()
