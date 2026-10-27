"""Train the no-language baseline on a local LeRobot v3.0 dataset (policy venv).

  policy\\.venv\\Scripts\\python.exe -m vla_policy.train_baseline --root datasets/<name> --out runs/baseline --epochs 6 \\
      --n-action-steps 20 --video-backend pyav

Action targets: next n_action_steps actions per row (LeRobot delta_timestamps; past the episode end the last action is
repeated and `action_is_pad` masks the loss).
Rows (GATES 0B row option B + 0C split, the same as vla_policy.train_smolvla): training samples one random row per
(episode, frame_idx) window per epoch from the `train`-split episodes only, with the SmolVLA trainer's WindowSampler
(same seed / epoch -> same rows); an epoch = one pass over those windows. The offline validation loss runs on the
dataset's own `val`-split episodes (0C: kept rank k % 20 == 19), one fixed row per window (sampler epoch 0); no train
episode is ever in it and no val row is ever trained on (both asserted per batch).
Timing (GATES GPU-h plan, baseline line): train steps 51-350 are timed (wall incl. data wait = the GPU-s a job is
charged, and compute only), written to train_summary.json.
TensorBoard + steps.jsonl (per step: epoch, loss, batch row indices, t = unix time at step end, w / c = wall /
compute seconds of the step). Checkpoints (GATES Stage 4 re-plan): `--save-fractions 0.25,0.5,0.75,1.0` saves
`ckpt_<step:06d>/` (BaselinePolicy.save dir, the server's `--ckpt`) at round(f * total_steps) and computes the offline
val loss at each (`vloss_<step:06d>.json`, also in train_summary.json); `--epoch-ckpt` additionally saves `epoch_XXX/` +
`last.pt` every epoch (needed for --resume). `--max-steps K` = bench mode (stop after K optimizer steps, no val, no
checkpoints). A non-finite loss stops the run (early abort (a)); (b) / (c) numbers are recorded in train_summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import train_smolvla as T
from .baseline import BaselineConfig, BaselinePolicy, loss_fn

TIMED_STEPS = 300  # GATES GPU-h plan: 300 steps after the 0B.6 warm-up (T.WARMUP_STEPS = 50)


def episode_labels(item: dict) -> np.ndarray:
    """(4,) int64 task labels of the episode a sample row belongs to. The single place that knows where the exporter
    puts them: export.py writes a per-row `labels` int64 (4,) column (constant within an episode). Repoint here."""
    return np.asarray(item["labels"], dtype=np.int64)


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ shift on (B,C,H,W): replicate-pad and crop at a random integer offset per sample."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad,) * 4, mode="replicate")
    oy, ox = torch.randint(0, 2 * pad + 1, (B,)), torch.randint(0, 2 * pad + 1, (B,))
    rows, cols = oy[:, None] + torch.arange(H), ox[:, None] + torch.arange(W)
    xr = xp.gather(2, rows[:, None, :, None].expand(B, C, H, W + 2 * pad))
    return xr.gather(3, cols[:, None, None, :].expand(B, C, H, W))


def make_ds(root: Path, episodes: list[int] | None, n: int, backend: str, fps: int | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    kw = dict(repo_id=f"local/{root.name}", root=root, episodes=episodes, video_backend=backend)
    if fps is None:
        return LeRobotDataset(**kw)
    return LeRobotDataset(**kw, delta_timestamps={"action": [i / fps for i in range(n)]})


def collate(items: list[dict], augment: bool) -> dict:
    side = torch.stack([i["observation.images.side"] for i in items])
    wrist = torch.stack([i["observation.images.wrist"] for i in items])
    if augment:
        side, wrist = random_shift(side), random_shift(wrist)
    return {"side": side, "wrist": wrist, "state": torch.stack([i["observation.state"] for i in items]),
            "labels": torch.from_numpy(np.stack([episode_labels(i) for i in items])),
            "action": torch.stack([i["action"] for i in items]),
            "pad": torch.stack([i["action_is_pad"] for i in items]),
            "index": torch.stack([torch.as_tensor(i["index"]) for i in items]).reshape(-1)}


def val_sampler(ds, is_val: np.ndarray, seed: int) -> T.WindowSampler:
    """Fixed validation rows: one per (episode, frame_idx) window of the `val`-split episodes (epoch 0 order)."""
    return T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), seed, keep=is_val)


def train_keep(ds, is_val: np.ndarray, train_eps: list[int], k: int | None) -> tuple[np.ndarray, list[int]]:
    """Rows the training sampler may draw from. k None: every train-split row (unchanged). k int: only the rows of the
    first k train episodes in ascending episode-index order (the overfit capacity check); val rows never."""
    if k is None:
        return ~is_val, train_eps
    kept = train_eps[:k]
    return np.isin(T.column(ds, "episode_index").reshape(-1), kept) & ~is_val, kept


def kept_ids(root: Path, eps: list[int]) -> list[dict]:
    """[{lerobot_episode_index, seed, episode_index}] for `eps` from meta/episodes_113.jsonl (seed / episode_index absent
    when the dataset has no sidecar)."""
    f = Path(root) / "meta" / "episodes_113.jsonl"
    side = {}
    if f.exists():
        side = {int(r["lerobot_episode_index"]): r for r in map(json.loads, f.read_text().splitlines())}
    return [{"lerobot_episode_index": e, **({"seed": int(side[e]["seed"]), "episode_index": int(side[e]["episode_index"])}
                                            if e in side else {})} for e in eps]


def run_epoch(model, ds, sampler, opt, sched, bs: int, dev: torch.device, workers: int, train: bool,
              is_val: np.ndarray, clock: dict, log=None, epoch: int = 0, on_step=None) -> dict:
    dl = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers, drop_last=train,
                    collate_fn=lambda items: collate(items, train))
    model.train(train)
    agg, nb = {"loss": 0.0, "mse": 0.0, "bce": 0.0}, 0
    with torch.set_grad_enabled(train):
        t0 = time.perf_counter()
        for b in dl:
            idx = b.pop("index").numpy()
            # GATES 0C: training never sees a val row; validation sees only val rows
            assert (not is_val[idx].any()) if train else is_val[idx].all(), f"split violation (train={train}): {idx}"
            t1 = time.perf_counter()
            b = {k: v.to(dev) for k, v in b.items()}
            loss, parts = loss_fn(model, b)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                sched.step()
            lv = loss.item()  # syncs the device
            t2 = time.perf_counter()
            if train:
                if not math.isfinite(lv):
                    raise FloatingPointError(f"non-finite loss {lv} at step {clock['step'] + 1}")
                clock["step"] += 1
                if T.WARMUP_STEPS < clock["step"] <= T.WARMUP_STEPS + TIMED_STEPS:
                    clock["wall_s"] += t2 - t0
                    clock["compute_s"] += t2 - t1
                    clock["samples"] += len(idx)
                    clock["steps"] += 1
                clock["losses"].append(lv)
                log.write(json.dumps({"step": clock["step"], "epoch": epoch, "loss": lv, "index": idx.tolist(),
                                      "t": time.time(), "w": t2 - t0, "c": t2 - t1}) + "\n")
            else:
                clock.setdefault("val_rows", set()).update(idx.tolist())
            agg["loss"] += lv
            agg["mse"] += parts["mse"]
            agg["bce"] += parts["bce"]
            nb += 1
            if train and on_step is not None:
                on_step(clock["step"])
                if clock.get("stop"):
                    break
            t0 = time.perf_counter()
    return {k: v / max(1, nb) for k, v in agg.items()}


def parse_fractions(txt: str) -> list[float]:
    return [float(x) for x in txt.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n-action-steps", type=int, default=10, help="GATES Stage 1+: N = 10")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="bench mode: stop after K optimizer steps (no val, no ckpt)")
    p.add_argument("--save-fractions", default="", help="e.g. 0.25,0.5,0.75,1.0: ckpt_<step>/ + val loss at round(f*total)")
    p.add_argument("--epoch-ckpt", action="store_true", help="also save epoch_XXX/ + last.pt (+ val loss) every epoch")
    p.add_argument("--max-train-episodes", type=int, default=None,
                   help="overfit check: train on the first K train episodes only (ascending episode index); val unchanged")
    p.add_argument("--expect-windows", type=int, default=None, help="assert train windows per epoch (S) == this")
    a = p.parse_args()

    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        assert torch.cuda.is_available(), "--device cuda but CUDA unavailable. STOP."
    print(f"[train] device={dev} cuda_available={torch.cuda.is_available()}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    meta_ds = make_ds(a.root, None, a.n_action_steps, a.video_backend)
    ds = make_ds(a.root, None, a.n_action_steps, a.video_backend, meta_ds.fps)  # all episodes: positions = indices
    is_val, train_eps, val_eps = T.split_episodes(ds)
    keep, kept = train_keep(ds, is_val, train_eps, a.max_train_episodes)
    kept_info = kept_ids(a.root, kept) if a.max_train_episodes is not None else None
    if kept_info is not None:
        print(f"[train] --max-train-episodes {a.max_train_episodes}: kept {len(kept)} episodes {kept_info}", flush=True)
    tr_sampler = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), a.seed, keep=keep)
    va_sampler = val_sampler(ds, is_val, a.seed) if val_eps else None
    spe = len(tr_sampler) // a.batch_size
    assert spe > 0, f"batch {a.batch_size} > train windows {len(tr_sampler)}"
    total = a.epochs * spe
    save_at = {round(f * total): f for f in parse_fractions(a.save_fractions)}
    print(f"[train] episodes train={len(train_eps)} val={len(val_eps)} rows={len(ds)} S(train windows/epoch)="
          f"{len(tr_sampler)} val windows={len(va_sampler) if va_sampler else 0} steps_per_epoch={spe} epochs={a.epochs} "
          f"total_steps={total} save_steps={sorted(save_at)} max_steps={a.max_steps}", flush=True)
    if a.expect_windows is not None and len(tr_sampler) != a.expect_windows:
        sys.exit(f"[train] ABORT before step 1: S={len(tr_sampler)} != --expect-windows {a.expect_windows}")

    model = BaselinePolicy(BaselineConfig(n_action_steps=a.n_action_steps))
    st, _ = T.norm_stats(ds, ~is_val)  # train split only, the SmolVLA trainer's helper (not LeRobot's all-row meta/stats)
    s = lambda k, f: torch.as_tensor(np.asarray(st[k][f], dtype=np.float32)).reshape(-1)
    model.set_norm(s("observation.state", "mean"), s("observation.state", "std"), s("action", "mean")[:4], s("action", "std")[:4])
    model.to(dev)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[train] params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    start = 0
    if a.resume and (a.out / "last.pt").exists():
        ck = torch.load(a.out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start = ck["epoch"] + 1
        print(f"[train] resumed at epoch {start}", flush=True)
    tr_sampler.epoch = start  # the sampler order is f(seed, epoch): resume continues the same row sequence
    clock = {"step": start * spe, "wall_s": 0.0, "compute_s": 0.0, "samples": 0, "steps": 0, "losses": []}

    tb = SummaryWriter(str(a.out / "tb"))
    log = (a.out / "steps.jsonl").open("a" if start else "w")
    vlosses: dict[int, dict] = {}
    v: dict = {}

    def validate() -> dict:
        va_sampler.epoch = 0  # the same validation rows every time
        r = run_epoch(model, ds, va_sampler, opt, sched, a.batch_size, dev, a.workers, False, is_val, clock)
        model.train(True)
        return r

    def on_step(step: int) -> None:
        if a.max_steps is not None and step >= a.max_steps:
            clock["stop"] = True
        if step in save_at:
            model.save(a.out / f"ckpt_{step:06d}")
            if va_sampler:
                vlosses[step] = {"step": step, "fraction": save_at[step], **validate()}
                (a.out / f"vloss_{step:06d}.json").write_text(json.dumps(vlosses[step], indent=2))
                tb.add_scalar("val/loss_at_save", vlosses[step]["loss"], step)
            print(f"[train] saved ckpt_{step:06d} (fraction {save_at[step]}) val {vlosses.get(step)}", flush=True)

    def summary(aborted: str | None) -> dict:
        n, L = clock["samples"], clock["losses"]
        by_frac = {f: k for k, f in save_at.items()}
        eb = ec = None
        if 0.25 in by_frac and by_frac[0.25] <= len(L):  # early abort (b): mean loss steps 1-50 vs last 100 before the 25 % save
            s25 = by_frac[0.25]
            first, last = float(np.mean(L[:50])), float(np.mean(L[max(0, s25 - 100):s25]))
            eb = {"step_25": s25, "mean_loss_steps_1_50": first, "mean_loss_last_100_before_25": last,
                  "ratio": last / first, "abort": not last < 0.5 * first}
        if 0.25 in by_frac and 0.5 in by_frac and by_frac[0.25] in vlosses and by_frac[0.5] in vlosses:  # (c)
            v25, v50 = vlosses[by_frac[0.25]]["loss"], vlosses[by_frac[0.5]]["loss"]
            ec = {"val_loss_25": v25, "val_loss_50": v50, "ratio": v50 / v25, "abort": v50 > 2 * v25}
        return {"git_sha": T.git_sha(), "argv": {k: str(x) for k, x in vars(a).items()}, "params": n_params,
                "aborted": aborted, "stats_rows": int((~is_val).sum()), "stats_split": "train",
                "norm_stats": {k: {f: st[k][f].tolist() for f in ("mean", "std")} for k in st},
                "n_train_episodes": len(train_eps),
                "max_train_episodes": a.max_train_episodes, "kept_train_episodes": kept_info, "n_val_episodes": len(val_eps), "rows": len(ds),
                "train_windows_per_epoch": len(tr_sampler), "S": len(tr_sampler), "steps_per_epoch": spe,
                "epochs": a.epochs, "total_steps": total, "save_steps": {str(k): f for k, f in sorted(save_at.items())},
                "final_step": clock["step"], "val_windows": len(va_sampler) if va_sampler else 0,
                "val_rows_seen": sorted(clock.get("val_rows", ())), "last_val": v,
                "val_at_checkpoints": {str(k): x for k, x in vlosses.items()},
                "early_abort": {"a_nan": aborted == "a_nan", "b": eb, "c": ec},
                "wall_s": time.time() - t_start, "device": str(dev),
                "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                "timing": {"warmup_steps": T.WARMUP_STEPS, "timed_steps": clock["steps"], "samples": n,
                           "gpu_s_per_sample": clock["wall_s"] / n if n else None,  # wall incl. data wait (charged)
                           "compute_s_per_sample": clock["compute_s"] / n if n else None,
                           "complete": clock["steps"] == TIMED_STEPS}}

    aborted = None
    try:
        for ep in range(start, a.epochs):
            r = run_epoch(model, ds, tr_sampler, opt, sched, a.batch_size, dev, a.workers, True, is_val, clock, log, ep,
                          on_step)
            for k, x in r.items():
                tb.add_scalar(f"train/{k}", x, ep)
            log.flush()
            if clock.get("stop"):
                break
            if a.epoch_ckpt:
                if va_sampler:
                    v = validate()
                    for k, x in v.items():
                        tb.add_scalar(f"val/{k}", x, ep)
                print(f"[train] epoch {ep} train {r} val {v}", flush=True)
                model.save(a.out / f"epoch_{ep:03d}")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep},
                           a.out / "last.pt")
        if a.max_steps is None and va_sampler and not a.epoch_ckpt:
            v = {k: x for k, x in vlosses[total].items() if k in ("loss", "mse", "bce")} if total in vlosses else validate()
    except FloatingPointError as e:
        aborted = "a_nan"
        print(f"[train] EARLY ABORT (a): {e}. No further steps; the latest saved checkpoint stays the evaluated one.", flush=True)
    log.close()
    tb.close()
    out = summary(aborted)
    (a.out / "train_summary.json").write_text(json.dumps(out, indent=2))
    print(f"[train] done: timing {out['timing']} early_abort {out['early_abort']}", flush=True)
    if aborted:
        sys.exit(3)


if __name__ == "__main__":
    main()
