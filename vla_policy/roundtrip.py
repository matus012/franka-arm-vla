"""0B.1 export round trip: contract HDF5 source vs the exported LeRobot dataset as LeRobotDataset returns it (pyav).

  python -m vla_policy.roundtrip --src data/0b_smoke/*.h5 --dataset datasets/0b_smoke --json results/0b1_export.json \
      --sheet results/frames/0b1/decoded_ep0.png

Checks (GATES 0B.1): (1) every non-image column equals the source exactly (state, action label, task, labels, split,
frame_idx); (2) decoded video vs source frame: PSNR per row, and frame identity — the decoded image is closer (MSE) to
its own source frame than to any other DIFFERENT captured frame of the episode; (3) rows per episode = T; (4) with the
SmolVLA delta timestamps, row t's action chunk = labels t .. t+49, padded (action_is_pad) past the episode end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_common.episode import read_episodes
from vla_policy.export import CAMERAS, CONTROL_HZ, load

CHUNK = 50
PSNR_BAR = 35.0  # GATES 0B.1 (2)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def check(files: list[Path], dataset: str | Path, keep_frames: int = 0) -> dict:
    """Returns the round-trip summary; `keep_frames` > 0 also returns (t, source, decoded) of episode 0 for a sheet."""
    ds = load(dataset, delta_timestamps={"action": [i / CONTROL_HZ for i in range(CHUNK)]}, return_uint8=True)
    eps = ds.meta.episodes
    out = {"episodes": 0, "rows": 0, "column_mismatch": [], "rows_per_episode_mismatch": [], "chunk_mismatch": 0,
           "pad_mismatch": 0, "identity_fail": 0, "psnr_min": float("inf"), "psnr_mean": 0.0, "sheet": []}
    psnrs = []
    e = 0
    for f in files:
        for group, a, at in read_episodes(f):
            T = a["state"].shape[0]
            lo, hi = int(eps[e]["dataset_from_index"]), int(eps[e]["dataset_to_index"])
            if hi - lo != T:
                out["rows_per_episode_mismatch"].append((str(f), group, hi - lo, T))
            labels = np.asarray(json.loads(at["labels"]) if isinstance(at["labels"], str) else at["labels"]).reshape(4)
            keep = set(np.linspace(0, T - 1, keep_frames).round().astype(int).tolist()) if e == 0 and keep_frames else ()
            for t in range(min(T, hi - lo)):
                r = ds[lo + t]
                k = int(a["frame_idx"][t])
                ok = (int(r["episode_index"]) == e and int(r["frame_index"]) == t
                      and np.array_equal(r["observation.state"].numpy(), a["state"][t])
                      and np.array_equal(r["action"][0].numpy(), a["action"][t])
                      and r["task"] == at["instruction"] and r["split"] == at["split"]
                      and np.array_equal(r["labels"].numpy(), labels) and int(r["frame_idx"]) == k)
                if not ok:
                    out["column_mismatch"].append((str(f), group, t))
                n = min(CHUNK, T - t)
                if not np.array_equal(r["action"][:n].numpy(), a["action"][t:t + n]):
                    out["chunk_mismatch"] += 1
                pad = r["action_is_pad"].numpy()
                if pad[:n].any() or not pad[n:].all():
                    out["pad_mismatch"] += 1
                for cam in CAMERAS:
                    dec = r[f"observation.images.{cam}"].permute(1, 2, 0).numpy()
                    src = a[f"{cam}_rgb"]
                    psnrs.append(psnr(dec, src[k]))
                    mse = ((src.astype(np.float32) - dec.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
                    other = np.array([not np.array_equal(src[j], src[k]) for j in range(src.shape[0])])
                    if other.any() and mse[k] >= mse[other].min():
                        out["identity_fail"] += 1
                    if t in keep:
                        out["sheet"].append((cam, t, k, a["action"][t].tolist(), src[k], dec))
                out["rows"] += 1
            e += 1
    out["episodes"] = e
    out["dataset_episodes"] = int(ds.meta.total_episodes)
    out["psnr_min"] = float(min(psnrs))
    out["psnr_mean"] = float(np.mean([p for p in psnrs if np.isfinite(p)])) if psnrs else float("nan")
    return out


def sheet(entries: list, path: Path, title: str) -> None:
    """Contact sheet: one row per kept step, columns source | decoded for side and wrist, frame index + action."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted({(t, k, tuple(round(float(x), 3) for x in act)) for _, t, k, act, _, _ in entries})
    img = {(cam, t): (s, d) for cam, t, _, _, s, d in entries}
    fig, ax = plt.subplots(len(steps), 4, figsize=(8, 2.4 * len(steps)), squeeze=False)
    for i, (t, k, act) in enumerate(steps):
        for j, (cam, which) in enumerate([(c, w) for c in CAMERAS for w in (0, 1)]):
            ax[i, j].imshow(img[(cam, t)][which], interpolation="nearest")
            head = (f"t={t} frame_idx={k}", "", "", f"a={list(act[:4])} claw={act[4]}")[j]
            ax[i, j].set_title(head + "\n" + f"{cam} {'source' if which == 0 else 'decoded'}", fontsize=6)
            ax[i, j].axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--sheet", required=True)
    args = p.parse_args()
    import subprocess
    import time

    t0 = time.time()
    files = sorted(Path(s) for s in args.src)
    out = check(files, args.dataset, keep_frames=6)
    sheet(out.pop("sheet"), Path(args.sheet), f"{args.dataset} episode 0: source vs decoded (pyav)")
    info = json.loads((Path(args.dataset) / "meta" / "info.json").read_text())
    T = sorted({int(a["state"].shape[0]) for f in files for _, a, _ in read_episodes(f)})
    out |= {"fps": info["fps"], "T_steps": T, "psnr_bar_db": PSNR_BAR, "sources": [str(f) for f in files],
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "wall_s": round(time.time() - t0, 1)}
    out["checks"] = {"1_columns_exact": not out["column_mismatch"],
                     "2a_psnr_min_ge_35dB": out["psnr_min"] >= PSNR_BAR,
                     "2b_frame_identity_real_frames": out["identity_fail"] == 0,
                     "3_rows_per_episode_eq_T": not out["rows_per_episode_mismatch"]
                     and out["dataset_episodes"] == out["episodes"] and out["rows"] == out["episodes"] * T[-1],
                     "4_chunk_eq_labels_t_t49_padded": out["chunk_mismatch"] == 0 and out["pad_mismatch"] == 0,
                     "fps_50": info["fps"] == CONTROL_HZ}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
