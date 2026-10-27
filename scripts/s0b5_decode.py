"""GATES 0B.5 item (1): the 0B.1 export round trip on PERUN with torchcodec, decoded frames vs the laptop's pyav decode.

  laptop (policy venv):  python scripts/s0b5_decode.py ref --dataset datasets/0b_smoke --out data/0b5_ref
      -> pyav_sha.json (sha256 of every decoded row image, per camera) + pyav_ep0.npz (episode 0 decoded pixels)
  PERUN (pol_exec):      python scripts/s0b5_decode.py perun --dataset /data/datasets/0b_smoke --src /data/0b5/src \
                             --ref /data/0b5/ref --out /runs/g0b5/decode.json

PERUN checks: (a) vla_policy.roundtrip.check with the torchcodec backend (columns exact vs the HDF5 source, frame
identity, rows per episode, action chunks + padding; PSNR vs the source frames reported); (b) every row decoded with
torchcodec vs the same row decoded with pyav on PERUN: PSNR per row image (bar 40 dB); (c) the PERUN pyav decode is
bit-identical to the laptop's pyav decode (per-row sha256), so (b) is the PSNR vs the laptop's pyav decode on every
row; (d) direct check for episode 0: torchcodec (PERUN) vs the laptop's decoded pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BAR_DB = 40.0  # GATES 0B.5 (1)
CAMS = ("side", "wrist")


def open_ds(d: Path, backend: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=f"local/{d.name}", root=d, video_backend=backend, return_uint8=True)


def decode_all(d: Path, backend: str):
    """Yields (row, episode_index, {cam: HxWx3 uint8}) for every row."""
    ds = open_ds(d, backend)
    for i in range(len(ds)):
        r = ds[i]
        yield i, int(r["episode_index"]), {c: r[f"observation.images.{c}"].permute(1, 2, 0).numpy() for c in CAMS}


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:24]


def ref(a: argparse.Namespace) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    hashes, ep0 = {c: [] for c in CAMS}, {c: [] for c in CAMS}
    for i, e, im in decode_all(a.dataset, "pyav"):
        for c in CAMS:
            hashes[c].append(sha(im[c]))
            if e == 0:
                ep0[c].append(im[c])
    (a.out / "pyav_sha.json").write_text(json.dumps({"backend": "pyav", "platform": sys.platform, "rows": len(hashes["side"]),
                                                     "sha256_24": hashes}))
    np.savez_compressed(a.out / "pyav_ep0.npz", **{c: np.stack(v) for c, v in ep0.items()})
    print(f"[ref] {len(hashes['side'])} rows hashed, episode 0 {len(ep0['side'])} rows saved")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def perun(a: argparse.Namespace) -> None:
    import vla_policy.export as EX
    from vla_policy import roundtrip as RT

    t0 = time.time()
    EX.VIDEO_BACKEND = "torchcodec"  # vla_policy.export.load (used by roundtrip.check) reads it at call time
    files = sorted(a.src.glob("*.h5"))
    rt = RT.check(files, a.dataset)
    rt.pop("sheet")
    refj = json.loads((a.ref / "pyav_sha.json").read_text())
    ref0 = np.load(a.ref / "pyav_ep0.npz")
    p_tc_pyav, sha_mismatch, p_ep0 = [], 0, []
    for (i, e, tc), (j, _, pa) in zip(decode_all(a.dataset, "torchcodec"), decode_all(a.dataset, "pyav")):
        assert i == j
        for c in CAMS:
            p_tc_pyav.append(psnr(tc[c], pa[c]))
            sha_mismatch += sha(pa[c]) != refj["sha256_24"][c][i]
            if e == 0:
                p_ep0.append(psnr(tc[c], ref0[c][i]))
    n = len(p_tc_pyav) // 2
    fin = [p for p in p_tc_pyav if np.isfinite(p)]
    out = {"gate": "0B.5 (1)", "backend": "torchcodec", "dataset": str(a.dataset), "sources": [str(f) for f in files],
           "roundtrip_torchcodec": rt,
           "rows": n, "laptop_ref_rows": refj["rows"],
           "psnr_torchcodec_vs_pyav_min_db": float(min(p_tc_pyav)),
           "psnr_torchcodec_vs_pyav_mean_finite_db": float(np.mean(fin)) if fin else float("inf"),
           "identical_images_frac": float(np.mean(~np.isfinite(p_tc_pyav))),
           "pyav_perun_vs_laptop_sha_mismatch": int(sha_mismatch),
           "psnr_torchcodec_vs_laptop_pyav_ep0_min_db": float(min(p_ep0)), "ep0_rows": len(p_ep0) // 2,
           "bar_db": BAR_DB, "wall_s": round(time.time() - t0, 1)}
    out["checks"] = {"1_columns_exact": not rt["column_mismatch"],
                     "frame_identity": rt["identity_fail"] == 0,
                     "rows_per_episode_eq_T": not rt["rows_per_episode_mismatch"] and rt["episodes"] == rt["dataset_episodes"],
                     "chunks_padded": rt["chunk_mismatch"] == 0 and rt["pad_mismatch"] == 0,
                     "same_row_count_as_laptop": n == refj["rows"],
                     "pyav_perun_eq_laptop": sha_mismatch == 0,
                     "psnr_vs_laptop_pyav_ge_40dB": out["psnr_torchcodec_vs_pyav_min_db"] >= BAR_DB
                     and sha_mismatch == 0 and out["psnr_torchcodec_vs_laptop_pyav_ep0_min_db"] >= BAR_DB}
    out["pass"] = all(out["checks"].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("DECODE=" + json.dumps({k: v for k, v in out.items() if k not in ("roundtrip_torchcodec", "sources")}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("ref", "perun"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--src", type=Path, help="perun: dir with the HDF5 sources")
    p.add_argument("--ref", type=Path, help="perun: dir with the laptop's pyav_sha.json + pyav_ep0.npz")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"ref": ref, "perun": perun}[a.mode](a)


if __name__ == "__main__":
    main()
