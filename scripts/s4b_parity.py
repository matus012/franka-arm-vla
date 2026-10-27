"""Instrument B-I2 (GATES Stage 4 re-plan): baseline input parity, CPU (policy venv; runs on PERUN on the real union).

  python scripts/s4b_parity.py --dataset /data/datasets/0c_s4_all_rl224_y15g --ckpt /runs/s4_baseline_seed0/ckpt_<step> \
      --out /runs/<unit>/b_i2_parity.json [--n 10 --image-size 224 --rows 32 --video-backend torchcodec]

On `--rows` (32) random `val` rows: the chunk the server HANDLER returns for the frames as the bridge sends them
(uint8 H x W x 3 = round(255 x decoded frame), state float32 19, labels int64 (4,)) must equal BaselinePolicy.predict on
the dataset's decoded frames (float C x H x W) to 1e-5 (arm; claw bitwise); the server's `info` must carry
chunk_size == N and image_size == H (and pass vla_common.protocol.check_info). `--mutant bgr|unnormalised|swapped_cams`
corrupts the bridge side of the comparison (the check must then report pass = false).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_common import protocol as P  # noqa: E402
from vla_policy import train_baseline as TB  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402
from vla_policy.baseline import BaselinePolicy  # noqa: E402
from vla_policy.server import BaselineServer, make_handler  # noqa: E402

TOL = 1e-5


def bridge_frames(x: np.ndarray, mutant: str) -> np.ndarray:
    """(B,3,H,W) float [0,1] decoded frames -> (B,H,W,3) uint8 as the sim sends them (mutants corrupt this step)."""
    u = np.rint(x * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)
    if mutant == "bgr":
        return np.ascontiguousarray(u[..., ::-1])
    if mutant == "unnormalised":  # float frames cast to uint8 without the x255 scaling
        return np.ascontiguousarray(x.astype(np.uint8).transpose(0, 2, 3, 1))
    return np.ascontiguousarray(u)


def check(ds, is_val: np.ndarray, ckpt: Path, n: int, H: int, rows: int, seed: int, mutant: str = "none") -> dict:
    pick = np.sort(np.random.default_rng(seed).choice(np.flatnonzero(is_val), size=rows, replace=False))
    items = [ds[int(i)] for i in pick]
    side = np.stack([i["observation.images.side"].numpy() for i in items])
    wrist = np.stack([i["observation.images.wrist"].numpy() for i in items])
    state = np.stack([i["observation.state"].numpy() for i in items]).astype(np.float32)
    labels = np.stack([TB.episode_labels(i) for i in items])
    assert side.shape[1:] == (3, H, H) and state.shape[1] == P.STATE_DIM, (side.shape, state.shape)
    ref = BaselinePolicy.load(ckpt, "cpu").predict(side, wrist, state, labels)  # float C x H x W, the trainer's view
    srv = BaselineServer(str(ckpt), n, "cpu", H)
    handle = make_handler(srv, 3, 1)
    info = handle({"op": "info"})
    exp = {"policy": "baseline", "n_action_steps": n, "chunk_size": n, "image_size": H, "state_dim": P.STATE_DIM,
           "action_dim": P.ACTION_DIM}
    info_ok = True
    try:
        P.check_info(info, exp)
    except RuntimeError as e:
        info_ok = str(e)
    s_, w_ = bridge_frames(side, mutant), bridge_frames(wrist, mutant)
    if mutant == "swapped_cams":
        s_, w_ = w_, s_
    msg = {"op": "act", "env_ids": np.arange(rows, dtype=np.int32), "side": s_, "wrist": w_, "state": state,
           "instruction": [""] * rows, "labels": labels}
    P.check_act(msg, True)  # the same whitelist the protocol enforces
    got = handle(msg)["actions"]
    arm = float(np.abs(got[..., :4] - ref[..., :4]).max())
    claw = int((got[..., 4] != ref[..., 4]).sum())
    roundtrip = float(np.abs(s_.transpose(0, 3, 1, 2) / 255.0 - side).max()) if mutant == "none" else None
    return {"mutant": mutant, "rows": pick.tolist(), "n_rows": rows, "N": n, "image_size": H, "tol": TOL,
            "max_abs_arm_diff": arm, "claw_mismatches": claw, "uint8_roundtrip_max_err": roundtrip,
            "info": {k: info[k] for k in ("policy", "n_action_steps", "chunk_size", "image_size", "ckpt_sha")},
            "info_ok": info_ok, "chunk_shape": list(got.shape), "chunk_nonconstant": bool(got.std() > 0),
            "pass": bool(arm <= TOL and claw == 0 and info_ok is True and info["chunk_size"] == n and info["image_size"] == H)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--mutant", default="none", choices=("none", "bgr", "unnormalised", "swapped_cams"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    meta = TB.make_ds(a.dataset, None, a.n, a.video_backend)
    ds = TB.make_ds(a.dataset, None, a.n, a.video_backend, meta.fps)
    is_val, _, _ = T.split_episodes(ds)
    r = check(ds, is_val, a.ckpt, a.n, a.image_size, a.rows, a.seed, a.mutant)
    r |= {"dataset": str(a.dataset), "ckpt": str(a.ckpt), "git_sha": T.git_sha()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2))
    print(json.dumps({k: r[k] for k in ("mutant", "max_abs_arm_diff", "claw_mismatches", "info_ok", "pass")}))
    if a.mutant == "none" and not r["pass"]:
        sys.exit("B-I2 FAILED")


if __name__ == "__main__":
    main()
