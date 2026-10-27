"""GATES 0B.5 measured: SmolVLA server latency per `act` at B = 64, 128, 256 on one H200 (policy venv, PERUN).

  python scripts/s0b5_server_latency.py --dataset /data/datasets/0b_smoke --batches 64 128 256 --out <json>

The server object itself (vla_policy.server.SmolVLAServer: strict fp32, TF32 off, per-env noise generators, pre /
postprocessors, ends with .cpu() = synchronous) on real dataset rows (images, states, instructions; rows cycled to fill
B); --warmup untimed calls, then --reps timed calls per B. Socket transfer is not included (the closed-loop eval JSONs
carry the client round trip).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--ckpt", default="lerobot/smolvla_base")
    p.add_argument("--n-action-steps", type=int, default=20)
    p.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable. STOP."
    from vla_policy.server import SmolVLAServer
    from vla_policy.train_smolvla import git_sha, open_dataset

    srv = SmolVLAServer(a.ckpt, a.n_action_steps, "cuda", str(a.dataset))
    ds = open_dataset(a.dataset, "torchcodec")
    rows = [ds[int(i)] for i in np.linspace(0, len(ds) - 1, 64).round()]
    res = {}
    for B in a.batches:
        rr = [rows[i % len(rows)] for i in range(B)]
        msg = {"env_ids": np.arange(B, dtype=np.int32),
               "side": np.stack([r["observation.images.side"].permute(1, 2, 0).numpy() for r in rr]),
               "wrist": np.stack([r["observation.images.wrist"].permute(1, 2, 0).numpy() for r in rr]),
               "state": np.stack([r["observation.state"].numpy() for r in rr]).astype(np.float32),
               "instruction": [r["task"] for r in rr]}
        srv.reset({"env_ids": msg["env_ids"], "seeds": np.arange(B, dtype=np.int64)})
        torch.cuda.reset_peak_memory_stats()
        for _ in range(a.warmup):
            srv.act(msg)
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter()
            out = srv.act(msg)
            ts.append(time.perf_counter() - t)
        assert out.shape == (B, a.n_action_steps, 5) and np.isfinite(out).all()
        res[str(B)] = {"server_s_p50": float(np.median(ts)), "server_s_p95": float(np.percentile(ts, 95)),
                       "server_s_mean": float(np.mean(ts)), "per_env_ms_p50": 1e3 * float(np.median(ts)) / B,
                       "reps": a.reps, "peak_mem_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        print(f"[latency] B={B}: {json.dumps(res[str(B)])}", flush=True)
    out = {"gate": "0B.5 measured", "ckpt": a.ckpt, "dataset": str(a.dataset), "n_action_steps": a.n_action_steps,
           "dtype": "float32 (TF32 off)", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "git_sha": git_sha(), "per_batch": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
