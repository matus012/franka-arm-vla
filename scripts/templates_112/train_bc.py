"""BC / DAgger refit of the student (vision_lift.train.train_bc).

Usage:
  python scripts/train_bc.py --data datasets/s2/bc --val_data datasets/s2/val_offline --out checkpoints/s2_bc \
      --depth 0 --samples 2400000 [--init checkpoints/.../epoch_xxx.pt] [--lr 3e-4] [--seed 0]
--samples = total training samples (steps) seen; epochs = ceil(samples / dataset size).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_lift.dataset import StepDataset, list_files  # noqa: E402
from vision_lift.train import train_bc  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="comma-separated dataset dirs")
    p.add_argument("--val_data", default=None, help="comma-separated dirs for the offline validation loss")
    p.add_argument("--out", required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--init", default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--w_aux", type=float, default=0.0)
    p.add_argument("--p_prev_drop", type=float, default=0.0)
    a = p.parse_args()
    tr = list_files([ROOT / d for d in a.data.split(",")])
    va = list_files([ROOT / d for d in a.val_data.split(",")]) if a.val_data else []
    n = len(StepDataset(tr))
    epochs = max(1, math.ceil(a.samples / n))
    print(f"[train_bc] {len(tr)} files, {n} samples, {epochs} epochs, init={a.init}", flush=True)
    res = train_bc(tr, va, ROOT / a.out, bool(a.depth), epochs, lr=a.lr, seed=a.seed, workers=a.workers, init=a.init,
                   w_aux=a.w_aux, p_prev_drop=a.p_prev_drop, log=lambda s: print(s, flush=True))
    res |= {"data": a.data, "samples_target": a.samples, "epochs": epochs, "init": a.init, "lr": a.lr, "seed": a.seed, "w_aux": a.w_aux,
            "p_prev_drop": a.p_prev_drop}
    (ROOT / a.out / "train_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
