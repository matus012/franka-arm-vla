"""Contact sheet of recorded training samples (what the student sees): side | wrist | side depth per row, labelled with
step, frame index (delayed), phase, claw label and the (clipped) teacher arm label.

Usage: python scripts/dataset_sheet.py --file datasets/x.h5 --out results/frames/x.png [--env 0] [--steps 0,60,...]
       python scripts/dataset_sheet.py --file datasets/x.h5 --out ... --envs 0,17,50,99 --step 120   (many envs, one step)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--file", required=True)
p.add_argument("--out", required=True)
p.add_argument("--env", type=int, default=0)
p.add_argument("--steps", default="0,5,6,60,61,62,63,150,399")
p.add_argument("--envs", default=None)
p.add_argument("--step", type=int, default=120)
a = p.parse_args()

h = h5py.File(ROOT / a.file, "r")
pairs = [(int(e), a.step) for e in a.envs.split(",")] if a.envs else [(a.env, int(t)) for t in a.steps.split(",")]
rows = []
for e, t in pairs:
    k = int(h["steps/frame"][t])
    side, wrist = h["frames/side_rgb"][e, k], h["frames/wrist_rgb"][e, k]
    d = np.nan_to_num(h["frames/side_depth"][e, k].astype(np.float32), nan=2.0)
    d = np.where(d <= 0, 2.0, d)
    d = np.repeat((np.clip(d, 0, 2) / 2 * 255).astype(np.uint8)[..., None], 3, 2)
    row = np.kron(np.concatenate([side, wrist, d], 1), np.ones((2, 2, 1), np.uint8))
    im = Image.fromarray(row)
    arm = np.round(h["steps/teacher_arm"][e, t], 2)
    ImageDraw.Draw(im).text((4, 4), f"env {e} step {t}: frame {k} (captured step {3 * k}), phase "
                            f"{int(h['privileged/step_phase'][e, t])}, claw label {int(h['steps/claw_label'][e, t])}, "
                            f"arm {arm}", fill=(255, 255, 0))
    rows.append(np.asarray(im))
out = ROOT / a.out
out.parent.mkdir(parents=True, exist_ok=True)
imageio.imwrite(out, np.concatenate(rows, 0))
print(out)
