"""0B: a small synthetic LeRobot export whose only job is to give the UNTRAINED bridge SmolVLA normalization stats
(policy venv). State ~ N(0, 0.1) per dim, arm action ~ U(-1, 1), claw ~ Bernoulli(0.5); frames = stripe patterns.

  policy\\.venv\\Scripts\\python.exe scripts\\s0b_synth_dataset.py --name s0b_synth128 --size 128
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from synth_episodes import attrs, episode_arrays  # noqa: E402
from vla_common.episode import EpisodeWriter  # noqa: E402
from vla_policy.export import export  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--size", type=int, default=128)
p.add_argument("--episodes", type=int, default=4)
p.add_argument("--steps", type=int, default=120)
args = p.parse_args()

rng = np.random.default_rng(0)
src = ROOT / "data" / args.name / "synth.h5"
with EpisodeWriter(src, {"git_sha": "synthetic", "created": "s0b_synth_dataset", "stage": "0B"}) as w:
    for ep in range(args.episodes):
        a = episode_arrays(args.steps, ep, rng, args.size)
        a["state"] = rng.normal(0, 0.1, a["state"].shape).astype(np.float32)
        a["action"][:, :4] = rng.uniform(-1, 1, (args.steps, 4)).astype(np.float32)
        a["action_exec"] = a["action"].copy()
        w.add(**a, attrs=attrs(ep, args.steps, args.size, env_id=ep, instruction="pick up the cube"))
shutil.rmtree(ROOT / "datasets" / args.name, ignore_errors=True)
ds = export([src], args.name)
print(ds.root, ds.meta.total_episodes, ds.meta.total_frames)
