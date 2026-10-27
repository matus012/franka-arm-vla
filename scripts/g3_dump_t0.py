"""Task 003 (GATES "Post-kill grounding diagnosis (003)"), sim venv: the t = 0 observation of the first policy query.

For the probe samples (GATES 003 "Samples"): S_p = paired stream 21645 (make_paired_stream, s2a / train, m 15), whole
layout groups from group 0 until |S_p| >= --min-paired; S_u = unpaired stream 21640 (make_stream) episodes
0 .. --total - |S_p| - 1. One run = one stream (its own env, cfg seed = the stream seed, as scripts/eval_policy.py).
Observation = what ChunkExecutor sends at t = 0: side + wrist through DelayedFrames(PERIOD, LATENCY) captured at the
t = 0 tick after the reset (visual nominal: no randomizer), state 19 = obs_buf["policy"][:, :19] at t = 0. Batches of
--envs in stream order, last batch padded with copies (dropped). One zero action is stepped per batch (unused).

  python scripts/g3_dump_t0.py --stream paired --seed 21645 --out /runs/g3_diag/t0_p21645.npz
  python scripts/g3_dump_t0.py --stream unpaired --seed 21640 --paired-seed 21645 --out /runs/g3_diag/t0_u21640.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

STAGE, SPLIT, YAW = "s2a", "train", 15.0
GROUP_OFFSET = 100_000  # group id of a paired sample = GROUP_OFFSET + layout_id; unpaired = its episode index


def paired_samples(seed: int, min_paired: int) -> list:
    """Whole groups from group 0 in order until >= min_paired episodes (GATES 003 S_p)."""
    from vla_tabletop.rollout import make_paired_stream

    out, g = [], 0
    while len(out) < min_paired:
        eps, _ = make_paired_stream(seed, g, 1, STAGE, SPLIT, YAW)
        out += eps
        g += 1
    return out


def unpaired_samples(seed: int, n: int) -> list:
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import make_stream

    return make_stream(seed, 0, n, lambda rng: sample_spec(STAGE, SPLIT, rng), split=SPLIT, cube_yaw_max_deg=YAW)


def meta(episodes: list, stream_seed: int, paired: bool) -> dict:
    """Per-sample identity / label arrays (pure; tested on CPU)."""
    from vla_common.catalog import COLORS
    from vla_tabletop.bridge import episode_seed

    return {"index": np.array([e.index for e in episodes], np.int64),
            "layout_id": np.array([e.layout_id if paired else -1 for e in episodes], np.int64),
            "member": np.array([e.member if paired else -1 for e in episodes], np.int64),
            "group": np.array([GROUP_OFFSET + e.layout_id if paired else e.index for e in episodes], np.int64),
            "label": np.array([COLORS.index(e.spec.target[0]) for e in episodes], np.int64),
            "target_size": np.array([e.spec.target[1] for e in episodes]),
            "instruction": np.array([e.instruction for e in episodes]),
            "spec_json": np.array([e.spec.to_json() for e in episodes]),
            "noise_seed": np.array([episode_seed(stream_seed, e.index) for e in episodes], np.int64),
            "stream_seed": np.int64(stream_seed), "paired": np.bool_(paired)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stream", choices=("paired", "unpaired"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--paired-seed", type=int, default=21645, help="unpaired run: sizes S_u = total - |S_p|")
    p.add_argument("--min-paired", type=int, default=1000)
    p.add_argument("--total", type=int, default=2000)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    if a.stream == "paired":
        eps = paired_samples(a.seed, a.min_paired)
    else:
        eps = unpaired_samples(a.seed, a.total - len(paired_samples(a.paired_seed, a.min_paired)))
    B, H = a.envs, a.image_size
    print(f"[g3_dump] {a.stream} seed {a.seed}: {len(eps)} samples, B {B}, H {H}", flush=True)

    import gymnasium as gym
    import torch

    import vla_tabletop.scene as S
    from collect import geometry_113
    from isaaclab_tasks.utils import launch_simulation
    from lift_rl.physics import assert_on_cuda, assert_run_preconditions
    from vision_lift import cameras as C
    from vision_lift.frames import DelayedFrames
    from vla_common import timing as TM
    from vla_tabletop.rollout import Runner

    cfg = S.make_cfg(B, a.seed, res=H)
    assert_run_preconditions(cfg)
    side, wrist, state = [], [], []
    with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
        t0 = time.time()
        env = gym.make(S.TASK_ID, cfg=cfg)
        u = env.unwrapped
        assert_on_cuda(env)
        run = Runner(env, geometry_113())
        for b0 in range(0, len(eps), B):
            batch = eps[b0:b0 + B]
            pad = batch + [batch[-1]] * (B - len(batch))
            got = {}

            class Capture:  # the first query's inputs, as PolicyCtrl / ChunkExecutor build them at t = 0
                def __init__(self, uu, b, s0):
                    self.u, self.fr = uu, DelayedFrames(TM.PERIOD, TM.LATENCY)

                def act(self, t, snap_fn):
                    assert t == 0
                    st19 = self.u.obs_buf["policy"].cpu().numpy()[:, :19].astype(np.float32)
                    self.fr.capture(t, (C.rgb(self.u, "side_cam").to(torch.uint8).cpu().numpy(),
                                        C.rgb(self.u, "wrist_cam").to(torch.uint8).cpu().numpy()))
                    k, (sd, wr) = self.fr.current()
                    assert k == 0
                    got.update(side=sd, wrist=wr, state=st19)
                    z = torch.zeros((B, 5), device=self.u.device)
                    return z, z

            run.run_batch(pad, 1, {}, controller=Capture)
            n = len(batch)
            side.append(got["side"][:n].copy())
            wrist.append(got["wrist"][:n].copy())
            state.append(got["state"][:n].copy())
            print(f"[g3_dump] batch {b0 // B}: {b0 + n}/{len(eps)} ({time.time() - t0:.0f} s)", flush=True)
        env.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    arr = meta(eps, a.seed, a.stream == "paired") | {"side": np.concatenate(side), "wrist": np.concatenate(wrist),
                                                      "state": np.concatenate(state)}
    assert arr["side"].shape == (len(eps), H, H, 3), arr["side"].shape
    np.savez(a.out, **arr)
    print("G3_DUMP=" + json.dumps({"stream": a.stream, "seed": a.seed, "n": len(eps),
                                   "groups": int(len(set(arr["group"].tolist()))), "out": str(a.out),
                                   "wall_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
