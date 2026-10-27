"""Contract HDF5 episodes (vla_common.episode) -> LeRobot dataset v3.0 under datasets/<name>/ (local only, never pushed).

Usage:  python -m vla_policy.export --src data/s1/*.h5 --name s1_pick [--rows control]

Row rate
--------
`--rows control` (the only implemented option): one row per 50 Hz control step t. The row's images are the frames the
policy SAW at t (`side_rgb[frame_idx[t]]`, i.e. delayed, repeated across the PERIOD steps that share a frame), the
row's `observation.state` is state[t] and its `action` is the clean label action[t] (contract v2 a; `action_exec`, the
noisy executed action, is not exported). SmolVLA's action chunk for row t is then rows t .. t+49 = 50 consecutive
control-rate labels, the same stream the policy executes open loop.

`--rows camera` is NOT expressible without changing the action semantics: LeRobot builds the chunk from consecutive
ROWS (`action_delta_indices = range(chunk_size)`, read at timestamps t + i / fps), so with one row per camera frame the
chunk would be [a_t, a_{t+P}, a_{t+2P}, ...] — every P-th control action, a 16.7 Hz action stream that the 50 Hz
executor cannot replay (or a stacked P*5-dim action, a different action space). It is therefore refused. The cost
saving it was meant to buy is obtained at TRAINING time instead (GATES 0B "Dataset row rate", default B): keep the
control-rate rows and draw one random row per camera-frame window per epoch (`FrameWindowSampler`; a window = the rows
of one episode that share `frame_idx`, i.e. show the same image). Storage of the control-rate rows is small: parquet
holds 19 + 5 + 4 numbers per row and the video repeats each frame ~PERIOD times, which the codec stores almost free.

Extra columns: `labels` int64 (4,) (baseline task labels), `frame_idx` int64 (1,) (index of the source frame, for the
alignment test), `split` string. Per-episode source attributes (+ `lerobot_episode_index`) go to
`meta/episodes_113.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vla_common.episode import file_attrs, read_episodes

ROOT = Path(__file__).resolve().parents[1]
CONTROL_HZ = 50  # contract: dt 0.01, decimation 2
# Video encoder. LeRobot's default (libsvtav1, yuv420p, crf 30) gives min PSNR 30.6 dB on 2,000 real 128 px sim rows
# (4:2:0 chroma on small, saturated objects; crf 20 / 10 only reach 31.2 / 31.6), below the 0B.1 bar of 35 dB.
# h264 4:4:4: crf 23 -> 31.8, crf 18 -> 35.3, crf 14 -> 38.2 dB min (42.98 mean), 3.2 MB per 2,000 rows x 2 cameras
# (AV1 crf 30: 2.4 MB). Keyframe interval g = 2 (LeRobot default, cheap random access).
ENCODER = {"vcodec": "h264", "pix_fmt": "yuv444p", "crf": 14}
VIDEO_BACKEND = "pyav"  # native Windows; PERUN (Linux) decodes with torchcodec
_JOINTS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
STATE_NAMES = [f"{j}.pos_rel" for j in _JOINTS] + [f"{j}.vel_rel" for j in _JOINTS] + ["gripper_width"]
ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "claw"]
CAMERAS = ("side", "wrist")


def features(image_size: int) -> dict:
    img = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
    return {
        **{f"observation.images.{c}": dict(img) for c in CAMERAS},
        "observation.state": {"dtype": "float32", "shape": (19,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (5,), "names": ACTION_NAMES},
        "labels": {"dtype": "int64", "shape": (4,), "names": ["task", "target_color", "target_size", "dest"]},
        "frame_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "split": {"dtype": "string", "shape": (1,), "names": None},
    }


def export(files: list[Path], name: str, rows: str = "control", root: Path | None = None,
           encoder: dict | None = None,
           select: Callable[[Path, str, dict], dict | None] | None = None) -> LeRobotDataset:
    """select(file, group, attrs) -> the attrs to export with (e.g. a split override) or None to skip the episode;
    default: every episode, attrs unchanged. Skipped episodes are still read (one at a time)."""
    if rows != "control":
        raise NotImplementedError("--rows camera changes the action-chunk semantics (see module docstring); use "
                                  "--rows control + FrameWindowSampler at training time")
    out = (root or ROOT / "datasets") / name
    first = next(read_episodes(files[0]))
    H = int(first[1]["side_rgb"].shape[1])
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=CONTROL_HZ, features=features(H), root=out,
                               robot_type="franka_panda_ik", use_videos=True, video_backend=VIDEO_BACKEND,
                               rgb_encoder=RGBEncoderConfig(**(encoder or ENCODER)),
                               # streaming: no temporary PNGs; the queue never fills, so no frame is ever dropped
                               streaming_encoding=True, encoder_queue_maxsize=1_000_000)
    sidecar = []
    for f in files:
        fa = file_attrs(f)
        for group, a, at in read_episodes(f):
            if select is not None:
                at = select(f, group, at)
                if at is None:
                    continue
            assert a["side_rgb"].shape[1] == H, f"{f}:{group} image size {a['side_rgb'].shape[1]} != {H}"
            labels = np.asarray(json.loads(at["labels"]) if isinstance(at["labels"], str) else at["labels"],
                                dtype=np.int64).reshape(4)
            for t in range(a["state"].shape[0]):
                k = int(a["frame_idx"][t])
                ds.add_frame({"observation.images.side": a["side_rgb"][k],
                              "observation.images.wrist": a["wrist_rgb"][k],
                              "observation.state": a["state"][t], "action": a["action"][t],
                              "labels": labels, "frame_idx": np.array([k], np.int64),
                              "split": str(at["split"]), "task": str(at["instruction"])})
            ds.save_episode(parallel_encoding=False)  # no process pool: Windows spawn re-imports callers
            sidecar.append({**at, "lerobot_episode_index": len(sidecar), "source": str(f), "group": group,
                            "file_attrs": fa, "n_rows": int(a["state"].shape[0])})
    ds.finalize()
    (out / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sidecar))
    return ds


def load(name_or_dir: str | Path, **kw) -> LeRobotDataset:
    d = Path(name_or_dir)
    d = d if d.is_dir() else ROOT / "datasets" / str(name_or_dir)
    return LeRobotDataset(repo_id=f"local/{d.name}", root=d, video_backend=VIDEO_BACKEND, **kw)


class FrameWindowSampler(torch.utils.data.Sampler[int]):
    """Row option B (GATES 0B): per epoch, one row drawn uniformly at random from every camera-frame window (rows of
    one episode with the same `frame_idx`), in shuffled order. Yields dataset positions (0 .. len(ds) - 1). Each
    sample keeps its 50 consecutive control-rate labels; across epochs every frame age the policy meets at query
    steps is covered. ~PERIOD x fewer samples per epoch than option A (every row)."""

    def __init__(self, ds: LeRobotDataset, seed: int = 0):
        hf = ds.hf_dataset
        key = np.stack([np.asarray(hf["episode_index"]).reshape(-1), np.asarray(hf["frame_idx"]).reshape(-1)], 1)
        _, self.window = np.unique(key, axis=0, return_inverse=True)
        self.window = self.window.reshape(-1)
        self.n = int(self.window.max()) + 1
        self.seed, self.epoch = seed, 0

    def __iter__(self):
        g = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        pick = np.full(self.n, -1)
        u = g.random(self.window.shape[0])  # random row per window = the row with the largest draw
        order = np.lexsort((u, self.window))
        last = np.r_[self.window[order][1:] != self.window[order][:-1], True]
        pick[self.window[order][last]] = order[last]
        return iter(pick[g.permutation(self.n)].tolist())

    def __len__(self) -> int:
        return self.n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--rows", default="control", choices=("control", "camera"))
    args = p.parse_args()
    files = sorted({Path(s) for pat in args.src for s in (Path().glob(pat) if any(c in pat for c in "*?[") else [pat])})
    ds = export(files, args.name, args.rows)
    print(json.dumps({"name": args.name, "episodes": ds.meta.total_episodes, "rows": ds.meta.total_frames,
                      "root": str(ds.root)}))


if __name__ == "__main__":
    main()
