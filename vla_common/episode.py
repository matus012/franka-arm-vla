"""Episode files, sim -> exporter (docs/CONTRACT.md "Episode files"). stdlib + numpy + h5py only.

One HDF5 file per chunk of episodes, one group `ep_<k>` per kept episode:
  side_rgb, wrist_rgb (F, H, H, 3) uint8   frames captured at camera ticks (step 0, PERIOD, 2*PERIOD, ...)
  state               (T, 19) float32      state at control step t, before the action
  action              (T, 5)  float32      clean label at step t (expert action clipped to the action box; v2 a)
  action_exec         (T, 5)  float32      action actually executed at step t (label + demo noise, clipped)
  frame_idx           (T,)    int32        index into the frames the policy sees at step t (delayed)
plus the episode attributes EP_ATTRS and the file attributes FILE_ATTRS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

EP_ATTRS = ("instruction", "split", "task", "spec_json", "labels", "seed", "env_id", "success", "metrics_json",
            "period", "latency", "image_size", "visual_mode", "action_noise_sigma", "expert_version",
            "episode_index", "T", "n_action_steps", "detector_version")
ARRAYS = ("side_rgb", "wrist_rgb", "state", "action", "action_exec", "frame_idx")
FILE_ATTRS = ("git_sha", "created", "stage")
STATE_DIM, ACTION_DIM = 19, 5
COMP = {"compression": "lzf"}


def check_episode(side_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray, action: np.ndarray,
                  action_exec: np.ndarray, frame_idx: np.ndarray) -> None:
    F, H = side_rgb.shape[0], side_rgb.shape[1]
    T = state.shape[0]
    for name, a in (("side_rgb", side_rgb), ("wrist_rgb", wrist_rgb)):
        assert a.dtype == np.uint8 and a.shape == (F, H, H, 3), (name, a.dtype, a.shape)
    assert state.dtype == np.float32 and state.shape == (T, STATE_DIM), (state.dtype, state.shape)
    for name, a in (("action", action), ("action_exec", action_exec)):
        assert a.dtype == np.float32 and a.shape == (T, ACTION_DIM), (name, a.dtype, a.shape)
    assert frame_idx.dtype == np.int32 and frame_idx.shape == (T,), (frame_idx.dtype, frame_idx.shape)
    assert T > 0 and frame_idx.min() >= 0 and frame_idx.max() < F, "frame_idx out of range"
    assert np.all(np.diff(frame_idx) >= 0), "frame_idx must be non-decreasing"


def _attr(v: Any) -> Any:
    return json.dumps(v) if isinstance(v, (dict, list, tuple)) or v is None else v


class EpisodeWriter:
    """`with EpisodeWriter(path, file_attrs) as w: w.add(side, wrist, state, action, frame_idx, attrs)`."""

    def __init__(self, path: str | Path, file_attrs: dict):
        missing = set(FILE_ATTRS) - set(file_attrs)
        assert not missing, f"file attrs missing: {sorted(missing)}"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w")
        for k, v in file_attrs.items():
            self.f.attrs[k] = _attr(v)
        self.n = 0

    def add(self, side_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray, action: np.ndarray,
            action_exec: np.ndarray, frame_idx: np.ndarray, attrs: dict) -> str:
        check_episode(side_rgb, wrist_rgb, state, action, action_exec, frame_idx)
        missing = set(EP_ATTRS) - set(attrs)
        assert not missing, f"episode attrs missing: {sorted(missing)}"
        assert int(attrs["T"]) == state.shape[0], ("attr T", attrs["T"], state.shape[0])
        H = side_rgb.shape[1]
        g = self.f.create_group(f"ep_{self.n}")
        for name, a in (("side_rgb", side_rgb), ("wrist_rgb", wrist_rgb)):
            g.create_dataset(name, data=a, chunks=(1, H, H, 3), **COMP)
        g["state"], g["action"], g["action_exec"], g["frame_idx"] = state, action, action_exec, frame_idx
        for k, v in attrs.items():
            g.attrs[k] = _attr(v)
        self.n += 1
        return g.name

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _py(v: Any) -> Any:
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def file_attrs(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: _py(v) for k, v in f.attrs.items()}


def read_episodes(path: str | Path) -> Iterator[tuple[str, dict[str, np.ndarray], dict]]:
    """Yields (group name, arrays, attrs) per episode in index order; one episode in RAM at a time."""
    with h5py.File(path, "r") as f:
        for name in sorted((k for k in f if k.startswith("ep_")), key=lambda k: int(k[3:])):
            g = f[name]
            arrays = {k: g[k][()] for k in ARRAYS}
            check_episode(**arrays)
            yield name, arrays, {k: _py(v) for k, v in g.attrs.items()}
