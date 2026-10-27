"""On-disk episodes (HDF5, chunked + compressed; never all in RAM) and the training-sample reader.

One file = one collection batch of E parallel episodes, T control steps, F = ceil(T / period) camera frames.

  /frames/{side,wrist}_rgb     (E, F, H, W, 3) uint8     chunk = one frame
  /frames/{side,wrist}_depth   (E, F, H, W)    float16   chunk = one frame (metres)
  /frames/proprio              (E, F, 27)  student proprio at the capture step (Stage 1 estimator input)
  /steps/proprio               (E, T, 27)  student proprio at each step
  /steps/frame                 (T,)        index of the frame the policy saw at step t (delayed, vision_lift.frames)
  /steps/teacher_arm           (E, T, 4)   teacher arm action on the true state (label)
  /steps/claw_label            (E, T)      teacher claw label: phase machine >= CLOSE (label)
  /steps/exec_action           (E, T, 5)   executed action (arm 4 + claw)
  /steps/driver                (E, T)      1 = teacher executed this step, 0 = student
  /privileged/...              cube pose at capture steps, phase, in_band, ... (labels / analysis ONLY)
  /episodes/...                per-episode outcomes
Student inputs are exactly INPUT_KEYS; nothing under /privileged is ever returned as an input.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

INPUT_KEYS = ("side_rgb", "wrist_rgb", "side_depth", "wrist_depth", "proprio")  # the only student inputs
LABEL_KEYS = ("teacher_arm", "claw_label")
COMP = {"compression": "lzf"}


class EpisodeWriter:
    """Streams one batch of E episodes to disk: frames as they are captured, per-step arrays at the end."""

    def __init__(self, path: Path, E: int, T: int, F: int, res: int, proprio_dim: int, attrs: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w")
        self.E, self.T, self.F = E, T, F
        for k, v in attrs.items():
            self.f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list, tuple)) or v is None else v
        g = self.f.create_group("frames")
        for cam in ("side", "wrist"):
            g.create_dataset(f"{cam}_rgb", (E, F, res, res, 3), np.uint8, chunks=(1, 1, res, res, 3), **COMP)
            g.create_dataset(f"{cam}_depth", (E, F, res, res), np.float16, chunks=(1, 1, res, res), **COMP)
        g.create_dataset("proprio", (E, F, proprio_dim), np.float32)
        self.steps: dict[str, list] = {}
        self.priv_frames: dict[str, list] = {}
        self.priv_steps: dict[str, list] = {}

    def add_frame(self, k: int, side_rgb, wrist_rgb, side_depth, wrist_depth, proprio, priv: dict | None = None):
        g = self.f["frames"]
        g["side_rgb"][:, k] = side_rgb
        g["wrist_rgb"][:, k] = wrist_rgb
        g["side_depth"][:, k] = np.asarray(side_depth).astype(np.float16)
        g["wrist_depth"][:, k] = np.asarray(wrist_depth).astype(np.float16)
        g["proprio"][:, k] = proprio
        for key, v in (priv or {}).items():
            self.priv_frames.setdefault(key, []).append(np.asarray(v))

    def add_step(self, data: dict, priv: dict | None = None):
        for key, v in data.items():
            self.steps.setdefault(key, []).append(np.asarray(v))
        for key, v in (priv or {}).items():
            self.priv_steps.setdefault(key, []).append(np.asarray(v))

    def close(self, frame_of_step: np.ndarray, episodes: dict):
        g = self.f.create_group("steps")
        g["frame"] = np.asarray(frame_of_step).astype(np.int32)
        for key, v in self.steps.items():
            g[key] = np.stack(v, 1)  # (E, T, ...)
        p = self.f.create_group("privileged")
        for key, v in self.priv_frames.items():
            p[f"frame_{key}"] = np.stack(v, 1)  # (E, F, ...)
        for key, v in self.priv_steps.items():
            p[f"step_{key}"] = np.stack(v, 1)  # (E, T, ...)
        e = self.f.create_group("episodes")
        for key, v in episodes.items():
            e[key] = np.asarray(v)
        self.f.close()


def list_files(dirs) -> list[Path]:
    out: list[Path] = []
    for d in [dirs] if isinstance(dirs, (str, Path)) else dirs:
        out += sorted(Path(d).glob("*.h5"))
    return out


class StepDataset(torch.utils.data.Dataset):
    """(episode, step) samples: the delayed side/wrist frames + step proprio -> teacher labels.

    `depth` selects RGB (False) or RGB-D (True). File handles open lazily (per DataLoader worker).
    """

    def __init__(self, files, depth: bool = False, episodes: dict | None = None):
        self.files = [Path(f) for f in files]
        self.depth = depth
        self.index: list[tuple[int, int, int]] = []  # (file, episode, step)
        self._frame_of_step = []
        for fi, f in enumerate(self.files):
            with h5py.File(f, "r") as h:
                E, T = h["steps/teacher_arm"].shape[:2]
                self._frame_of_step.append(h["steps/frame"][:])
            es = episodes.get(fi, range(E)) if episodes else range(E)
            self.index += [(fi, e, t) for e in es for t in range(T)]
        self._h: dict[int, h5py.File] = {}

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        if fi not in self._h:
            self._h[fi] = h5py.File(self.files[fi], "r")
        return self._h[fi]

    def __getitem__(self, i):
        fi, e, t = self.index[i]
        h = self._file(fi)
        k = int(self._frame_of_step[fi][t])
        out = {}
        for cam in ("side", "wrist"):
            out[f"{cam}_rgb"] = torch.from_numpy(h[f"frames/{cam}_rgb"][e, k]).permute(2, 0, 1)  # (3, H, W) uint8
            if self.depth:
                out[f"{cam}_depth"] = torch.from_numpy(h[f"frames/{cam}_depth"][e, k].astype(np.float32))[None]
        out["proprio"] = torch.from_numpy(h["steps/proprio"][e, t])
        out["teacher_arm"] = torch.from_numpy(h["steps/teacher_arm"][e, t])
        out["claw_label"] = torch.tensor(float(h["steps/claw_label"][e, t]))
        if "privileged/step_rel_yaw_deg" in h:  # auxiliary training TARGET (label), never a model input
            out["aux_rel_yaw_deg"] = torch.tensor(float(h["privileged/step_rel_yaw_deg"][e, t]))
        out["_where"] = torch.tensor([fi, e, t, k])
        return out


def model_inputs(batch: dict) -> dict:
    """The ONLY path from a batch to the student model: whitelisted keys, nothing else."""
    return {k: batch[k] for k in INPUT_KEYS if k in batch}


class FrameDataset(torch.utils.data.Dataset):
    """Stage 1 perception samples, one per captured camera frame: side/wrist frame k + student proprio at the capture
    step -> cube position (env frame) and yaw at that instant (privileged LABELS, never inputs). Also returns the
    phase at the capture step (for the per-phase report only)."""

    def __init__(self, files, depth: bool = False, pregrasp: bool = False):
        """pregrasp: keep only frames whose capture-step phase is HOVER / DESCEND / CLOSE and that come before the
        episode's first LIFT step (= before the first confirmed grasp; pre-publish addendum 2)."""
        self.files = [Path(f) for f in files]
        self.depth = depth
        self.index: list[tuple[int, int, int]] = []
        for fi, f in enumerate(self.files):
            with h5py.File(f, "r") as h:
                E, Fn = h["frames/proprio"].shape[:2]
                if not pregrasp:
                    self.index += [(fi, e, k) for e in range(E) for k in range(Fn)]
                    continue
                period = int(h.attrs.get("period", 3)) if "period" in h.attrs else 3
                sp = np.asarray(h["privileged/step_phase"][:])  # (E, steps)
            for e in range(E):
                lift = np.nonzero(sp[e] >= 3)[0]  # 3 = LIFT (lift_rl.phases)
                first_lift = int(lift[0]) if lift.size else sp.shape[1]
                self.index += [(fi, e, k) for k in range(Fn) if k * period < first_lift and sp[e, k * period] <= 2]
        self._h: dict[int, h5py.File] = {}

    def __len__(self):
        return len(self.index)

    def _file(self, fi):
        if fi not in self._h:
            self._h[fi] = h5py.File(self.files[fi], "r")
        return self._h[fi]

    def __getitem__(self, i):
        fi, e, k = self.index[i]
        h = self._file(fi)
        out = {}
        for cam in ("side", "wrist"):
            out[f"{cam}_rgb"] = torch.from_numpy(h[f"frames/{cam}_rgb"][e, k]).permute(2, 0, 1)
            if self.depth:
                out[f"{cam}_depth"] = torch.from_numpy(h[f"frames/{cam}_depth"][e, k].astype(np.float32))[None]
        out["proprio"] = torch.from_numpy(h["frames/proprio"][e, k])
        out["cube_pos"] = torch.from_numpy(h["privileged/frame_cube_pos"][e, k])
        out["cube_yaw"] = torch.tensor(float(h["privileged/frame_cube_yaw"][e, k]))
        period = int(h.attrs.get("period", 3)) if "period" in h.attrs else 3
        out["phase"] = torch.tensor(int(h["privileged/step_phase"][e, k * period]))
        out["_where"] = torch.tensor([fi, e, k])
        return out
