"""Synthetic contract episodes whose frames encode (frame index, episode id, camera) as flat grey stripes, so a decoded
video frame can be mapped back to its source frame despite lossy coding. Used by the protocol-free 0B tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_lift.frames import frame_index
from vla_common.episode import EpisodeWriter

STEP, BASE, NV = 8, 16, 28  # grey level = BASE + STEP * digit, digit in [0, NV); decoding needs |codec error| < STEP / 2
H = 48  # default image size: 3 stripes of H // 3 px
PERIOD, LATENCY = 3, 1


def _grey(d: int) -> int:
    return BASE + STEP * d


def make_frame(k: int, ep: int, cam: int, h: int = H) -> np.ndarray:
    img = np.empty((h, h, 3), np.uint8)
    w = h // 3
    for s, d in enumerate((k % NV, k // NV, (ep * 2 + cam) % NV)):
        img[:, w * s:w * (s + 1)] = _grey(d)
    img[:, 3 * w:] = 0
    return img


def decode_frame(img: np.ndarray) -> tuple[int, int, int, float]:
    """(frame index, episode id, camera, max interior error) from a decoded (H, H, 3) uint8 frame."""
    digits, err, w = [], 0.0, img.shape[0] // 3
    for s in range(3):
        block = img[4:-4, w * s + 4:w * (s + 1) - 4].astype(np.float64)
        d = int(round((block.mean() - BASE) / STEP))
        digits.append(d)
        err = max(err, float(np.abs(block - _grey(d)).max()))
    return digits[0] + NV * digits[1], digits[2] // 2, digits[2] % 2, err


def episode_arrays(T: int, ep: int, rng: np.random.Generator, h: int = H) -> dict[str, np.ndarray]:
    F = (T + PERIOD - 1) // PERIOD
    fidx = np.array([frame_index(t, PERIOD, LATENCY) for t in range(T)], np.int32)
    action = rng.uniform(-1, 1, (T, 5)).astype(np.float32)
    action[:, 0] = np.arange(T) + 1000 * ep  # the executed action at step t encodes (episode, t)
    action[:, 4] = (rng.random(T) > 0.5).astype(np.float32)
    action_exec = action.copy()
    action_exec[:, 1:4] = np.clip(action[:, 1:4] + rng.normal(0, 0.1, (T, 3)), -1, 1)  # != label: exporter must use label
    state = rng.normal(size=(T, 19)).astype(np.float32)
    state[:, 0] = np.arange(T) + 1000 * ep
    return {"side_rgb": np.stack([make_frame(k, ep, 0, h) for k in range(F)]),
            "wrist_rgb": np.stack([make_frame(k, ep, 1, h) for k in range(F)]),
            "state": state, "action": action, "action_exec": action_exec.astype(np.float32), "frame_idx": fidx}


def attrs(ep: int, T: int, h: int, env_id: int = 0, seed: int = 0, instruction: str | None = None) -> dict:
    return {"instruction": instruction or f"pick up the cube number {ep}", "split": "train", "task": "pick",
            "spec_json": {"ep": ep}, "labels": [0, ep % 4, ep % 2, 12], "seed": seed, "env_id": env_id,
            "success": True, "metrics_json": {}, "period": PERIOD, "latency": LATENCY, "image_size": h,
            "visual_mode": "nominal", "action_noise_sigma": 0.1, "expert_version": "synthetic", "episode_index": ep,
            "T": T, "n_action_steps": 10, "detector_version": "none"}


def write_file(path: Path, lengths: list[int], seed: int = 0, ep0: int = 0, h: int = H) -> list[dict]:
    rng = np.random.default_rng(seed)
    eps = []
    with EpisodeWriter(path, {"git_sha": "test", "created": "test", "stage": "0B"}) as w:
        for i, T in enumerate(lengths):
            ep = ep0 + i
            a = episode_arrays(T, ep, rng, h)
            at = attrs(ep, T, h, env_id=i, seed=seed)
            w.add(**a, attrs=at)
            eps.append({"arrays": a, "attrs": at})
    return eps
