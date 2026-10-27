"""Contact-sheet helpers for gate frames: a close-up camera (the 111 video camera config, aimed per env), text labels,
and grids of images. Rendering only; never touches physics."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from lift_rl.video import DIR, cam_cfg

CLOSEUP = "video_cam"


def add_closeup(cfg, w: int = 480, h: int = 360, focal: float = 18.0) -> None:
    cfg.scene.video_cam = cam_cfg(focal, w, h)


def aim_closeup(u, look_env: torch.Tensor, dist: float = 0.55) -> None:
    """Point the close-up camera of every env at look_env (N, 3) (env frame) from the 111 video direction."""
    o = u.scene.env_origins
    look = look_env + o
    u.scene[CLOSEUP].set_world_poses_from_view(look + dist * DIR.to(u.device), look)


def grab(u, name: str) -> np.ndarray:
    rgb = u.scene[name].data.output["rgb"]
    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
    return rgb[..., :3].to(torch.uint8).cpu().numpy()


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(img))
    dr = ImageDraw.Draw(im, "RGBA")
    y = 3
    for s in lines:
        bb = dr.textbbox((4, y), s)
        dr.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(0, 0, 0, 160))
        dr.text((4, y), s, fill=(255, 255, 255, 255))
        y = bb[3] + 3
    return np.asarray(im)


def fit(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    w = max(1, round(img.shape[1] * h / img.shape[0]))
    return np.asarray(Image.fromarray(img).resize((w, h), Image.NEAREST))


def grid(rows: list[list[np.ndarray]], h: int = 240, pad: int = 4) -> np.ndarray:
    out_rows = []
    for row in rows:
        cells = []
        for c in row:
            cells += [fit(c, h), np.full((h, pad, 3), 255, np.uint8)]
        out_rows.append(np.concatenate(cells[:-1], 1))
    W = max(r.shape[1] for r in out_rows)
    out = []
    for r in out_rows:
        r = np.concatenate([r, np.full((h, W - r.shape[1], 3), 255, np.uint8)], 1)
        out += [r, np.full((pad, W, 3), 255, np.uint8)]
    return np.concatenate(out[:-1], 0)
