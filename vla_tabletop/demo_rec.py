"""Demo recording (task 011, viewer only; never used by a gate or a training run).

eval_policy.py --record: every kept slot of a new run gets, per camera tick, a frame of a fixed higher-resolution viewer
camera (the close-up camera, aimed once per env at the scene at t = 0; it does NOT feed the policy), the policy's own
side / wrist frames at the trained resolution, and the evaluator's own per-step signals (vla_tabletop.detectors) plus
the viewer-pixel positions of the target cube, the destination and the fingertip centre. Written per episode to
<out dir>/demo_rec/<out stem>/ep<index>/{viewer,side,wrist}.mp4 + signals.json; the verdict is the evaluator's record.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VIEW_W, VIEW_H, VIEW_FOCAL, VIEW_DIST = 960, 720, 18.0, 1.15
SIGNALS = ("confirmed", "held", "success_now", "wrong_grasp", "drop", "slam", "jam", "early_close", "rest_dest",
           "clear")


def project(K: torch.Tensor, pos_w: torch.Tensor, quat_w_ros: torch.Tensor, p_w: torch.Tensor) -> torch.Tensor:
    """Pinhole projection of world points p_w (N, M, 3) for N cameras (ROS optical frame: x right, y down, z forward;
    quat_w_ros xyzw = camera -> world) -> (N, M, 3) = (u px, v px, depth m)."""
    from isaaclab.utils.math import matrix_from_quat

    R = matrix_from_quat(quat_w_ros)  # (N, 3, 3) columns = camera axes in the world
    pc = torch.einsum("nji,nkj->nki", R, p_w - pos_w[:, None])  # R^T (p - c)
    uv = torch.einsum("nij,nkj->nki", K, pc)
    return torch.cat([uv[..., :2] / uv[..., 2:3].clamp_min(1e-6), pc[..., 2:3]], -1)


def jpeg(img: np.ndarray, q: int = 92) -> bytes:
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(b, "JPEG", quality=q)
    return b.getvalue()


def unjpeg(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


class Recorder:
    """One round: tick() once per camera tick from eval_policy's on_step; write() per kept slot after the round."""

    def __init__(self, slots: list[int]):
        self.slots = slots
        self.frames = {j: {"viewer": [], "side": [], "wrist": []} for j in slots}
        self.ticks = {j: [] for j in slots}
        self.aimed = False

    def aim(self, u, s: dict, det) -> None:
        """Fixed viewer pose per env: look at the mean of the on-table cubes (+ the destination pad for place), from
        the 111 video direction."""
        from lift_rl.video import DIR

        from .sheets import CLOSEUP

        ar = torch.arange(det.N, device=u.device)
        cp = s["cube_pos"]
        m = det.on_table.float()[..., None]
        look = (cp * m).sum(1) / m.sum(1).clamp_min(1.0)
        pad = s["pad_pos"][ar, det.dest.clamp(0, 3)]
        look = torch.where(det.is_place[:, None], 0.5 * (look + pad), look)
        look[:, 2] = cp[ar, det.target, 2] + 0.08
        o = u.scene.env_origins
        u.scene[CLOSEUP].set_world_poses_from_view(o + look + VIEW_DIST * DIR.to(u.device), o + look)
        self.aimed = True

    def tick(self, t: int, u, s: dict, exe: torch.Tensor, det, side: np.ndarray, wrist: np.ndarray) -> None:
        from .sheets import CLOSEUP, grab

        if not self.aimed:
            self.aim(u, s, det)
        view = grab(u, CLOSEUP)
        cam = u.scene[CLOSEUP]
        ar = torch.arange(det.N, device=u.device)
        o = u.scene.env_origins
        dest = torch.where(det.is_place[:, None], s["pad_pos"][ar, det.dest.clamp(0, 3)],
                           s["cube_pos"][ar, det.dest.clamp(0, 7)])
        pts = torch.stack([s["cube_pos"][ar, det.target], dest, s["tcp"]], 1) + o[:, None]
        tt = lambda x: x.torch if hasattr(x, "torch") else x  # noqa: E731 - Isaac Lab 3 camera data are warp proxies
        px = project(*(tt(x).float() for x in (cam.data.intrinsic_matrices, cam.data.pos_w, cam.data.quat_w_ros)),
                     pts).cpu().numpy()
        z_t = s["cube_pos"][ar, det.target, 2]
        lift = (z_t - det.geo.z_rest.to(z_t.device)[det.target]).cpu().numpy()
        sig = {"confirmed": det.confirmed, "held": getattr(det, "held_t_now", None), "success_now": det.success_now,
               "wrong_grasp": det.wrong_grasp, "drop": det.drop, "slam": det.slam, "jam": det.jam,
               "early_close": det.early_close, "rest_dest": getattr(det, "rest_dest_now", None),
               "clear": getattr(det, "clear_now", None)}
        sig = {k: (np.zeros(det.N, bool) if v is None else v.cpu().numpy().astype(bool)) for k, v in sig.items()}
        claw = exe[:, 4].cpu().numpy() > 0.5
        for j in self.slots:
            f = self.frames[j]
            f["viewer"].append(jpeg(view[j]))
            f["side"].append(jpeg(side[j], 95))
            f["wrist"].append(jpeg(wrist[j], 95))
            self.ticks[j].append({"t": t, "claw_closed": bool(claw[j]), "lift_m": round(float(lift[j]), 4),
                                  **{k: bool(sig[k][j]) for k in SIGNALS},
                                  "px_target": [round(float(x), 1) for x in px[j, 0]],
                                  "px_dest": [round(float(x), 1) for x in px[j, 1]],
                                  "px_tcp": [round(float(x), 1) for x in px[j, 2]]})

    def write(self, j: int, ep_dir: Path, meta: dict, fps: float) -> None:
        import imageio

        ep_dir.mkdir(parents=True, exist_ok=True)
        for name, fr in self.frames[j].items():
            imgs = [unjpeg(b) for b in fr]
            if name == "viewer" and len(imgs) > 1:
                imgs[0] = imgs[1]  # the viewer pose set at t = 0 renders from the next tick on
            imageio.mimwrite(ep_dir / f"{name}.mp4", imgs, fps=fps, quality=9, macro_block_size=1)
        ticks = self.ticks[j]
        if len(ticks) > 1:
            ticks[0] |= {k: ticks[1][k] for k in ("px_target", "px_dest", "px_tcp")}  # same: pose read at the next tick
        (ep_dir / "signals.json").write_text(json.dumps(meta | {"fps": fps, "viewer_wh": [VIEW_W, VIEW_H],
                                                                "ticks": ticks}, indent=0))
        self.frames[j] = {}
