"""Stage 4 visual randomization, limited to what the Newton Warp renderer supports (isaaclab_newton / newton 1.2.1):

  supported, used:
    - cube and table colour: per env, per episode — the renderer's albedo = shape colour x texture, so a colour tint on
      the cube's and the table's visual shapes (the cube's letters stay visible);
    - lighting: the direction of the one directional light — GLOBAL (one light list shared by all envs), per batch;
    - camera pose jitter: per env, per episode — side camera eye +-1 cm per axis and view direction +-2 deg; wrist
      camera mount +-1 cm per axis and +-2 deg about each axis (the wrist camera is a child of the hand; its jittered
      world pose is written once after the reset, so the jittered mount rides with the hand for the whole episode).
  not supported by the renderer (documented, not randomized): light intensity / colour and ambient level (ambient is
  compiled in at 0.5), per-env lights, swapping textures at run time (textures load once at start), table texture
  patterns beyond a colour tint.

Modes: "nominal" (no change: default colours, default light, no jitter), "train", "heldout".
Fix after Stage 4 run 1 (the default scene's hue ~0.585 fell inside the old held-out band): the TRAIN distribution
contains the nominal scene — the default tints (hue 0.585, saturation 0.56 / 0.60, brightness 0.91 / 0.67) lie inside
the train bands, and NOMINAL_FRACTION of the envs per episode get the exact default colours with zero camera jitter,
and the same fraction of batches the exact default light (elevation 35.3 deg, inside the train range). The HELD-OUT
bands are disjoint from train (hue margin 0.03, brightness margin 0.05, light-elevation margin 5 deg) and therefore
from nominal (tests/test_visual_rand.py). Jitter magnitudes are the same in train and held-out (+-1 cm / +-2 deg).
"""

from __future__ import annotations

import math

import torch

from . import cameras as C

# (hue bands in [0, 1), brightness range) for the tints; light elevation range [deg]; light azimuth is free
MODES = {
    "train": {"hues": ((0.00, 0.35), (0.50, 1.00)), "sat": (0.30, 0.90), "value": (0.55, 1.00), "light_elev": (30.0, 80.0)},
    "heldout": {"hues": ((0.38, 0.47),), "sat": (0.30, 0.90), "value": (0.30, 0.50), "light_elev": (12.0, 25.0)},
}
NOMINAL_FRACTION = 0.2  # train mode: share of envs (colours + jitter) and batches (light) given the exact nominal scene
JITTER_POS_M = 0.01
JITTER_ROT_DEG = 2.0
DEFAULT_LIGHT = (-0.57735026, 0.57735026, -0.57735026)  # elevation asin(0.577) = 35.3 deg


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb.unbind(-1)
    mx, mn = rgb.max(-1).values, rgb.min(-1).values
    d = (mx - mn).clamp_min(1e-9)
    h = torch.where(mx == r, ((g - b) / d) % 6, torch.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    return torch.stack([h % 1.0, torch.where(mx > 0, (mx - mn) / mx.clamp_min(1e-9), torch.zeros_like(mx)), mx], -1)


def light_elevation_deg(d) -> float:
    d = torch.as_tensor(d, dtype=torch.float32)
    return float(torch.rad2deg(torch.asin(-d[2] / torch.linalg.norm(d))))


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def _hsv_to_rgb(h, s, v):
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], -1).gather(-1, i[..., None])[..., 0]
    g = torch.stack([t, v, v, q, p, p], -1).gather(-1, i[..., None])[..., 0]
    b = torch.stack([p, p, t, v, v, q], -1).gather(-1, i[..., None])[..., 0]
    return torch.stack([r, g, b], -1)


def _small_rotation(n: int, deg: float, gen, device, scale: torch.Tensor | None = None) -> torch.Tensor:
    """(n, 3, 3) rotations with each Euler angle uniform in +-deg (times a per-row scale, e.g. 0 for nominal envs)."""
    a = (torch.rand(n, 3, generator=gen, device=device) * 2 - 1) * math.radians(deg)
    if scale is not None:
        a = a * scale[:, None]
    cx, sx, cy, sy, cz, sz = a[:, 0].cos(), a[:, 0].sin(), a[:, 1].cos(), a[:, 1].sin(), a[:, 2].cos(), a[:, 2].sin()
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).view(n, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(n, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).view(n, 3, 3)
    return Rz @ Ry @ Rx


class VisualRandomizer:
    def __init__(self, u, mode: str, seed: int):
        assert mode in ("nominal", "train", "heldout")
        self.u, self.mode, self.n, self.dev = u, mode, u.num_envs, u.device
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        from isaaclab_newton.physics import NewtonManager

        self.model = NewtonManager.get_model()
        labels = self.model.shape_label
        self.cube_ids = C.cube_shape_ids(u)
        self.table_ids = torch.tensor([next(k for k, lab in enumerate(labels)
                                            if lab.startswith(f"/World/envs/env_{i}/Table/") and "visual" in lab.lower())
                                       for i in range(self.n)], device=self.dev)
        import warp as wp

        self._colors = wp.to_torch(self.model.shape_color)  # (S, 3) float, shared with the renderer
        self.default_cube = self._colors[self.cube_ids].clone()
        self.default_table = self._colors[self.table_ids].clone()
        self.last = {}

    def _tint(self) -> torch.Tensor:
        cfg = MODES[self.mode]
        bands = torch.tensor(cfg["hues"], device=self.dev)
        widths = bands[:, 1] - bands[:, 0]
        k = torch.multinomial(widths / widths.sum(), self.n, replacement=True, generator=self.gen)
        h = bands[k, 0] + torch.rand(self.n, generator=self.gen, device=self.dev) * widths[k]
        s0, s1 = cfg["sat"]
        s = s0 + (s1 - s0) * torch.rand(self.n, generator=self.gen, device=self.dev)
        v0, v1 = cfg["value"]
        v = v0 + (v1 - v0) * torch.rand(self.n, generator=self.gen, device=self.dev)
        return _hsv_to_rgb(h, s, v)

    def _light(self):
        renderer = self.u.scene["side_cam"]._renderer
        import warp as wp

        use_default = self.mode == "nominal" or (
            self.mode == "train" and float(torch.rand(1, generator=self.gen, device=self.dev)) < NOMINAL_FRACTION)
        if use_default:
            d = torch.tensor(DEFAULT_LIGHT)
        else:
            e0, e1 = MODES[self.mode]["light_elev"]
            el = math.radians(e0 + (e1 - e0) * float(torch.rand(1, generator=self.gen, device=self.dev)))
            az = 2 * math.pi * float(torch.rand(1, generator=self.gen, device=self.dev))
            d = torch.tensor([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
        renderer.newton_sensor.utils.create_default_light(enable_shadows=renderer.cfg.enable_shadows,
                                                         direction=wp.vec3f(*[float(x) for x in d]))
        return d.tolist()

    def apply(self) -> dict:
        """Call right after an episode reset, before the first render."""
        u = self.u
        nominal = self.mode == "nominal"
        # per env: nominal scene (default colours, zero jitter) always in nominal mode, with NOMINAL_FRACTION in train
        if nominal:
            nom = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        elif self.mode == "train":
            nom = torch.rand(self.n, generator=self.gen, device=self.dev) < NOMINAL_FRACTION
        else:
            nom = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        jp, jr = JITTER_POS_M, JITTER_ROT_DEG
        cube_t, table_t = (self.default_cube, self.default_table) if nominal else (self._tint(), self._tint())
        self._colors[self.cube_ids] = torch.where(nom[:, None], self.default_cube, cube_t)
        self._colors[self.table_ids] = torch.where(nom[:, None], self.default_table, table_t)
        light = self._light()
        keep = (~nom).float()[:, None]  # zero jitter for nominal envs
        # side camera: eye +-1 cm per axis, view direction +-2 deg (look-at point moved perpendicular to the view axis)
        o = u.scene.env_origins
        eye = torch.tensor(C.SIDE_EYE, device=self.dev) + (torch.rand(self.n, 3, generator=self.gen, device=self.dev) * 2 - 1) * jp * keep
        look = torch.tensor(C.SIDE_LOOK, device=self.dev).expand(self.n, 3)
        d = look - eye
        R = _small_rotation(self.n, jr, self.gen, self.dev, scale=keep[:, 0])
        look = eye + (R @ d[..., None])[..., 0]
        u.scene["side_cam"].set_world_poses_from_view(o + eye, o + look)
        # wrist camera: hand pose x jittered mount, written once (the prim is a child of the hand)
        from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

        r = u.scene["robot"]
        h = r.body_names.index("panda_hand")
        hp, hq = _t(r.data.body_link_pos_w)[:, h], _t(r.data.body_link_quat_w)[:, h]
        Rh = matrix_from_quat(hq)
        mount_pos = torch.tensor(C.WRIST_POS, device=self.dev) + (torch.rand(self.n, 3, generator=self.gen, device=self.dev) * 2 - 1) * jp * keep
        Rm = matrix_from_quat(torch.tensor(C.tilt_quat_xyzw(C.WRIST_TILT_DEG), device=self.dev).expand(self.n, 4))
        Rj = _small_rotation(self.n, jr, self.gen, self.dev, scale=keep[:, 0])
        pos = hp + (Rh @ mount_pos[..., None])[..., 0]
        q_ros = quat_from_matrix(Rh @ Rj @ Rm)
        u.scene["wrist_cam"].set_world_poses(pos, q_ros, convention="ros")
        self.last = {"light": light, "mode": self.mode, "nominal_envs": int(nom.sum())}
        return self.last
