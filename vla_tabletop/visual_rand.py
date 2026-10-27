"""113 visual randomization: the 112 randomizer (vision_lift/visual_rand.py) with colour made part of the task.

Differences to 112 (brief section 3 [consultant]):
  - cube and pad hues are FIXED at their named colours (vla_common.catalog.NOMINAL_RGB); only small saturation / brightness
    jitter (OBJ_SAT_SCALE / OBJ_VAL_SCALE), identical in train and heldout; tests prove every jittered colour stays nearest
    its own named colour;
  - hue randomization applies to the TABLE only (112's bands, margins and nominal handling kept);
  - light direction (global, per batch) and camera-pose jitter (per env) as in 112.
Modes: "nominal", "train", "heldout". Train contains the nominal scene (NOMINAL_FRACTION of envs get exact nominal colours
and zero camera jitter; the same fraction of batches the exact default light). Held-out table hue band, low table
brightness and low light elevation are disjoint from train with margins (tests/test_visual_rand.py).

Interface: params = sample_params(mode, n, gen) is pure CPU torch; apply(env, params, env_ids) writes them to the sim.
apply() expects the 112 env `u` (u.scene["side_cam"], u.scene["wrist_cam"], u.scene["robot"], u.scene.env_origins) and
shapes found by vla_tabletop.scene.shape_ids (visible-flag shapes under "/World/envs/env_<i>/<name>/") for name in CUBE_NAMES + PAD_NAMES + ["Table"]
(scene names: cube_<color>_<size>, pad_<color>; the table prim is "Table" as in 112). Row j of params belongs to env_ids[j].
VERIFIED IN SIM BEFORE STAGE 4: apply() cannot be GPU-tested in the CPU suite (colour write-through, camera poses).

s4r table palette (owner decision 2026-10-03, relayed by consultant 113; Stage 4 / s4r ONLY, explicit opt-in): modes
"s4r_nominal", "s4r_train", "s4r_heldout" (CLI: --table-palette s4r, see run_mode). The modes above stay bit-identical
(tests/test_visual_rand.py pins their output); Stages 1-3 data and their nominal scene never use these modes.
  - nominal table S4R_DEFAULT_TABLE = (0.5, 0.5, 0.5): neutral mid grey (HSV value 0.5, saturation 0);
  - train: table hue free in [0, 1), saturation <= S4R_SAT_CAP = 0.06 (a faint tint at most), value (grey level /
    brightness) in [0.35, 0.75] (contains the nominal 0.5); light elevation 30-80 deg (as 112 train);
  - heldout: same saturation cap, value in a darker band [0.06, 0.20] or a lighter band [0.82, 0.95] (0.15 / 0.07
    from the train range; test margin 0.05 as for the 112 bands), plus low light: elevation 22-25 deg (5 deg below
    train; table shading ~0.58-0.63 vs 0.71-1.0 in train). 112's 12-25 deg was tried first: under the tests' light
    model a shaded dark-yellow cube against a matched dark grey then reaches CIEDE2000 7.1 < the bar of 10;
  - the saturation cap 0.06 is what keeps every s4r table >= 20 CIEDE2000 from every cube / pad at the nominal light
    (0.10 gives 18.7, blue vs a bluish grey; the 112 nominal table gives blue 12.2, the finding behind this palette);
  - cube / pad colours, their jitter, the nominal fraction and camera jitter are those of the modes above; the random
    draws are made in the same order, so for one generator state an s4r mode differs from its 112 twin only in the
    table colour and the light (tested).
Contrast / colour-name tests for this palette (CIEDE2000 under a multiplicative light model calibrated on
results/visual_rand_check.json and results/frames/s4r_palette/check.json): tests/test_visual_rand.py.
"""

from __future__ import annotations

import math

import torch

from vla_common.catalog import ALL_CUBES, COLORS, CUBE_NAMES, NOMINAL_RGB, PAD_NAMES

MODES = {  # table tint (hue bands in [0,1), sat, value) and light elevation [deg]; light azimuth is free
    "train": {"hues": ((0.00, 0.35), (0.50, 1.00)), "sat": (0.30, 0.90), "value": (0.55, 1.00), "light_elev": (30.0, 80.0)},
    "heldout": {"hues": ((0.38, 0.47),), "sat": (0.30, 0.90), "value": (0.30, 0.50), "light_elev": (12.0, 25.0)},
}
OBJ_SAT_SCALE = (0.85, 1.0)  # cube / pad saturation = nominal x U(range)
OBJ_VAL_SCALE = (0.85, 1.0)  # cube / pad brightness = nominal x U(range)
NOMINAL_FRACTION = 0.2
JITTER_POS_M = 0.01
JITTER_ROT_DEG = 2.0
DEFAULT_LIGHT = (-0.57735026, 0.57735026, -0.57735026)  # elevation 35.3 deg
DEFAULT_TABLE = (0.267, 0.467, 0.667)  # measured 112 default table tint (inside the train bands)
S4R_SAT_CAP = 0.06
S4R_MODES = {  # s4r palette (module doc): any hue, saturation <= cap, value bands (grey level), light elevation [deg]
    "train": {"sat": (0.0, S4R_SAT_CAP), "values": ((0.35, 0.75),), "light_elev": (30.0, 80.0)},
    "heldout": {"sat": (0.0, S4R_SAT_CAP), "values": ((0.06, 0.20), (0.82, 0.95)), "light_elev": (22.0, 25.0)},
}
S4R_DEFAULT_TABLE = (0.5, 0.5, 0.5)
VISUAL_MODES = ("nominal", "train", "heldout", "s4r_nominal", "s4r_train", "s4r_heldout")
TABLE_PALETTES = ("112", "s4r")  # CLI --table-palette; "112" = the default, unchanged


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb.unbind(-1)
    mx, mn = rgb.max(-1).values, rgb.min(-1).values
    d = (mx - mn).clamp_min(1e-9)
    h = torch.where(mx == r, ((g - b) / d) % 6, torch.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    return torch.stack([h % 1.0, torch.where(mx > 0, (mx - mn) / mx.clamp_min(1e-9), torch.zeros_like(mx)), mx], -1)


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], -1).gather(-1, i[..., None])[..., 0]
    g = torch.stack([t, v, v, q, p, p], -1).gather(-1, i[..., None])[..., 0]
    b = torch.stack([p, p, t, v, v, q], -1).gather(-1, i[..., None])[..., 0]
    return torch.stack([r, g, b], -1)


def light_elevation_deg(d) -> float:
    d = torch.as_tensor(d, dtype=torch.float32)
    return float(torch.rad2deg(torch.asin(-d[2] / torch.linalg.norm(d))))


def small_rotation(n: int, deg: float, gen: torch.Generator, scale: torch.Tensor) -> torch.Tensor:
    """(n, 3, 3) rotations, each Euler angle uniform in +-deg times a per-row scale (0 for nominal envs)."""
    a = (torch.rand(n, 3, generator=gen) * 2 - 1) * math.radians(deg) * scale[:, None]
    cx, sx, cy, sy, cz, sz = a[:, 0].cos(), a[:, 0].sin(), a[:, 1].cos(), a[:, 1].sin(), a[:, 2].cos(), a[:, 2].sin()
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).view(n, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(n, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).view(n, 3, 3)
    return Rz @ Ry @ Rx


def _jitter_objects(base: torch.Tensor, n: int, gen: torch.Generator, nominal: torch.Tensor) -> torch.Tensor:
    """base (K, 3) nominal RGB -> (n, K, 3): hue unchanged, saturation and brightness scaled; exact nominal where flagged."""
    hsv = rgb_to_hsv(base)[None].expand(n, -1, -1)
    k = base.shape[0]
    s = hsv[..., 1] * (OBJ_SAT_SCALE[0] + (OBJ_SAT_SCALE[1] - OBJ_SAT_SCALE[0]) * torch.rand(n, k, generator=gen))
    v = hsv[..., 2] * (OBJ_VAL_SCALE[0] + (OBJ_VAL_SCALE[1] - OBJ_VAL_SCALE[0]) * torch.rand(n, k, generator=gen))
    rgb = hsv_to_rgb(hsv[..., 0], s, v)
    return torch.where(nominal[:, None, None], base[None].expand(n, -1, -1), rgb)


def _table_tint(mode: str, n: int, gen: torch.Generator) -> torch.Tensor:
    cfg = MODES[mode]
    bands = torch.tensor(cfg["hues"])
    widths = bands[:, 1] - bands[:, 0]
    k = torch.multinomial(widths / widths.sum(), n, replacement=True, generator=gen)
    h = bands[k, 0] + torch.rand(n, generator=gen) * widths[k]
    s0, s1 = cfg["sat"]
    v0, v1 = cfg["value"]
    return hsv_to_rgb(h, s0 + (s1 - s0) * torch.rand(n, generator=gen), v0 + (v1 - v0) * torch.rand(n, generator=gen))


def _table_grey(mode: str, n: int, gen: torch.Generator) -> torch.Tensor:
    """s4r palette table: the same draws as _table_tint (band pick, h, s, v); value bands instead of hue bands."""
    cfg = S4R_MODES[mode]
    bands = torch.tensor(cfg["values"])
    widths = bands[:, 1] - bands[:, 0]
    k = torch.multinomial(widths / widths.sum(), n, replacement=True, generator=gen)
    h = torch.rand(n, generator=gen)
    s0, s1 = cfg["sat"]
    s = s0 + (s1 - s0) * torch.rand(n, generator=gen)
    return hsv_to_rgb(h, s, bands[k, 0] + torch.rand(n, generator=gen) * widths[k])


def run_mode(visual: str, table_palette: str) -> str | None:
    """CLI --visual (nominal | train_rand | heldout) + --table-palette -> sample_params mode; None = no randomizer
    (the scene's own nominal colours, scene.apply_nominal_colors)."""
    assert table_palette in TABLE_PALETTES, table_palette
    base = {"nominal": None, "train_rand": "train", "heldout": "heldout"}[visual]
    return base if table_palette == "112" else f"s4r_{base or 'nominal'}"


def sample_params(mode: str, n: int, gen: torch.Generator) -> dict:
    """Per-env parameters for n envs (CPU). Row j is applied to env_ids[j] by apply()."""
    assert mode in VISUAL_MODES, mode
    s4r = mode.startswith("s4r_")
    out_mode, mode = mode, mode.removeprefix("s4r_")
    if mode == "nominal":
        nom = torch.ones(n, dtype=torch.bool)
    elif mode == "train":
        nom = torch.rand(n, generator=gen) < NOMINAL_FRACTION
    else:
        nom = torch.zeros(n, dtype=torch.bool)
    cube_base = torch.tensor([NOMINAL_RGB[c] for c, _ in ALL_CUBES])  # (8, 3), order = ALL_CUBES = CUBE_NAMES
    pad_base = torch.tensor([NOMINAL_RGB[c] for c in COLORS])  # (4, 3)
    tm = "train" if mode == "nominal" else mode
    table = (torch.where(nom[:, None], torch.tensor(S4R_DEFAULT_TABLE), _table_grey(tm, n, gen)) if s4r else
             torch.where(nom[:, None], torch.tensor(DEFAULT_TABLE), _table_tint(tm, n, gen)))
    if mode == "nominal" or (mode == "train" and float(torch.rand(1, generator=gen)) < NOMINAL_FRACTION):
        light = torch.tensor(DEFAULT_LIGHT)
    else:
        e0, e1 = (S4R_MODES if s4r else MODES)[mode]["light_elev"]
        el = math.radians(e0 + (e1 - e0) * float(torch.rand(1, generator=gen)))
        az = 2 * math.pi * float(torch.rand(1, generator=gen))
        light = torch.tensor([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)])
    keep = (~nom).float()  # zero camera jitter for nominal envs
    return {
        "mode": out_mode, "nominal": nom, "light": light, "table_rgb": table,
        "cube_rgb": _jitter_objects(cube_base, n, gen, nom), "pad_rgb": _jitter_objects(pad_base, n, gen, nom),
        "side_eye_off": (torch.rand(n, 3, generator=gen) * 2 - 1) * JITTER_POS_M * keep[:, None],
        "side_R": small_rotation(n, JITTER_ROT_DEG, gen, keep),
        "wrist_off": (torch.rand(n, 3, generator=gen) * 2 - 1) * JITTER_POS_M * keep[:, None],
        "wrist_R": small_rotation(n, JITTER_ROT_DEG, gen, keep),
    }


def apply(env, params: dict, env_ids: torch.Tensor) -> dict:
    """Call right after an episode reset of env_ids, before the first render. `env` = the 112-style env `u` (see module doc)."""
    import warp as wp
    from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
    from isaaclab_newton.physics import NewtonManager

    from vision_lift import cameras as C

    from . import scene

    if not hasattr(env, "_vr_cache"):
        colors = wp.to_torch(NewtonManager.get_model().shape_color)  # (S, 3), shared with the renderer
        env._vr_cache = {"colors": colors, "ids": {n: scene.shape_ids(env, n) for n in (*CUBE_NAMES, *PAD_NAMES, "Table")}}
    colors, ids = env._vr_cache["colors"], env._vr_cache["ids"]
    dev = env.device
    P = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in params.items()}
    ei = env_ids.to(dev).long()
    for j, name in enumerate(CUBE_NAMES):
        colors[ids[name][ei]] = P["cube_rgb"][:, j, None]
    for j, name in enumerate(PAD_NAMES):
        colors[ids[name][ei]] = P["pad_rgb"][:, j, None]
    colors[ids["Table"][ei]] = P["table_rgb"][:, None]
    # light: global per batch
    renderer = env.scene["side_cam"]._renderer
    d = [float(x) for x in params["light"]]
    renderer.newton_sensor.utils.create_default_light(enable_shadows=renderer.cfg.enable_shadows, direction=wp.vec3f(*d))
    n = len(ei)
    # side camera: eye +-1 cm, view direction +-2 deg
    eye = torch.tensor(C.SIDE_EYE, device=dev) + P["side_eye_off"]
    look = eye + (P["side_R"] @ (torch.tensor(C.SIDE_LOOK, device=dev).expand(n, 3) - eye)[..., None])[..., 0]
    o = env.scene.env_origins[ei]
    env.scene["side_cam"].set_world_poses_from_view(o + eye, o + look, env_ids=ei)
    # wrist camera: hand pose x jittered mount, written once (child of the hand)
    r = env.scene["robot"]
    h = r.body_names.index("panda_hand")
    pos_w, quat_w = r.data.body_link_pos_w, r.data.body_link_quat_w
    hp = (pos_w.torch if hasattr(pos_w, "torch") else pos_w)[ei, h]
    Rh = matrix_from_quat((quat_w.torch if hasattr(quat_w, "torch") else quat_w)[ei, h])
    mount = torch.tensor(C.WRIST_POS, device=dev) + P["wrist_off"]
    Rm = matrix_from_quat(torch.tensor(C.tilt_quat_xyzw(C.WRIST_TILT_DEG), device=dev).expand(n, 4))
    env.scene["wrist_cam"].set_world_poses(hp + (Rh @ mount[..., None])[..., 0], quat_from_matrix(Rh @ P["wrist_R"] @ Rm),
                                           env_ids=ei, convention="ros")
    return {"light": d, "mode": params["mode"], "nominal_envs": int(params["nominal"].sum())}
