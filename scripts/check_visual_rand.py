"""Sim check of vla_tabletop.visual_rand.apply() (needed before Stage 4). 8 envs, one fixed layout set, modes nominal /
train / heldout applied in turn to the same env:

  python scripts/check_visual_rand.py [--envs 8] [--seed 5] [--res 128]
  -> results/frames/visual_rand_check/sheet.png   (rows: side + wrist per mode, one column per env)
  -> results/visual_rand_check.json               (per-object measured mean colour vs named colour, table hue, camera shift)
  python scripts/check_visual_rand.py --table-palette s4r      (the s4r palette, visual_rand module doc)
  -> results/frames/s4r_palette/sheet.png, <condition>.png, check.json
     conditions: nominal, train, heldout (as sampled), heldout_dark / heldout_light (the sampled held-out tables forced
     to spread over the dark / light value band, default light), low_light (nominal grey table, light at the held-out
     minimum elevation), light_12deg (the same at 12 deg, below every s4r range: light-model calibration point for
     tests/test_visual_rand.py). Also recorded: per object the shading factor (measured / sampled mean RGB, the table's
     likewise) and the CIEDE2000 between its measured colour and its env's measured table colour.

Colours are measured from the instance-segmentation masks (side and wrist cameras) as the mean rendered RGB, and
compared to the four named colours by nearest CIELAB distance (the unit test's space; the rendered colour is shaded, so
a chroma-only (a, b) nearest is reported next to it). Table: mean rendered hue vs the sampled table hue. Camera
jitter: centroid shift (px) of the on-table objects' masks in the side camera relative to the nominal render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=8)
p.add_argument("--seed", type=int, default=5)
p.add_argument("--res", type=int, default=128)
p.add_argument("--table-palette", default="112", choices=("112", "s4r"))
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from tests.test_visual_rand import _de00, _lab  # noqa: E402
from vision_lift import cameras as C  # noqa: E402
from vla_common.catalog import ALL_CUBES, COLORS, CUBE_NAMES, NOMINAL_RGB, PAD_NAMES  # noqa: E402
from vla_common.tasks import sample_spec  # noqa: E402
from vla_tabletop import visual_rand as VR  # noqa: E402
from vla_tabletop.layout import layout_array, sample_layout  # noqa: E402
from vla_tabletop.sheets import grid, label  # noqa: E402

B = args.envs
S4R = args.table_palette == "s4r"
OUT_PNG = ROOT / "results" / "frames" / ("s4r_palette" if S4R else "visual_rand_check") / "sheet.png"
OUT_JSON = OUT_PNG.parent / "check.json" if S4R else ROOT / "results" / "visual_rand_check.json"
MIN_PX = 12
ref = _lab(np.array([NOMINAL_RGB[c] for c in COLORS], dtype=np.float64))
cfg = S.make_cfg(B, args.seed, res=args.res, seg=True)
assert_run_preconditions(cfg)


def nearest(rgb01: np.ndarray, chroma_only: bool) -> int:
    d = _lab(rgb01.astype(np.float64)) - ref
    return int(np.linalg.norm(d[:, 1:] if chroma_only else d, axis=-1).argmin())


with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    rng = np.random.default_rng(args.seed)
    lays = []
    for _ in range(B):
        sp = sample_spec("s4", "train", rng)
        lays.append(sample_layout(sp, rng)[0])
    u._layout = torch.as_tensor(layout_array(lays), device=u.device)
    env.reset()
    C.aim_side(u)
    zero = torch.zeros(B, 5, device=u.device)
    for _ in range(10):
        env.step(zero)
    on = (u._layout[:, :12, 4] > 0.5).cpu().numpy()  # (B, 12) OBJ order = cubes then pads
    ids = {n: S.shape_ids(u, n) for n in (*CUBE_NAMES, *PAD_NAMES, "Table")}
    ar = torch.arange(B, device=u.device)

    def render() -> dict:
        for _ in range(2):
            env.step(zero)
        return {c: (C.rgb(u, f"{c}_cam").to(torch.uint8).cpu().numpy(),
                    S._t(u.scene[f"{c}_cam"].data.output["instance_segmentation_fast"])[..., 0].long()) for c in ("side", "wrist")}

    def mask(seg: torch.Tensor, name: str) -> np.ndarray:  # (B, H, W) bool
        return (seg[:, None] == ids[name][:, :, None, None]).any(1).cpu().numpy()

    tiles: dict = {}
    result: dict = {"config": vars(args), "modes": {}}
    base_centroid = None
    def conditions():
        if not S4R:
            for mode in ("nominal", "train", "heldout"):
                yield mode, VR.sample_params(mode, B, torch.Generator().manual_seed(1000 + args.seed))
            return
        for mode in ("nominal", "train", "heldout"):
            yield mode, VR.sample_params(f"s4r_{mode}", B, torch.Generator().manual_seed(1000 + args.seed))
        for name, (lo, hi) in zip(("heldout_dark", "heldout_light"), VR.S4R_MODES["heldout"]["values"]):
            q = VR.sample_params("s4r_heldout", B, torch.Generator().manual_seed(1000 + args.seed))
            hsv = VR.rgb_to_hsv(q["table_rgb"])
            q["table_rgb"] = VR.hsv_to_rgb(hsv[:, 0], hsv[:, 1], torch.linspace(lo, hi, B))
            q["light"] = torch.tensor(VR.DEFAULT_LIGHT)
            yield name, q
        for name, deg in (("low_light", VR.S4R_MODES["heldout"]["light_elev"][0]), ("light_12deg", 12.0)):
            q = VR.sample_params("s4r_nominal", B, torch.Generator().manual_seed(1000 + args.seed))
            el, az = np.radians(deg), np.radians(-45.0)  # DEFAULT_LIGHT's azimuth
            q["light"] = torch.tensor([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), -np.sin(el)], dtype=torch.float32)
            yield name, q

    for mode, params in conditions():
        info = VR.apply(u, params, ar)
        shots = render()
        tiles[mode] = {c: shots[c][0] for c in shots}
        stats = {"n_checked": 0, "n_skipped_small_mask": 0, "nearest_lab_ok": 0, "nearest_chroma_ok": 0, "rows": []}
        shade = {"object": [], "table": [], "de00_obj_table": []}
        for cam, (rgb, seg) in shots.items():
            tmc = mask(seg, "Table")
            tab_mean = [rgb[e][tmc[e]].mean(0) / 255.0 if tmc[e].sum() >= MIN_PX else None for e in range(B)]
            for e in range(B):
                if tab_mean[e] is not None:
                    shade["table"].append(float(tab_mean[e].mean() / params["table_rgb"][e].mean()))
            for j, name in enumerate((*CUBE_NAMES, *PAD_NAMES)):
                true = COLORS.index(name.split("_")[1])
                m = mask(seg, name)
                for e in range(B):
                    if not on[e, j]:
                        continue
                    if m[e].sum() < MIN_PX:
                        stats["n_skipped_small_mask"] += 1
                        continue
                    mean = rgb[e][m[e]].mean(0) / 255.0
                    nl, nc = nearest(mean[None], False), nearest(mean[None], True)
                    samp = params["cube_rgb"][e, j] if j < 8 else params["pad_rgb"][e, j - 8]
                    shade["object"].append(float(mean.mean() / samp.mean()))
                    if tab_mean[e] is not None:
                        shade["de00_obj_table"].append((float(_de00(_lab(mean.astype(np.float64)),
                                                                     _lab(tab_mean[e].astype(np.float64)))), cam, e, name))
                    stats["n_checked"] += 1
                    stats["nearest_lab_ok"] += int(nl == true)
                    stats["nearest_chroma_ok"] += int(nc == true)
                    if nl != true or nc != true or len(stats["rows"]) < 6:
                        stats["rows"].append({"cam": cam, "env": e, "object": name, "px": int(m[e].sum()),
                                              "named_rgb": NOMINAL_RGB[COLORS[true]], "sampled_rgb": [round(float(x), 3) for x in
                                              (params["cube_rgb"][e, j] if j < 8 else params["pad_rgb"][e, j - 8])],
                                              "measured_rgb": [round(float(x), 3) for x in mean],
                                              "nearest_lab": COLORS[nl], "nearest_chroma": COLORS[nc]})
        rgb, seg = shots["side"]
        tm = mask(seg, "Table")
        th, ts = [], []
        for e in range(B):
            mean = torch.tensor(rgb[e][tm[e]].mean(0) / 255.0)
            hsv = VR.rgb_to_hsv(mean)
            sam = VR.rgb_to_hsv(params["table_rgb"][e])
            th.append({"env": e, "measured_hue": round(float(hsv[0]), 3), "sampled_hue": round(float(sam[0]), 3),
                       "measured_value": round(float(hsv[2]), 3), "sampled_value": round(float(sam[2]), 3)})
        cen = np.full((B, 2), np.nan)
        for e in range(B):
            mk = np.zeros(rgb.shape[1:3], bool)
            for j, name in enumerate((*CUBE_NAMES, *PAD_NAMES)):
                if on[e, j]:
                    mk |= mask(seg, name)[e]
            yy, xx = np.nonzero(mk)
            if len(xx):
                cen[e] = (xx.mean(), yy.mean())
        if base_centroid is None:
            base_centroid = cen
        shift = np.linalg.norm(cen - base_centroid, axis=1)
        result["modes"][mode] = {
            "apply_info": info, "light_elev_deg": VR.light_elevation_deg(params["light"]), "colour": stats,
            "table": th, "mean_image_value": round(float(rgb.mean() / 255), 3),
            "side_centroid_shift_px": [round(float(x), 2) for x in shift], "max_shift_px": round(float(np.nanmax(shift)), 2),
            "nominal_env_flags": params["nominal"].tolist()}
        if S4R:
            worst = min(shade["de00_obj_table"])
            result["modes"][mode]["shading"] = {
                "object_factor_min_max": [round(min(shade["object"]), 3), round(max(shade["object"]), 3)],
                "table_factor_min_max": [round(min(shade["table"]), 3), round(max(shade["table"]), 3)],
                "object_over_table_min_max": [round(min(shade["object"]) / max(shade["table"]), 3),
                                              round(max(shade["object"]) / min(shade["table"]), 3)],
                "de00_obj_table_min": {"de00": round(worst[0], 2), "cam": worst[1], "env": worst[2], "object": worst[3]},
                "de00_obj_table_p05": round(float(np.percentile([d[0] for d in shade["de00_obj_table"]], 5)), 2)}
            print(f"[vr] {mode} shading {json.dumps(result['modes'][mode]['shading'])}", flush=True)
        print(f"[vr] {mode}: colour nearest(LAB) {stats['nearest_lab_ok']}/{stats['n_checked']}, chroma {stats['nearest_chroma_ok']}/"
              f"{stats['n_checked']}, skipped {stats['n_skipped_small_mask']}; light elev {result['modes'][mode]['light_elev_deg']:.1f}; "
              f"max camera shift {result['modes'][mode]['max_shift_px']:.1f} px", flush=True)

    rows = [[label(tiles[m][c][e], [f"{m} {c}" if e == 0 else f"{m}"]) for e in range(B)]
            for m in tiles for c in ("side", "wrist")]
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(OUT_PNG, grid(rows, h=args.res))
    if S4R:  # one still per condition: side row over wrist row, every env
        for m in tiles:
            imageio.imwrite(OUT_PNG.parent / f"{m}.png", grid([[tiles[m][c][e] for e in range(B)] for c in ("side", "wrist")],
                                                               h=2 * args.res))
    OUT_JSON.write_text(json.dumps(result, indent=1))
    print(f"[vr] wrote {OUT_PNG} and {OUT_JSON}", flush=True)
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
