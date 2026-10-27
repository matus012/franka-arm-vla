"""Stage 0.2: student cameras. 50 teacher episodes (full phase-6 setting), side + wrist camera at --res, cube
visibility from instance segmentation (visible = >= 20 cube pixels), per phase; mounting checks (wrist camera follows
the hand; each env's camera sees only its own env); a contact sheet per phase for both cameras.

Usage: python scripts/s02_cameras.py --out results/s02_cameras.json [--res 128]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--res", type=int, default=128)
p.add_argument("--num_envs", type=int, default=50)
p.add_argument("--seed", type=int, default=12345)
p.add_argument("--sheet_envs", type=int, default=6)
p.add_argument("--min_px", type=int, default=20)
p.add_argument("--sheet_dir", default=None, help="default results/frames/s02_res<res> (tests pass a temp dir)")
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab.utils.math import matrix_from_quat, quat_mul  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.evaluation import make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import summarize  # noqa: E402
from lift_rl.phases import PHASES  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift import cameras as C  # noqa: E402
from vision_lift.teacher import TEACHER_TASK, load_teacher  # noqa: E402

cfg = make_eval_cfg(load_cfg_from_registry(TEACHER_TASK, "env_cfg_entry_point"), args.num_envs, args.seed)
C.add_student_cameras(cfg, res=args.res)
assert_run_preconditions(cfg)
out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
sheet_dir = Path(args.sheet_dir) if args.sheet_dir else ROOT / "results" / "frames" / f"s02_res{args.res}"
sheet_dir.mkdir(parents=True, exist_ok=True)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    t0 = time.time()
    env = gym.make(TEACHER_TASK, cfg=cfg)
    u = env.unwrapped
    actor = load_teacher(u.device)
    assert_on_cuda(env, actor)
    ids = C.cube_shape_ids(u)
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    labels = model.shape_label
    own = torch.zeros(u.num_envs, model.shape_count + 1, dtype=torch.bool, device=u.device)  # allowed ids per env
    for k, lab in enumerate(labels):
        for i in range(u.num_envs):
            if lab.startswith(f"/World/envs/env_{i}/"):
                own[i, k] = True
        if not lab.startswith("/World/envs/"):
            own[:, k] = True  # global shapes (ground plane) are shared by every world
    r = u.scene["robot"]
    h = r.body_names.index("panda_hand")
    mount_pos = torch.tensor(C.WRIST_POS, device=u.device)
    mount_q = torch.tensor(C.tilt_quat_xyzw(C.WRIST_TILT_DEG), device=u.device)
    SE = args.sheet_envs
    px = {"side_cam": [], "wrist_cam": []}
    ph_log = []
    frames = {"side_cam": [], "wrist_cam": []}
    chk = {"wrist_pos_err_max_m": 0.0, "wrist_rot_err_max_deg": 0.0, "foreign_pixels": 0, "steps": 0}
    st = {"aimed": False}

    def pol(obs):
        if not st["aimed"]:
            C.aim_side(u)
            st["aimed"] = True
        return actor(obs["policy"])

    def on_step(t, uu):
        ph_log.append(uu._phase_machine.phase.clone())
        for name in px:
            px[name].append(C.cube_pixels(uu, name, ids))
            frames[name].append(C.rgb(uu, name)[:SE].to(torch.uint8).clone())
            seg = _t(uu.scene[name].data.output["instance_segmentation_fast"])[..., 0].long()
            seg = seg.clamp(0, model.shape_count)
            allowed = own.gather(1, seg.reshape(uu.num_envs, -1)).reshape(seg.shape) | (seg == 0)
            chk["foreign_pixels"] += int((~allowed).sum())
        # wrist camera pose vs hand pose (x) mount offset
        hp = _t(r.data.body_link_pos_w)[:, h]
        hq = _t(r.data.body_link_quat_w)[:, h]
        exp_pos = hp + (matrix_from_quat(hq) @ mount_pos[:, None])[..., 0]
        cd = uu.scene["wrist_cam"].data
        pos_err = torch.linalg.norm(_t(cd.pos_w) - exp_pos, dim=-1).max()
        exp_q = quat_mul(hq, mount_q.expand_as(hq))
        dot = (_t(cd.quat_w_ros) * exp_q).sum(-1).abs().clamp(max=1.0)
        rot_err = torch.rad2deg(2 * torch.arccos(dot)).max()
        chk["wrist_pos_err_max_m"] = max(chk["wrist_pos_err_max_m"], float(pos_err))
        chk["wrist_rot_err_max_deg"] = max(chk["wrist_rot_err_max_deg"], float(rot_err))
        chk["steps"] += 1

    res = rollout(env, pol, None, on_step=on_step)
    ph = torch.stack(ph_log)  # (T, N)
    vis = {}
    for name in px:
        pp = torch.stack(px[name])
        v = pp >= args.min_px
        vis[name] = {"all_frames": float(v.float().mean()), "median_px": float(pp.float().median()),
                     "p10_px": float(torch.quantile(pp.float(), 0.1)),
                     "approach_grasp_lift": float(v[ph <= 3].float().mean()),
                     "invisible_frames_with_zero_px_frac": float(((pp == 0) & ~v).sum() / max(1, int((~v).sum()))),
                     "per_phase": {PHASES[j]: (float(v[ph == j].float().mean()) if bool((ph == j).any()) else None)
                                   for j in range(6)}}
    passed = vis["side_cam"]["all_frames"] >= 0.90 and vis["wrist_cam"]["approach_grasp_lift"] >= 0.80
    mount_ok = chk["wrist_pos_err_max_m"] < 1e-3 and chk["wrist_rot_err_max_deg"] < 0.5 and chk["foreign_pixels"] == 0
    # contact sheets: one per phase, rows = envs (first SE), each cell = side | wrist at the phase's middle step
    sheets = []
    for j, name in enumerate(PHASES):
        rows = []
        for i in range(SE):
            steps = (ph[:, i] == j).nonzero()[:, 0]
            if len(steps) == 0:
                continue
            s = int(steps[len(steps) // 2])
            cells = [frames[c][s][i].cpu().numpy() for c in ("side_cam", "wrist_cam")]
            rows.append(np.concatenate([cells[0], np.full((args.res, 4, 3), 255, np.uint8), cells[1]], 1))
        if rows:
            sheet = np.concatenate([np.concatenate([rw, np.full((4, rw.shape[1], 3), 255, np.uint8)], 0) for rw in rows], 0)
            sheet = np.kron(sheet, np.ones((2, 2, 1), np.uint8))
            f = sheet_dir / f"phase{j + 1}_{name}_side_wrist.png"
            imageio.imwrite(f, sheet)
            sheets.append(str(f))
    summary = summarize(res)
    out_d = {"stage": "0.2", "res": args.res, "seed": args.seed, "episodes": args.num_envs, "min_px": args.min_px,
             "side_eye": C.SIDE_EYE, "side_look": C.SIDE_LOOK, "side_focal_mm": C.SIDE_FOCAL,
             "wrist_mount_pos": C.WRIST_POS, "wrist_tilt_deg": C.WRIST_TILT_DEG, "wrist_focal_mm": C.WRIST_FOCAL,
             "visibility": vis, "gate_pass": passed, "mounting_checks": {**chk, "pass": mount_ok},
             "teacher_success": summary["success"]["rate"], "sheets": sheets, "wall_s": round(time.time() - t0, 1),
             "nan_guard_events": u.nan_guard_events}
    print(json.dumps(out_d, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_d, indent=1))
    env.close()
