"""Stage 0A.4 (GATES.md): camera image size. The 0A.2 expert (unchunked) drives the task episodes; side + wrist cameras
(112 mounts, side aimed explicitly) render rgb + instance segmentation at --res; frames are read at the PERIOD ticks.

Counted per run (one task, one size; per-episode-pooled counters are saved so tasks can be pooled in the summary):
  (a) side: for every cube on the table, frames before the first grasp confirmation in which it has >= 20 px;
  (b) wrist: target >= 20 px in frames whose expert phase is HOVER / DESCEND / CLOSE;
  (c) side: every pad >= 20 px in frames before the first claw opening after the grasp (pick: whole episode);
  reported: wrist sees the destination (pad / bottom cube) during CARRY / LOWER / RELEASE; median pixel counts;
  mounting: wrist pose error, foreign pixels, parked objects 0 px. Contact sheet (first 4 envs: side | wrist at hover,
  close, carry, release / end).
  -> results/0a4_cam_<task>_res<res>_s<seed>_<start>.json, results/frames/0a4/<task>/res<res>.png
summary mode pools the three tasks per size, applies the gate and the smart length rule, picks the smallest passing
size -> results/0a4_cameras_res<res>.json, results/0a4_cameras.json.

Usage: python scripts/s0a4_cameras.py --task pick --res 128,224,256 --seed 11400 --n 300 (one camera pair per size, same run)
       python scripts/s0a4_cameras.py --summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["pick", "place", "stack"])
p.add_argument("--res", default="128,224,256", help="comma list; all sizes rendered in the same run (same trajectories)")
p.add_argument("--seed", type=int)
p.add_argument("--n", type=int, default=300)
p.add_argument("--start", type=int, default=0)
p.add_argument("--num_envs", type=int, default=150)
p.add_argument("--summary", action="store_true")
p.add_argument("--min_px", type=int, default=20)
args = p.parse_args()

import numpy as np  # noqa: E402

RES = ROOT / "results"
SEEDS = {"pick": 11400, "place": 11410, "stack": 11420}


def sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


if args.summary:
    out_all = {"git_sha": sha(), "sizes": {}}
    for res in (128, 224, 256):
        runs = {t: [json.loads(f.read_text()) for f in sorted(RES.glob(f"0a4_cam_{t}_res{res}_s{SEEDS[t]}_*.json"))]
                for t in SEEDS}
        if not all(runs.values()):
            continue

        def pool(key: str, tasks=tuple(SEEDS), first: int | None = None):
            num = den = 0
            per = None
            for t in tasks:
                for r in runs[t]:
                    a = np.asarray(r["per_episode"][key])  # (E, 2[, K]) visible, total
                    idx = np.asarray(r["per_episode"]["index"])
                    m = idx < first if first else np.ones_like(idx, bool)
                    s = a[m].sum(0)
                    per = s if per is None else per + s
            return per

        a = pool("side_cube")  # (2, 8)
        a300 = pool("side_cube", first=300)
        b, b300 = pool("wrist_target"), pool("wrist_target", first=300)
        c, c300 = pool("side_pad", ("place",)), pool("side_pad", ("place",), first=300)
        c_all = pool("side_pad")
        dst = pool("wrist_dest", ("place", "stack"))
        fa = a[0] / np.maximum(a[1], 1)
        fc = c[0] / np.maximum(c[1], 1)
        fb = float(b[0] / max(b[1], 1))
        bars = {"a_min_cube_side": (float(fa.min()), 0.90), "b_wrist_target": (fb, 0.80), "c_min_pad_side": (float(fc.min()), 0.90)}
        f300 = {"a_min_cube_side": float((a300[0] / np.maximum(a300[1], 1)).min()), "b_wrist_target": float(b300[0] / max(b300[1], 1)),
                "c_min_pad_side": float((c300[0] / np.maximum(c300[1], 1)).min())}
        n_eps = min(len(sum((r["per_episode"]["index"] for r in runs[t]), [])) for t in SEEDS)
        near = any(abs(100 * (f300[k] - bar)) <= 5 for k, (_, bar) in bars.items())
        decision = "extend to 1000" if near and n_eps < 1000 else ("decided at 1000" if near else "decided at 300")
        use = bars if decision != "decided at 300" else {k: (f300[k], bars[k][1]) for k in bars}
        passed = all(v >= bar for v, bar in use.values()) if decision != "extend to 1000" else None
        mount = [r["mounting"] for t in SEEDS for r in runs[t]]
        mount_ok = all(m["wrist_pos_err_max_m"] < 1e-3 and m["wrist_rot_err_max_deg"] < 0.5 and m["foreign_pixels"] == 0
                       and m["parked_pixels"] == 0 for m in mount)
        o = {"res": res, "episodes_per_task": n_eps, "smart_length": decision, "gate_values": {k: v for k, (v, _) in use.items()},
             "bars": {k: bar for k, (_, bar) in bars.items()}, "first300": f300,
             "a_per_cube_side": fa.tolist(), "c_per_pad_side_place": fc.tolist(),
             "c_per_pad_side_all_tasks": (c_all[0] / np.maximum(c_all[1], 1)).tolist(),
             "wrist_dest_carry_release": float(dst[0] / max(dst[1], 1)),
             "per_task": {t: {k: [r[k] for r in runs[t]] for k in ("median_px", "wall_s", "sheet")} for t in SEEDS},
             "mounting_ok": mount_ok, "mounting": mount, "pass": passed if passed is None else bool(passed and mount_ok)}
        (RES / f"0a4_cameras_res{res}.json").write_text(json.dumps(o, indent=1))
        out_all["sizes"][res] = {"pass": o["pass"], "smart_length": decision, **o["gate_values"]}
    passing = [r for r, v in out_all["sizes"].items() if v["pass"]]
    out_all["chosen_size"] = min(passing) if passing else None
    (RES / "0a4_cameras.json").write_text(json.dumps(out_all, indent=1))
    print(json.dumps(out_all, indent=1))
    sys.exit(0)

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_mul  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift import cameras as C  # noqa: E402
from vla_common import timing as TM  # noqa: E402
from vla_common.catalog import ALL_CUBES, COLORS  # noqa: E402
from vla_common.tasks import sample_spec  # noqa: E402
from vla_tabletop.expert import CARRY, CLOSE, DESCEND, HOVER, LOWER, RELEASE  # noqa: E402
from vla_tabletop.layout import OBJ_NAMES  # noqa: E402
from vla_tabletop.rollout import Runner, geometry, make_stream  # noqa: E402
from vla_tabletop.sheets import grid, label  # noqa: E402

STAGE = {"pick": "s2", "place": "s3a", "stack": "s3b"}[args.task]
RESL = [int(x) for x in str(args.res).split(",")]
meas = json.loads((RES / "0a1_measure.json").read_text())
tj = json.loads((RES / f"0a2_{args.task}_timing.json").read_text())
T = tj["T_steps"]
geo = geometry(meas["z_rest_m"], h_pad=float(np.mean(list(meas["h_pad_m"].values()))))
grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES])
N = min(args.num_envs, args.n)
cfg = S.make_cfg(N, args.seed)
DT = ("rgb", "instance_segmentation_fast")
for rr in RESL:  # one camera pair per size in the same env: identical trajectories across sizes by construction
    setattr(cfg.scene, f"side_cam_{rr}", C.side_cam_cfg(rr, data_types=DT, name=f"SideCam{rr}"))
    setattr(cfg.scene, f"wrist_cam_{rr}", C.wrist_cam_cfg(rr, data_types=DT, name=f"WristCam{rr}"))
assert_run_preconditions(cfg)
FR = RES / "frames" / "0a4" / args.task
FR.mkdir(parents=True, exist_ok=True)

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    t0 = time.time()
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    print(f"[0a4] device={u.device} envs={N} task={args.task} res={RESL}", flush=True)
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    ids = torch.stack([S.shape_ids(u, n)[:, 0] for n in OBJ_NAMES], 1)  # (N, 12) visible shape id per object
    own = torch.zeros(N, model.shape_count + 1, dtype=torch.bool, device=u.device)
    for k, lab in enumerate(model.shape_label):
        for i in range(N):
            if lab.startswith(f"/World/envs/env_{i}/"):
                own[i, k] = True
        if not lab.startswith("/World/envs/"):
            own[:, k] = True
    r = u.scene["robot"]
    h = r.body_names.index("panda_hand")
    mount_pos = torch.tensor(C.WRIST_POS, device=u.device)
    mount_q = torch.tensor(C.tilt_quat_xyzw(C.WRIST_TILT_DEG), device=u.device)
    mnt = {rr: {"wrist_pos_err_max_m": 0.0, "wrist_rot_err_max_deg": 0.0, "foreign_pixels": 0, "parked_pixels": 0}
           for rr in RESL}
    ar = torch.arange(N, device=u.device)
    acc: dict = {rr: {} for rr in RESL}
    shared: dict = {}
    sheet = {rr: {i: [] for i in range(min(4, N))} for rr in RESL}

    def counts(name: str, rr: int) -> torch.Tensor:
        seg = S._t(u.scene[name].data.output["instance_segmentation_fast"])[..., 0].long()
        segc = seg.clamp(0, model.shape_count)
        allowed = own.gather(1, segc.reshape(N, -1)).reshape(seg.shape) | (seg <= 0)
        mnt[rr]["foreign_pixels"] += int((~allowed).sum())
        flat = seg.reshape(N, -1)
        return torch.stack([(flat == ids[:, j:j + 1]).sum(1) for j in range(len(OBJ_NAMES))], 1)  # (N, 12)

    def on_step(t, uu, s, lab, exe, ctrl, det):
        if t == 0:
            for rr in RESL:
                C.aim_side(uu, name=f"side_cam_{rr}")
                acc[rr].update(side_cube=torch.zeros(N, 2, 8, device=uu.device),
                               side_pad=torch.zeros(N, 2, 4, device=uu.device),
                               wrist_target=torch.zeros(N, 2, device=uu.device),
                               wrist_dest=torch.zeros(N, 2, device=uu.device), px_side=[], px_wrist=[])
            shared.update(released=torch.zeros(N, dtype=torch.bool, device=uu.device), prev_phase=None)
        shared["released"] |= det.confirmed & (exe[:, 4] < 0.5) & (ctrl.task != 0)
        if t % TM.PERIOD:
            return
        on = uu._layout[:, :, 4] > 0.5
        ph = ctrl.phase
        pp = shared["prev_phase"]
        hp, hq = S._t(r.data.body_link_pos_w)[:, h], S._t(r.data.body_link_quat_w)[:, h]
        appr = (ph == HOVER) | (ph == DESCEND) | (ph == CLOSE)
        dj = torch.where(ctrl.task == 1, 8 + ctrl.dest.clamp(0, 3), ctrl.dest.clamp(0, 7))
        carry = ((ph == CARRY) | (ph == LOWER) | (ph == RELEASE)) & (ctrl.task != 0)
        for rr in RESL:
            A, M = acc[rr], mnt[rr]
            sn, wn = f"side_cam_{rr}", f"wrist_cam_{rr}"
            cs, cw = counts(sn, rr), counts(wn, rr)
            M["parked_pixels"] += int((cs[~on]).sum() + (cw[~on]).sum())
            vis_s, vis_w = cs >= args.min_px, cw >= args.min_px
            m = on[:, :8] & ~det.confirmed[:, None]
            A["side_cube"][:, 0] += (vis_s[:, :8] & m).float()
            A["side_cube"][:, 1] += m.float()
            mp = on[:, 8:] & ~shared["released"][:, None]
            A["side_pad"][:, 0] += (vis_s[:, 8:] & mp).float()
            A["side_pad"][:, 1] += mp.float()
            A["wrist_target"][:, 0] += (vis_w[ar, ctrl.target] & appr).float()
            A["wrist_target"][:, 1] += appr.float()
            A["wrist_dest"][:, 0] += (vis_w[ar, dj] & carry).float()
            A["wrist_dest"][:, 1] += carry.float()
            A["px_side"].append(cs[:, :8][on[:, :8]].float().cpu())
            A["px_wrist"].append(cw[ar, ctrl.target][appr].float().cpu())
            cd = uu.scene[wn].data
            M["wrist_pos_err_max_m"] = max(M["wrist_pos_err_max_m"], float(torch.linalg.norm(
                S._t(cd.pos_w) - (hp + (matrix_from_quat(hq) @ mount_pos[:, None])[..., 0]), dim=-1).max()))
            dot = (S._t(cd.quat_w_ros) * quat_mul(hq, mount_q.expand_as(hq))).sum(-1).abs().clamp(max=1.0)
            M["wrist_rot_err_max_deg"] = max(M["wrist_rot_err_max_deg"],
                                             float(torch.rad2deg(2 * torch.arccos(dot)).max()))
            snap_rows = [i for i in sheet[rr] if t == 0 or t >= T - TM.PERIOD or
                         (pp is not None and int(ph[i]) != int(pp[i]) and int(ph[i]) in (CLOSE, CARRY, RELEASE))]
            if snap_rows:
                side = S._t(uu.scene[sn].data.output["rgb"])[..., :3].to(torch.uint8).cpu().numpy()
                wrist = S._t(uu.scene[wn].data.output["rgb"])[..., :3].to(torch.uint8).cpu().numpy()
                for i in snap_rows:
                    sheet[rr][i].append(np.concatenate([label(side[i], [f"t={t} ph={int(ph[i])} {rr}px"]), wrist[i]], 1))
        shared["prev_phase"] = ph.clone()

    eps = make_stream(args.seed, args.start, args.n, lambda rng: sample_spec(STAGE, "train", rng))
    run = Runner(env, geo, grasp_dz=grasp_dz.to(u.device))
    per = {rr: {"index": [], "side_cube": [], "side_pad": [], "wrist_target": [], "wrist_dest": []} for rr in RESL}
    pxs, pxw = {rr: [] for rr in RESL}, {rr: [] for rr in RESL}
    finals, succ = [], []
    for b0 in range(0, len(eps), N):
        batch = eps[b0:b0 + N]
        padded = batch + [batch[-1]] * (N - len(batch))
        o = run.run_batch(padded, T, {"fresh": True}, on_step=on_step)
        k = len(batch)
        for rr in RESL:
            per[rr]["index"] += [e.index for e in batch]
            for key in ("side_cube", "side_pad", "wrist_target", "wrist_dest"):
                per[rr][key] += acc[rr][key][:k].cpu().tolist()
            pxs[rr].append(torch.cat(acc[rr]["px_side"]))
            pxw[rr].append(torch.cat(acc[rr]["px_wrist"]))
        finals += o["final_cube_pos"][:k].cpu().tolist()
        succ += o["success"][:k].cpu().tolist()
    for rr in RESL:
        shot = FR / f"res{rr}.png"
        imageio.imwrite(shot, grid([sheet[rr][i][:6] for i in sheet[rr]], h=max(160, rr)))
        ps, pw = torch.cat(pxs[rr]), torch.cat(pxw[rr])
        out = {"stage": "0A.4", "task": args.task, "res": rr, "seed": args.seed, "start": args.start, "n": args.n,
               "T": T, "num_envs": N, "git_sha": sha(), "min_px": args.min_px, "period": TM.PERIOD,
               "side_eye": C.SIDE_EYE, "side_look": C.SIDE_LOOK, "wrist_tilt_deg": C.WRIST_TILT_DEG,
               "sizes_in_same_run": RESL, "expert_success": float(np.mean(succ)),
               "median_px": {"side_cube_on_table": float(ps.median()), "wrist_target_approach": float(pw.median())},
               "mounting": mnt[rr], "sheet": str(shot.relative_to(ROOT)), "wall_s": round(time.time() - t0, 1),
               "per_episode": per[rr], "final_cube_pos": finals}
        (RES / f"0a4_cam_{args.task}_res{rr}_s{args.seed}_{args.start}.json").write_text(json.dumps(out))
        print(json.dumps({k: v for k, v in out.items() if k not in ("per_episode", "final_cube_pos")}, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
