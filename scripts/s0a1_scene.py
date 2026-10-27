"""Stage 0A.1 (GATES.md): scene load + measurements + the per-size scripted grasp.

Parts (one simulator per process):
  measure : z_rest per size (cube dropped 1 cm onto the table, settled 1 s, 100 envs, median), table-top agreement,
            h_pad (cube resting on a pad), masses / friction / edges read back from the Newton model, the
            large-on-small and small-on-large teleport stack settle test (100 each, still stacked after 2 s), and the
            placement-sampler rejection rate (10,000 specs per stage; pure python)  -> results/0a1_measure.json
  grasp   : the gate trials for one size: one cube of that size (colour from the 4 colours, x 0.40-0.60, y +-0.25,
            yaw +-pi = the 111 reach box), pads parked, the 111 scripted controller with per-size geometry through the
            student IK + claw (approach, grasp, lift 15 cm, hold), 400 steps; episodes [start, start+n) of the seed's
            stream -> results/0a1_grasp_<size>_s<seed>_<start>.json (per-episode outcomes)
  cams    : load checks with cameras at 128 + instance segmentation (validation seed): distinct segmentation ids,
            parked objects 0 px and outside the reach box and still (< 1 mm), wrist mount error, foreign pixels;
            contact sheets (side | wrist | close-up per size at grasp and at hold; an env with all 8 cubes + 4 pads)
            -> results/0a1_cams.json, results/frames/0a1/*.png
  summary : merges the part JSONs into results/0a1_scene.json (smart length per size, GATES conventions).

Usage: python scripts/s0a1_scene.py --part grasp --size small --seed 11100 --n 300 --start 0
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--part", required=True, choices=["measure", "grasp", "cams", "summary"])
p.add_argument("--size", choices=["small", "large"])
p.add_argument("--seed", type=int, default=21100)
p.add_argument("--n", type=int, default=300)
p.add_argument("--start", type=int, default=0)
p.add_argument("--num_envs", type=int, default=300)
p.add_argument("--tag", default="")
args = p.parse_args()

import numpy as np  # noqa: E402

from vla_common.catalog import ALL_CUBES, COLORS, EDGE_M, PAD_EDGE_M, PAD_THICKNESS_M, cube_name  # noqa: E402
from vla_common.tasks import TaskSpec, sample_spec  # noqa: E402

RES = ROOT / "results"
FR = RES / "frames" / "0a1"
T_GRASP = 400


def sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def grasp_spec(size: str):
    def f(rng: np.random.Generator):
        c = (COLORS[int(rng.integers(4))], size)
        lay = {cube_name(c): (float(rng.uniform(0.40, 0.60)), float(rng.uniform(-0.25, 0.25)),
                              float(rng.uniform(-math.pi, math.pi)))}
        return TaskSpec("pick", c, None, [c], []), lay, {}
    return f


def wilson(k: int, n: int) -> list[float]:
    from lift_rl.metrics import wilson_ci

    return list(wilson_ci(k, n))


def rate(x) -> dict:
    x = np.asarray(x, bool)
    return {"count": int(x.sum()), "n": int(x.size), "rate": float(x.mean()) if x.size else float("nan"),
            "ci95": wilson(int(x.sum()), int(x.size))}


# --------------------------------------------------------------------------- pure-python part
if args.part == "summary":
    out = {"stage": "0A.1", "git_sha": sha()}
    m = json.loads((RES / "0a1_measure.json").read_text())
    out["measure"] = m
    if (RES / "0a1_cams.json").exists():
        out["cams"] = json.loads((RES / "0a1_cams.json").read_text())
    for size, seed in (("small", 11100), ("large", 11110)):
        files = sorted(RES.glob(f"0a1_grasp_{size}_s*.json"))
        runs = [json.loads(f.read_text()) for f in files]
        runs = [r for r in runs if r["seed"] in (seed + k for k in range(10))]
        if not runs:
            continue
        sd = max(r["seed"] for r in runs)  # the latest attempt
        rr = sorted([r for r in runs if r["seed"] == sd], key=lambda r: r["start"])
        ep = {k: sum((r["episodes"][k] for r in rr), []) for k in rr[0]["episodes"]}
        idx = ep["index"]
        assert idx == list(range(len(idx))), "episode stream must be contiguous from 0"
        n = len(idx)
        hold = rate(ep["pick_hold_last50"])
        h300 = rate(ep["pick_hold_last50"][:300])
        d = 100 * (h300["rate"] - 0.95)
        decision = "FAIL at 300" if d < -5 else ("needs 1000" if n < 1000 else "decided at 1000")
        passed = hold["rate"] >= 0.95 if (n >= 1000 or d < -5) else None
        if d < -5:
            passed = False
        out[size] = {"seed": sd, "episodes": n, "holds_last50": hold, "holds_last50_first300": h300,
                     "smart_length": decision, "gate_pass": passed,
                     **{k: rate(ep[k]) for k in ("success", "slam", "jam", "drop", "clean", "clean_c1_no_table_hit",
                                                 "clean_c2_top_down", "clean_c3_both_pads_at_liftoff", "clean_c4_no_palm",
                                                 "early_close", "invalid", "grasp_confirmed")},
                     "pad_force_median_holding_N": float(np.nanmedian(ep["pad_force_hold_median"])),
                     "confirm_tcp_minus_cube_z_median_m": float(np.nanmedian(ep["confirm_tcp_minus_cube_z"])),
                     "wall_s": sum(r["wall_s"] for r in rr), "files": [str(f.name) for f in files]}
    (RES / "0a1_scene.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k in ("small", "large")}, indent=1))
    sys.exit(0)

# --------------------------------------------------------------------------- simulator parts
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vla_tabletop.detectors import Predicates, Consts113, DETECTOR_VERSION  # noqa: E402
from vla_tabletop.expert import EXPERT_VERSION, CLOSE, LIFT  # noqa: E402
from vla_tabletop.layout import OBJ_NAMES, REACH_BOX_HI, REACH_BOX_LO, TABLE_Z, sample_layout  # noqa: E402
from vla_tabletop.rollout import Episode, Runner, geometry, make_stream, read_state, to_step_state  # noqa: E402
from lift_rl.metrics import Thresholds  # noqa: E402

CAMS = args.part == "cams"
N = args.num_envs
cfg = S.make_cfg(N, args.seed, res=128 if CAMS else None, seg=CAMS)
if CAMS:
    from vla_tabletop.sheets import add_closeup

    add_closeup(cfg)
assert_run_preconditions(cfg)


def zero_action(u) -> torch.Tensor:
    return torch.zeros(u.num_envs, 5, device=u.device)


with launch_simulation(cfg, {"headless": True, "enable_cameras": CAMS}):
    t0 = time.time()
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    print(f"[0a1] device={u.device} envs={u.num_envs} part={args.part}", flush=True)
    meas_path = RES / "0a1_measure.json"
    geo = geometry(json.loads(meas_path.read_text())["z_rest_m"] if meas_path.exists() and args.part != "measure" else None)

    if args.part == "measure":
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        rng = np.random.default_rng(args.seed)
        # batch A: drop test. Per env: red small + blue large on the table, green small on pad red, yellow large on
        # pad blue, all dropped from 1 cm above their resting height
        lays = []
        for i in range(N):
            j = rng.uniform(-0.02, 0.02, size=2)
            lays.append({"cube_red_small": (0.42 + j[0], -0.22, rng.uniform(-3, 3)),
                         "cube_blue_large": (0.58, -0.22 + j[1], rng.uniform(-3, 3)),
                         "pad_red": (0.42, 0.05, 0.0), "pad_blue": (0.58, 0.20, 0.0),
                         "cube_green_small": (0.42 + j[1], 0.05 + j[0], rng.uniform(-3, 3)),
                         "cube_yellow_large": (0.58 - j[0], 0.20 - j[1], rng.uniform(-3, 3))})
        from vla_tabletop.layout import layout_array

        lay = torch.as_tensor(layout_array(lays, drop=0.01), device=u.device)
        on_pad = [OBJ_NAMES.index("cube_green_small"), OBJ_NAMES.index("cube_yellow_large")]
        lay[:, on_pad, 2] += PAD_THICKNESS_M
        u._layout = lay
        env.reset()
        for _ in range(50):
            env.step(zero_action(u))
        s = read_state(u)
        z = s["cube_pos"][..., 2].cpu().numpy()
        ci = {n: OBJ_NAMES.index(n) for n in ("cube_red_small", "cube_blue_large", "cube_green_small", "cube_yellow_large")}
        z_rest = {"small": float(np.median(z[:, ci["cube_red_small"]])), "large": float(np.median(z[:, ci["cube_blue_large"]]))}
        z_tab = {k: z_rest[k] - EDGE_M[k] / 2 for k in z_rest}
        h_pad = {"small": float(np.median(z[:, ci["cube_green_small"]])) - z_rest["small"],
                 "large": float(np.median(z[:, ci["cube_yellow_large"]])) - z_rest["large"]}
        vmax = float(torch.linalg.norm(s["cube_vel"], dim=-1).amax())
        # model read-back: edges (box half extents), masses, friction
        labels = model.shape_label
        scale, mu, flags = model.shape_scale.numpy(), model.shape_material_mu.numpy(), model.shape_flags.numpy()
        edges, mus = {}, {}
        for c in ALL_CUBES:
            k = [k for k, lab in enumerate(labels) if lab.startswith(f"/World/envs/env_0/{cube_name(c)}/") and flags[k] & 2][0]
            edges[cube_name(c)] = [2 * float(x) for x in scale[k]]
            mus[cube_name(c)] = float(mu[k])
        pk = [k for k, lab in enumerate(labels) if lab.startswith("/World/envs/env_0/pad_red/") and flags[k] & 2][0]
        pad_dims = [2 * float(x) for x in scale[pk]]
        masses = {n: float(S._t(u.scene[n].data.body_mass).reshape(N, -1)[0].sum()) for n in OBJ_NAMES}
        from vla_common.catalog import DENSITY

        dens_err = {n: masses[n] / (DENSITY * EDGE_M[n.rsplit("_", 1)[1]] ** 3) - 1 for n in OBJ_NAMES if n.startswith("cube_")}
        # batch B: teleport stack settle test, per env: red large on blue small, green small on yellow large
        lays = []
        for i in range(N):
            lays.append({"cube_blue_small": (0.42, -0.15, rng.uniform(-3, 3)), "cube_red_large": (0.42, -0.15, rng.uniform(-3, 3)),
                         "cube_yellow_large": (0.58, 0.15, rng.uniform(-3, 3)), "cube_green_small": (0.58, 0.15, rng.uniform(-3, 3))})
        lay = torch.as_tensor(layout_array(lays), device=u.device)
        rl, bs = OBJ_NAMES.index("cube_red_large"), OBJ_NAMES.index("cube_blue_small")
        gs, yl = OBJ_NAMES.index("cube_green_small"), OBJ_NAMES.index("cube_yellow_large")
        lay[:, rl, 2] = z_rest["small"] + EDGE_M["small"] / 2 + EDGE_M["large"] / 2 + 0.001
        lay[:, gs, 2] = z_rest["large"] + EDGE_M["large"] / 2 + EDGE_M["small"] / 2 + 0.001
        u._layout = lay
        env.reset()
        for _ in range(100):
            env.step(zero_action(u))
        geo = geometry(z_rest, h_pad=float(np.mean(list(h_pad.values()))))
        P = Predicates(to_step_state(read_state(u), torch.zeros(N, device=u.device)), geo, Thresholds(), Consts113())
        roc = P.rests_on_cube.cpu().numpy()
        stack = {"large_on_small": float(roc[:, rl, bs].mean()), "small_on_large": float(roc[:, gs, yl].mean()), "n": N}
        # placement sampler rejection rate (pure python), 10,000 specs per stage
        samp = {}
        for stage in ("s1", "s2", "s3a", "s3b"):
            r2 = np.random.default_rng([args.seed, 7])
            d = o = rs = 0
            inbox = True
            for _ in range(10000):
                spec = sample_spec(stage, "train", r2)
                layd, inf = sample_layout(spec, r2)
                d, o, rs = d + inf["draws"], o + inf["objects"], rs + inf["restarts"]
                for (x, y, _) in layd.values():
                    inbox &= REACH_BOX_LO[0] <= x <= REACH_BOX_HI[0] and REACH_BOX_LO[1] <= y <= REACH_BOX_HI[1]
            samp[stage] = {"specs": 10000, "per_object_rejection_rate": 1 - o / d, "layout_restarts": rs,
                           "all_in_reach_box": bool(inbox)}
        e_s, e_l = EDGE_M["small"], EDGE_M["large"]
        out = {"stage": "0A.1 measure", "seed": args.seed, "num_envs": N, "git_sha": sha(),
               "edge_m_catalog": EDGE_M, "edges_m_model": edges, "small_edge_from_111_collision_box_m": 0.048,
               "large_rule": {"edge_le_0.060": e_l <= 0.060 + 1e-9,
                              "edge_cos10_plus_sin10_m": e_l * (math.cos(math.radians(10)) + math.sin(math.radians(10))),
                              "le_0.070": e_l * (math.cos(math.radians(10)) + math.sin(math.radians(10))) <= 0.070},
               "masses_kg": masses, "mass_vs_density_rel_err": dens_err, "mass_ok_1pct": all(abs(v) < 0.01 for v in dens_err.values()),
               "friction_mu": mus, "z_rest_m": z_rest, "z_tab_m": z_tab, "z_tab_agree_m": abs(z_tab["small"] - z_tab["large"]),
               "z_tab_agree_ok_1mm": abs(z_tab["small"] - z_tab["large"]) <= 0.001, "h_pad_m": h_pad,
               "pad_dims_model_m": pad_dims, "l_pad_m": PAD_EDGE_M, "l_pad_ge_large_plus_2cm": PAD_EDGE_M >= e_l + 0.02 - 1e-9,
               "max_cube_speed_after_settle_mps": vmax, "stack_settle_2s": stack, "placement_sampler": samp,
               "wall_s": round(time.time() - t0, 1)}
        print(json.dumps(out, indent=1))
        meas_path.write_text(json.dumps(out, indent=1))

    elif args.part == "grasp":
        eps = make_stream(args.seed, args.start, args.n, grasp_spec(args.size))
        run = Runner(env, geo)
        holdf = []

        def on_step(t, uu, s, label, exe, ctrl, det):
            if t >= T_GRASP - 50:
                tg = ctrl.target
                ar = torch.arange(uu.num_envs, device=uu.device)
                f = torch.stack([s["f_l"][ar, tg], s["f_r"][ar, tg]], 1)
                holdf.append(f)

        res = run.run(eps, T_GRASP, on_step=on_step)
        hf = torch.stack(holdf, 0)  # (50, B_total?, 2) of the last batch only when several batches
        keys = ("index", "pick_hold_last50", "success", "slam", "jam", "drop", "clean", "clean_c1_no_table_hit",
                "clean_c2_top_down", "clean_c3_both_pads_at_liftoff", "clean_c4_no_palm", "early_close", "invalid",
                "grasp_confirmed", "confirm_tcp_minus_cube_z", "completion_step")
        epd = {k: res[k].tolist() for k in keys}
        # median pad force over the last 50 steps on steps where both pads touch, per episode (batches concatenated)
        nb = (len(eps) + N - 1) // N
        hf = hf.reshape(nb, 50, N, 2).permute(0, 2, 1, 3).reshape(nb * N, 50, 2)[: len(eps)].cpu()
        med = []
        for e in hf:
            m = (e > 1.0).all(-1)
            med.append(float(e[m].median()) if bool(m.any()) else float("nan"))
        epd["pad_force_hold_median"] = med
        out = {"stage": "0A.1 grasp", "size": args.size, "seed": args.seed, "start": args.start, "n": args.n,
               "T": T_GRASP, "num_envs": N, "git_sha": sha(), "expert_version": EXPERT_VERSION,
               "detector_version": DETECTOR_VERSION, "thresholds": Thresholds().to_dict(), "consts113": Consts113().to_dict(),
               "geometry": {"edge": geo.edge.tolist(), "z_rest": geo.z_rest.tolist(), "h_pad": geo.h_pad, "l_pad": geo.l_pad},
               "holds_last50_rate": float(np.mean(epd["pick_hold_last50"])), "nan_guard_events": u.nan_guard_events,
               "wall_s": round(time.time() - t0, 1), "episodes": epd}
        f = RES / f"0a1_grasp_{args.size}_s{args.seed}_{args.start}{args.tag}.json"
        f.write_text(json.dumps(out))
        print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=1))

    elif args.part == "cams":
        import imageio

        from isaaclab.utils.math import matrix_from_quat, quat_mul
        from isaaclab_newton.physics import NewtonManager
        from vision_lift import cameras as C
        from vla_tabletop.sheets import CLOSEUP, aim_closeup, grab, grid, label

        FR.mkdir(parents=True, exist_ok=True)
        model = NewtonManager.get_model()
        labels = model.shape_label
        obj_ids = {n: S.shape_ids(u, n) for n in OBJ_NAMES}  # (N, k) visible shapes
        distinct = all(len(set(int(obj_ids[n][i, 0]) for n in OBJ_NAMES)) == len(OBJ_NAMES) for i in range(N))
        own = torch.zeros(N, model.shape_count + 1, dtype=torch.bool, device=u.device)
        for k, lab in enumerate(labels):
            for i in range(N):
                if lab.startswith(f"/World/envs/env_{i}/"):
                    own[i, k] = True
            if not lab.startswith("/World/envs/"):
                own[:, k] = True
        r = u.scene["robot"]
        h = r.body_names.index("panda_hand")
        mount_pos = torch.tensor(C.WRIST_POS, device=u.device)
        mount_q = torch.tensor(C.tilt_quat_xyzw(C.WRIST_TILT_DEG), device=u.device)
        chk = {"wrist_pos_err_max_m": 0.0, "wrist_rot_err_max_deg": 0.0, "foreign_pixels": 0, "parked_pixels": 0,
               "parked_max_move_m": 0.0, "parked_outside_reach_box": True, "frames": 0}
        lo, hi = torch.tensor(REACH_BOX_LO, device=u.device), torch.tensor(REACH_BOX_HI, device=u.device)
        st: dict = {}

        def checks(uu, s, lay):
            parked = lay[:, :, 4] < 0.5  # (N, 12)
            for name in ("side_cam", "wrist_cam"):
                seg = S._t(uu.scene[name].data.output["instance_segmentation_fast"])[..., 0].long()
                seg = seg.clamp(-1, model.shape_count)
                segc = seg.clamp(0, model.shape_count)
                allowed = own.gather(1, segc.reshape(N, -1)).reshape(seg.shape) | (seg <= 0)
                chk["foreign_pixels"] += int((~allowed).sum())
                for j, n in enumerate(OBJ_NAMES):
                    px = (seg[..., None] == obj_ids[n][:, None, None, :]).any(-1).sum((1, 2))
                    chk["parked_pixels"] += int(px[parked[:, j]].sum())
            pos = torch.cat([s["cube_pos"], s["pad_pos"]], 1)
            if "p0" not in st:
                st["p0"] = pos.clone()
            mv = torch.linalg.norm(pos - st["p0"], dim=-1)[parked]
            chk["parked_max_move_m"] = max(chk["parked_max_move_m"], float(mv.max()) if mv.numel() else 0.0)
            inbox = ((pos >= lo) & (pos <= hi)).all(-1) & parked
            chk["parked_outside_reach_box"] &= not bool(inbox.any())
            hp = S._t(r.data.body_link_pos_w)[:, h]
            hq = S._t(r.data.body_link_quat_w)[:, h]
            exp_pos = hp + (matrix_from_quat(hq) @ mount_pos[:, None])[..., 0]
            cd = uu.scene["wrist_cam"].data
            chk["wrist_pos_err_max_m"] = max(chk["wrist_pos_err_max_m"], float(torch.linalg.norm(S._t(cd.pos_w) - exp_pos, dim=-1).max()))
            exp_q = quat_mul(hq, mount_q.expand_as(hq))
            dot = (S._t(cd.quat_w_ros) * exp_q).sum(-1).abs().clamp(max=1.0)
            chk["wrist_rot_err_max_deg"] = max(chk["wrist_rot_err_max_deg"], float(torch.rad2deg(2 * torch.arccos(dot)).max()))
            chk["frames"] += 1

        run = Runner(env, geo)
        sheets = []
        # (1) grasp trials per size with frames at grasp confirmation (CLOSE -> LIFT) and at the hold (last step)
        for size in ("small", "large"):
            eps = make_stream(args.seed + (0 if size == "small" else 10), 0, N, grasp_spec(size))
            shots = {"grasp": [None] * N, "hold": [None] * N}

            def on_step(t, uu, s, label_, exe, ctrl, det, shots=shots):
                if t == 0:
                    st.pop("p0", None)
                    aim_closeup(uu, s["cube_pos"][torch.arange(N, device=uu.device), ctrl.target])
                if t % 3 == 0:
                    checks(uu, s, uu._layout)
                for i in range(N):
                    if (shots["grasp"][i] is None and int(ctrl.phase[i]) == LIFT) or (t == T_GRASP - 1 and shots["hold"][i] is None):
                        key = "grasp" if shots["grasp"][i] is None and int(ctrl.phase[i]) == LIFT and t < T_GRASP - 1 else "hold"
                        imgs = [grab(uu, n)[i] for n in ("side_cam", "wrist_cam", CLOSEUP)]
                        shots[key][i] = [label(imgs[0], [f"{size} env{i} t={t} {key}"]), imgs[1], imgs[2]]

            res = run.run(eps, T_GRASP, on_step=on_step)
            rows = []
            for i in range(min(N, 4)):
                row = []
                for key in ("grasp", "hold"):
                    if shots[key][i] is not None:
                        row += shots[key][i]
                rows.append(row)
            f = FR / f"grasp_{size}_side_wrist_closeup.png"
            imageio.imwrite(f, grid(rows, h=200))
            sheets.append(str(f.relative_to(ROOT)))
            st[f"grasp_{size}_holds"] = res["pick_hold_last50"].tolist()
        # (2) task scenes (place, all pads) + one env with all 8 cubes and 4 pads on the table
        def all_spec(rng):
            sp = sample_spec("s3a", "train", rng)
            return sp

        eps = make_stream(args.seed + 5, 0, N, all_spec)
        r8 = np.random.default_rng([args.seed, 99])
        full = TaskSpec("pick", ALL_CUBES[0], None, list(ALL_CUBES), list(COLORS))
        try:
            lay8, _ = sample_layout(full, r8, max_restarts=5000)
            eps[0] = Episode(0, full, "all objects", lay8)
        except RuntimeError:
            pass
        shots8 = {}

        def on_step2(t, uu, s, label_, exe, ctrl, det):
            if t == 0:
                st.pop("p0", None)
                aim_closeup(uu, torch.tensor([[0.5, 0.0, 0.05]], device=uu.device).expand(N, 3), dist=0.9)
            if t % 3 == 0:
                checks(uu, s, uu._layout)
            if t in (1, 150, 299):
                shots8[t] = [grab(uu, n)[:4] for n in ("side_cam", "wrist_cam", CLOSEUP)]

        res2 = run.run(eps, 300, on_step=on_step2)
        rows = [[label(shots8[t][0][i], [f"env{i} t={t} {eps[i].spec.task}"]), shots8[t][1][i], shots8[t][2][i]]
                for i in range(min(N, 4)) for t in (1, 299)]
        f = FR / "scenes_side_wrist_closeup.png"
        imageio.imwrite(f, grid(rows, h=200))
        sheets.append(str(f.relative_to(ROOT)))
        out = {"stage": "0A.1 cams", "seed": args.seed, "res": 128, "num_envs": N, "git_sha": sha(),
               "distinct_seg_ids_all_envs": distinct, "checks": chk,
               "pass": distinct and chk["foreign_pixels"] == 0 and chk["parked_pixels"] == 0 and chk["parked_max_move_m"] < 1e-3
               and chk["parked_outside_reach_box"] and chk["wrist_pos_err_max_m"] < 1e-3 and chk["wrist_rot_err_max_deg"] < 0.5,
               "all_objects_env0_spec": eps[0].spec.to_json(), "sheets": sheets,
               "grasp_frames_holds": {k: v for k, v in st.items() if k.startswith("grasp_")},
               "wall_s": round(time.time() - t0, 1)}
        print(json.dumps(out, indent=1))
        (RES / "0a1_cams.json").write_text(json.dumps(out, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
