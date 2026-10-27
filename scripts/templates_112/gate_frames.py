"""Close-up gate frames (every gate, brief section 2): replay a policy on a few envs AFTER training (never during)
and write per-phase contact sheets: close-up scene view with the phase overlay (+ the student's side and wrist views
once the student cameras exist, --student_cams).

Usage: python scripts/gate_frames.py --policy teacher --task LiftRL-Phased-P6-TS-v0 --out results/frames/s01 [--num_envs 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--policy", default="teacher", help="'teacher' or a student checkpoint path")
p.add_argument("--world", default=None, help="student replay: Fixed/P3/P4/P5/P6 (student env)")
p.add_argument("--task", default="LiftRL-Phased-P6-TS-v0")
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=3)
p.add_argument("--seed", type=int, default=2024)
p.add_argument("--student_cams", action="store_true")
p.add_argument("--tag", default="")
p.add_argument("--cube_period", type=int, default=0, help="teacher: Stage-0.3 cube estimate every P steps (latency 1, 1 cm noise, action sigma 0.1)")
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.evaluation import cube_and_target, make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import Thresholds  # noqa: E402
from lift_rl.phases import PHASES  # noqa: E402
from lift_rl.physics import assert_run_preconditions  # noqa: E402
from lift_rl.video import aim_task, cam_cfg, grab, overlay, project, target_points  # noqa: E402
from vision_lift.teacher import load_teacher  # noqa: E402

N = args.num_envs
STUDENT = args.policy != "teacher"
if STUDENT:
    import vision_lift.envs as E

    args.task = E.student_id(args.world)
cfg = make_eval_cfg(load_cfg_from_registry(args.task, "env_cfg_entry_point"), N, args.seed)
cfg.scene.video_cam = cam_cfg(18.0, 960, 540)
if args.student_cams and not STUDENT:
    from vision_lift.cameras import add_student_cameras

    add_student_cameras(cfg)
assert_run_preconditions(cfg)
th = Thresholds()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    if args.policy == "teacher":
        actor = load_teacher(u.device)
        policy = lambda obs: actor(obs["policy"])  # noqa: E731
        label = "teacher (111 phase 6, frozen)"
        if args.cube_period:
            from vision_lift.cube_estimate import CubeEstimate, replace_cube_terms, true_cube

            ce = CubeEstimate(N, u.device, period=args.cube_period, latency=1, pos_sigma=0.01, seed=args.seed + 1)
            gen = torch.Generator(device=u.device)
            gen.manual_seed(args.seed + 2)

            def policy(obs):
                est = ce.step(*true_cube(u))
                a = actor(replace_cube_terms(obs["policy"], u, *est))
                return (a + torch.randn(a.shape, generator=gen, device=u.device) * 0.1).clamp(-1.0, 1.0)

            label = f"teacher, cube obs {50 / args.cube_period:.1f} Hz +1 frame, 1 cm, act noise 0.1"
    else:
        from vision_lift.collect import Runner
        from vision_lift.train import load_student

        student, depth = load_student(ROOT / args.policy, u.device)
        runner = Runner(env, load_teacher(u.device), student, depth, "student", 0.0, 0.0, seed=args.seed)
        label = f"student {Path(args.policy).parent.name}/{Path(args.policy).stem}"
    state = {"aimed": False}
    steps = cfg.eval_steps
    phase_log = np.zeros((steps, N), np.int64)
    frames: dict[int, list[np.ndarray]] = {i: [] for i in range(N)}

    def pol(obs):
        if not state["aimed"]:
            aim_task(u, u.scene["video_cam"])
            if args.student_cams:
                from vision_lift.cameras import aim_side

                aim_side(u)
            state["aimed"] = True
        return policy(obs)

    def on_step(t, uu):
        if STUDENT and t == 0:
            aim_task(uu, uu.scene["video_cam"])  # after the episode reset: frame this episode's cube and target
        pm = uu._phase_machine
        ph = pm.phase.cpu().numpy()
        phase_log[t] = ph
        cube, goal = cube_and_target(uu)
        hit = (torch.linalg.norm(cube - goal, dim=-1) < th.success_r).cpu().numpy()
        fr = grab(uu, "video_cam")
        uvd = project(uu, "video_cam", target_points(uu))
        views = None
        if STUDENT:
            _, fed = runner.last_fed  # the delayed frames the student actually saw at this step
            both = torch.cat([fed["side_rgb"], fed["wrist_rgb"]], dim=1)
            k = max(1, fr.shape[1] // both.shape[1])
            both = both.repeat_interleave(k, 1).repeat_interleave(k, 2)
            views = np.zeros((N, fr.shape[1], both.shape[2], 3), np.uint8)
            views[:, : both.shape[1]] = both.cpu().numpy()
        elif args.student_cams:
            from vision_lift.cameras import student_views

            views = student_views(uu, height=fr.shape[1])
        for i in range(N):
            f = overlay(fr[i], uvd[i], [f"phase {ph[i] + 1}/6: {PHASES[ph[i]].upper()}" + ("  (grasp)" if bool(pm.grasp[i]) else ""),
                                        f"{label}   t = {(t + 1) * uu.step_dt:.2f} s"], bool(hit[i]))
            if views is not None:
                f = np.concatenate([f, views[i]], axis=1)
            frames[i].append(f[::2, ::2])

    if STUDENT:
        res = runner.run_batch(on_step_extra=on_step)
    else:
        res = rollout(env, pol, th, on_step=on_step)
    sheets = {}
    for i in range(N):
        sheets[i] = []
        for k, name in enumerate(PHASES):
            st = np.flatnonzero(phase_log[:, i] == k)
            if len(st) == 0:
                continue
            pick = sorted({int(st[0]), int(st[len(st) // 2]), int(st[-1])})
            fname = f"{args.tag}env{i}_phase{k + 1}_{name}_steps_{'-'.join(map(str, pick))}.png"
            sheet = np.concatenate([frames[i][s] for s in pick], axis=0)
            imageio.imwrite(out / fname, sheet)
            sheets[i].append(fname)
    per_env = [{"env": i, "success": bool(res["success"][i]), "dist_m": round(float(res["dist"][i]), 4),
                "slam": bool(res["slam"][i]), "jam": bool(res["jam"][i]), "sheets": sheets[i]} for i in range(N)]
    (out / f"{args.tag}frames.json").write_text(json.dumps({"task": args.task, "policy": args.policy, "seed": args.seed,
                                                              "per_env": per_env}, indent=1))
    print(json.dumps(per_env, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
