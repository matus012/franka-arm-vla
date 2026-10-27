"""Demo video worker (one sim process): replays a policy AFTER training (never during) on N envs and writes one MP4 per
env: close-up scene view with the phase overlay | what the student saw (delayed side + wrist frames). 720p, first
2 s at 0.25x (slow-motion grasp), then 1x (lift_rl.camera.SlowMoWriter). Also writes <out>/episodes.json.

  --policy student --ckpt X.pt   camera-only student (student env)
  --policy baseline              Stage-1 estimator -> frozen teacher (teacher env, reflex; vision_lift.baseline)
  --visual none|nominal|train|heldout  (Stage 4 randomization)
Usage: python scripts/demo_worker.py --policy student --ckpt ckpt.pt --world P6 --num_envs 9 --seed 12345 --out videos/raw/x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--policy", choices=("student", "baseline"), required=True)
p.add_argument("--ckpt", default=None)
p.add_argument("--estimator", default="checkpoints/s1_rgb/picked.pt")
p.add_argument("--world", default="P6")
p.add_argument("--num_envs", type=int, default=9)
p.add_argument("--seed", type=int, default=12345)
p.add_argument("--visual", default="none", choices=("none", "nominal", "train", "heldout"))
p.add_argument("--visual_seed", type=int, default=0)
p.add_argument("--label", default=None)
p.add_argument("--out", required=True)
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
import vision_lift.envs as E  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.camera import SlowMoWriter  # noqa: E402
from lift_rl.evaluation import cube_and_target, make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import Thresholds  # noqa: E402
from lift_rl.phases import PHASES  # noqa: E402
from lift_rl.video import aim_task, cam_cfg, grab, overlay, project, target_points  # noqa: E402
from vision_lift import cameras as C  # noqa: E402

task = E.student_id(args.world) if args.policy == "student" else E.teacher_id(args.world)
cfg = make_eval_cfg(load_cfg_from_registry(task, "env_cfg_entry_point"), args.num_envs, args.seed)
cfg.scene.video_cam = cam_cfg(18.0, 960, 720)
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
th = Thresholds()


def panel(fed: dict, i: int, h: int) -> np.ndarray:
    """Student view panel: side (top) and wrist (bottom) delayed frames, nearest-upscaled to height h."""
    both = torch.cat([fed["side_rgb"][i], fed["wrist_rgb"][i]], 0)  # (2H, W, 3)
    k = max(1, h // both.shape[0])
    both = both.repeat_interleave(k, 0).repeat_interleave(k, 1).to(torch.uint8).cpu().numpy()
    pad = np.zeros((h, both.shape[1], 3), np.uint8)
    pad[: both.shape[0]] = both
    return pad


with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    N = u.num_envs
    visual = None
    if args.visual != "none":
        from vision_lift.visual_rand import VisualRandomizer

        visual = VisualRandomizer(u, args.visual, args.visual_seed)
    writers = [SlowMoWriter(out / f"env{i}.mp4", slow_steps=100) for i in range(N)]
    phase_seen = {"fed": None}
    if args.policy == "student":
        from vision_lift.collect import Runner
        from vision_lift.teacher import load_teacher
        from vision_lift.train import load_student

        student, depth = load_student(ROOT / args.ckpt, u.device)
        runner = Runner(env, load_teacher(u.device), student, depth, "student", 0.0, 0.0, seed=args.seed, visual=visual)
        label = args.label or f"camera-only student ({Path(args.ckpt).parent.name})"
    else:
        from vision_lift.baseline import estimate_to_pose, install_estimated_phase_state, teacher_obs_with_estimate
        from vision_lift.frames import DelayedFrames
        from vision_lift.student import CubeEstimator
        from vision_lift.teacher import OBS_SLICES, load_teacher

        install_estimated_phase_state(env)
        teacher = load_teacher(u.device)
        ck = torch.load(ROOT / args.estimator, map_location="cpu", weights_only=False)
        est = CubeEstimator(depth=ck["depth"])
        est.load_state_dict(ck["model"])
        est = est.to(u.device).eval()
        r = u.scene["robot"]
        fid = r.find_joints("panda_finger.*")[0]
        buf = DelayedFrames()
        st = {"t": 0}
        label = args.label or "modular baseline (image cube estimator -> teacher)"

        @torch.inference_mode()
        def base_policy(obs):
            from lift_rl.phases import CLOSE

            t = st["t"]
            if t == 0:
                visual.apply() if visual is not None else C.aim_side(u)
            if buf.is_tick(t):
                po = obs["policy"]
                w = (r.data.joint_pos.torch if hasattr(r.data.joint_pos, "torch") else r.data.joint_pos)[:, fid].sum(1, keepdim=True)
                prop = torch.cat([po[:, OBS_SLICES["joint_pos"]], po[:, OBS_SLICES["joint_vel"]], w, po[:, OBS_SLICES["target"]],
                                  po[:, OBS_SLICES["last_arm_action"]], (u._phase_machine.phase >= CLOSE).float()[:, None]], 1)
                fr = {"side_rgb": C.rgb(u, "side_cam").to(torch.uint8), "wrist_rgb": C.rgb(u, "wrist_cam").to(torch.uint8)}
                pos, sc = est({"side_rgb": fr["side_rgb"].permute(0, 3, 1, 2), "wrist_rgb": fr["wrist_rgb"].permute(0, 3, 1, 2),
                               "proprio": prop})
                buf.capture(t, (fr, estimate_to_pose(pos, sc)))
            _, (fr, (e_pos, e_yaw)) = buf.current()
            phase_seen["fed"] = fr
            u._cube_est = (e_pos, e_yaw)
            st["t"] += 1
            return teacher(teacher_obs_with_estimate(obs["policy"], u, e_pos, e_yaw))

    def on_step(t, uu):
        if t == 0:
            aim_task(uu, uu.scene["video_cam"])
        pm = uu._phase_machine
        ph = pm.phase.cpu().numpy()
        cube, goal = cube_and_target(uu)
        hit = (torch.linalg.norm(cube - goal, dim=-1) < th.success_r).cpu().numpy()
        fr = grab(uu, "video_cam")
        uvd = project(uu, "video_cam", target_points(uu))
        fed = runner.last_fed[1] if args.policy == "student" else phase_seen["fed"]
        for i in range(N):
            f = overlay(fr[i], uvd[i], [f"phase {ph[i] + 1}/6: {PHASES[ph[i]].upper()}" + ("  (grasp)" if bool(pm.grasp[i]) else ""),
                                        f"{label}   t = {(t + 1) * uu.step_dt:.2f} s"], bool(hit[i]))
            writers[i].add(np.concatenate([f, panel(fed, i, f.shape[0])], 1))

    if args.policy == "student":
        res = runner.run_batch(on_step_extra=on_step)
    else:
        res = rollout(env, base_policy, th, on_step=on_step)
    for w in writers:
        w.close()
    eps = [{"env": i, "success": bool(res["success"][i]), "dist_m": round(float(res["dist"][i]), 4),
            "slam": bool(res["slam"][i]), "jam": bool(res["jam"][i]), "file": f"env{i}.mp4"} for i in range(N)]
    (out / "episodes.json").write_text(json.dumps({"policy": args.policy, "ckpt": args.ckpt, "world": args.world,
                                                   "seed": args.seed, "visual": args.visual, "label": label,
                                                   "episodes": eps}, indent=1))
    print(json.dumps(eps))
    env.close()
