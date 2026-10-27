"""PERUN render check: does Isaac Lab + Newton camera rendering work headless on this node, and how fast?

Builds the frozen-teacher task env (default LiftRL-Phased-P6-TS-v0, from the bind-mounted repo) with one
Newton-Warp RGB(+depth) camera per env, steps N control steps with zero actions, and prints images/s
(= num_envs * steps / wall). Saves one frame PNG for eyeballing. Runs identically on the laptop (compare).

  python hpc/render_check.py --num_envs 64 --steps 200 --width 128 --height 128 --out results/render_check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (bind-mounted)

import gymnasium as gym
import torch

p = argparse.ArgumentParser()
p.add_argument("--task", default="LiftRL-Phased-P6-TS-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--width", type=int, default=128)
p.add_argument("--height", type=int, default=128)
p.add_argument("--depth", action="store_true", help="also render depth")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results/render_check")
args = p.parse_args()

assert torch.cuda.is_available(), "CUDA unavailable -- STOP (no CPU fallback for GPU work)"
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda} "
      f"DISPLAY={os.environ.get('DISPLAY')!r}", flush=True)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import lift_rl  # noqa: E402,F401  (registers tasks)
from lift_rl.evaluation import make_eval_cfg  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402

cfg = make_eval_cfg(load_cfg_from_registry(args.task, "env_cfg_entry_point"), args.num_envs, args.seed, sensors=False)
cfg.scene.cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/RenderCheckCam", update_latest_camera_pose=True,
    data_types=["rgb"] + (["depth"] if args.depth else []),
    spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, focus_distance=400.0, horizontal_aperture=20.955,
                                     clipping_range=(0.05, 20.0)),
    width=args.width, height=args.height, renderer_cfg=NewtonWarpRendererCfg())
assert_run_preconditions(cfg)


def t(x):
    return x.torch if hasattr(x, "torch") else x


with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    cam = u.scene["cam"]
    o = u.scene.env_origins
    cam.set_world_poses_from_view(o + torch.tensor([1.10, 0.80, 0.62], device=u.device),
                                  o + torch.tensor([0.32, 0.0, 0.22], device=u.device))
    env.reset()
    act = torch.zeros(args.num_envs, u.action_manager.total_action_dim, device=u.device)

    def step() -> torch.Tensor:
        env.step(act)
        return t(cam.data.output["rgb"])

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        rgb = step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    frame = rgb[0, ..., :3].to(torch.uint8).cpu().numpy()
    assert frame.std() > 1.0, "frame 0 is (nearly) constant -- renderer produced no image"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.imwrite(out / "frame0.png", frame)
    res = dict(node=os.uname().nodename if hasattr(os, "uname") else "win", gpu=torch.cuda.get_device_name(0),
               num_envs=args.num_envs, steps=args.steps, res=[args.width, args.height], depth=args.depth,
               wall_s=round(dt, 3), control_steps_per_s=round(args.steps / dt, 2),
               images_per_s=round(args.num_envs * args.steps / dt, 1), frame_std=float(frame.std()),
               peak_vram_mib=int(torch.cuda.max_memory_allocated() / 2**20))
    (out / "render_check.json").write_text(json.dumps(res, indent=1))
    print("RENDER_CHECK", json.dumps(res), flush=True)
    env.close()
