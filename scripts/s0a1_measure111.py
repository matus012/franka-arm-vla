"""0A.1 (measurement): the real 111 cube (DexCube x 0.8) and table-top height, read from the Newton model of the
111 env (LiftRL-Own-v0: same scene + physics fix 2 as every 111/112 env).

Records: cube collision-shape type / scale / local AABB (-> edge), body mass, friction mu, the cube's settled rest
height, and the table-top height (= rest height - half edge). Writes results/s0a1_measure111.json.

Usage: python scripts/s0a1_measure111.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402

TASK = "LiftRL-Own-v0"
N = 4
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = N
cfg.seed = 7
cfg.sim.device = "cuda:0"
cfg.episode_length_s = 1000.0
cfg.events.reset_object_pose = None  # cube at its default pose (0.5, 0, 0.055), yaw 0
assert_run_preconditions(cfg)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


with launch_simulation(cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    env.reset()
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    labels = model.shape_label
    stype = model.shape_type.numpy()
    sscale = model.shape_scale.numpy()
    lo = model.shape_collision_aabb_lower.numpy() if model.shape_collision_aabb_lower is not None else None
    hi = model.shape_collision_aabb_upper.numpy() if model.shape_collision_aabb_upper is not None else None
    mu = model.shape_material_mu.numpy()
    flags = model.shape_flags.numpy()
    shapes = []
    for k, lab in enumerate(labels):
        if lab.startswith("/World/envs/env_0/Object") or lab.startswith("/World/envs/env_0/Table"):
            src = model.shape_source[k]
            verts = getattr(src, "vertices", None)
            vb = None
            if verts is not None:
                v = np.asarray(verts) * sscale[k]
                vb = {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "extent": (v.max(0) - v.min(0)).tolist()}
            shapes.append({"k": k, "label": lab, "type": int(stype[k]), "scale": sscale[k].tolist(), "flags": int(flags[k]),
                           "aabb_lo": lo[k].tolist() if lo is not None else None,
                           "aabb_hi": hi[k].tolist() if hi is not None else None,
                           "mesh_vertex_bbox": vb, "mu": float(mu[k])})
    obj = u.scene["object"]
    mass = float(_t(obj.data.body_mass).reshape(N, -1)[0].sum())
    zero = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)
    zs = []
    for t in range(150):
        env.step(zero)
        zs.append(float((_t(obj.data.root_pos_w) - u.scene.env_origins)[:, 2].mean()))
    rest_z = float(np.mean(zs[-20:]))
    rest_z_std = float(np.std(zs[-20:]))
    cube_coll = [s for s in shapes if s["label"].startswith("/World/envs/env_0/Object") and s["flags"] & 2]
    out = {"task": TASK, "seed": cfg.seed, "num_envs": N, "cube_mass_kg": mass, "rest_z_m": rest_z,
           "rest_z_std_m": rest_z_std, "shapes": shapes,
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    p = ROOT / "results" / "s0a1_measure111.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    env.close()
