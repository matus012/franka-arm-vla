"""Scene check for tests/test_sim_113.py (GPU, 4 envs, no cameras): prints CHECK_SCENE=<json> with the policy
observation terms / dims, the parked objects' displacement over 100 steps and whether they stay outside the reach box,
and the layout objects' rest heights after settling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vla_common.tasks import sample_spec  # noqa: E402
from vla_tabletop.layout import REACH_BOX_HI, REACH_BOX_LO, layout_array, sample_layout  # noqa: E402

N = 4
cfg = S.make_cfg(N, 3)
assert_run_preconditions(cfg)
with launch_simulation(cfg, {"headless": True}):
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    rng = np.random.default_rng(3)
    u._layout = torch.as_tensor(layout_array([sample_layout(sample_spec("s3a", "train", rng), rng)[0] for _ in range(N)]),
                                device=u.device)
    obs, _ = env.reset()
    p0, _ = S.object_poses(u)
    for _ in range(100):
        obs = env.step(torch.zeros(N, 5, device=u.device))[0]
    p1, _ = S.object_poses(u)
    # GATES 0B.4 item 1: teleporting every cube and pad (no physics step in between) changes the policy state by 0.
    s0 = u.observation_manager.compute()["policy"].clone()
    ids = torch.arange(N, device=u.device)
    for j, name in enumerate(S.OBJ_NAMES):
        a = u.scene[name]
        pose = torch.cat([S._t(a.data.root_pos_w), S._t(a.data.root_quat_w)], 1).clone()
        pose[:, :3] += torch.tensor([0.07 + 0.01 * j, -0.05, 0.03], device=u.device)
        a.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    p_tel, _ = S.object_poses(u)
    s1 = u.observation_manager.compute()["policy"]
    teleport = {"teleport_moved_min_m": float(torch.linalg.norm(p_tel - p1, dim=-1).min()),
                "teleport_state_max_abs_change": float((s1 - s0).abs().max())}
    parked = u._layout[:, :, 4] < 0.5
    lo, hi = torch.tensor(REACH_BOX_LO, device=u.device), torch.tensor(REACH_BOX_HI, device=u.device)
    out = {"policy_terms": u.observation_manager.active_terms["policy"],
           "policy_dims": [list(d) for d in u.observation_manager.group_obs_term_dim["policy"]],
           "policy_shape": list(obs["policy"].shape), "obs_groups": list(obs.keys()),
           "parked_max_move_m": float(torch.linalg.norm(p1 - p0, dim=-1)[parked].max()) if bool(parked.any()) else 0.0,
           "parked_in_reach_box": bool((((p1 >= lo) & (p1 <= hi)).all(-1) & parked).any()),
           "on_table_max_move_m": float(torch.linalg.norm(p1 - p0, dim=-1)[~parked].max()), **teleport}
    print("CHECK_SCENE=" + json.dumps(out), flush=True)
    env.close()
