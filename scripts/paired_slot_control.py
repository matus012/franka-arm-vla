"""Control (sim venv, laptop via ops/run_job.ps1): is the residual t=0 render difference between members of a paired group
determined by the env SLOT or by the TARGET?  For each of several paired groups (same scene, members m = 0..k-1):
  X  every slot holds member 0
  Y  slot s holds member (s mod k) of the same group
Both batches use the collect path (Runner.run_batch, t=0 capture before the first step, both cameras).
Per camera: (1) X[s] == Y[s] for the slots where Y holds member 0 (same slot, same member: trivially equal) AND for the slots
where Y holds another member (same slot, other target) -> target has no effect; (2) for every slot pair (i, j):
diff(X_i, X_j) == diff(X_i, Y_j) elementwise, i.e. the residual is the slot's, not the target's.
  python scripts/paired_slot_control.py --envs 16 --image-size 224 --seed 32800 --groups 4 --out results/paired_slot_control.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=16)
p.add_argument("--image-size", type=int, default=224)
p.add_argument("--seed", type=int, default=32800)
p.add_argument("--groups", type=int, default=4, help="how many groups with >= 3 members to test")
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import vla_tabletop.scene as S  # noqa: E402
from collect import geometry_113, task_timing  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift import cameras as C  # noqa: E402
from vla_common.catalog import ALL_CUBES  # noqa: E402
from vla_tabletop.rollout import Runner, make_paired_stream  # noqa: E402

B, H = args.envs, args.image_size
cfg = S.make_cfg(B, args.seed, res=H)
assert_run_preconditions(cfg)
eps, _ = make_paired_stream(args.seed, 0, 60, "s2a", "train", 15.0)
groups: dict[int, list] = {}
for e in eps:
    groups.setdefault(e.layout_id, []).append(e)
chosen = [g for g in sorted(groups) if len(groups[g]) >= 3][: args.groups]

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    run = Runner(env, geometry_113())
    tj = task_timing("pick")
    run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)

    def t0(batch: list) -> dict[str, np.ndarray]:
        rec = {}

        def on_step(t, uu, s, label, exe, ctrl, det):
            if t == 0:
                rec["side"] = C.rgb(uu, "side_cam").to(torch.uint8).cpu().numpy().astype(np.int16)
                rec["wrist"] = C.rgb(uu, "wrist_cam").to(torch.uint8).cpu().numpy().astype(np.int16)

        run.run(batch, 2, on_step=on_step)
        return rec

    res = {"envs": B, "image_size": H, "device": str(u.device), "groups": []}
    for g in chosen:
        ms = groups[g]
        X = t0([ms[0]] * B)
        Y = t0([ms[s % len(ms)] for s in range(B)])
        row = {"layout_id": g, "members": len(ms), "targets": ["_".join(m.spec.target) for m in ms]}
        for cam in ("side", "wrist"):
            x, y = X[cam], Y[cam]
            same_slot = [bool(np.array_equal(x[s], y[s])) for s in range(B)]
            other_target = [s for s in range(B) if s % len(ms) != 0]
            pair_eq = all(np.array_equal(x[i] - x[j], x[i] - y[j]) for i in range(B) for j in range(B))
            px = lambda a, b: int((np.abs(a - b).max(-1) > 0).sum())  # noqa: E731
            row[cam] = {"X_slot_equals_Y_slot_all": all(same_slot), "X_slot_equals_Y_slot_other_target_slots":
                        all(same_slot[s] for s in other_target), "diff_Xi_Xj_equals_diff_Xi_Yj_all_pairs": pair_eq,
                        "slots_where_X_slot_differs_from_X_slot0": [s for s in range(B) if not np.array_equal(x[s], x[0])],
                        "max_abs_diff_between_X_slots": int(max(np.abs(x[i] - x[j]).max() for i in range(B) for j in range(B))),
                        "n_px_X0_vs_X_slots": [px(x[0], x[s]) for s in range(B)]}
        res["groups"].append(row)
        print(json.dumps(row), flush=True)
    res["residual_is_slot_determined"] = all(r[c][k] for r in res["groups"] for c in ("side", "wrist")
                                             for k in ("X_slot_equals_Y_slot_all", "diff_Xi_Xj_equals_diff_Xi_Yj_all_pairs"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("CONTROL_DONE residual_is_slot_determined=", res["residual_is_slot_determined"], flush=True)
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
