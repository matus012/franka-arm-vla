"""Diagnostic (sim venv, via ops/run_job.ps1): why are the t=0 frames of one paired group not identical?
The collect path (Runner.run_batch + the same on_step capture at t=0) is run on B slots in four batches:
  A  the SAME episode copied into all B slots           -> pure slot effect on the t=0 frame
  A2 batch A again (a second reset of the same slots)   -> reset / stale-buffer effect (frame vs run A, same slot)
  P  paired groups packed into the slots (members adjacent) -> what collect does
  S  a DIFFERENT episode in every slot, then A again    -> stale previous-episode effect on the t=0 frame
Per batch and camera: number of distinct sha256 over slots, max / mean abs diff to slot 0, and a per-slot pixel diff map
summary. Also captures the frame at t=3 (the first tick after one PERIOD of physics) for the same comparison.
  python scripts/paired_t0_probe.py --envs 16 --image-size 224 --seed 32800 --out results/paired_t0_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
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
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import vla_tabletop.scene as S  # noqa: E402
from collect import geometry_113, task_timing  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vision_lift import cameras as C  # noqa: E402
from vla_common import timing as TM  # noqa: E402
from vla_common.catalog import ALL_CUBES  # noqa: E402
from vla_tabletop.rollout import Runner, make_paired_stream  # noqa: E402

B, H = args.envs, args.image_size
cfg = S.make_cfg(B, args.seed, res=H)
assert_run_preconditions(cfg)
eps, _ = make_paired_stream(args.seed, 0, 40, "s2a", "train", 15.0)
groups = {}
for e in eps:
    groups.setdefault(e.layout_id, []).append(e)
packed = [e for g in sorted(groups) for e in groups[g]][:B]
distinct = [groups[g][0] for g in sorted(groups)][:B]
same = [eps[0]] * B


def report(frames: dict[str, np.ndarray]) -> dict:
    out = {}
    for cam, f in frames.items():  # f: (B, H, W, 3) uint8
        sha = [hashlib.sha256(f[j].tobytes()).hexdigest()[:10] for j in range(len(f))]
        d = np.abs(f.astype(np.int16) - f[0].astype(np.int16))
        out[cam] = {"distinct_sha": len(set(sha)), "max_abs_diff": int(d.max()),
                    "mean_abs_diff_per_slot": [round(float(d[j].mean()), 4) for j in range(len(f))],
                    "n_px_differing_per_slot": [int((d[j].max(-1) > 0).sum()) for j in range(len(f))]}
    return out


with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    run = Runner(env, geometry_113())
    tj = task_timing("pick")
    run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)
    res = {"envs": B, "image_size": H, "device": str(u.device)}

    def go(name: str, batch: list) -> dict[str, np.ndarray]:
        rec: dict[int, dict[str, np.ndarray]] = {}

        def on_step(t, uu, s, label, exe, ctrl, det):
            if t in (0, 3):
                rec[t] = {"side": C.rgb(uu, "side_cam").to(torch.uint8).cpu().numpy(),
                          "wrist": C.rgb(uu, "wrist_cam").to(torch.uint8).cpu().numpy()}

        run.run(batch + [batch[-1]] * (B - len(batch)), 4, on_step=on_step)
        res[name] = {f"t{t}": report(r) for t, r in rec.items()}
        lid = [e.layout_id for e in batch]
        res[name]["within_group_t0"] = {  # diff of each member to the first member of its own group, per camera
            c: [{"layout_id": g, "slots": [j for j, x in enumerate(lid) if x == g],
                 "max_abs_diff": int(max(np.abs(rec[0][c][j].astype(np.int16) - rec[0][c][lid.index(g)].astype(np.int16)).max()
                                         for j, x in enumerate(lid) if x == g)),
                 "n_px_differing": [int((np.abs(rec[0][c][j].astype(np.int16) - rec[0][c][lid.index(g)].astype(np.int16)).max(-1) > 0).sum())
                                    for j, x in enumerate(lid) if x == g]} for g in dict.fromkeys(lid)]
            for c in ("side", "wrist")}
        res[name]["slots"] = [{"layout_id": e.layout_id, "member": e.member} for e in batch]
        print(name, json.dumps({k: {c: (v["distinct_sha"], v["max_abs_diff"]) for c, v in r.items()}
                                for k, r in res[name].items() if k not in ("slots", "within_group_t0")}), flush=True)
        return rec[0]

    a = go("A_same_episode_all_slots", same)
    a2 = go("A2_same_again", same)
    res["A2_vs_A_t0_max_abs_diff"] = {c: int(np.abs(a[c].astype(np.int16) - a2[c].astype(np.int16)).max()) for c in a}
    go("P_paired_groups_packed", packed)
    go("S_distinct_episodes", distinct)
    a3 = go("A3_same_after_S", same)
    res["A3_vs_A_t0_max_abs_diff"] = {c: int(np.abs(a[c].astype(np.int16) - a3[c].astype(np.int16)).max()) for c in a}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("PROBE_DONE", flush=True)
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
