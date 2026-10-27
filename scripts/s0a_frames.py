"""Close-up gate frames for 0A.2 / 0A.3 (rendering only, after the counted run): replays chosen episodes of a seed's
stream with the same expert settings and writes one sheet per episode: rows = key steps (expert phase entries: grasp
(lift), release, last step; plus approach / carry), columns = close-up | side | wrist. Also checks the replay reproduces
the counted run's outcome (success flag per episode, from the run JSON when given).

Usage: python scripts/s0a_frames.py --task stack --seed 11220 --indices 133,164,0,1 --out results/frames/0a2/stack
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["pick", "place", "stack"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--indices", required=True)
p.add_argument("--out", required=True)
p.add_argument("--chunk", type=int, default=0)
p.add_argument("--split", default="train")
p.add_argument("--res", type=int, default=128)
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import vla_tabletop.scene as S  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from vla_common import timing as TM  # noqa: E402
from vla_common.catalog import ALL_CUBES  # noqa: E402
from vla_common.tasks import sample_spec  # noqa: E402
from vla_tabletop.expert import CARRY, HOVER, LIFT, PHASES, RELEASE  # noqa: E402
from vla_tabletop.rollout import Runner, geometry, make_stream  # noqa: E402
from lift_rl.video import project  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from vla_tabletop.sheets import CLOSEUP, add_closeup, aim_closeup, grab, grid, label  # noqa: E402


def ghost(img: np.ndarray, uv_true, uv_snap) -> np.ndarray:
    """Mark the true target cube (green circle) and the expert's stale snapshot of it (magenta cross)."""
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    x, y = uv_true
    d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(0, 255, 0), width=2)
    x, y = uv_snap
    d.line([x - 9, y - 9, x + 9, y + 9], fill=(255, 0, 255), width=2)
    d.line([x - 9, y + 9, x + 9, y - 9], fill=(255, 0, 255), width=2)
    return np.asarray(im)

STAGE = {"pick": "s2", "place": "s3a", "stack": "s3b"}[args.task]
idx = [int(i) for i in args.indices.split(",")]
N = len(idx)
RES = ROOT / "results"
meas = json.loads((RES / "0a1_measure.json").read_text())
tj = json.loads((RES / f"0a2_{args.task}_timing.json").read_text())
T = tj["T_steps"]
geo = geometry(meas["z_rest_m"], h_pad=float(np.mean(list(meas["h_pad_m"].values()))))
grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES])
expert_kw = {"fresh": True} if args.chunk == 0 else {"fresh": False, "n_query": args.chunk, "period": TM.PERIOD,
                                                       "latency": TM.LATENCY}
cfg = S.make_cfg(N, args.seed, res=args.res)
add_closeup(cfg, 320, 240, focal=18.0)
assert_run_preconditions(cfg)
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(S.TASK_ID, cfg=cfg)
    u = env.unwrapped
    assert_on_cuda(env)
    eps = [make_stream(args.seed, i, 1, lambda rng: sample_spec(STAGE, args.split, rng), split=args.split)[0] for i in idx]
    run = Runner(env, geo, grasp_dz=grasp_dz.to(u.device))
    shots: dict[int, list] = {i: [] for i in range(N)}
    prev = {"ph": None}

    def on_step(t, uu, s, lab, exe, ctrl, det):
        ar = torch.arange(N, device=uu.device)
        tgt = s["cube_pos"][ar, ctrl.target]
        if t == 0 or t % 25 == 0:
            aim_closeup(uu, tgt + torch.tensor([0.0, 0.0, 0.03], device=uu.device), dist=0.45)
        ph = ctrl.phase.clone()
        key = []
        for i in range(N):
            new = prev["ph"] is not None and int(ph[i]) != int(prev["ph"][i])
            if (new and int(ph[i]) in (LIFT, CARRY, RELEASE)) or t in (40, T - 1) or (t == 1 and int(ph[i]) == HOVER):
                key.append(i)
        prev["ph"] = ph
        if args.chunk and t % args.chunk == 0 and ctrl.snap is not None and len(shots[0]) < 14 and t % (2 * args.chunk) == 0:
            key = list(range(N))
        if key:
            ims = {n: grab(uu, n) for n in (CLOSEUP, "side_cam", "wrist_cam")}
            if args.chunk and ctrl.snap is not None:
                pts = torch.stack([tgt, ctrl.snap["cube_pos"][ar, ctrl.target]], 1)
                uvd = project(uu, CLOSEUP, pts)
                ims[CLOSEUP] = np.stack([ghost(ims[CLOSEUP][i], uvd[i, 0, :2], uvd[i, 1, :2]) for i in range(N)])
            for i in key:
                stale = f" snap@{ctrl.snap_step}" if args.chunk and ctrl.snap is not None else ""
                shots[i].append([label(ims[CLOSEUP][i], [f"ep {idx[i]} t={t} {PHASES[int(ph[i])]}{stale}"]),
                                 ims["side_cam"][i], ims["wrist_cam"][i]])

    res = run.run(eps, T, expert_kw=expert_kw, on_step=on_step)
    files = []
    for i in range(N):
        ok = bool(res["success"][i])
        f = out / f"{args.task}_s{args.seed}_ep{idx[i]}_{'success' if ok else 'FAIL'}{'_N' + str(args.chunk) if args.chunk else ''}.png"
        imageio.imwrite(f, grid(shots[i], h=200))
        files.append({"episode": idx[i], "success": ok, "drop": bool(res["drop"][i]),
                      "release_failure": bool(res["release_failure"][i]), "sheet": str(f.relative_to(ROOT)),
                      "spec": eps[i].spec.to_json()})
    print(json.dumps(files, indent=1))
    (out / f"frames_{args.task}_s{args.seed}{'_N' + str(args.chunk) if args.chunk else ''}.json").write_text(json.dumps(files, indent=1))
    env.close()
