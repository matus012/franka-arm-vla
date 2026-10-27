"""One-command demo (sim venv): an instruction + a seed -> an MP4 (and a JSON next to it).

  python scripts/demo.py --instruction "pick up the red cube" --seed 1 --policy expert --out videos/pick_red.mp4
  python scripts/demo.py --instruction "stack the blue cube on the small green cube" --seed 3 --policy smolvla \
      --ckpt <pretrained_model dir> --visual heldout --out videos/stack.mp4          (baseline: --policy baseline)

Scene: layouts are drawn from the seed's stream (episode j = default_rng([seed, j]): task-matched spec, layout); the
first layout in which the instruction names exactly one cube (and one destination: pad for place, other cube for
stack) is used, the spec's target / dest are set to those. Fails after --max-layouts. The baseline gets the 4 labels
derived from that spec. Policy noise seed = bridge.episode_seed(seed, j) (a different j gives a different noise stream).

Video (25 fps = one frame per 2 control steps): close-up scene camera (render-only, fixed aim at the work area, never
fed to the policy) | side | wrist (the delayed 128-px frames the policy sees, upscaled); instruction, step and phase
(expert) / grasp state (policy) overlaid; the outcome is stamped on a 1.5 s hold of the last frame. JSON: spec,
instruction, layout index, success + detector flags, config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FPS, STEP_EVERY = 25, 2  # 50 Hz control / 2
FONT_PATHS = ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf")


def resolve(instruction: str, seed: int, max_layouts: int, stage: str = "s4"):
    """-> (j, TaskSpec, layout dict): first layout j of the seed's stream in which `instruction` names one cube (+ one dest)."""
    from vla_common.tasks import TaskSpec, _parse, matches, matches_dest, sample_spec
    from vla_tabletop.layout import sample_layout

    if stage == "s1":  # Stage 1: one cube on the table, "the cube" is unique; layout j=0 of the seed (as eval_policy episode 0)
        rng = np.random.default_rng([seed, 0])
        spec = sample_spec("s1", "train", rng)
        lay, _ = sample_layout(spec, rng)
        return 0, spec, lay
    nps = _parse(instruction)
    assert nps and nps[0][2] == "cube", f"instruction must start with a cube noun phrase ('the <colour> cube'): {instruction!r}"
    task = "pick" if len(nps) == 1 else ("place" if nps[1][2] == "pad" else "stack")
    w = tuple(float(task == t) for t in ("pick", "place", "stack"))
    for j in range(max_layouts):
        rng = np.random.default_rng([seed, j])
        spec = sample_spec("s4", "train", rng, w)
        lay, _ = sample_layout(spec, rng)
        tg = matches(spec, instruction)
        if len(tg) != 1:
            continue
        dest = None
        if task != "pick":
            d = matches_dest(spec, instruction)
            if len(d) != 1 or (task == "stack" and d[0] == tg[0]):
                continue
            dest = d[0]
        return j, TaskSpec(task, tg[0], dest, spec.cubes, spec.pads), lay
    raise SystemExit(f"no layout among the first {max_layouts} of seed {seed} makes {instruction!r} name exactly one "
                     f"cube{'' if task == 'pick' else ' and one destination'} (the instruction is ambiguous or names "
                     "objects that never co-occur; add the size, e.g. 'the small red cube')")


def font(size: int):
    from PIL import ImageFont

    for f in FONT_PATHS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def banner(width: int, lines: list[tuple[str, int, tuple]]) -> np.ndarray:
    from PIL import Image, ImageDraw

    h = sum(int(s * 1.35) for _, s, _ in lines) + 10
    im = Image.new("RGB", (width, h), (20, 20, 24))
    dr = ImageDraw.Draw(im)
    y = 5
    for text, size, col in lines:
        dr.text((10, y), text, fill=col, font=font(size))
        y += int(size * 1.35)
    return np.asarray(im)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--policy", choices=("smolvla", "baseline", "expert"), required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--dataset", default="", help="dataset dir with the normalization stats (base weights only)")
    p.add_argument("--stage", choices=("s1", "s4"), default="s4", help="s1: single-cube scene (instruction 'pick up the cube')")
    p.add_argument("--visual", choices=("nominal", "train_rand", "heldout"), default="nominal")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n", type=int, default=0, help="query period (default vla_common.timing.N)")
    p.add_argument("--max-layouts", type=int, default=2000)
    p.add_argument("--policy-python", default=os.environ.get("VLA_POLICY_PY", ""))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert args.out.suffix == ".mp4"
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    from vla_common import timing as TM

    N = args.n or TM.N
    H = args.image_size
    j_ep, spec, layout = resolve(args.instruction, args.seed, args.max_layouts, args.stage)
    print(f"[demo] layout {j_ep} of seed {args.seed}: {spec.task} target={spec.target} dest={spec.dest} "
          f"cubes={spec.cubes}", flush=True)

    bridge = None
    if args.policy != "expert":
        import vla_tabletop.bridge as BR
        from eval_policy import Labelled, free_port

        if args.policy_python:
            BR.POLICY_PY = Path(args.policy_python)
        srv = ["--policy", args.policy, "--ckpt", args.ckpt, "--n-action-steps", str(N), "--period", str(TM.PERIOD),
               "--latency", str(TM.LATENCY), "--device", "cuda"]
        if args.dataset:
            srv += ["--dataset", str(Path(args.dataset).resolve())]
        bridge = BR.PolicyBridge({"policy": args.policy, "n_action_steps": N, "period": TM.PERIOD,
                                  "latency": TM.LATENCY, "image_size": H, "state_dim": 19, "action_dim": 5,
                                  "chunk_size": 50}, srv, free_port(), log_path=out.with_name(out.stem + "_server.log"))
        if args.policy == "baseline":
            bridge.client = Labelled(bridge.client)
        print(f"[demo] server up after {bridge.load_s:.1f} s", flush=True)

    import gymnasium as gym
    import imageio
    import torch

    import vla_tabletop.scene as S
    from collect import geometry_113, git_sha, task_timing, visual_params
    from eval_policy import RECORD_KEYS
    from isaaclab_tasks.utils import launch_simulation
    from lift_rl.physics import assert_on_cuda, assert_run_preconditions
    from vision_lift import cameras as C
    from vision_lift.frames import DelayedFrames
    from vla_common.catalog import ALL_CUBES, COLORS
    from vla_common.tasks import labels as task_labels
    from vla_tabletop import visual_rand as VR
    from vla_tabletop.bridge import ChunkExecutor, episode_seed
    from vla_tabletop.expert import PHASES
    from vla_tabletop.rollout import Episode, Runner
    from vla_tabletop.sheets import CLOSEUP, add_closeup, aim_closeup, fit, grab

    ep = Episode(j_ep, spec, args.instruction, layout)
    cfg = S.make_cfg(1, args.seed, res=H)
    add_closeup(cfg, 480, 360, focal=18.0)
    assert_run_preconditions(cfg)
    mode = {"nominal": None, "train_rand": "train", "heldout": "heldout"}[args.visual]
    tj = task_timing(spec.task)
    T = tj["T_steps"]
    label = {"expert": "scripted expert (0A.2, chunked)", "smolvla": "SmolVLA", "baseline": "modular baseline"}[args.policy]
    frames_v: list[np.ndarray] = []
    shown = {"w": 0}

    with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
        t_start = time.time()
        env = gym.make(S.TASK_ID, cfg=cfg)
        u = env.unwrapped
        assert_on_cuda(env)
        print(f"[demo] device={u.device} policy={args.policy} N={N} T={T} visual={args.visual}", flush=True)
        run = Runner(env, geometry_113())
        run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)
        vis = visual_params(mode, [j_ep], args.seed) if mode else None
        st = {"visual": False, "t": -1, "frames": None}
        view = DelayedFrames(TM.PERIOD, TM.LATENCY)
        tgt_i = ALL_CUBES.index(tuple(spec.target))

        def ensure_visual(uu) -> None:
            if not st["visual"]:
                if vis is not None:
                    VR.apply(uu, vis, torch.arange(1, device=uu.device))
                st["visual"] = True

        def frames(t: int, uu):
            ensure_visual(uu)
            if st["t"] != t:
                st["t"] = t
                st["frames"] = (C.rgb(uu, "side_cam").to(torch.uint8).cpu().numpy(),
                                C.rgb(uu, "wrist_cam").to(torch.uint8).cpu().numpy())
            return st["frames"]

        class PolicyCtrl:
            def __init__(self, uu, b, s0):
                self.u = uu
                self.ex = ChunkExecutor(bridge, N, TM.PERIOD, TM.LATENCY, [args.instruction])
                if args.policy == "baseline":
                    bridge.client.labels = task_labels(spec)[None].astype(np.int64)
                self.ex.reset(np.array([episode_seed(args.seed, j_ep)], np.int64))

            def act(self, t, snap_fn):
                a = self.ex.step(t, lambda: frames(t, self.u), self.u.obs_buf["policy"].cpu().numpy())
                a = torch.from_numpy(a).to(self.u.device)
                return a, a

        grasped = {"v": False}

        def on_step(t, uu, s, lab, exe, ctrl, det):
            if t == 0:
                view.reset()
                # fixed aim at the work area: midpoint of target and destination (initial positions), env frame
                pts = [s["cube_pos"][0, tgt_i]]
                if spec.task == "place":
                    pts.append(s["pad_pos"][0, COLORS.index(spec.dest)])
                elif spec.task == "stack":
                    pts.append(s["cube_pos"][0, ALL_CUBES.index(tuple(spec.dest))])
                mid = torch.stack(pts).mean(0, keepdim=True) + torch.tensor([0.0, 0.0, 0.07], device=uu.device)
                sep = float(torch.linalg.norm(pts[0] - pts[-1]))
                aim_closeup(uu, mid, dist=0.6 + 1.2 * sep)
            side, wrist = frames(t, uu)
            if t % TM.PERIOD == 0:
                view.capture(t, (side[0], wrist[0]))
            grasped["v"] |= bool(det.confirmed[0])
            if t % STEP_EVERY:
                return
            _, (vs, vw) = view.current()
            cu = grab(uu, CLOSEUP)[0]
            h = 360
            row = np.concatenate([fit(cu, h), np.full((h, 4, 3), 255, np.uint8), fit(vs, h), np.full((h, 4, 3), 255, np.uint8),
                                  fit(vw, h)], 1)
            phase = PHASES[int(ctrl.phase[0])].upper() if args.policy == "expert" else \
                ("grasp confirmed" if grasped["v"] else "approaching / grasping")
            bn = banner(row.shape[1], [
                (f"\"{args.instruction}\"", 30, (255, 255, 255)),
                (f"{label}   |   {spec.task}   |   seed {args.seed}   |   visual: {args.visual}", 18, (170, 190, 220)),
                (f"t = {t * uu.step_dt:5.2f} s  (step {t + 1}/{T})   |   {phase}", 20, (255, 215, 90))])
            frames_v.append(np.concatenate([bn, row], 0))
            shown["w"] = row.shape[1]

        ekw = {"fresh": False, "n_query": N, "period": TM.PERIOD, "latency": TM.LATENCY, "sigma": 0.0}
        res = run.run([ep], T, expert_kw=ekw, on_step=on_step, controller=None if args.policy == "expert" else PolicyCtrl)
        wall = time.time() - t_start
        rec = {k: res[k][0].item() for k in RECORD_KEYS}
        if bridge is not None:
            bridge.close()
        ok = bool(rec["success"])
        last = frames_v[-1].copy()
        from PIL import Image, ImageDraw

        im = Image.fromarray(last)
        dr = ImageDraw.Draw(im, "RGBA")
        txt = "SUCCESS" if ok else "FAILED"
        dr.rectangle([0, last.shape[0] - 70, last.shape[1], last.shape[0]], fill=(0, 0, 0, 170))
        dr.text((12, last.shape[0] - 64), txt, fill=(80, 255, 120) if ok else (255, 90, 80), font=font(48))
        frames_v += [np.asarray(im)] * int(1.5 * FPS)
        # H.264 needs even sizes
        frames_v = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames_v]
        imageio.mimwrite(out, frames_v, fps=FPS, macro_block_size=1)
        det_flags = {k: rec[k] for k in RECORD_KEYS}
        out.with_suffix(".json").write_text(json.dumps({
            "instruction": args.instruction, "seed": args.seed, "layout_index": j_ep, "policy": args.policy,
            "ckpt": args.ckpt, "visual": args.visual, "image_size": H, "n": N, "T": T, "spec": json.loads(spec.to_json()),
            "success": ok, "detectors": det_flags, "frames": len(frames_v), "fps": FPS, "video": str(out),
            "wall_s": round(wall, 1), "git_sha": git_sha()}, indent=1))
        print(f"[demo] {txt} ({spec.task}) -> {out} ({len(frames_v)} frames, {wall:.0f} s)", flush=True)
        print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
        env.close()


if __name__ == "__main__":
    main()
