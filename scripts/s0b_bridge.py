"""GATES 0B.2 bridge run on the 113 Stage 1 scene (one cube from SEEN_CUBES + 4 pads), driven by an UNTRAINED SmolVLA
(lerobot/smolvla_base, strict fp32 server, normalization stats from --dataset) through the policy server, which this
script spawns with the policy-venv python (server first: model load peaks in host RAM).

  powershell -File ops\\run_job.ps1 -Job g0b2_b16 -LogFile logs\\jobs\\g0b2_b16.log -PyArgs
      "scripts\\s0b_bridge.py --envs 16 --dataset datasets/0b_smoke --json results/0b2_bridge_b16.json --invariance
       --frames results/frames/0b2"
  python scripts/s0b_bridge.py --merge results/0b2_bridge_b1.json results/0b2_bridge_b16.json results/0b2_bridge_b64.json
      --json results/0b2_bridge.json

Episodes: --batches rounds of B envs; round b slot j = episode b * B + j of the s1 / train stream of --seed (31500),
T = T_pick (results/0a2_pick_timing.json), N / PERIOD / LATENCY from vla_common.timing, policy noise seeds
vla_tabletop.bridge.episode_seed(seed, i). Checks (GATES 0B.2): every response (B, N, 5) finite (bridge.act), executed
claw in {0, 1} and arm in [-1, 1]; queries only at t % N == 0; at each query the frame index sent = frames.py's rule and
the pixels sent = the frame captured at that tick (sha1); --invariance: the first 4 requests of round 0 re-served per
env alone (B = 1, fresh reset with the same noise seed) equal the batched actions within 1e-4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

INV_TOL = 1e-4  # GATES 0B.2 (2)


def pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(x), q)) if len(x) else None


def merge(paths: list[Path], out: Path) -> None:
    runs = [json.loads(p.read_text()) for p in paths]
    by_b = {r["B"]: r for r in runs}
    inv = [r["batch_invariance"] for r in runs if r["batch_invariance"]]
    checks = {"1_all_B_complete_shapes_finite_ranges": all(
                  r["complete"] and r["action_check_fail_steps"] == 0 for r in runs) and sorted(by_b) == [1, 16, 64],
              "2_batch_invariance_1e-4": bool(inv) and all(i["pass"] for i in inv),
              "3_query_steps_and_delayed_frames": all(r["query_steps_ok"] and r["frame_check_fail"] == 0 for r in runs)}
    keys = ("B", "episodes", "server_s_p50", "server_s_p95", "rtt_s_p50", "rtt_s_p95", "control_steps_per_s",
            "env_steps_per_s", "episodes_per_h", "rollout_wall_s", "peak_vram_torch_mib", "success")
    o = {"gate": "0B.2", "checks": checks, "pass": all(checks.values()), "batch_invariance": inv,
         "measured": {str(r["B"]): {k: r[k] for k in keys} for r in runs}, "sources": [str(p) for p in paths],
         "git_sha": runs[0]["git_sha"], "config": {k: runs[0][k] for k in ("seed", "stage", "split", "N", "period",
                                                                          "latency", "H", "T", "ckpt", "dataset")},
         "ckpt_sha": runs[0]["server_info"]["ckpt_sha"], "stats_sha": runs[0]["stats_sha"],
         "wall_s": sum(r["wall_s"] for r in runs)}
    out.write_text(json.dumps(o, indent=1))
    print(json.dumps({k: o[k] for k in ("checks", "pass")}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int)
    p.add_argument("--batches", type=int, default=2, help="rounds of B envs (GATES 0B.2: 2 episodes per env slot)")
    p.add_argument("--seed", type=int, default=31500)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--ckpt", default="lerobot/smolvla_base")
    p.add_argument("--dataset", default="datasets/0b_smoke", help="LeRobot dataset dir giving the normalization stats")
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--invariance", action="store_true")
    p.add_argument("--frames", default="", help="contact sheet dir (env 0 of round 0)")
    p.add_argument("--merge", nargs="+", type=Path)
    args = p.parse_args()
    if args.merge:
        return merge(args.merge, args.json)

    import vla_tabletop.bridge as BR
    from collect import geometry_113, git_sha, task_timing
    from eval_policy import free_port, stats_sha
    from vision_lift.frames import frame_index
    from vla_common import timing as TM

    B, H, N, PER, LAT = args.envs, args.image_size, TM.N, TM.PERIOD, TM.LATENCY
    tj = task_timing("pick")
    T = tj["T_steps"]
    job = args.json.stem
    t_start = time.time()
    srv = ["--policy", "smolvla", "--ckpt", args.ckpt, "--dataset", str((ROOT / args.dataset).resolve()),
           "--n-action-steps", str(N), "--period", str(PER), "--latency", str(LAT), "--device", "cuda"]
    bridge = BR.PolicyBridge({"policy": "smolvla", "n_action_steps": N, "period": PER, "latency": LAT, "image_size": H,
                              "state_dim": 19, "action_dim": 5, "chunk_size": 50}, srv, free_port(),
                             log_path=ROOT / "logs" / "jobs" / f"{job}_server.log")
    print(f"[0b2] server up after {bridge.load_s:.1f} s: {json.dumps(bridge.info)}", flush=True)

    import gymnasium as gym
    import torch

    import vla_tabletop.scene as S
    from isaaclab_tasks.utils import launch_simulation
    from lift_rl.physics import assert_on_cuda, assert_run_preconditions
    from vision_lift import cameras as C
    from vla_common.catalog import ALL_CUBES
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import Runner, make_stream

    cfg = S.make_cfg(B, args.seed, res=H)
    assert_run_preconditions(cfg)

    def h(a: np.ndarray) -> str:
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()

    with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
        env = gym.make(S.TASK_ID, cfg=cfg)
        u = env.unwrapped
        assert_on_cuda(env)
        assert torch.cuda.is_available() and str(u.device).startswith("cuda")
        print(f"[0b2] sim device {u.device} ({torch.cuda.get_device_name(0)}), B={B}, N={N}, PERIOD={PER}, "
              f"LATENCY={LAT}, H={H}, T={T}", flush=True)
        run = Runner(env, geometry_113())
        run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)
        cnt = {"frame_check_fail": 0, "action_check_fail_steps": 0}
        rounds, query_log, first_requests, sheet = [], [], [], {"side": [], "wrist": [], "marks": []}
        for b in range(args.batches):
            batch = make_stream(args.seed, b * B, B, lambda rng: sample_spec("s1", "train", rng), split="train")
            st = {"hashes": []}

            class Ctrl:
                def __init__(self, uu, eps, s0):
                    self.u = uu
                    self.ex = BR.ChunkExecutor(bridge, N, PER, LAT, [e.instruction for e in eps])
                    self.ex.reset(np.array([BR.episode_seed(args.seed, e.index) for e in eps], np.int64))

                def capture(self):
                    side = C.rgb(self.u, "side_cam").to(torch.uint8).cpu().numpy()
                    wrist = C.rgb(self.u, "wrist_cam").to(torch.uint8).cpu().numpy()
                    st["hashes"].append(h(side) + h(wrist))
                    if b == 0:
                        sheet["side"].append(side[0].copy())
                        sheet["wrist"].append(wrist[0].copy())
                    return side, wrist

                def act(self, t, snap_fn):
                    state = self.u.obs_buf["policy"].cpu().numpy()
                    a = self.ex.step(t, self.capture, state)
                    if t % N == 0:
                        q = self.ex.queries[-1]
                        k, (side, wrist) = self.ex.frames.current()
                        ok = (k == q["expected_frame_index"] == frame_index(t, PER, LAT)
                              and h(side) + h(wrist) == st["hashes"][k])
                        cnt["frame_check_fail"] += int(not ok)
                        query_log.append({"round": b, "t": t, "frame_index": k, "expected": frame_index(t, PER, LAT),
                                          "pixels_match_capture": h(side) + h(wrist) == st["hashes"][k],
                                          "server_s": round(q["server_s"], 4), "rtt_s": round(q["rtt_s"], 4)})
                        if b == 0 and len(first_requests) < 4:
                            first_requests.append({"side": side, "wrist": wrist, "state": state[:, :19].copy(),
                                                   "actions": self.ex.chunk.copy()})
                        if b == 0:
                            sheet["marks"].append((t, k, self.ex.chunk[0, 0].tolist()))
                    cnt["action_check_fail_steps"] += int(not (
                        a.shape == (B, 5) and np.isfinite(a).all() and np.abs(a[:, :4]).max() <= 1.0
                        and set(np.unique(a[:, 4]).tolist()) <= {0.0, 1.0}))
                    at = torch.from_numpy(a).to(self.u.device)
                    return at, at

            t0 = time.time()
            res = run.run(batch, T, controller=Ctrl)
            wall = time.time() - t0
            q = run.ctrl.ex.queries
            rounds.append({"round": b, "episodes": [e.index for e in batch], "instructions": [e.instruction for e in batch],
                           "wall_s": wall, "queries": len(q), "query_steps": [x["t"] for x in q],
                           "query_steps_ok": [x["t"] for x in q] == list(range(0, T, N)),
                           "server_s": [x["server_s"] for x in q], "rtt_s": [x["rtt_s"] for x in q],
                           "success": int(res["success"].sum()), "invalid": int(res["invalid"].sum())})
            print(f"[0b2] round {b}: {wall:.1f} s, {B * T / wall:.0f} env-steps/s, server p50 "
                  f"{pct(rounds[-1]['server_s'][1:], 50)} s, rtt p50 {pct(rounds[-1]['rtt_s'][1:], 50)} s, "
                  f"success {rounds[-1]['success']}/{B}", flush=True)

        inv = None
        if args.invariance:
            diffs = np.zeros((B, len(first_requests)))
            for i in range(B):
                bridge.reset(np.array([i], np.int32), np.array([BR.episode_seed(args.seed, i)], np.int64))
                for j, r in enumerate(first_requests):
                    a1, _, _ = bridge.act(np.array([i], np.int32), r["side"][i:i + 1], r["wrist"][i:i + 1],
                                          r["state"][i:i + 1], [rounds[0]["instructions"][i]])
                    diffs[i, j] = float(np.abs(a1[0] - r["actions"][i]).max())
            inv = {"B_batched": B, "envs": B, "requests": len(first_requests), "max_abs_diff": float(diffs.max()),
                   "max_abs_diff_per_request": diffs.max(0).tolist(), "tol": INV_TOL,
                   "pass": bool(diffs.max() <= INV_TOL)}
            print(f"[0b2] batch invariance: {inv}", flush=True)
        bridge.close()

        if args.frames:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            marks = sheet["marks"]
            sel = [marks[j] for j in np.linspace(0, len(marks) - 1, 9).round().astype(int)]
            fig, ax = plt.subplots(2, len(sel), figsize=(2 * len(sel), 4.8))
            for j, (t, k, a0) in enumerate(sel):
                for r, img in enumerate((sheet["side"][k], sheet["wrist"][k])):
                    ax[r, j].imshow(img, interpolation="nearest")
                    ax[r, j].axis("off")
                ax[0, j].set_title(f"query t={t}\nframe {k} (step {k * PER})\na0=" +
                                   ",".join(f"{v:.2f}" for v in a0), fontsize=6)
            fig.suptitle(f"0B.2 B={B} untrained SmolVLA, ep 0 '{rounds[0]['instructions'][0]}': side (top) / wrist "
                         f"(bottom) frame sent at each query", fontsize=8)
            fig.tight_layout()
            d = ROOT / args.frames
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{job}_ep0.png", dpi=110)

        srv_s = [s for r in rounds for s in r["server_s"][1:]]  # first query of each round = warm-up, excluded
        rtt = [s for r in rounds for s in r["rtt_s"][1:]]
        roll = sum(r["wall_s"] for r in rounds)
        out = {"gate": "0B.2", "job": job, "git_sha": git_sha(), "B": B, "batches": args.batches,
               "episodes": B * args.batches, "seed": args.seed, "stage": "s1", "split": "train", "N": N,
               "period": PER, "latency": LAT, "H": H, "T": T, "ckpt": args.ckpt, "dataset": args.dataset,
               "server_info": bridge.info, "stats_sha": stats_sha(args.ckpt, str(ROOT / args.dataset)),
               "server_load_s": bridge.load_s, "complete": len(rounds) == args.batches,
               "server_s_p50": pct(srv_s, 50), "server_s_p95": pct(srv_s, 95), "rtt_s_p50": pct(rtt, 50),
               "rtt_s_p95": pct(rtt, 95), "control_steps_per_s": T * args.batches / roll,
               "env_steps_per_s": B * T * args.batches / roll, "episodes_per_h": B * args.batches / roll * 3600,
               "rollout_wall_s": roll, "query_steps_ok": all(r["query_steps_ok"] for r in rounds), **cnt,
               "batch_invariance": inv, "success": sum(r["success"] for r in rounds),
               "invalid": sum(r["invalid"] for r in rounds), "rounds": rounds, "query_log": query_log,
               "peak_vram_torch_mib": torch.cuda.max_memory_allocated() / 2 ** 20, "device": str(u.device),
               "nan_guard_events": u.nan_guard_events, "wall_s": time.time() - t_start}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k not in ("rounds", "server_info", "query_log")}), flush=True)
        print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
        env.close()


if __name__ == "__main__":
    main()
