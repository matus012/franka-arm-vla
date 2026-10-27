"""GATES 0B.3 closed loop on the 10 overfit states (sim venv; PERUN `sim_exec`, spawns the policy server with
/opt/policy/bin/python like scripts/eval_policy.py).

  python scripts/s0b3_overfit.py replay --actions /runs/0b3_prereq/actions.npz --out /runs/0b3_replay/replay.json
  python scripts/s0b3_overfit.py eval --ckpt <pretrained_model> --demo /runs/0b3_replay/replay.json \
      --out /runs/0b3_eval/0b3_overfit.json --policy-python /opt/policy/bin/python

States: episodes 0-9 of seed 31600 (Stage 1 sampler, train split) = the demos of `hpc/steps/0b3_collect.txt`, same
make_cfg(B, 31600, res=128), T_pick, detectors at T_pick, through the bridge client's ChunkExecutor (query every N
control steps with the delayed frame, N actions open loop, arm clipped, claw > 0.5).
- replay (prerequisite a): B = 10 slots = episodes 0-9 in slot order, exactly the collection layout (the GPU solver is
  not bit-reproducible across batch compositions); the "server" is in-process and returns the exported `action`
  column of the slot's demo, rows qN .. qN+N-1 at query q. Gate: 10 / 10 success.
- eval (the gate): B = 128 slots (protocol B); slot v holds state v % 10 with policy noise seed v // 10 (sent as the
  reset seed, server-side generator per (seed, query)); slots 0-99 = the 10 states x noise seeds 0-9 are scored,
  slots 100-127 (noise seeds 10-12) run and are discarded (protocol: a round always starts all B slots). >= 80 / 100.
Frames: a close-up camera (rendering only) is aimed at the target cube; for slots 0-9 (noise seed 0 in eval) the
close-up | side | wrist images at the first grasp confirmation and at the last camera tick are kept; replay writes
them to <out stem>_frames.npz, eval reads the replay's to draw rollout vs demo per state -> --frames-dir.
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

SEED, N_STATES, N_NOISE, B_EVAL, BAR = 31600, 10, 10, 128, 80  # GATES 0B.3
KEYS = ("success", "slam", "slam_10N", "slam_50N", "jam", "early_close", "drop", "grasp_confirmed", "clean",
        "invalid", "completion_step", "max_hand_table_or_pad_force", "max_jam_run")


class ReplayBridge:
    """In-process stand-in for PolicyBridge: query q of slot j returns actions[j][qN:(q+1)N] (last row repeated)."""

    def __init__(self, actions: list[np.ndarray], n: int):
        self.a, self.n, self.q = actions, n, 0
        self.info = {"ckpt_sha": None}

    def reset(self, env_ids, seeds) -> None:
        self.q = 0

    def act(self, env_ids, side, wrist, state, instruction):
        lo = self.q * self.n
        out = np.stack([np.concatenate([a, np.repeat(a[-1:], self.n, 0)])[lo:lo + self.n] for a in self.a])
        self.q += 1
        return out.astype(np.float32), 0.0, 0.0


class HybridBridge:
    """Diagnosis (substitute the suspect input): the real server answers every query (its noise query counter advances
    as in the gate run), but for queries q < k each slot's chunk is replaced by its demo's labels (ReplayBridge)."""

    def __init__(self, bridge, demo: ReplayBridge, k: int):
        self.b, self.demo, self.k, self.q = bridge, demo, k, 0
        self.info = bridge.info

    def reset(self, env_ids, seeds) -> None:
        self.q = 0
        self.b.reset(env_ids, seeds)
        self.demo.reset(env_ids, seeds)

    def act(self, *args):
        a, s, r = self.b.act(*args)
        d = self.demo.act(*args)[0]
        self.q += 1
        return (d if self.q <= self.k else a), s, r

    def close(self) -> None:
        self.b.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("replay", "eval"))
    p.add_argument("--actions", type=Path, help="replay: actions.npz from scripts/s0b3_prep.py prereq")
    p.add_argument("--ckpt", default="", help="eval: the final-step pretrained_model dir")
    p.add_argument("--demo", type=Path, help="eval: the replay JSON (its _frames.npz = demo frames)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--policy-python", default=os.environ.get("VLA_POLICY_PY", ""))
    p.add_argument("--demo-queries", type=int, default=0,
                   help="diagnosis: the first k queries of every slot execute its demo's labels (needs --actions)")
    p.add_argument("--show", type=int, default=N_STATES, help="slots 0..show-1 get close-up frames (+ --dump)")
    p.add_argument("--dump", type=Path, default=None,
                   help="diagnosis: npz of the shown slots' per-step state + executed action and per-query frames / chunk")
    a = p.parse_args()

    from vla_common import timing as TM
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import make_stream

    N, H = TM.N, a.image_size
    states = make_stream(SEED, 0, N_STATES, lambda rng: sample_spec("s1", "train", rng), split="train")
    if a.mode == "replay":
        npz = np.load(a.actions)
        assert sorted(int(k) for k in npz.files) == list(range(N_STATES)), f"demos {npz.files} != episodes 0-9"
        B, slots, nseed = N_STATES, list(range(N_STATES)), [0] * N_STATES
        bridge = ReplayBridge([npz[str(i)] for i in slots], N)
    else:
        import vla_tabletop.bridge as BR
        from eval_policy import free_port

        if a.policy_python:
            BR.POLICY_PY = Path(a.policy_python)
        B, slots, nseed = B_EVAL, [v % N_STATES for v in range(B_EVAL)], [v // N_STATES for v in range(B_EVAL)]
        srv = ["--policy", "smolvla", "--ckpt", a.ckpt, "--n-action-steps", str(N), "--period", str(TM.PERIOD),
               "--latency", str(TM.LATENCY), "--device", "cuda"]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        bridge = BR.PolicyBridge({"policy": "smolvla", "n_action_steps": N, "period": TM.PERIOD,
                                  "latency": TM.LATENCY, "image_size": H, "state_dim": 19, "action_dim": 5,
                                  "chunk_size": 50}, srv, free_port(), log_path=a.out.with_name(a.out.stem + "_server.log"))
        print(f"[0b3] server up after {bridge.load_s:.1f} s: {json.dumps(bridge.info)}", flush=True)
        if a.demo_queries:
            npz = np.load(a.actions)
            bridge = HybridBridge(bridge, ReplayBridge([npz[str(k)] for k in slots], N), a.demo_queries)

    import gymnasium as gym
    import imageio
    import torch

    import vla_tabletop.scene as S
    from collect import geometry_113, git_sha, task_timing
    from eval_policy import stats_sha
    from isaaclab_tasks.utils import launch_simulation
    from lift_rl.physics import assert_on_cuda, assert_run_preconditions
    from vision_lift import cameras as C
    from vla_common.catalog import ALL_CUBES
    from vla_tabletop.bridge import ChunkExecutor
    from vla_tabletop.rollout import Runner
    from vla_tabletop.sheets import CLOSEUP, add_closeup, aim_closeup, grab, grid, label

    cfg = S.make_cfg(B, SEED, res=H)
    add_closeup(cfg, 320, 240, focal=18.0)
    assert_run_preconditions(cfg)
    tj = task_timing("pick")
    T = tj["T_steps"]
    last_tick = (T - 1) // TM.PERIOD * TM.PERIOD
    batch = [states[k] for k in slots]
    shown = list(range(a.show if a.mode == "eval" else N_STATES))  # slots with frames (eval slot j: state j % 10, seed j // 10)
    dump = {"state": np.zeros((len(shown), T, 19), np.float32), "action": np.zeros((len(shown), T, 5), np.float32),
            "q_t": [], "q_side": [], "q_wrist": [], "q_state": [], "q_chunk": [], "q_frame_index": []}

    with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
        t_start = time.time()
        env = gym.make(S.TASK_ID, cfg=cfg)
        u = env.unwrapped
        assert_on_cuda(env)
        print(f"[0b3] {a.mode}: device={u.device} B={B} H={H} N={N} T={T}", flush=True)
        run = Runner(env, geometry_113())
        run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)
        st = {"t": -1, "frames": None, "grasp": np.zeros(B, bool), "ev": {j: {} for j in shown}}

        def frames(t: int, uu):
            if st["t"] != t:
                st["t"] = t
                st["frames"] = (C.rgb(uu, "side_cam").to(torch.uint8).cpu().numpy(),
                                C.rgb(uu, "wrist_cam").to(torch.uint8).cpu().numpy())
            return st["frames"]

        class Ctrl:
            def __init__(self, uu, b, s0):
                self.u = uu
                self.ex = ChunkExecutor(bridge, N, TM.PERIOD, TM.LATENCY, [e.instruction for e in b])
                self.ex.reset(np.array(nseed, np.int64))

            def act(self, t, snap_fn):
                obs = self.u.obs_buf["policy"].cpu().numpy()
                x = self.ex.step(t, lambda: frames(t, self.u), obs)
                dump["state"][:, t], dump["action"][:, t] = obs[shown, :19], x[shown]
                if t % N == 0:
                    k, (sd, wr) = self.ex.frames.current()
                    for key, v in (("q_t", t), ("q_side", sd[shown]), ("q_wrist", wr[shown]), ("q_state", obs[shown, :19]),
                                   ("q_chunk", self.ex.chunk[shown]), ("q_frame_index", k)):
                        dump[key].append(v)
                x = torch.from_numpy(x).to(self.u.device)
                return x, x

        def on_step(t, uu, s, label_, exe, ctrl, det):
            if t % TM.PERIOD:
                return
            side, wrist = frames(t, uu)
            ar = torch.arange(B, device=uu.device)
            aim_closeup(uu, s["cube_pos"][ar, det.target] + torch.tensor([0.0, 0.0, 0.03], device=uu.device), dist=0.45)
            conf = det.confirmed.cpu().numpy()
            new = conf & ~st["grasp"]
            st["grasp"] |= conf
            hits = [("grasp", j) for j in shown if new[j]] + ([("end", j) for j in shown] if t == last_tick else [])
            if hits:
                cu = grab(uu, CLOSEUP)
                for name, j in hits:
                    st["ev"][j][name] = (t, cu[j], side[j], wrist[j])

        res = run.run(batch, T, on_step=on_step, controller=Ctrl)
        wall = time.time() - t_start
        q = run.ctrl.ex.queries
        assert all(x["frame_index"] == x["expected_frame_index"] for x in q), "delayed-frame check failed"
        recs = [{"slot": v, "state": k, "noise_seed": nseed[v], "instruction": states[k].instruction,
                 "target": "_".join(states[k].spec.target), **{x: res[x][v].item() for x in KEYS}}
                for v, k in enumerate(slots)]
        scored = recs if a.mode == "replay" else [r for r in recs if r["noise_seed"] < N_NOISE]
        per_state = {k: int(sum(r["success"] for r in scored if r["state"] == k)) for k in range(N_STATES)}
        n_succ = int(sum(r["success"] for r in scored))
        from collect import wilson

        out = {"gate": "0B.3", "mode": a.mode, "seed": SEED, "states": list(range(N_STATES)), "B": B, "N": N, "H": H,
               "T_steps": T, "period": TM.PERIOD, "latency": TM.LATENCY,
               "layout": "replay: slot j = episode j (collection layout)" if a.mode == "replay" else
               "slot v = state v % 10, noise seed v // 10; slots 0-99 scored, 100-127 run and discarded",
               "n_scored": len(scored), "success": n_succ, "success_rate": n_succ / len(scored),
               "ci95": wilson(n_succ, len(scored)), "per_state_success": per_state,
               **{f"n_{x}": int(sum(r[x] for r in scored)) for x in ("slam", "slam_10N", "slam_50N", "jam",
                                                                     "early_close", "drop", "invalid")},
               "bar": BAR if a.mode == "eval" else N_STATES,
               "pass": n_succ >= (BAR if a.mode == "eval" else N_STATES),
               "ckpt": a.ckpt or None, "ckpt_sha": bridge.info["ckpt_sha"],
               "stats_sha": stats_sha(a.ckpt, "") if a.ckpt else None,
               "server_s_p50": float(np.median([x["server_s"] for x in q[1:]])) if a.mode == "eval" else None,
               "server_s_p95": float(np.percentile([x["server_s"] for x in q[1:]], 95)) if a.mode == "eval" else None,
               "queries": len(q), "wall_s": round(wall, 1), "gpu_h": wall / 3600, "git_sha": git_sha(),
               "gpu": torch.cuda.get_device_name(0), "nan_guard_events": u.nan_guard_events,
               "frames_note": "close-up camera rendered in this run (rendering only, no physics change)",
               "demo_queries": a.demo_queries,
               "episodes": recs}
        if a.mode == "eval":
            bridge.close()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        fr = {f"{j}_{name}_{c}": img for j in shown for name, (t, *ims) in st["ev"][j].items()
              for c, img in zip(("closeup", "side", "wrist"), ims)}
        fr |= {f"{j}_{name}_t": np.array(t) for j in shown for name, (t, *_) in st["ev"][j].items()}
        np.savez_compressed(a.out.with_name(a.out.stem + "_frames.npz"), **fr)
        if a.dump:
            np.savez_compressed(a.dump, slots=np.array(shown), states=np.array([slots[j] for j in shown]),
                                noise_seed=np.array([nseed[j] for j in shown]),
                                **{k: np.asarray(v) for k, v in dump.items()})
        if a.mode == "eval" and a.frames_dir:
            demo = np.load(a.demo.with_name(a.demo.stem + "_frames.npz"))
            a.frames_dir.mkdir(parents=True, exist_ok=True)
            blank = np.full((240, 320, 3), 255, np.uint8)
            for j in shown:
                k, sd = slots[j], nseed[j]
                rows = []
                for name in ("grasp", "end"):
                    def cells(src, i, who, tag):
                        if f"{i}_{name}_closeup" not in src:
                            return [label(blank, [f"state {k} {who}: no {name} event"]), blank[:128, :128],
                                    blank[:128, :128]]
                        return [label(src[f"{i}_{name}_closeup"], [f"state {k} {who} {name} t={int(src[f'{i}_{name}_t'])}",
                                                                     tag]),
                                src[f"{i}_{name}_side"], src[f"{i}_{name}_wrist"]]
                    r = recs[j]
                    rows.append(cells(demo, k, "demo (replay)", states[k].instruction[:40]) +
                                cells(fr, j, f"rollout seed {sd}", f"success={r['success']} slam={r['slam']} jam={r['jam']} "
                                      f"early_close={r['early_close']}"))
                imageio.imwrite(a.frames_dir / (f"0b3_state{k}" + (f"_seed{sd}" if sd else "") + ".png"), grid(rows, h=200))
        print("S0B3=" + json.dumps({k: out[k] for k in ("mode", "n_scored", "success", "per_state_success", "n_slam",
                                                         "n_jam", "n_early_close", "pass", "wall_s")}), flush=True)
        print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
        env.close()


if __name__ == "__main__":
    main()
