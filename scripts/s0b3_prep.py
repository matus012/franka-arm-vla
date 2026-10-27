"""GATES 0B.3 policy-venv pieces (PERUN container `pol_exec` or the laptop policy venv).

  python scripts/s0b3_prep.py prereq --dataset /data/datasets/0b3_overfit --out /runs/0b3_prereq
      -> actions.npz (the exported `action` column per demo, keyed by stream episode index; input of the replay
         ceiling, prerequisite a) + prereq_b.json (prerequisite b: normalization round trip + near-constant dims)
  python scripts/s0b3_prep.py offline --dataset /data/datasets/0b3_overfit --ckpt <pretrained_model> --out <json>
      -> offline action MAE (arm, normalized units) + claw accuracy on the 10 demos (reported, not gated)

(b) Every row of the dataset as the trainer's DataLoader returns it (action chunk of 50, padded past the end) goes
through the trainer's preprocessor (MEAN_STD stats re-estimated from this dataset by train_smolvla.norm_stats, the
stats the trainer uses) and back through its postprocessor; max |raw - round trip| must be <= 1e-5 and the claw
threshold (> 0.5) unchanged on every element; state / action dims with std < 1e-6 are listed (handled: std := 1).
Offline: the evaluated checkpoint served exactly as in the closed loop (vla_policy.server.SmolVLAServer, strict fp32,
noise seed 0 per demo) on the demo rows at the query steps t = 0, N, 2N, ... (the row = the delayed frame + state the
policy saw at t); its N returned actions vs the labels t .. t+N-1. MAE arm in units of the dataset std; claw
accuracy = served claw (thresholded) == label claw > 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_policy import train_smolvla as T  # noqa: E402

TOL = 1e-5  # GATES 0B.3 (b)


def stream_index(dpath: Path) -> dict[int, int]:
    """LeRobot episode index -> collector stream episode index (meta/episodes_113.jsonl)."""
    rows = [json.loads(x) for x in (dpath / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    return {int(r["lerobot_episode_index"]): int(r["episode_index"]) for r in rows}


def prereq(a: argparse.Namespace) -> None:
    dpath = T.resolve_dataset(a.dataset, None)
    ds = T.open_dataset(dpath, a.video_backend)
    a.out.mkdir(parents=True, exist_ok=True)
    ep = T.column(ds, "episode_index").reshape(-1)
    act = T.column(ds, "action").astype(np.float32)
    m = stream_index(dpath)
    np.savez(a.out / "actions.npz", **{str(m[e]): act[ep == e] for e in sorted(set(ep.tolist()))})

    stats, report = T.norm_stats(ds)
    _, pre, post, _ = T.build_policy(a.init, ds, stats, a.n_action_steps, True, True, a.device)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=a.workers, shuffle=False)
    err, claw_flip, n = 0.0, 0, 0
    for b in dl:
        raw = b["action"].clone()
        norm = pre(T.to_float_images(b))["action"]
        back = post(norm).cpu()
        err = max(err, float((back - raw).abs().max()))
        claw_flip += int(((back[..., 4] > 0.5) != (raw[..., 4] > 0.5)).sum())
        n += raw.shape[0]
    res = {"dataset": str(dpath), "rows": n, "max_abs_roundtrip_err": err, "tol": TOL, "claw_threshold_flips": claw_flip,
           "norm_report": report, "stats": {k: {s: v.tolist() for s, v in d.items() if s in ("mean", "std")}
                                            for k, d in stats.items()},
           "pass": err <= TOL and claw_flip == 0, "episodes_dumped": sorted(m.values()), "git_sha": T.git_sha()}
    (a.out / "prereq_b.json").write_text(json.dumps(res, indent=1))
    print("PREREQ_B=" + json.dumps({k: res[k] for k in ("rows", "max_abs_roundtrip_err", "claw_threshold_flips",
                                                         "norm_report", "pass")}), flush=True)


def offline(a: argparse.Namespace) -> None:
    from vla_policy.server import SmolVLAServer

    dpath = T.resolve_dataset(a.dataset, None)
    ds = T.open_dataset(dpath, a.video_backend)
    srv = SmolVLAServer(a.ckpt, a.n_action_steps, a.device)
    stats, _ = T.norm_stats(ds)
    std = stats["action"]["std"][:4]
    ep = T.column(ds, "episode_index").reshape(-1)
    m = stream_index(dpath)
    N = a.n_action_steps
    per = {}
    for e in (T.split_episodes(ds)[2] if a.val_only else sorted(set(ep.tolist()))):
        rows = np.flatnonzero(ep == e)
        labels = T.column(ds, "action")[rows].astype(np.float32)
        srv.reset({"env_ids": np.array([0], np.int32), "seeds": np.array([0], np.int64)})
        mae, claw, cnt = [], [], 0
        for t in range(0, len(rows), N):
            r = ds[int(rows[t])]
            msg = {"env_ids": np.array([0], np.int32), "instruction": [r["task"]],
                   "side": r["observation.images.side"].permute(1, 2, 0).numpy()[None],
                   "wrist": r["observation.images.wrist"].permute(1, 2, 0).numpy()[None],
                   "state": r["observation.state"].numpy()[None].astype(np.float32)}
            pred = srv.act(msg)[0]
            lab = labels[t:t + N]
            k = len(lab)
            mae.append(np.abs(pred[:k, :4] - lab[:, :4]) / std)
            claw.append(pred[:k, 4] == (lab[:, 4] > 0.5))
        per_q = [float(x.mean()) for x in mae]
        mae, claw = np.concatenate(mae), np.concatenate(claw)
        per[m[e]] = {"mae_arm_norm": float(mae.mean()), "mae_arm_norm_per_dim": mae.mean(0).tolist(),
                     "mae_arm_norm_per_query": per_q,
                     "claw_acc": float(claw.mean()), "steps": int(len(claw))}
    res = {"ckpt": a.ckpt, "ckpt_sha": srv.ckpt_sha, "dataset": str(dpath), "n_action_steps": N, "noise_seed": 0,
           "per_episode": per, "mae_arm_norm": float(np.mean([v["mae_arm_norm"] for v in per.values()])),
           "claw_acc": float(np.mean([v["claw_acc"] for v in per.values()])),
           "mae_arm_norm_per_query": np.mean([v["mae_arm_norm_per_query"] for v in per.values()], 0).tolist(),
           "git_sha": T.git_sha()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("OFFLINE=" + json.dumps({k: res[k] for k in ("mae_arm_norm", "claw_acc", "mae_arm_norm_per_query")}), flush=True)


def compare(a: argparse.Namespace) -> None:
    """Diagnosis (measure the design): the closed-loop rollouts dumped by `s0b3_overfit.py eval --dump` vs the demos.
    Per shown slot and query q (t = qN): runtime obs vs the demo row t (state in dataset std units, image PSNR), the
    served chunk vs the demo labels t..t+N-1 (arm MAE in std units, claw agreement); the chunk re-served offline at
    B = 1 from the runtime obs (same reset seed and query order -> must equal the closed-loop chunk) and from the demo
    obs at the same t (substitute the suspect input); first claw-close step, runtime vs demo."""
    from vla_policy.server import SmolVLAServer

    d = np.load(a.dump)
    dpath = T.resolve_dataset(a.dataset, None)
    ds = T.open_dataset(dpath, a.video_backend)
    srv = SmolVLAServer(a.ckpt, a.n_action_steps, a.device)
    stats, _ = T.norm_stats(ds)
    sstd, astd = stats["observation.state"]["std"], stats["action"]["std"][:4]
    ep = T.column(ds, "episode_index").reshape(-1)
    inv = {v: k for k, v in stream_index(dpath).items()}
    N, out = a.n_action_steps, []

    def psnr(x, y):
        m = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
        return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)

    def serve(seed, msgs):
        srv.reset({"env_ids": np.array([0], np.int32), "seeds": np.array([seed], np.int64)})
        return [srv.act(m)[0] for m in msgs]

    for i, (slot, k, ns) in enumerate(zip(d["slots"], d["states"], d["noise_seed"])):
        rows = np.flatnonzero(ep == inv[int(k)])
        lab = T.column(ds, "action")[rows].astype(np.float32)
        dst = T.column(ds, "observation.state")[rows].astype(np.float32)
        qs, rt_msgs, demo_msgs, demo_rows = [], [], [], []
        for q, t in enumerate(d["q_t"]):
            r = ds[int(rows[t])]
            dsd, dwr = (r[f"observation.images.{c}"].permute(1, 2, 0).numpy() for c in ("side", "wrist"))
            instr = r["task"]
            rt_msgs.append({"env_ids": np.array([0], np.int32), "instruction": [instr], "side": d["q_side"][q, i][None],
                            "wrist": d["q_wrist"][q, i][None], "state": d["q_state"][q, i][None]})
            demo_msgs.append({"env_ids": np.array([0], np.int32), "instruction": [instr], "side": dsd[None],
                              "wrist": dwr[None], "state": dst[t][None]})
            demo_rows.append((dsd, dwr))
        re_rt, re_demo = serve(int(ns), rt_msgs), serve(int(ns), demo_msgs)
        for q, t in enumerate(d["q_t"]):
            t = int(t)
            l = lab[t:t + N]
            ch = d["q_chunk"][q, i][:len(l)]
            qs.append({"t": t,
                       "state_dev_std_max": float(np.max(np.abs(d["q_state"][q, i] - dst[t]) / sstd)),
                       "psnr_side": psnr(d["q_side"][q, i], demo_rows[q][0]),
                       "psnr_wrist": psnr(d["q_wrist"][q, i], demo_rows[q][1]),
                       "chunk_vs_label_mae_std": float(np.mean(np.abs(ch[:, :4] - l[:, :4]) / astd)),
                       "chunk_claw_agree": float(np.mean(ch[:, 4] == (l[:, 4] > 0.5))),
                       "reserve_b1_eq_closed_loop_maxdiff": float(np.max(np.abs(re_rt[q][:len(l)] - ch))),
                       "demo_obs_chunk_vs_label_mae_std": float(np.mean(np.abs(re_demo[q][:len(l), :4] - l[:, :4]) / astd)),
                       "demo_obs_claw_agree": float(np.mean(re_demo[q][:len(l), 4] == (l[:, 4] > 0.5)))})
        cl_rt = np.flatnonzero(d["action"][i, :, 4] > 0.5)
        cl_demo = np.flatnonzero(lab[:, 4] > 0.5)
        dev = np.max(np.abs(d["state"][i] - dst[:d["state"].shape[1]]) / sstd, axis=1)
        out.append({"slot": int(slot), "state": int(k), "noise_seed": int(ns),
                    "first_close_runtime": int(cl_rt[0]) if cl_rt.size else None,
                    "first_close_demo": int(cl_demo[0]) if cl_demo.size else None,
                    "first_step_state_dev_gt_0p5std": int(np.argmax(dev > 0.5)) if (dev > 0.5).any() else None,
                    "first_step_state_dev_gt_2std": int(np.argmax(dev > 2)) if (dev > 2).any() else None,
                    "queries": qs})
        print(f"[compare] slot {slot} state {k} seed {ns}: close rt {out[-1]['first_close_runtime']} demo "
              f"{out[-1]['first_close_demo']}, dev>0.5std at {out[-1]['first_step_state_dev_gt_0p5std']}; q0 "
              f"{json.dumps(qs[0])}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"ckpt": a.ckpt, "dump": str(a.dump), "slots": out, "git_sha": T.git_sha()}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prereq", "offline", "compare"))
    p.add_argument("--dump", type=Path, default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--init", default="lerobot/smolvla_base")
    p.add_argument("--n-action-steps", type=int, default=20)
    p.add_argument("--video-backend", default="torchcodec")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--val-only", action="store_true", help="offline: only the dataset's val episodes (GATES Stage 1)")
    a = p.parse_args()
    assert torch.cuda.is_available() or a.device == "cpu", "CUDA unavailable. STOP."
    print(f"[s0b3_prep] {a.mode} device={a.device}", flush=True)
    {"prereq": prereq, "offline": offline, "compare": compare}[a.mode](a)


if __name__ == "__main__":
    main()
