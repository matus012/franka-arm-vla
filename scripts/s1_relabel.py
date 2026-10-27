"""Stage 1 fix diagnosis D2 (validation only): relabel the sigma-0.1 demos without the expert's deadbeat cancellation of
the previous step's executed noise.

Definition (arm dims 0-3; claw unchanged; r_0 = a_0):
    r_t = clip(a_t + (x_{t-1} - a_{t-1}), -1, 1)
a = stored clean label, x = stored executed action (action_exec), so x_{t-1} - a_{t-1} = the effective (post-clip)
executed noise of the previous step. Why this one: the stored label is a_t = (setpoint_t - cmd_t) / STEP where the
action term's commanded target cmd_t = cmd_{t-1} + x_{t-1} * STEP already contains last step's noise; r_t is exactly the
action the expert would have issued at t had step t-1 executed its own label (cmd_t - noise * STEP), i.e. its setpoint
step setpoint_t - setpoint_{t-1} when no clip binds. It is a contract action (task-space delta in [-1, 1] + binary
claw, the same space and units as the stored label), its running sum is the setpoint path, so executed open loop at
sigma 0 it drives the commanded target along the demo's intended path (within one noise step of the demo's executed
command), and it is computable from stored data only. The alternative "setpoint relative to the observed TCP" is an
absolute position error, not a delta of the contract action space (and carries the IK tracking lag).

  python scripts/s1_relabel.py stats --src /data/0c/s1/b000 /data/c3r1_demos --out /runs/d2fix/stats.json   (any venv)
  python scripts/s1_relabel.py build --dataset /data/datasets/0c_s1_pick --out /runs/d2fix/relabel.npz      (policy)
  python scripts/s1_relabel.py replay --stages s1:21600 s2:21620 --episodes 64 --envs 64 --out r.json        (sim)
  python scripts/s1_relabel.py dscheck --dataset /data/datasets/0c_s1_pick_rl ... --expect-original /data/datasets/0c_s1_pick --out c.json  (policy)
    -> passes on the same validation episodes / batch: A noisy expert sigma 0.1 (records a, x); B relabel r, C executed
       x, D stored labels a, each open loop from the same initial states at sigma 0; E clean expert (sigma 0).
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


def relabel(action: np.ndarray, action_exec: np.ndarray) -> np.ndarray:
    """(T, 5) stored label + executed -> (T, 5) relabel (module docstring)."""
    r = action.copy()
    r[1:, :4] = np.clip(action[1:, :4] + (action_exec[:-1, :4] - action[:-1, :4]), -1.0, 1.0)
    return r


def label_stats(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """s1_diag_labels statistics of a label stream vs the previous step's executed noise n_{t-1} = x_{t-1} - a_{t-1}
    (a = the ORIGINAL label in both cases): diff slope / R^2, level slope, std of the label and of its first diff."""
    L, D, N = [], [], []
    for lab, act, exe in pairs:
        L.append(lab[1:, :4])
        D.append(lab[1:, :4] - lab[:-1, :4])
        N.append(exe[:-1, :4] - act[:-1, :4])
    L, D, N = (np.concatenate(x).astype(np.float64) for x in (L, D, N))
    nn = np.maximum((N * N).sum(0), 1e-12)
    slope = (D * N).sum(0) / nn
    r2 = 1 - ((D - slope * N) ** 2).sum(0) / np.maximum(((D - D.mean(0)) ** 2).sum(0), 1e-12)
    lslope = ((L - L.mean(0)) * N).sum(0) / nn
    return {"rows": len(L), "std_label": L.std(0).tolist(), "std_label_diff": D.std(0).tolist(),
            "slope_diff_on_prev_noise": slope.tolist(), "r2": r2.tolist(),
            "slope_level_on_prev_noise": lslope.tolist(), "std_noise": N.std(0).tolist()}


def stats(a: argparse.Namespace) -> None:
    from s1_diag_labels import episodes

    res = {}
    for src in a.src:
        eps = list(episodes(src))
        res[str(src)] = {"episodes": len(eps),
                         "original": label_stats([(act, act, exe) for act, exe in eps]),
                         "relabel": label_stats([(relabel(act, exe), act, exe) for act, exe in eps])}
        print(src, json.dumps(res[str(src)]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


def build(a: argparse.Namespace) -> None:
    """Relabelled action per dataset row (aligned with the LeRobot `index`), from each episode's source HDF5 group
    (meta/episodes_113.jsonl); the stored label must equal the dataset's action column bit for bit."""
    import h5py

    from vla_policy import train_smolvla as T

    d = T.resolve_dataset(str(a.dataset[0]), None)
    ds = T.open_dataset(d, "torchcodec")
    col = T.column(ds, "action").astype(np.float32).reshape(len(ds), 5)
    ep = T.column(ds, "episode_index").reshape(-1)
    idx = T.column(ds, "index").reshape(-1)
    assert np.array_equal(idx, np.arange(len(ds))), "dataset rows are not in index order"
    side = [json.loads(x) for x in (d / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    R = np.full_like(col, np.nan)
    for s in side:
        rows = np.flatnonzero(ep == s["lerobot_episode_index"])
        with h5py.File(s["source"], "r") as h:
            g = h[s["group"]]
            act, exe = g["action"][()].astype(np.float32), g["action_exec"][()].astype(np.float32)
        assert len(rows) == len(act) == s["n_rows"], (s["lerobot_episode_index"], len(rows), len(act))
        assert np.array_equal(col[rows], act), f"episode {s['lerobot_episode_index']}: label != dataset action"
        R[rows] = relabel(act, exe)
    assert np.isfinite(R).all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, action=R, dataset=str(d), rows=len(ds))
    print(f"[relabel] {len(side)} episodes, {len(ds)} rows -> {a.out}; mean |r - a| arm "
          f"{np.abs(R[:, :4] - col[:, :4]).mean(0).round(4).tolist()}", flush=True)


class OpenLoop:
    """Runner controller: executes a fixed (B, T, 5) action sequence (arm in [-1, 1], binary claw) open loop."""

    def __init__(self, acts):
        self.a = acts

    def act(self, t: int, snap_fn):
        return self.a[:, t], self.a[:, t]


def replay(a: argparse.Namespace) -> None:
    """Executability per stage on validation streams (--stages s1:21600 s2:21620 ...): one env (a.envs slots), per stage
    the passes A-E of the module docstring. Stage task = the task of its stream (all episodes of a 0C set share one)."""
    import time

    import torch

    import vla_tabletop.scene as S
    from collect import NoisyExpert, geometry_113, git_sha, noise_table, task_timing
    from isaaclab_tasks.utils import launch_simulation
    from lift_rl.physics import assert_on_cuda, assert_run_preconditions
    from vla_common.catalog import ALL_CUBES
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import Runner, make_stream

    import gymnasium as gym

    B = a.envs
    pairs = [(x.split(":")[0], int(x.split(":")[1])) for x in a.stages]
    assert all(21600 <= sd <= 21999 for _, sd in pairs), "validation seeds only"
    cfg = S.make_cfg(B, pairs[0][1], res=None)
    assert_run_preconditions(cfg)
    geo = geometry_113()
    result: dict = {"envs": B, "episodes": a.episodes, "sigma": a.sigma, "cube_yaw_max_deg": a.cube_yaw_max_deg, "git_sha": git_sha(),
              "stages": {}}
    with launch_simulation(cfg, {"headless": True, "enable_cameras": False}):
        t00 = time.time()
        env = gym.make(S.TASK_ID, cfg=cfg)
        u = env.unwrapped
        assert_on_cuda(env)
        print(f"[replay] device={u.device} B={B} stages {pairs}", flush=True)
        run = Runner(env, geo)
        for stage, seed in pairs:
            t0 = time.time()
            eps = make_stream(seed, 0, a.episodes, lambda rng: sample_spec(stage, "train", rng), split="train",
                              cube_yaw_max_deg=a.cube_yaw_max_deg)
            task = eps[0].spec.task
            assert all(e.spec.task == task for e in eps)
            tj = task_timing(task)
            T = tj["T_steps"]
            run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)
            out: dict = {"stage": stage, "task": task, "seed": seed, "episodes": a.episodes, "T": T}
            pad = [e.index for e in eps] + [eps[-1].index] * (-len(eps) % B)
            noise = noise_table(seed, pad, T, a.sigma)
            rec = {"a": np.empty((len(pad), T, 5), np.float32), "x": np.empty((len(pad), T, 5), np.float32)}
            off = {"v": 0}

            def on_step(t, uu, s, label, exe, ctrl, det):
                rec["a"][off["v"]:off["v"] + B, t] = label.cpu().numpy()
                rec["x"][off["v"]:off["v"] + B, t] = exe.cpu().numpy()

            def noisy(uu, b, s0):
                return NoisyExpert(uu, b, s0, geo, noise[off["v"]:off["v"] + B])

            def summarize(res: dict) -> dict:
                k = {f: res[f].numpy().astype(bool) for f in ("success", "slam", "jam", "drop", "early_close")}
                return {f: int(v.sum()) for f, v in k.items()} | {"n": len(k["success"]),
                                                                  "success_per_ep": k["success"].tolist()}

            resA = []  # pass A: noisy expert (records a, x), batch by batch so the noise rows follow the batch
            for b0 in range(0, len(eps), B):
                off["v"] = b0
                resA.append(run.run(eps[b0:b0 + B], T, on_step=on_step, controller=noisy))
            out["A_noisy_expert"] = summarize({k: torch.cat([r[k] for r in resA]) for k in resA[0]})
            R = np.stack([relabel(ai, xi) for ai, xi in zip(rec["a"], rec["x"])])
            seqs = {"B_relabel_open_loop": R, "C_executed_open_loop": rec["x"], "D_labels_open_loop": rec["a"]}
            for name, arr in seqs.items():
                parts = []
                for b0 in range(0, len(eps), B):
                    acts = torch.as_tensor(arr[b0:b0 + B], device=u.device)
                    parts.append(run.run(eps[b0:b0 + B], T, controller=lambda uu, b, s0, A_=acts: OpenLoop(A_)))
                out[name] = summarize({k: torch.cat([p[k] for p in parts]) for k in parts[0]})
            clean = np.empty((len(pad), T, 5), np.float32)
            partsE = []
            for b0 in range(0, len(eps), B):
                def on_clean(t, uu, s, label, exe, ctrl, det, b0=b0):
                    clean[b0:b0 + B, t] = label.cpu().numpy()
                partsE.append(run.run(eps[b0:b0 + B], T, on_step=on_clean))
            out["E_clean_expert"] = summarize({k: torch.cat([p[k] for p in partsE]) for k in partsE[0]})
            n = len(eps)
            out["label_stats"] = {"A_labels": label_stats([(a_, a_, x_) for a_, x_ in zip(rec["a"][:n], rec["x"][:n])]),
                                  "A_relabel": label_stats([(r_, a_, x_) for r_, a_, x_ in
                                                            zip(R[:n], rec["a"][:n], rec["x"][:n])]),
                                  "E_clean_labels": label_stats([(c_, c_, c_) for c_ in clean[:n]])}
            demo_ok = np.array(out["A_noisy_expert"]["success_per_ep"])
            for name in (*seqs, "E_clean_expert"):
                sx = np.array(out[name]["success_per_ep"])
                out[name]["success_rate"] = float(sx.mean())
                out[name]["success_on_demo_successes"] = [int(sx[demo_ok].sum()), int(demo_ok.sum())]
            out["wall_s"] = time.time() - t0
            result["stages"][stage] = out
            print(f"REPLAY[{stage}]=" + json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "success_per_ep"}
                                                         if isinstance(v, dict) else v) for k, v in out.items()
                                                    if k != "label_stats"}), flush=True)
        result["wall_s"] = time.time() - t00
        env.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))


SLOPE_MAX = 0.1  # GATES 0C attempt 1 item 4(c): |slope of the label diff on the previous executed noise| < 0.1


def relabel_shift(action: np.ndarray, action_exec: np.ndarray) -> np.ndarray:
    """Mutant m-b: this step's noise x_t - a_t instead of the previous step's."""
    r = action.copy()
    r[1:, :4] = np.clip(action[1:, :4] + (action_exec[1:, :4] - action[1:, :4]), -1.0, 1.0)
    return r


def relabel_claw(action: np.ndarray, action_exec: np.ndarray) -> np.ndarray:
    """Mutant m-c as written: the relabel applied to all 5 dims (claw included)."""
    r = action.copy()
    r[1:] = np.clip(action[1:] + (action_exec[:-1] - action[:-1]), -1.0, 1.0)
    return r


def relabel_claw_offset(action: np.ndarray, action_exec: np.ndarray) -> np.ndarray:
    """Mutant m-c' (non-equivalent claw mutant, used only if m-c changes no row): the arm correction written one dim
    to the right (dims 1-4), so the dyaw correction lands on the claw."""
    r = action.copy()
    r[1:, 1:] = np.clip(action[1:, 1:] + (action_exec[:-1, :4] - action[:-1, :4]), -1.0, 1.0)
    return r


MUTANTS = {"shift": relabel_shift, "claw": relabel_claw, "claw_offset": relabel_claw_offset}


def dscheck(a: argparse.Namespace) -> None:
    """Per exported dataset, from its sidecar's HDF5 sources: (i) identity (item 4(b)): the LeRobot `action` column ==
    relabel(HDF5 action, action_exec) on EVERY row (bitwise float32), claw column == stored claw, first row of every
    episode == the clean label; (ii) label statistics of the original vs the relabelled stream and the item-4(c) gate
    |slope| < 0.1 per arm dim. `--expect-original` datasets must FAIL (i) (mutant m-a). `--mutant shift|claw|claw_offset`:
    the shared relabel function is replaced by the mutant for BOTH the export (the action column is the mutant's
    output, i.e. the set an exporter calling the mutated function would write) and the check; the mutant must be
    flagged by (i) or (c). Exit 1 if a --dataset fails or a mutant passes."""
    import h5py
    import pyarrow.parquet as pq

    fn = MUTANTS[a.mutant] if a.mutant else relabel

    def check(d: Path, mutated: bool = False) -> dict:
        side = [json.loads(x) for x in (d / "meta" / "episodes_113.jsonl").read_text().splitlines()]
        tabs = [pq.read_table(f, columns=["index", "episode_index", "action"])
                for f in sorted((d / "data").rglob("*.parquet"))]
        idx = np.concatenate([t.column("index").to_numpy() for t in tabs])
        ep = np.concatenate([t.column("episode_index").to_numpy() for t in tabs])
        col = np.concatenate([np.stack(t.column("action").to_numpy(zero_copy_only=False)) for t in tabs])
        o = np.argsort(idx)
        idx, ep, col = idx[o], ep[o], col[o].astype(np.float32)
        assert np.array_equal(idx, np.arange(len(idx))), "rows not contiguous"
        bad_rel = bad_orig = bad_claw = bad_first = rows = mut_rows = 0
        triples = []
        for s in side:
            r = np.flatnonzero(ep == s["lerobot_episode_index"])
            with h5py.File(s["source"], "r") as h:
                g = h[s["group"]]
                act, exe = g["action"][()].astype(np.float32), g["action_exec"][()].astype(np.float32)
            assert len(r) == len(act), (s["lerobot_episode_index"], len(r), len(act))
            rl = fn(act, exe)
            c = rl if mutated else col[r]  # mutant: the column an exporter calling the mutated function writes
            ne_true = int((rl != relabel(act, exe)).any(1).sum())  # rows where the mutant changes the label at all
            bad_rel += int((c != rl).any(1).sum())
            bad_orig += int((c != act).any(1).sum())
            bad_claw += int((c[:, 4] != act[:, 4]).sum())
            bad_first += int((c[0] != act[0]).any())
            rows += len(r)
            triples.append((act, exe))
            mut_rows += ne_true
        pre = np.concatenate([x[1:, :4] + (e[:-1, :4] - x[:-1, :4]) for x, e in triples])  # r before its clip
        lab = np.concatenate([x[1:, :4] for x, e in triples])
        clip = {"rows": len(pre), "frac_rows_clip_binds_r": float((np.abs(pre) > 1).any(1).mean()),
                "frac_rows_clip_binds_a": float((np.abs(lab) >= 1).any(1).mean()),
                "per_dim_r": (np.abs(pre) > 1).mean(0).tolist(), "per_dim_a": (np.abs(lab) >= 1).mean(0).tolist()}
        st = label_stats([(fn(x, e), x, e) for x, e in triples])
        slope_ok = all(abs(v) < SLOPE_MAX for v in st["slope_diff_on_prev_noise"])
        identity_ok = bad_rel == 0 and bad_claw == 0 and bad_first == 0
        return {"episodes": len(side), "rows": rows, "label": side[0].get("label", "action"),
                "relabel_fn": fn.__name__, "rows_ne_relabel": bad_rel, "rows_ne_original": bad_orig,
                "claw_rows_ne_stored": bad_claw, "episodes_first_row_ne_clean": bad_first,
                "identity_ok": identity_ok, "slope_ok_4c": slope_ok, "pass_4b_4c": identity_ok and slope_ok,
                **({"mutant_rows_changed_vs_true_relabel": mut_rows} if mutated else {}), "clip": clip,
                "original": label_stats([(x, x, e) for x, e in triples]), "relabel": st}

    res = {str(d): check(d, mutated=bool(a.mutant)) for d in a.dataset}
    mut = {str(d): check(d) for d in a.expect_original}
    for k, v in {**res, **mut}.items():
        print(k, json.dumps({x: v[x] for x in v if x not in ("original",)} | {
            "relabel": {x: v["relabel"][x] for x in ("slope_diff_on_prev_noise", "std_label_diff")}}), flush=True)
    if a.mutant:  # every --dataset must be flagged by (i) identity or (c) slope
        ok = not any(v["pass_4b_4c"] for v in res.values())
    else:
        ok = all(v["pass_4b_4c"] for v in res.values()) and not any(v["identity_ok"] for v in mut.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"mutant": a.mutant, "datasets": res, "mutants_original_labels": mut, "pass": ok},
                                indent=1))
    print("DSCHECK_PASS" if ok else "DSCHECK_FAIL", flush=True)
    sys.exit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("stats", "build", "replay", "dscheck"))
    p.add_argument("--src", type=Path, nargs="+")
    p.add_argument("--dataset", nargs="+", type=Path, default=[Path("/data/datasets/0c_s1_pick")])
    p.add_argument("--expect-original", nargs="*", type=Path, default=[])
    p.add_argument("--stages", nargs="+", default=["s1:21600"], help="replay: stage:validation_seed")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--cube-yaw-max-deg", type=float, default=180.0, help="replay: cube spawn yaw U(-m, m) deg")
    p.add_argument("--mutant", choices=tuple(MUTANTS), default=None, help="dscheck: mutated shared relabel function")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"stats": stats, "build": build, "replay": replay, "dscheck": dscheck}[a.mode](a)


if __name__ == "__main__":
    main()
