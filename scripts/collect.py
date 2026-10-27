"""Stage 0C bulk demo collector (sim venv; laptop or PERUN `sim_exec`). Brief section 8 0C, GATES "Stage 0C".

  python scripts/collect.py --stage s2 --split train --start 0 --episodes 400 --seed 31800 --envs 64 \
      --noise-sigma 0.1 --image-size 128 --visual nominal --out data/0c/s2/b000

One invocation = one GATES "batch": the contiguous episode range [start, start + episodes) of one dataset's stream.
- Episode stream (GATES conventions): episode i of --seed draws spec, layout and instruction from rng([seed, i])
  (vla_tabletop.rollout.make_stream with vla_common.tasks.sample_spec(stage, split)). --cube-yaw-max-deg m (default
  180 = the old U(-pi, pi) stream, bit for bit) narrows only the cube spawn yaw to U(-m, m) deg; recorded in RUN.json,
  the file attrs and every episode's attrs (`cube_yaw_max_deg`).
- The range is partitioned by task (s4 mixes tasks with different T) and, per task in stream order, grouped into
  CHUNKS of <= --envs episodes; one chunk = one simulator batch at that task's T_task (results/0a2_<task>_timing.json).
  Chunk id `<task>_e<first episode index:06d>`.
- Expert: the 0A.2 expert, unchunked (vla_tabletop.expert.Expert, fresh=True, sigma 0 inside). Demo noise (GATES 0C):
  executed = clip(label + sigma * z, -1, 1) on the 4 arm dims, claw not noised, z from a generator per (data seed,
  episode index) (`noise_table`). The expert dead-reckons the action term's commanded target / yaw from what it
  executed; because the noise is added outside it, `NoisyExpert` re-syncs that estimate every step from the term's
  commanded target / yaw (the snapshot's cmd_target / cmd_yaw, the expert's own command state, GATES 0A.3 attempt 1).
  The term's update is the same box clamp + 5 cm lead clamp the expert replays, so this equals the expert's own
  noise path; with sigma = 0 the re-sync is a no-op (sidecar `cmd_resync_max_m`, checked ~0 in the laptop smoke).
  Stored `action` = clean clipped label (the training label), `action_exec` = executed (contract v2 a).
- Cameras side + wrist at H x H, a frame on every PERIOD-th step; `frame_idx` = the delayed frame the policy sees at t
  (vla_common.timing PERIOD / LATENCY passed explicitly); `state` = the 19-dim policy observation before the action.
  --visual train_rand: the 113 randomizer in "train" mode, per-episode parameters from a generator per (seed, i) (the
  light is global per batch: the chunk's first episode's), applied after the reset, before the first frame, stored
  per episode in the `visual_json` attribute (audited: train mode, outside the held-out band).
  --table-palette s4r (Stage 4 / s4r only; default 112 = unchanged): the s4r grey-table palette (vla_tabletop.
  visual_rand module doc); --visual nominal then applies the s4r nominal scene (grey table) through the randomizer
  too. Recorded like cube_yaw_max_deg (RUN.json, file attrs, episode attrs `table_palette`).
- Only successful episodes (vla_tabletop.detectors at T_task) are written (contract HDF5, vla_common.episode) to
  `<prefix>_<chunk>_p<j>.h5.tmp`; a part rolls over once it reaches PART_BYTES (150 MiB), and every part is renamed to
  `.h5` when the chunk closes; then the chunk sidecar `<prefix>_<chunk>.json` (attempted, kept, failure classes,
  kept indices, files). A chunk is finished iff its sidecar exists with its listed files. Resume = re-run the same
  command: finished chunks are skipped (their recorded indices must equal the plan), leftovers of unfinished ones are
  deleted and redone.
- At the end: `collect_summary.json` (attempted / kept / rate + Wilson CI per task, failure classes, throughput) and
  `MANIFEST.sha256` (sha256sum format, every .h5 and JSON in --out). Use one --out directory per invocation.
--paired-targets (s2a, s2, s3a or s3b; counterfactual paired scenes, vla_tabletop.rollout.make_paired_stream): the batch is the group range
[--start-group, + --groups) instead of [--start, + --episodes); one episode per valid target of each scene, global episode
index = running count over all earlier non-skipped groups (a later batch continues the same index space). Files are
`<stage>_paired_<split>_s<seed>_*` (s2a_paired_.. / s2_paired_..; s2: a target is a (colour, size) cube, held-out cubes stay
distractors; s3a / s3b (task 007, split train): a member is a (cube, pad) / (top, bottom) pair, 2-3 per scene,
rollout.pair_members); every episode carries paired / layout_id / member / slot attrs (s3a / s3b also `relation`, the
member's type vs member 0); `paired.json` in --out maps group -> episode indices (members, targets; s3a / s3b also dests
and `pairs` = [member i, member j, relation type] for every member pair), lists the skipped groups and each episode's
chunk + env slot. Demo noise
stays per episode index; a --visual train_rand look is keyed by layout_id (the light stays one per chunk).
Determinism: specs, layouts, instructions, visual parameters and demo noise depend on (seed, i) only; the physics also
on the batch composition (GPU solver, GATES 0A.2 instrument note), so a resume must use the same --envs (RUN.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = ("s1", "s2", "s2a", "s3a", "s3b", "s3a1", "s3b1", "s4")
TASK_ORDER = ("pick", "place", "stack")
PART_BYTES = 150 * 2 ** 20  # part rollover size (task file: 100-200 MB per file)
FLAG_KEYS = ("wrong_object", "wrong_grasp", "wrong_dest", "slam", "jam", "drop", "early_close", "release_failure",
             "invalid")
SIDECAR_KEYS = ("chunk", "task", "indices", "n", "n_success", "success_rate", "kept_indices", "files", "flags_all",
                "failures", "failed_unflagged", "failing_indices", "T", "cmd_resync_max_m", "wall_s", "git_sha",
                "config")
RUN_KEYS = ("stage", "split", "seed", "envs", "noise_sigma", "image_size", "visual", "mix",
            "cube_yaw_max_deg", "table_palette", "paired_targets")  # must match on resume
RUN_LEGACY = {"cube_yaw_max_deg": 180.0, "table_palette": "112", "paired_targets": False}  # value of a key absent from a RUN.json written before it existed


# --------------------------------------------------------------------------- pure helpers (tested on CPU)
def plan_chunks(tasks: list[str], B: int, start: int = 0) -> list[dict]:
    """tasks[j] = task of stream episode start + j. Per task (pick, place, stack), its episodes in stream order grouped
    into chunks of <= B (absolute indices). A longer range keeps every earlier full chunk unchanged."""
    out = []
    for task in TASK_ORDER:
        idx = [start + j for j, t in enumerate(tasks) if t == task]
        for k in range(0, len(idx), B):
            out.append({"chunk": f"{task}_e{idx[k]:06d}", "task": task, "indices": idx[k:k + B]})
    return out


def noise_table(seed: int, indices: list[int], T: int, sigma: float) -> np.ndarray:
    """(len(indices), T, 4) float32 demo noise, row r from a generator per (data seed, episode indices[r])."""
    return np.stack([np.random.default_rng([seed, i, 2]).standard_normal((T, 4)) * sigma for i in indices]).astype(
        np.float32)


def prefix(stage: str, split: str, seed: int, paired: bool = False) -> str:
    return f"{stage}_paired_{split}_s{seed}" if paired else f"{stage}_{split}_s{seed}"


def paired_manifest(eps: list, skipped: list[dict], plan: list[dict], seed: int, start_group: int, groups: int,
                    stage: str, split: str) -> dict:
    """paired.json of one --paired-targets batch: group -> episode indices (+ members, targets), the skipped groups,
    and where each episode runs (chunk id, env slot = position in the chunk, < --envs)."""
    where = {i: {"chunk": c["chunk"], "slot": j} for c in plan for j, i in enumerate(c["indices"])}
    gm: dict[str, dict] = {}
    for e in eps:
        g = gm.setdefault(str(e.layout_id), {"episodes": [], "members": [], "targets": []})
        g["episodes"].append(e.index)
        g["members"].append(e.member)
        g["targets"].append(list(e.spec.target))
        if e.relation is not None:  # s3a / s3b: destinations + every member pair's relation type
            g.setdefault("dests", []).append(list(e.spec.dest) if e.spec.task == "stack" else e.spec.dest)
    from vla_tabletop.rollout import relation

    for g in gm.values():
        if "dests" in g:
            pr = list(zip(map(tuple, g["targets"]), [tuple(d) if isinstance(d, list) else d for d in g["dests"]]))
            g["pairs"] = [[g["members"][i], g["members"][j], relation(pr[i], pr[j])]
                          for i in range(len(pr)) for j in range(i + 1, len(pr))]
    return {"seed": seed, "stage": stage, "split": split, "start_group": start_group, "groups": groups,
            "n_episodes": len(eps), "groups_map": gm, "skipped_groups": skipped,
            "episode_slots": {str(i): where[i] for i in sorted(where)}}


def part_name(pre: str, chunk: str, j: int, tmp: bool = False) -> str:
    return f"{pre}_{chunk}_p{j}.h5" + (".tmp" if tmp else "")


def sidecar_path(out: Path, pre: str, chunk: str) -> Path:
    return out / f"{pre}_{chunk}.json"


def chunk_done(out: Path, pre: str, chunk: dict) -> bool:
    """Finished iff the sidecar exists and lists every file present; a finished chunk must match the plan."""
    sc = sidecar_path(out, pre, chunk["chunk"])
    if not sc.exists():
        return False
    d = json.loads(sc.read_text())
    if d["indices"] != chunk["indices"]:
        raise RuntimeError(f"{sc.name}: finished chunk holds episodes {d['indices'][:3]}.. but the plan says "
                           f"{chunk['indices'][:3]}.. (different range / --envs?); use a new --out")
    missing = [f for f in d["files"] if not (out / f).exists()]
    if missing:
        raise RuntimeError(f"{sc.name}: listed files missing {missing}")
    return True


def clean_chunk(out: Path, pre: str, chunk: str) -> list[str]:
    """Delete leftovers (parts / tmp parts) of an unfinished chunk; returns their names."""
    gone = []
    for f in sorted(out.glob(f"{pre}_{chunk}_p*.h5*")):
        f.unlink()
        gone.append(f.name)
    return gone


class PartWriter:
    """Kept episodes of one chunk -> <pre>_<chunk>_p<j>.h5.tmp, rolling to part j+1 once part j reaches part_bytes;
    close() renames every part to .h5 (os.replace) and returns the final names (none if nothing was kept)."""

    def __init__(self, out: Path, pre: str, chunk: str, file_attrs: dict, part_bytes: int = PART_BYTES):
        self.out, self.pre, self.chunk, self.fa, self.part_bytes = out, pre, chunk, file_attrs, part_bytes
        self.w = None
        self.parts: list[int] = []

    def add(self, arrays: dict[str, np.ndarray], attrs: dict) -> None:
        from vla_common.episode import EpisodeWriter

        if self.w is not None:
            self.w.f.flush()
            if (self.out / part_name(self.pre, self.chunk, self.parts[-1], tmp=True)).stat().st_size >= self.part_bytes:
                self.w.close()
                self.w = None
        if self.w is None:
            j = len(self.parts)
            self.parts.append(j)
            self.w = EpisodeWriter(self.out / part_name(self.pre, self.chunk, j, tmp=True),
                                   self.fa | {"part": j, "chunk": self.chunk})
        self.w.add(arrays["side_rgb"], arrays["wrist_rgb"], arrays["state"], arrays["action"], arrays["action_exec"],
                   arrays["frame_idx"], attrs)

    def close(self) -> list[str]:
        if self.w is not None:
            self.w.close()
            self.w = None
        names = []
        for j in self.parts:
            os.replace(self.out / part_name(self.pre, self.chunk, j, tmp=True), self.out / part_name(self.pre, self.chunk, j))
            names.append(part_name(self.pre, self.chunk, j))
        return names


def build_sidecar(chunk: dict, outcomes: dict[str, np.ndarray], files: list[str], T: int, resync_err: float,
                  wall_s: float, git_sha: str, config: dict) -> dict:
    """outcomes: per-episode arrays (rows = chunk['indices']) with 'success' + FLAG_KEYS."""
    succ = np.asarray(outcomes["success"], bool)
    fail = ~succ
    flags = {k: np.asarray(outcomes[k], bool) for k in FLAG_KEYS}
    any_flag = np.any(np.stack(list(flags.values())), 0)
    idx = chunk["indices"]
    d = {"chunk": chunk["chunk"], "task": chunk["task"], "indices": idx, "n": len(idx), "n_success": int(succ.sum()),
         "success_rate": float(succ.mean()) if len(idx) else float("nan"),
         "kept_indices": [i for i, s in zip(idx, succ) if s], "files": files,
         "flags_all": {k: int(v.sum()) for k, v in flags.items()},
         "failures": {k: int((v & fail).sum()) for k, v in flags.items()},
         "failed_unflagged": int((fail & ~any_flag).sum()),
         "failing_indices": [i for i, s in zip(idx, succ) if not s],
         "T": T, "cmd_resync_max_m": resync_err, "wall_s": round(wall_s, 1), "git_sha": git_sha, "config": config}
    assert tuple(d) == SIDECAR_KEYS
    return d


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95 % Wilson interval (= lift_rl.metrics.wilson_ci; copied: this module's pure part must not need lift_rl)."""
    if n == 0:
        return [float("nan"), float("nan")]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def summarize(sides: list[dict]) -> dict:
    out = {}
    for t in [x for x in TASK_ORDER if any(s["task"] == x for s in sides)]:
        ss = [s for s in sides if s["task"] == t]
        n, k = sum(s["n"] for s in ss), sum(s["n_success"] for s in ss)
        out[t] = {"attempted": n, "kept": k, "rate": k / n, "ci95": wilson(k, n),
                  "failures": {f: sum(s["failures"][f] for s in ss) for f in FLAG_KEYS},
                  "failed_unflagged": sum(s["failed_unflagged"] for s in ss)}
    return out


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(out: Path) -> Path:
    """MANIFEST.sha256 (sha256sum -c format) over every finished .h5 and every JSON in out."""
    files = sorted([f for f in out.iterdir() if f.suffix in (".h5", ".json") and f.is_file()], key=lambda f: f.name)
    m = out / "MANIFEST.sha256"
    m.write_text("".join(f"{file_sha256(f)}  {f.name}\n" for f in files))
    return m


def check_run_config(out: Path, cfg: dict) -> None:
    """First run writes RUN.json; a resume must use the same RUN_KEYS."""
    f = out / "RUN.json"
    if f.exists():
        old = json.loads(f.read_text())
        old = RUN_LEGACY | old
        cfg_k = RUN_LEGACY | cfg
        bad = {k: (old[k], cfg_k[k]) for k in RUN_KEYS if old[k] != cfg_k[k]}
        if bad:
            raise RuntimeError(f"{f}: resume with different settings (old, new): {bad}")
    f.write_text(json.dumps(cfg, indent=1))


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    g = ROOT / "GIT_HEAD"  # PERUN: git archive copy (hpc/push_repo.sh)
    return g.read_text().strip() if g.exists() else "unknown"


def visual_params(mode: str, indices: list[int], seed: int) -> dict:
    """113 randomizer parameters, one row per episode from a generator per (seed, i, 1); light from row 0."""
    import torch

    from vla_tabletop import visual_rand as VR

    rows = [VR.sample_params(mode, 1, torch.Generator().manual_seed(
        int(np.random.SeedSequence([seed, i, 1]).generate_state(1, np.uint64)[0] >> 1))) for i in indices]
    out = {k: torch.cat([r[k] for r in rows]) for k in rows[0] if k not in ("mode", "light")}
    out["mode"], out["light"] = mode, rows[0]["light"]
    return out


def visual_row(vis: dict, j: int) -> dict:
    """Episode j's visual parameters as JSON (stored per episode, read by the data audit)."""
    return {"mode": vis["mode"], "light": vis["light"].tolist(),
            **{k: vis[k][j].tolist() for k in ("nominal", "table_rgb", "cube_rgb", "pad_rgb")}}


def task_timing(task: str) -> dict:
    return json.loads((ROOT / "results" / f"0a2_{task}_timing.json").read_text())


def geometry_113():
    from vla_tabletop.rollout import geometry

    meas = json.loads((ROOT / "results" / "0a1_measure.json").read_text())
    return geometry(meas["z_rest_m"], h_pad=float(np.mean(list(meas["h_pad_m"].values()))))


class NoisyExpert:
    """Runner controller: the 0A.2 expert (fresh, sigma 0) + per-episode demo noise added outside it, with the
    expert's command estimate re-synced from the action term every step (module docstring)."""

    def __init__(self, u, batch, s0, geo, noise):
        import torch

        from vla_tabletop.expert import Expert
        from vla_tabletop.layout import TABLE_Z
        from vla_tabletop.rollout import spec_indices, to_snapshot

        task, target, dest = spec_indices([e.spec for e in batch], u.device)
        self.e = Expert(task, target, dest, geo.edge, TABLE_Z, geo.h_pad, u.step_dt, fresh=True)
        self.e.reset(to_snapshot(s0))
        self.noise = torch.as_tensor(noise, device=u.device)
        self.resync_err = 0.0

    def act(self, t: int, snap_fn):
        if t > 0:
            sn = snap_fn()
            self.resync_err = max(self.resync_err, float((self.e.target_cmd - sn["cmd_target"]).abs().max()))
            self.e.target_cmd, self.e.yaw_cmd = sn["cmd_target"].clone(), sn["cmd_yaw"].clone()
        label, _ = self.e.act(t, snap_fn)
        exe = label.clone()
        exe[:, :4] = (label[:, :4] + self.noise[:, t]).clamp(-1.0, 1.0)
        return label, exe


# --------------------------------------------------------------------------- simulator part
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--split", required=True, choices=("train", "heldout_comp", "paraphrase"))
    p.add_argument("--start", type=int, default=0, help="first stream episode index of this batch")
    p.add_argument("--episodes", type=int, help="episodes attempted: [start, start + episodes)")
    p.add_argument("--seed", type=int, required=True, help="data seed (GATES 0C table)")
    p.add_argument("--envs", type=int, required=True, help="parallel envs B = chunk size")
    p.add_argument("--noise-sigma", type=float, required=True, help="demo noise (GATES 0A.2 / 0C: 0.1)")
    p.add_argument("--image-size", type=int, required=True)
    p.add_argument("--visual", required=True, choices=("nominal", "train_rand"))
    p.add_argument("--mix", default="1,1,1", help="s4 task weights pick,place,stack (sample_spec weights)")
    p.add_argument("--cube-yaw-max-deg", type=float, default=180.0,
                   help="cube spawn yaw U(-m, m) deg (vla_tabletop.layout; 180 = U(-pi, pi), unchanged)")
    p.add_argument("--table-palette", default="112", choices=("112", "s4r"),
                   help="randomizer table palette (vla_tabletop.visual_rand; s4r = Stage 4 / s4r only)")
    p.add_argument("--paired-targets", action="store_true",
                   help="counterfactual paired scenes (vla_tabletop.rollout.make_paired_stream): groups --start-group .. "
                        "+ --groups instead of --start / --episodes; one episode per valid target of each scene "
                        "(layout_id / member in the attrs, paired.json); s2a, s2, s3a, s3b, s3a1 or s3b1")
    p.add_argument("--start-group", type=int, default=0, help="--paired-targets: first group of this batch")
    p.add_argument("--groups", type=int, default=0, help="--paired-targets: groups attempted")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    assert args.paired_targets == (args.episodes is None) and (args.paired_targets or args.start_group == args.groups == 0), \
        "give --episodes (with --start), or --paired-targets with --groups (with --start-group), not both"
    assert not args.paired_targets or ((args.stage in ("s2a", "s2") or args.stage in ("s3a", "s3b")
                                        or args.stage in ("s3a1", "s3b1")) and args.groups > 0), \
        "--paired-targets: stage s2a, s2, s3a, s3b, s3a1 or s3b1, --groups > 0"

    from vla_common import timing as TM
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import make_paired_stream, make_stream

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    skipped: list[dict] = []
    weights = tuple(float(x) for x in args.mix.split(","))
    if args.paired_targets:
        plist, skipped = make_paired_stream(args.seed, args.start_group, args.groups, args.stage, args.split,
                                            args.cube_yaw_max_deg)
        assert plist, "no episodes in these groups"
        eps = {e.index: e for e in plist}
        start, n_att = plist[0].index, len(plist)
    else:
        start, n_att = args.start, args.episodes
        eps = {e.index: e for e in make_stream(args.seed, start, n_att,
                                               lambda rng: sample_spec(args.stage, args.split, rng, weights),
                                               split=args.split, cube_yaw_max_deg=args.cube_yaw_max_deg)}
    cfg_run = {"stage": args.stage, "split": args.split, "seed": args.seed, "envs": args.envs,
               "noise_sigma": args.noise_sigma, "image_size": args.image_size, "visual": args.visual,
               "mix": args.mix, "cube_yaw_max_deg": args.cube_yaw_max_deg,
               "table_palette": args.table_palette, "start": start, "episodes": n_att, "period": TM.PERIOD,
               "latency": TM.LATENCY, "n_action_steps": TM.N, "part_bytes": PART_BYTES, "git_sha": sha}
    if args.paired_targets:  # keys absent from every non-paired RUN.json (RUN_LEGACY: False)
        cfg_run |= {"paired_targets": True, "start_group": args.start_group, "groups": args.groups}
    check_run_config(out, cfg_run)
    pre = prefix(args.stage, args.split, args.seed, args.paired_targets)
    plan = plan_chunks([eps[i].spec.task for i in sorted(eps)], args.envs, start)
    if args.paired_targets:
        (out / "paired.json").write_text(json.dumps(paired_manifest(
            list(eps.values()), skipped, plan, args.seed, args.start_group, args.groups, args.stage, args.split), indent=1))
    todo = [c for c in plan if not chunk_done(out, pre, c)]
    print(f"[collect] {pre} episodes {start}..{start + n_att - 1}: {len(plan)} chunks, "
          f"{len(plan) - len(todo)} finished, {len(todo)} to run", flush=True)
    if not todo and (out / "MANIFEST.sha256").exists():
        return

    import gymnasium as gym
    import torch

    import vla_tabletop.scene as S
    from isaaclab_tasks.utils import launch_simulation
    from lift_rl.physics import assert_on_cuda, assert_run_preconditions
    from vision_lift import cameras as C
    from vision_lift.frames import frame_index, n_frames
    from vla_common.catalog import ALL_CUBES
    from vla_common.tasks import labels as task_labels
    from vla_tabletop import visual_rand as VR
    from vla_tabletop.detectors import DETECTOR_VERSION
    from vla_tabletop.expert import EXPERT_VERSION
    from vla_tabletop.rollout import Runner

    cfg_run |= {"expert_version": EXPERT_VERSION, "detector_version": DETECTOR_VERSION}
    B, H = args.envs, args.image_size
    cfg = S.make_cfg(B, args.seed, res=H)
    assert_run_preconditions(cfg)
    geo = geometry_113()
    mode = VR.run_mode(args.visual, args.table_palette)

    with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
        t_start = time.time()
        env = gym.make(S.TASK_ID, cfg=cfg)
        u = env.unwrapped
        assert_on_cuda(env)
        print(f"[collect] device={u.device} envs={B} H={H} visual={args.visual} sigma={args.noise_sigma}", flush=True)
        run = Runner(env, geo)
        n_run = 0
        for c in todo:
            gone = clean_chunk(out, pre, c["chunk"])
            if gone:
                print(f"[collect] {c['chunk']}: removed leftovers {gone}", flush=True)
            t0 = time.time()
            tj = task_timing(c["task"])
            T, F = tj["T_steps"], n_frames(tj["T_steps"], TM.PERIOD)
            run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)
            batch = [eps[i] for i in c["indices"]]
            padded = c["indices"] + [c["indices"][-1]] * (B - len(batch))  # Runner pads with copies of the last
            vkey = [eps[i].layout_id for i in padded] if args.paired_targets else padded  # paired: one look per scene
            vis = visual_params(mode, vkey, args.seed) if mode else None
            noise = noise_table(args.seed, padded, T, args.noise_sigma)
            rec = {"side_rgb": np.empty((B, F, H, H, 3), np.uint8), "wrist_rgb": np.empty((B, F, H, H, 3), np.uint8),
                   "state": np.empty((B, T, 19), np.float32), "action": np.empty((B, T, 5), np.float32),
                   "action_exec": np.empty((B, T, 5), np.float32)}

            def on_step(t, uu, s, label, exe, ctrl, det):
                if t == 0 and vis is not None:
                    VR.apply(uu, vis, torch.arange(B, device=uu.device))
                rec["state"][:, t] = uu.obs_buf["policy"][:, :19].cpu().numpy()
                rec["action"][:, t] = label.cpu().numpy()
                rec["action_exec"][:, t] = exe.cpu().numpy()
                if t % TM.PERIOD == 0:
                    rec["side_rgb"][:, t // TM.PERIOD] = C.rgb(uu, "side_cam").to(torch.uint8).cpu().numpy()
                    rec["wrist_rgb"][:, t // TM.PERIOD] = C.rgb(uu, "wrist_cam").to(torch.uint8).cpu().numpy()

            res = run.run(batch, T, on_step=on_step,
                          controller=lambda uu, b, s0: NoisyExpert(uu, b, s0, geo, noise))
            fidx = np.array([frame_index(t, TM.PERIOD, TM.LATENCY) for t in range(T)], np.int32)
            outc = {k: v.numpy() for k, v in res.items()}
            w = PartWriter(out, pre, c["chunk"], {"git_sha": sha, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "stage": args.stage, "split": args.split, "seed": args.seed,
                                                  "mix": args.mix, "cube_yaw_max_deg": args.cube_yaw_max_deg,
                                                  "table_palette": args.table_palette})
            for j, e in enumerate(batch):
                if not outc["success"][j]:
                    continue
                metrics = {k: (v[j].item() if v[j].ndim == 0 else v[j].tolist()) for k, v in outc.items()
                           if k not in ("final_cube_pos", "spawn_yaw_deg", "spawn_yaw_fold_deg")}
                w.add({k: rec[k][j] for k in rec} | {"frame_idx": fidx},
                      {"instruction": e.instruction, "split": args.split, "task": e.spec.task,
                       "spec_json": e.spec.to_json(), "labels": task_labels(e.spec).tolist(), "seed": args.seed,
                       "env_id": j, **({"paired": True, "layout_id": e.layout_id, "member": e.member, "slot": j}
                                       if args.paired_targets else {}),
                       **({"relation": e.relation} if e.relation is not None else {}), "success": True, "metrics_json": metrics, "period": TM.PERIOD,
                       "latency": TM.LATENCY, "image_size": H, "visual_mode": args.visual,
                       "action_noise_sigma": args.noise_sigma, "expert_version": EXPERT_VERSION,
                       "episode_index": e.index, "T": T, "n_action_steps": TM.N, "detector_version": DETECTOR_VERSION,
                       "stage": args.stage, "cube_yaw_max_deg": args.cube_yaw_max_deg, "table_palette": args.table_palette,
                       "spawn_yaw_deg": float(outc["spawn_yaw_deg"][j]),
                       "spawn_yaw_fold_deg": float(outc["spawn_yaw_fold_deg"][j]), "visual_json": visual_row(vis, j) if vis else {"mode": "nominal"}})
            files = w.close()
            sc = build_sidecar(c, {k: outc[k] for k in ("success", *FLAG_KEYS)}, files, T, run.ctrl.resync_err,
                               time.time() - t0, sha, cfg_run)
            sidecar_path(out, pre, c["chunk"]).write_text(json.dumps(sc, indent=1))
            n_run += len(batch)
            el = time.time() - t_start
            print(f"[collect] {c['chunk']}: {sc['n_success']}/{sc['n']} kept, {sc['wall_s']} s, resync err "
                  f"{sc['cmd_resync_max_m']:.2e} m, files {files}; {n_run} eps in {el:.0f} s = "
                  f"{n_run / el * 3600:.0f} eps/h", flush=True)
        wall = time.time() - t_start
        sides = [json.loads(sidecar_path(out, pre, c["chunk"]).read_text()) for c in plan]
        disk = sum((out / f).stat().st_size for s in sides for f in s["files"])
        kept = sum(s["n_success"] for s in sides)
        summary = {"prefix": pre, "config": cfg_run, "chunks": len(plan), "attempted": n_att, "kept": kept,
                   "expert_success": kept / n_att, "ci95": wilson(kept, n_att),
                   "per_task": summarize(sides), "files": sorted(f for s in sides for f in s["files"]),
                   "disk_bytes": disk, "mb_per_kept_episode": disk / 1e6 / max(kept, 1),
                   "cmd_resync_max_m": max(s["cmd_resync_max_m"] for s in sides),
                   "this_invocation": {"episodes_run": n_run, "wall_s": round(wall, 1),
                                       "episodes_per_h": n_run / wall * 3600 if n_run else None,
                                       "torch_peak_vram_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                                       "device": str(u.device), "gpu": torch.cuda.get_device_name(0)},
                   "nan_guard_events": u.nan_guard_events}
        (out / "collect_summary.json").write_text(json.dumps(summary, indent=1))
        write_manifest(out)
        print("COLLECT_SUMMARY=" + json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)
        print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
        env.close()


if __name__ == "__main__":
    main()
