"""Closed-loop policy evaluation (sim venv; spawns the policy server with the policy-venv python, like
scripts/s0b_bridge.py). Implements GATES "Closed-loop policy evaluation protocol" + conventions (episode stream,
determinism, smart length, detectors, rates).

  python scripts/eval_policy.py --policy smolvla --ckpt <pretrained_model dir> --stage s2 --split train \
      --visual nominal --episodes 300 --seed 21620 --envs 128 --image-size 128 --out results/<name>.json
  ... --extend-to 1000 --out results/<name>.json     (smart length: episodes 300..999 of the same stream, merged)
  python scripts/eval_policy.py --replay results/<name>.json --sheets 3 [--videos 5,17]   (close-up frames)
  python scripts/eval_policy.py --policy expert ...  (the 0A.2 expert in its 0A.3 chunked mode = ceiling reference)

- Stream: episode i of --seed = sample_spec(stage, split, rng([seed, i])) + layout + instruction of that split
  (vla_tabletop.rollout.make_stream; --cube-yaw-max-deg m narrows only the cube spawn yaw to U(-m, m) deg, default
  180 = the old stream bit for bit, recorded as config.cube_yaw_max_deg); policy noise seed of episode i = vla_tabletop.bridge.episode_seed(seed, i), sent
  with `reset` (per-env generators server-side, query index added there).
- Batching (protocol): per task, the task's episodes in stream order; round r = task positions [rB, (r+1)B), slot =
  position mod B (single-task stream: round i // B, slot i mod B). Every round runs all B slots with real stream
  episodes; slots outside the requested range run and are discarded. Episode i's batch composition is therefore fixed
  for a given (seed, stage, split, B) — the same for every policy, checkpoint and for --extend-to — so comparisons on
  one stream are paired despite the solver's batch-composition non-determinism. A round = one task, one T_task.
- Policies: one `act` per query step t % N == 0 for all B envs with the delayed frame (vla_tabletop.bridge.
  ChunkExecutor, PERIOD / LATENCY from vla_common.timing), N actions executed open loop, arm clipped, claw > 0.5;
  the client checks `info` against N / PERIOD / LATENCY / H / dims and every query's frame index against frames.py.
  Baseline: the task labels ride along in `act`. `--policy expert`: the 0A.2 expert, fresh=False, n_query=N (0A.3).
- --visual nominal | train_rand | heldout: the 113 randomizer ("train" / "heldout"), per-episode parameters from
  (seed, i) (scripts/collect.visual_params), applied after the reset, before the first frame. --table-palette s4r
  (Stage 4 only; default 112 = unchanged): the s4r grey-table palette, nominal included (visual_rand.run_mode);
  recorded as config.table_palette (absent in older JSONs = 112).
- Output JSON: config, policy checkpoint + ckpt_sha + stats sha, per-episode records (detector flags), aggregates
  overall / per task / per target cube / per cube count (count, n, rate, Wilson CI), server_s and round trip p50 / p95,
  wall, GPU-h, episodes per GPU-h, and with --bars the GATES smart-length decision.
- --paired-targets --groups N (Stage 2a probe; default off = everything above unchanged): the stream is the counterfactual
  paired stream of vla_tabletop.rollout.make_paired_stream(seed, 0, N, stage s2a, split): group g = ONE scene (cubes,
  positions, yaws, pads) drawn from rng([seed, g]) with one episode per valid target (instruction from rng([seed, g, m])),
  global episode index = running count, so --episodes / --extend-to are not used and the run is groups 0..N-1 (the last
  round's spare slots take the next groups' episodes and are discarded, like any partial round). Records carry
  `layout_id` and `member`; config carries `paired_targets` and `groups`. Policy noise seed is per episode index as
  always. scripts/s1_report.py probe_s2a <eval json> reads it.
  Stages s2 / s3a / s3b / s3a1 / s3b1 are accepted too (make_paired_stream); two-noun records also carry `relation` (anchor /
  same_target / same_dest / both_differ) and `rest_cube` / `rest_dest` (every record: the latest settled released cube,
  ALL_CUBES index, and the pad COLORS index / bottom-cube ALL_CUBES index it rests on; -1 none) -> s1_report.py probe_s3. A --visual train_rand look is keyed by layout_id.
- --dump-slots K (instrument D1, default 0 = off and byte-identical behaviour): `<out stem>_dump.npz` with, for the first K
  slots of round 0, the first query's side / wrist frames, state, labels (baseline), returned chunk, and the whole
  episode's executed actions (K, T, 5) and states (K, T, 19), plus seed / episode index / spec json
  (scripts/s4b_d1_compare.py reads it against the training dataset).
- --guidance-w w (task 004 guidance probe, validation only; default 0 = off and unchanged): the server runs the guided
  sampler of vla_policy.guidance at weight w; after each round's reset the client sends every slot's "other"
  instructions (vla_common.guidance.other_instructions of the instruction sent, over the spec's cube colours: scene
  information from the simulator, disclosed in GATES 004). smolvla, stage s2a, no instruction swap. Recorded as
  config.guidance_w (absent = off); --replay of such a JSON re-runs with the same w.
- Frames (protocol): evaluations render only the policy cameras. `--replay <json>` re-runs the rounds containing the
  chosen episodes (first --sheets successes + the first episode of every failure type, + --videos indices) with the
  same seeds / layouts / slots and a close-up camera on, writes close-up | side | wrist sheets at the grasp (first
  grasp confirmation), release (first open claw after it; place / stack) and the last tick, and optional MP4s
  (16.7 fps) -> <out dir>/frames/<stem>/; each replayed episode's outcome is compared to the recorded one
  (`<stem>_frames.json`, `reproduced`). Never used during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TASK_ORDER = ("pick", "place", "stack")
METRICS = ("success", "wrong_grasp", "wrong_dest", "wrong_object", "slam", "slam_10N", "slam_50N", "jam",
           "release_failure", "release_fail_not_released", "release_fail_off_destination", "release_fail_not_clear",
           "release_fail_unstable", "early_close", "drop", "distractor_disturbed", "clean", "invalid")
RECORD_KEYS = METRICS + ("stack_release_rest_seen", "stack_stable_after_release", "grasp_confirmed",
                         "completion_step", "max_jam_run", "max_hand_table_or_pad_force",
                         "spawn_yaw_deg", "spawn_yaw_fold_deg", "wrong_grasp_cube", "rest_cube", "rest_dest")
FAIL_TYPES = ("wrong_object", "slam", "jam", "release_failure", "drop", "early_close", "invalid")
CONFIG_KEYS = ("policy", "ckpt", "stage", "mix", "split", "visual", "seed", "envs", "image_size", "n", "period",
               "latency", "dataset", "cube_yaw_max_deg", "expert_fresh", "instruction_swap",
               "table_palette", "paired_targets", "groups")  # must match when extending
CONFIG_LEGACY = {"cube_yaw_max_deg": 180.0, "expert_fresh": False, "instruction_swap": "none", "table_palette": "112",
                 "paired_targets": False, "groups": 0}  # value of a key absent from a JSON written before it existed
N300, N1000, MARGIN = 300, 1000, 0.05  # GATES "Smart test length"


# --------------------------------------------------------------------------- pure helpers (tested on CPU)
def plan_rounds(task_of, start: int, stop: int, B: int) -> list[dict]:
    """Protocol batching. task_of(i) -> task of stream episode i. Returns the rounds holding any episode of
    [start, stop): {"task", "round", "indices" (B stream indices, slot order), "keep" (B bools: index in range)}."""
    pos: dict[str, list[int]] = {t: [] for t in TASK_ORDER}
    for i in range(stop):
        pos[task_of(i)].append(i)
    out = []
    for t in TASK_ORDER:
        rounds = sorted({k // B for k, i in enumerate(pos[t]) if i >= start})
        j = stop
        for r in rounds:
            while len(pos[t]) < (r + 1) * B:  # fill the round's remaining slots with the task's next stream episodes
                if task_of(j) == t:
                    pos[t].append(j)
                j += 1
            idx = pos[t][r * B:(r + 1) * B]
            out.append({"task": t, "round": r, "indices": idx, "keep": [start <= i < stop for i in idx]})
    return out


def plan_instruction_swap(episodes: list, mode: str) -> tuple[dict[int, str], dict | None]:
    """Instrument (not a protocol change): {stream index: instruction to SEND} for `episodes` (stream order, indices
    0..K-1 of one run) and the swap report. none -> ({}, None); shuffle -> the donor permutation of vla_common.swap
    (default_rng(0); different target, target cube absent from this scene where possible). The detectors keep scoring
    the episode's own spec."""
    if mode == "none":
        return {}, None
    from vla_common.swap import donors, swap_report

    instr, tg, cubes = ([e.instruction for e in episodes], [tuple(e.spec.target) for e in episodes],
                        [[tuple(x) for x in e.spec.cubes] for e in episodes])
    d = donors(instr, tg, cubes, seed=0)
    return ({episodes[i].index: instr[j] for i, j in enumerate(d)},
            swap_report(instr, tg, d, cubes) | {"donor_of": {str(episodes[i].index): episodes[j].index for i, j in enumerate(d)}})


def server_args(c: dict) -> list[str]:
    """argv of `python -m vla_policy.server` for run config `c`."""
    from vla_common import timing as TM

    srv = ["--policy", c["policy"], "--ckpt", c["ckpt"], "--n-action-steps", str(c["n"]), "--period", str(TM.PERIOD),
           "--latency", str(TM.LATENCY), "--device", "cuda"]
    if c["dataset"]:
        srv += ["--dataset", str(Path(c["dataset"]).resolve())]
    if c["policy"] == "baseline":  # the network is size agnostic: the server needs the size the sim sends
        srv += ["--image-size", str(c["image_size"])]
    if c.get("sampling", "default") != "default":
        srv += ["--sampling", c["sampling"], "--k", str(c["k"])]
    if c.get("guidance_w"):
        srv += ["--guidance-w", repr(float(c["guidance_w"]))]
    return srv


def expected_info(c: dict) -> dict:
    """What the sim requires of the server's `info` (vla_common.protocol.check_info). SmolVLA predicts a 50-step chunk
    and executes N; the baseline's chunk IS N."""
    from vla_common import timing as TM

    return {"policy": c["policy"], "n_action_steps": c["n"], "period": TM.PERIOD, "latency": TM.LATENCY,
            "image_size": c["image_size"], "state_dim": 19, "action_dim": 5,
            "chunk_size": c["n"] if c["policy"] == "baseline" else 50}


def labels_sent(batch: list, swap_rep: dict | None, episode, task_labels) -> np.ndarray:
    """(B, 4) int64 labels the baseline is SENT for `batch`: labels(spec) of the episode itself, or, under
    --instruction-swap shuffle, of its DONOR episode (the same permutation as the instructions; slots without a donor
    entry -- discarded slots past --episodes -- keep their own). The detectors score the episode's own spec."""
    d = {} if swap_rep is None else swap_rep["donor_of"]
    return np.stack([task_labels(episode(d[str(e.index)]).spec if str(e.index) in d else e.spec)
                     for e in batch]).astype(np.int64)


def visual_record(vis: dict, j: int) -> dict:
    """Per-episode visual parameters AS APPLIED to slot j (row j of the params VR.apply received; the light is one per
    round): visual mode, table HSV value and saturation, light elevation [deg]. Not read back from the sim."""
    from vla_tabletop import visual_rand as VR

    hsv = VR.rgb_to_hsv(vis["table_rgb"][j])
    return {"visual_mode": vis["mode"], "table_value": round(float(hsv[2]), 6), "table_sat": round(float(hsv[1]), 6),
            "light_elev_deg": round(VR.light_elevation_deg(vis["light"]), 4)}


def rate(x) -> dict:
    x = np.asarray(x, bool)
    k, n = int(x.sum()), int(x.size)
    from collect import wilson

    return {"count": k, "n": n, "rate": k / n if n else float("nan"), "ci95": wilson(k, n)}


def aggregate(records: list[dict]) -> dict:
    """Rates per task and over all episodes; stack stability among stack episodes released resting on the bottom;
    success by target cube and by number of cubes on the table."""
    out = {}
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in records)]
    for task in ["all"] + tasks:
        rs = records if task == "all" else [r for r in records if r["task"] == task]
        d = {"n": len(rs), **{m: rate([r[m] for r in rs]) for m in METRICS}}
        st = [r for r in rs if r["task"] == "stack" and r["stack_release_rest_seen"]]
        if any(r["task"] == "stack" for r in rs):
            d["stack_stability"] = rate([r["stack_stable_after_release"] for r in st])
        cs = np.asarray([r["completion_step"] for r in rs if r["completion_step"] >= 0])
        d["completion_s_p50"] = float(np.percentile(cs, 50)) / 50 if cs.size else None
        if task != "all":
            d["success_by_target"] = {c: rate([r["success"] for r in rs if r["target"] == c])
                                      for c in sorted({r["target"] for r in rs})}
            d["success_by_n_cubes"] = {str(n): rate([r["success"] for r in rs if r["n_cubes"] == n])
                                       for n in sorted({r["n_cubes"] for r in rs})}
        out[task] = d
    return out


def parse_bars(s: str) -> list[tuple[str, str, str, float]]:
    """"pick.success>=0.85,pick.wrong_grasp<0.05" -> [(task, metric, op, bar)]; task may be 'all'."""
    out = []
    for item in filter(None, s.split(",")):
        op = ">=" if ">=" in item else "<"
        lhs, bar = item.split(op)
        task, metric = lhs.strip().split(".")
        out.append((task, metric, op, float(bar)))
    return out


def margins(agg: dict, bars: list) -> list[float]:
    """Signed distance to each bar in rate units, positive = passing side."""
    return [(agg[t][m]["rate"] - b) if op == ">=" else (b - agg[t][m]["rate"]) for t, m, op, b in bars]


def decide(records: list[dict], bars: list) -> dict:
    """GATES smart length on the stream prefix: at 300 FAIL if any gated metric is > 5 points on the failing side,
    PASS if every one is > 5 points on the passing side, else extend to 1,000 and decide there (no further extension)."""
    idx = [r["index"] for r in records]
    assert idx == list(range(len(idx))), "records must be the contiguous stream prefix 0..n-1"
    if len(records) < N300:
        return {"decision": "incomplete", "n": len(records)}
    m300 = margins(aggregate(records[:N300]), bars)
    if any(x < -MARGIN for x in m300):
        return {"decision": "FAIL decided at 300", "pass": False, "margins_300": m300}
    if all(x > MARGIN for x in m300):
        return {"decision": "PASS decided at 300", "pass": True, "margins_300": m300}
    if len(records) < N1000:
        return {"decision": "extend to 1000", "pass": None, "margins_300": m300}
    m1000 = margins(aggregate(records[:N1000]), bars)
    ok = all(x >= 0 if op == ">=" else x > 0 for x, (_, _, op, _) in zip(m1000, bars))
    return {"decision": f"{'PASS' if ok else 'FAIL'} decided at 1000", "pass": ok, "margins_300": m300,
            "margins_1000": m1000}


def fail_types(r: dict) -> list[str]:
    if r["success"]:
        return ["success"]
    t = [k for k in FAIL_TYPES if r[k]]
    return t or ["fail_other"]


def pick_for_frames(records: list[dict], n_success: int) -> list[int]:
    """Stream indices for contact sheets: the first n_success successes and the first episode of every failure type."""
    have: dict[str, int] = {}
    out = []
    for r in sorted(records, key=lambda r: r["index"]):
        new = [t for t in fail_types(r) if have.get(t, 0) < (n_success if t == "success" else 1)]
        if new:
            out.append(r["index"])
            for t in fail_types(r):
                have[t] = have.get(t, 0) + 1
    return out


def pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(x), q)) if len(x) else None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stats_sha(ckpt: str, dataset: str) -> str | None:
    """sha256 over the normalization stats the server uses: the checkpoint's normalizer files, or (base weights +
    --dataset) the dataset's meta/stats.json."""
    files = sorted(Path(ckpt).glob("*normalizer*.safetensors")) if ckpt and Path(ckpt).is_dir() else []
    if dataset:
        files = [Path(dataset) / "meta" / "stats.json"]
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()


class Labelled:
    """Wraps the bridge's protocol client so every `act` carries the batch's task labels (baseline only)."""

    def __init__(self, client):
        self.client, self.labels = client, None

    def call(self, msg: dict) -> dict:
        if msg.get("op") == "act":
            msg = msg | {"labels": self.labels[msg["env_ids"]]}
        return self.client.call(msg)

    def close(self) -> None:
        self.client.close()


class SlotRecorder:
    """--dump-slots instrument (pure numpy; nothing is created when the flag is 0): the first K slots of one round.
    `record` is called once per control step AFTER ChunkExecutor.step: the state the step was given, the action it
    returned, and at t = 0 (the first query) the delayed frames sent, the labels sent and the chunk returned."""

    def __init__(self, K: int):
        self.K, self.states, self.actions, self.first = K, [], [], {}

    def record(self, t: int, state: np.ndarray, action: np.ndarray, ex) -> None:
        K = self.K
        self.states.append(np.asarray(state)[:K, :19].astype(np.float32).copy())
        self.actions.append(np.asarray(action)[:K].astype(np.float32).copy())
        if t == 0:
            side, wrist = ex.frames.current()[1]
            self.first = {"side0": side[:K].copy(), "wrist0": wrist[:K].copy(), "state0": self.states[0].copy(),
                          "chunk0": ex.chunk[:K].astype(np.float32).copy()}
            if ex.labels is not None:
                self.first["labels0"] = np.asarray(ex.labels)[:K].astype(np.int64).copy()

    def arrays(self, batch: list, seed: int) -> dict:
        """npz payload: first-query inputs / chunk, whole-episode actions (K, T, 5) and states (K, T, 19), identity."""
        K = self.K
        return {**self.first, "actions": np.stack(self.actions, 1), "states": np.stack(self.states, 1),
                "seed": np.int64(seed), "episode_index": np.array([e.index for e in batch[:K]], np.int64),
                "slot": np.arange(K, dtype=np.int64), "spec_json": np.array([e.spec.to_json() for e in batch[:K]])}


# --------------------------------------------------------------------------- simulator part
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("smolvla", "baseline", "pi05", "expert"))
    p.add_argument("--ckpt", default="", help="policy checkpoint dir / hub id (not for expert)")
    p.add_argument("--dataset", default="", help="LeRobot dataset dir with the normalization stats (base weights only)")
    p.add_argument("--stage", choices=("s1", "s2", "s2a", "s3a", "s3b", "s3a1", "s3b1", "s4"))
    p.add_argument("--mix", default="1,1,1", help="s4 task weights pick,place,stack")
    p.add_argument("--split", choices=("train", "heldout_comp", "paraphrase"))
    p.add_argument("--visual", choices=("nominal", "train_rand", "heldout"))
    p.add_argument("--table-palette", default="112", choices=("112", "s4r"),
                   help="randomizer table palette (vla_tabletop.visual_rand; s4r = Stage 4 only)")
    p.add_argument("--episodes", type=int, default=0, help="stream episodes 0..K-1 (new run)")
    p.add_argument("--extend-to", type=int, default=0, help="continue --out's stream up to this many episodes")
    p.add_argument("--paired-targets", action="store_true",
                   help="evaluate counterfactual paired groups (module docstring); needs --groups, not --episodes")
    p.add_argument("--groups", type=int, default=0, help="--paired-targets: groups 0..N-1 of the seed's paired stream")
    p.add_argument("--seed", type=int)
    p.add_argument("--envs", type=int, help="B env slots per round (protocol: 128)")
    p.add_argument("--image-size", type=int)
    p.add_argument("--n", type=int, default=0, help="query period / n_action_steps (default vla_common.timing.N)")
    p.add_argument("--cube-yaw-max-deg", type=float, default=180.0,
                   help="cube spawn yaw U(-m, m) deg (vla_tabletop.layout; 180 = U(-pi, pi), unchanged)")
    p.add_argument("--expert-fresh", action="store_true",
                   help="--policy expert only: unchunked expert, fresh true state every step (s0a2_expert --chunk 0)")
    p.add_argument("--instruction-swap", choices=("none", "shuffle"), default="none",
                   help="instrument: send another episode's instruction (fixed permutation, default_rng(0)); the "
                        "detectors still score the episode's own target. Not with --extend-to.")
    p.add_argument("--log-instructions", type=int, default=0,
                   help="write, for the first K episodes, the instruction sent and the server's decoded tokens")
    p.add_argument("--dump-slots", type=int, default=0,
                   help="instrument D1 (default 0 = off, nothing changes): save <out stem>_dump.npz for the first K slots "
                        "of round 0 (first-query frames / state / labels / chunk, per-step executed action and state)")
    p.add_argument("--out", type=Path)
    p.add_argument("--bars", default="", help='gated metrics, e.g. "pick.success>=0.85,pick.wrong_grasp<0.05"')
    p.add_argument("--replay", type=Path, help="eval JSON whose rounds are replayed with the close-up camera")
    p.add_argument("--sheets", type=int, default=3, help="--replay: successes with a contact sheet")
    p.add_argument("--videos", default="", help="--replay: stream indices with a per-episode MP4")
    p.add_argument("--record", action="store_true",
                   help="task 011 demo (new run only): per kept slot a fixed viewer camera + the policy frames + the "
                        "evaluator's per-tick signals -> <out dir>/demo_rec/<out stem>/ep<i>/ (vla_tabletop.demo_rec)")
    p.add_argument("--sampling", default="default", choices=("default", "zero", "meanK"),
                   help="smolvla server start-noise mode (vla_policy.server --sampling; default unchanged)")
    p.add_argument("--k", type=int, default=8, help="--sampling meanK: chunks averaged per query")
    p.add_argument("--guidance-w", type=float, default=0.0,
                   help="task 004: instruction guidance weight w >= 1 (smolvla, s2a; 0 = off)")
    p.add_argument("--policy-python", default=os.environ.get("VLA_POLICY_PY", ""),
                   help="policy-venv python (default policy/.venv; PERUN container: /opt/policy/bin/python)")
    args = p.parse_args()
    assert not args.expert_fresh or args.policy == "expert", "--expert-fresh is valid only with --policy expert"
    assert not args.dump_slots or (args.policy in ("smolvla", "baseline") and not args.replay and not args.extend_to
                                   and 0 < args.dump_slots <= (args.envs or 0)),         "--dump-slots K: smolvla / baseline, a new run (no --replay / --extend-to), 0 < K <= --envs"

    from vla_common import timing as TM

    prev = None
    if args.replay:
        prev = json.loads(args.replay.read_text())
        cfg_run = prev["config"]
        out = args.replay
        records_prev = {r["index"]: r for r in prev["episodes"]}
        chosen = sorted(set(pick_for_frames(prev["episodes"], args.sheets)) |
                        {int(i) for i in args.videos.split(",") if i})
        start, stop = 0, prev["n_episodes"]
    else:
        if args.paired_targets:
            assert (args.groups > 0 and not args.episodes and not args.extend_to
                    and args.stage in ("s2a", "s2", "s3a", "s3b", "s3a1", "s3b1")), \
                "--paired-targets: stage s2a, s2, s3a, s3b, s3a1 or s3b1, --groups N, no --episodes / --extend-to"
        else:
            assert not args.groups, "--groups needs --paired-targets"
            assert bool(args.episodes) != bool(args.extend_to), "give exactly one of --episodes / --extend-to"
        N = args.n or TM.N
        assert N, "vla_common.timing.N is not set yet (0A.3); pass --n explicitly"
        out = args.out if args.out.is_absolute() else ROOT / args.out
        cfg_run = {"policy": args.policy, "ckpt": args.ckpt, "stage": args.stage, "mix": args.mix,
                   "split": args.split, "visual": args.visual, "seed": args.seed, "envs": args.envs,
                   "image_size": args.image_size, "n": N, "period": TM.PERIOD, "latency": TM.LATENCY,
                   "dataset": args.dataset, "provisional_n": TM.N is None,
                   "cube_yaw_max_deg": args.cube_yaw_max_deg, "expert_fresh": args.expert_fresh,
                   "instruction_swap": args.instruction_swap, "table_palette": args.table_palette}
        if args.sampling != "default":  # absent key = default (every JSON written before the flag existed)
            cfg_run |= {"sampling": args.sampling, "k": args.k}
        if args.guidance_w:  # absent key = off
            assert (args.guidance_w >= 1.0 and args.policy == "smolvla" and args.stage == "s2a"
                    and args.instruction_swap == "none" and args.sampling == "default" and not args.extend_to),                 "--guidance-w w >= 1: smolvla, stage s2a, default sampling, no instruction swap, new run"
            cfg_run |= {"guidance_w": args.guidance_w}
        if args.paired_targets:  # absent keys = unpaired (CONFIG_LEGACY)
            cfg_run |= {"paired_targets": True, "groups": args.groups}
        if args.extend_to:
            assert args.instruction_swap == "none", "--instruction-swap with --extend-to (the permutation spans the run)"
            prev = json.loads(out.read_text())
            pc = CONFIG_LEGACY | prev["config"]
            ck = CONFIG_LEGACY | cfg_run
            bad = {k: (pc[k], ck[k]) for k in CONFIG_KEYS if pc[k] != ck[k]}
            assert not bad, f"--extend-to with a different config (old, new): {bad}"
            start, stop = prev["n_episodes"], args.extend_to
            assert stop > start, f"{out} already has {start} episodes"
        else:
            assert not out.exists(), f"{out} exists (use --extend-to, or a new --out)"
            start, stop = 0, args.episodes
    c = CONFIG_LEGACY | cfg_run  # --replay of a JSON written before cube_yaw_max_deg existed: 180
    N, B, H = c["n"], c["envs"], c["image_size"]
    assert (c["period"], c["latency"]) == (TM.PERIOD, TM.LATENCY), "vla_common.timing changed since the first run"

    # Server first (model load peaks in host RAM), then the simulator (s0b_bridge pattern).
    bridge = None
    if c["policy"] != "expert":
        import vla_tabletop.bridge as BR

        if args.policy_python:
            BR.POLICY_PY = Path(args.policy_python)
        srv = server_args(c)
        log = out.parent / f"{out.stem}{'_replay' if args.replay else ''}_server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        bridge = BR.PolicyBridge(expected_info(c), srv, free_port(), log_path=log)
        if prev is not None:
            assert bridge.info["ckpt_sha"] == prev["server_info"]["ckpt_sha"], "checkpoint changed since the first run"
        if c["policy"] == "smolvla":
            assert bridge.info.get("guidance_w") == (float(c["guidance_w"]) if c.get("guidance_w") else None),                 (bridge.info.get("guidance_w"), c.get("guidance_w"))
        if c["policy"] == "baseline":
            bridge.client = Labelled(bridge.client)
        print(f"[eval] server up after {bridge.load_s:.1f} s: {json.dumps(bridge.info)}", flush=True)

    import gymnasium as gym
    import imageio
    import torch

    import vla_tabletop.scene as S
    from collect import geometry_113, git_sha, task_timing, visual_params
    from isaaclab_tasks.utils import launch_simulation
    from lift_rl.physics import assert_on_cuda, assert_run_preconditions
    from vision_lift import cameras as C
    from vla_common.catalog import ALL_CUBES
    from vla_common.guidance import other_instructions
    from vla_common.tasks import labels as task_labels
    from vla_common.tasks import sample_spec
    from vla_tabletop import visual_rand as VR
    from vla_tabletop.bridge import ChunkExecutor, episode_seed
    from vla_tabletop.detectors import DETECTOR_VERSION
    from vla_tabletop.expert import EXPERT_VERSION
    from vla_tabletop.rollout import Runner, make_stream
    from vla_tabletop.sheets import CLOSEUP, add_closeup, aim_closeup, grab, grid, label

    weights = tuple(float(x) for x in c["mix"].split(","))
    cache: dict = {}
    if c["paired_targets"]:  # stream = the paired episodes of groups 0..groups-1 + B spare groups for the last round's padding
        from vla_tabletop.rollout import make_paired_stream

        pep, _ = make_paired_stream(c["seed"], 0, c["groups"] + B, c["stage"], c["split"], c["cube_yaw_max_deg"])
        cache = {e.index: e for e in pep}
        if not args.replay:
            stop = sum(e.layout_id < c["groups"] for e in pep)

    def episode(i: int):
        if i not in cache:
            cache[i] = make_stream(c["seed"], i, 1, lambda rng: sample_spec(c["stage"], c["split"], rng, weights),
                                   split=c["split"], cube_yaw_max_deg=c["cube_yaw_max_deg"])[0]
        return cache[i]

    sent, swap_rep = plan_instruction_swap([episode(i) for i in range(stop)] if c["instruction_swap"] != "none" else [],
                                           c["instruction_swap"])
    sent_of = lambda e: sent.get(e.index, e.instruction)  # noqa: E731 - slots past `stop` are discarded: own text
    instr_log: list[dict] = []
    rounds = plan_rounds(lambda i: episode(i).spec.task, start, stop, B)
    if args.replay:
        rounds = [r for r in rounds if set(r["indices"]) & set(chosen)]
        r_keep = [[i in chosen for i in r["indices"]] for r in rounds]
    else:
        r_keep = [r["keep"] for r in rounds]
    vid = {int(i) for i in args.videos.split(",") if i}
    assert not args.record or not (args.replay or args.extend_to), "--record: a new run only"
    cams = c["policy"] != "expert" or bool(args.replay) or args.record
    cfg = S.make_cfg(B, c["seed"], res=H if cams else None)
    if args.replay:
        add_closeup(cfg, 320, 240, focal=18.0)
    if args.record:
        from vla_tabletop import demo_rec as DR

        add_closeup(cfg, DR.VIEW_W, DR.VIEW_H, focal=DR.VIEW_FOCAL)
    assert_run_preconditions(cfg)
    mode = VR.run_mode(c["visual"], c["table_palette"])
    frames_dir = out.parent / "frames" / out.stem  # results/<name>.json -> results/frames/<name>/

    with launch_simulation(cfg, {"headless": True, "enable_cameras": cams}):
        t_start = time.time()
        env = gym.make(S.TASK_ID, cfg=cfg)
        u = env.unwrapped
        assert_on_cuda(env)
        print(f"[eval] device={u.device} B={B} H={H} policy={c['policy']} N={N} episodes {start}..{stop - 1} in "
              f"{len(rounds)} rounds{' (replay)' if args.replay else ''}", flush=True)
        run = Runner(env, geometry_113())
        records, server_s, rtt_s, frames_out = [], [], [], []
        dump = None
        for rd, keep in zip(rounds, r_keep):
            t0 = time.time()
            tj = task_timing(rd["task"])
            T = tj["T_steps"]
            run.grasp_dz = torch.tensor([tj["grasp_dz_m"][s] for _, s in ALL_CUBES], device=u.device)
            batch = [episode(i) for i in rd["indices"]]
            dump_round = bool(args.dump_slots) and rd is rounds[0]
            vkey = [e.layout_id for e in batch] if c["paired_targets"] else rd["indices"]  # paired: one look per scene
            vis = visual_params(mode, vkey, c["seed"]) if mode and cams else None
            last_tick = (T - 1) // TM.PERIOD * TM.PERIOD
            sel = [j for j, k in enumerate(keep) if k] if args.replay else []
            st = {"t": -1, "frames": None, "visual": False, "events": {j: [] for j in sel},
                  "video": {j: [] for j in sel if batch[j].index in vid}, "grasp": np.zeros(B, bool),
                  "release": np.zeros(B, bool)}
            rec = DR.Recorder([j for j, k in enumerate(keep) if k]) if args.record else None

            def frames(t: int, uu) -> tuple[np.ndarray, np.ndarray]:
                """Render side + wrist once per camera tick (after this round's visual randomization)."""
                if not st["visual"]:
                    if vis is not None:
                        VR.apply(uu, vis, torch.arange(B, device=uu.device))
                    st["visual"] = True
                if st["t"] != t:
                    st["t"] = t
                    st["frames"] = (C.rgb(uu, "side_cam").to(torch.uint8).cpu().numpy(),
                                    C.rgb(uu, "wrist_cam").to(torch.uint8).cpu().numpy())
                return st["frames"]

            class PolicyCtrl:
                def __init__(self, uu, b, s0):
                    self.u = uu
                    others = ([other_instructions(sent_of(e), e.spec.target[0], [x[0] for x in e.spec.cubes])
                               for e in b] if c.get("guidance_w") else None)
                    self.ex = ChunkExecutor(bridge, N, TM.PERIOD, TM.LATENCY, [sent_of(e) for e in b], others=others)
                    self.ex.log_first = bool(args.log_instructions) and not args.replay
                    if c["policy"] == "baseline":
                        self.ex.labels = bridge.client.labels = labels_sent(b, swap_rep, episode, task_labels)
                    self.ex.reset(np.array([episode_seed(c["seed"], e.index) for e in b], np.int64))
                    self.rec = SlotRecorder(args.dump_slots) if dump_round else None

                def act(self, t, snap_fn):
                    st19 = self.u.obs_buf["policy"].cpu().numpy()
                    a = self.ex.step(t, lambda: frames(t, self.u), st19)
                    if self.rec is not None:
                        self.rec.record(t, st19, a, self.ex)
                    a = torch.from_numpy(a).to(self.u.device)
                    return a, a

            def on_step(t, uu, s, label_, exe, ctrl, det):
                if not cams or t % TM.PERIOD:
                    return
                side, wrist = frames(t, uu)
                if rec is not None:
                    rec.tick(t, uu, s, exe, det, side, wrist)
                if not sel:
                    return
                ar = torch.arange(B, device=uu.device)
                aim_closeup(uu, s["cube_pos"][ar, det.target] + torch.tensor([0.0, 0.0, 0.03], device=uu.device),
                            dist=0.45)
                conf = det.confirmed.cpu().numpy()
                claw = exe[:, 4].cpu().numpy() > 0.5
                ev = {"grasp": conf & ~st["grasp"], "release": conf & ~claw & ~st["release"] & (det.task != 0).cpu().numpy(),
                      "end": np.full(B, t == last_tick)}
                st["grasp"] |= ev["grasp"]
                st["release"] |= ev["release"]
                hits = [(name, j) for name, m in ev.items() for j in sel if m[j]]
                if hits or st["video"]:
                    cu = grab(uu, CLOSEUP)
                    for j in st["video"]:
                        st["video"][j].append(grid([[cu[j], side[j], wrist[j]]], h=240))
                    for name, j in hits:
                        st["events"][j].append((name, t, cu[j], side[j], wrist[j]))

            ctrl = None if c["policy"] == "expert" else PolicyCtrl
            ekw = ({"fresh": True, "sigma": 0.0} if c["expert_fresh"] else
                   {"fresh": False, "n_query": N, "period": TM.PERIOD, "latency": TM.LATENCY, "sigma": 0.0})
            res = run.run(batch, T, expert_kw=ekw, on_step=on_step, controller=ctrl)
            if dump_round:
                dump = run.ctrl.rec.arrays(batch, c["seed"])
            for j, e in enumerate(batch):
                if not keep[j]:
                    continue
                r = {"index": e.index, "task": e.spec.task, "target": "_".join(e.spec.target),
                     "n_cubes": len(e.spec.cubes), "instruction": e.instruction, "spec": e.spec.to_json(), "T": T,
                     "round": rd["round"], "slot": j, **{k: res[k][j].item() for k in RECORD_KEYS}}
                if c["paired_targets"]:
                    r |= {"layout_id": e.layout_id, "member": e.member}
                    if e.relation is not None:  # s3a / s3b only
                        r["relation"] = e.relation
                if sent:
                    r["instruction_sent"] = sent_of(e)
                if c.get("guidance_w"):
                    r["others"] = run.ctrl.ex.others[j]
                if vis is not None:
                    r |= visual_record(vis, j)
                if args.log_instructions and not args.replay and e.index < args.log_instructions and bridge is not None:
                    fq = run.ctrl.ex.first_query
                    ent = {"index": e.index, "slot": j, "target": list(e.spec.target),
                           "instruction": e.instruction, "instruction_sent": fq["sent"][j]}
                    if c["policy"] == "baseline":  # instrument B-I1: the labels the server RECEIVED vs labels(spec)
                        exp = labels_sent([e], swap_rep, episode, task_labels)[0].tolist()
                        got = fq["labels"][j].tolist()
                        ent |= {"labels_server": got, "labels_spec": task_labels(e.spec).tolist(),
                                "labels_expected_sent": exp, "labels_ok": got == exp}
                    else:
                        ent["server_tokens"] = fq["tokens"][j]
                    instr_log.append(ent)
                records.append(r)
                if rec is not None:
                    rec.write(j, out.parent / "demo_rec" / out.stem / f"ep{e.index}",
                              r | {"fail_types": fail_types(r), "ckpt": c["ckpt"], "seed": c["seed"],
                                   "stage": c["stage"], "split": c["split"], "visual": c["visual"],
                                   "table_palette": c["table_palette"]}, 50 / TM.PERIOD)
            for j in sel:
                e, r = batch[j], records[-len(sel) + sel.index(j)]
                old = records_prev[e.index]
                item = {"index": e.index, "types_recorded": fail_types(old), "types_replay": fail_types(r),
                        "reproduced": all(old[k] == r[k] for k in METRICS)}
                frames_dir.mkdir(parents=True, exist_ok=True)
                if st["events"][j]:
                    rows = [[label(cu, [f"ep {e.index} {e.spec.task} {'/'.join(fail_types(old))}"
                                        f"{'' if item['reproduced'] else ' (replay differs)'}", e.instruction[:48],
                                        f"{name} t={t}"]), sd, wr] for name, t, cu, sd, wr in st["events"][j]]
                    f = frames_dir / f"{out.stem}_ep{e.index}_{e.spec.task}_{'_'.join(fail_types(old))}.png"
                    imageio.imwrite(f, grid(rows, h=200))
                    item["sheet"] = str(f)
                if st["video"].get(j):
                    f = frames_dir / f"{out.stem}_ep{e.index}.mp4"
                    imageio.mimwrite(f, st["video"][j], fps=50 / TM.PERIOD, macro_block_size=1)
                    item["video"] = str(f)
                frames_out.append(item)
            if bridge is not None:
                q = run.ctrl.ex.queries
                server_s += [x["server_s"] for x in q[1:]]  # first query of a round excluded (warm-up)
                rtt_s += [x["rtt_s"] for x in q[1:]]
                assert all(x["frame_index"] == x["expected_frame_index"] for x in q), "delayed-frame check failed"
            ks = records[len(records) - sum(keep):]
            print(f"[eval] {rd['task']} round {rd['round']}: {sum(r['success'] for r in ks)}/{len(ks)} success "
                  f"(kept slots), {time.time() - t0:.0f} s; {len(records)} eps, "
                  f"{len(records) / (time.time() - t_start) * 3600:.0f} eps/h", flush=True)
        wall = time.time() - t_start
        if bridge is not None:
            bridge.close()
        if args.replay:
            fj = out.parent / f"{out.stem}_frames.json"
            fj.write_text(json.dumps({"source": str(out), "chosen": chosen, "rounds": len(rounds),
                                      "wall_s": round(wall, 1), "git_sha": git_sha(), "frames": frames_out}, indent=1))
            print("REPLAY_SUMMARY=" + json.dumps(frames_out), flush=True)
            print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
            env.close()
            return
        allrec = sorted((prev or {}).get("episodes", []) + records, key=lambda r: r["index"])
        bars = parse_bars(args.bars) if args.bars else [tuple(b) for b in (prev or {}).get("bars", [])]
        runs = (prev or {}).get("runs", []) + [{
            "episodes": [start, stop], "rounds": len(rounds), "slots_run": len(rounds) * B, "wall_s": round(wall, 1),
            "gpu_h": wall / 3600, "episodes_per_gpu_h": len(records) / wall * 3600,
            "slots_per_gpu_h": len(rounds) * B / wall * 3600, "git_sha": git_sha(), "device": str(u.device),
            "gpu": torch.cuda.get_device_name(0), "torch_peak_vram_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
            "nan_guard_events": u.nan_guard_events,
            "raw_latency": {"server_s": [round(x, 5) for x in server_s], "rtt_s": [round(x, 5) for x in rtt_s]}}]
        all_srv = [x for r in runs for x in r["raw_latency"]["server_s"]]
        all_rtt = [x for r in runs for x in r["raw_latency"]["rtt_s"]]
        res_json = {
            "config": cfg_run, "cube_yaw_max_deg": c["cube_yaw_max_deg"], "n_episodes": len(allrec), "server_info": bridge.info if bridge else None,
            "ckpt": c["ckpt"], "ckpt_sha": bridge.info["ckpt_sha"] if bridge else None,
            "stats_sha": stats_sha(c["ckpt"], c["dataset"]),
            "expert_version": EXPERT_VERSION, "detector_version": DETECTOR_VERSION,
            "T_steps": {t: task_timing(t)["T_steps"] for t in sorted({r["task"] for r in allrec})},
            "aggregate": aggregate(allrec),
            "latency": {"server_s_p50": pct(all_srv, 50), "server_s_p95": pct(all_srv, 95),
                        "rtt_s_p50": pct(all_rtt, 50), "rtt_s_p95": pct(all_rtt, 95), "queries": len(all_srv)},
            "wall_s": sum(r["wall_s"] for r in runs), "gpu_h": sum(r["gpu_h"] for r in runs),
            "episodes_per_gpu_h": len(allrec) / sum(r["wall_s"] for r in runs) * 3600,
            "bars": [list(b) for b in bars], "smart_length": decide(allrec, bars) if bars else None,
            "runs": runs, "episodes": allrec}
        if swap_rep is not None:
            res_json["instruction_swap_report"] = swap_rep
        if instr_log:
            res_json["instruction_log"] = sorted(instr_log, key=lambda x: x["index"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res_json, indent=1))
        if dump is not None:
            np.savez(out.with_name(f"{out.stem}_dump.npz"), **dump)
        a = res_json["aggregate"]
        print("EVAL_SUMMARY=" + json.dumps({"n": len(allrec), "success": {t: a[t]["success"]["rate"] for t in a},
                                            "latency": res_json["latency"], "wall_s": round(wall, 1),
                                            "episodes_per_gpu_h": runs[-1]["episodes_per_gpu_h"],
                                            "slots_per_gpu_h": runs[-1]["slots_per_gpu_h"],
                                            "smart_length": res_json["smart_length"]}), flush=True)
        print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
        env.close()


if __name__ == "__main__":
    main()
