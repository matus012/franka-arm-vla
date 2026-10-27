"""GATES 0C gate item 2: held-out audit on EXPORTED training data (LeRobot v3.0 dirs written by
scripts/export_dataset.py). Reads the exported columns (task string, labels, split, rows per episode) and the
per-episode HDF5 attributes carried in meta/episodes_113.jsonl (spec_json, instruction, seed, visual parameters).

  python tests/test_data_audit.py --root <dataset dir> [<dataset dir> ...]     (real data; exit 1 on any violation)
  AUDIT_ROOT=<dir>[;<dir>] pytest tests/test_data_audit.py                      (same, inside pytest)

Checks per dataset: every episode passes scripts/export_dataset.audit_episode (sampler split train; no held-out target
as the manipulated cube; no held-out place / stack pair; train template, no paraphrase template incl. filler-word
skeletons; matches() = exactly the target / destination; labels = labels(spec); s1 one SEEN cube; data-range seed;
visual mode / train-mode parameters outside the held-out band); exported columns agree with the attributes (task
column = instruction, labels column = labels attr, split column constant per episode, rows = T); the val split
follows the 0C rule (per stream, kept rank k is val iff k % 20 == 19) and meta/split_113.json matches; each episode's
spec + layout regenerates from (seed, episode index, sampler) and equals (1 mm / 1 deg) none of episodes 0-999 of the
test / validation streams assigned in GATES.md. Mutants (synthetic data) are flagged, see the tests below.

Paired streams (counterfactual scenes, collect.py --paired-targets; stream tuple (seed, stage, split, m, "paired")):
group g = the scene of rng([seed, g]) with one episode per valid target (vla_tabletop.rollout.make_paired_stream), so a
stream's "episodes 0-999" are its first 1000 GROUPS. A paired data episode regenerates from (seed, layout_id, member), its
split rule is per kept group (rank k % 20 == 19), and its collision check is by SCENE (same cube set + same layout within
1 mm / 1 deg, whatever the target): a scene of a test / validation stream must not be trained on under any target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_dataset import VAL_EVERY, audit_episode  # noqa: E402
from vla_common.tasks import COLORS, HELD_OUT_PLACE_PAIRS, HELD_OUT_TARGETS, TaskSpec, instruction, sample_spec  # noqa: E402
from vla_tabletop.layout import sample_layout  # noqa: E402
from vla_tabletop.rollout import make_paired_stream  # noqa: E402


S4_SAMPLERS = ("s2", "s3a", "s3b")  # pick / place / stack seeds 11610 + 10 (3 g + j)
S4_TEST_SPLITS = ("train", "train", "heldout_comp", "paraphrase", "train")  # g = 0 (a), 1 (b), 2 (c), 3 paraphrase, 4 (b')


def _streams() -> list[tuple[int, str, str, float]]:
    """(seed, stage sampler, split, cube_yaw_max_deg) of every test / validation stream assigned in GATES.md that uses sample_spec;
    test seeds with attempts +0..+9. (0A.1 grasp and the 0B vision_lift streams use other samplers.)"""
    test = [(11200, "s2", "train"), (11210, "s3a", "train"), (11220, "s3b", "train"),
            (11300, "s2", "train"), (11310, "s3a", "train"), (11320, "s3b", "train"),
            (11400, "s2", "train"), (11410, "s3a", "train"), (11420, "s3b", "train"),
            (11500, "s1", "train"), (11520, "s2", "train"), (11530, "s2", "heldout_comp"), (11540, "s2a", "train"),
            (11550, "s2", "train"), (11560, "s2", "heldout_comp")]
    val = [(21200, "s2", "train"), (21210, "s3a", "train"), (21220, "s3b", "train"),
           (21205, "s2", "heldout_comp"), (21215, "s3a", "heldout_comp"), (21225, "s3b", "heldout_comp"),
           (21300, "s2", "train"), (21310, "s3a", "train"), (21320, "s3b", "train"),
           (21400, "s2", "train"), (21410, "s3a", "train"), (21420, "s3b", "train"),
           (21405, "s2", "train"), (21415, "s3a", "train"), (21425, "s3b", "train"),
           (21600, "s1", "train"), (21610, "s1", "train"), (21620, "s2", "train"), (21630, "s2", "heldout_comp"),
           (21640, "s2a", "train"), (21650, "s2", "train")]
    # Stage 1 redesign #2 (GATES V15 / V0): the tested variant is not known yet, so both m for 11504; expert /
    # reachability 11780 (m 15) / 11790 (m 0); validation 21790 and 21600 at both m (21600 at m 180 is above).
    y2 = [(11504, "s1", "train", 15.0), (11504, "s1", "train", 0.0), (11780, "s1", "train", 15.0),
          (11790, "s1", "train", 0.0), (21790, "s1", "train", 15.0), (21790, "s1", "train", 0.0),
          (21600, "s1", "train", 15.0), (21600, "s1", "train", 0.0)]
    # Stage 2 re-plan (GATES, every stream at m 15): test seeds +0..+9; expert / reachability 11800 (m 15);
    # validation 21620 / 21630; fallback 2a test 11540 / validation 21640; 2b test 11550 / 21650 / held-out 11560.
    s2 = ([(s + k, st, sp, 15.0) for s, st, sp in [(11520, "s2", "train"), (11530, "s2", "heldout_comp"),
                                                   (11540, "s2a", "train"), (11550, "s2", "train"),
                                                   (11560, "s2", "heldout_comp")] for k in range(10)]
          + [(11800, "s2", "train", 15.0), (21620, "s2", "train", 15.0), (21630, "s2", "heldout_comp", 15.0),
             (21640, "s2a", "train", 15.0), (21650, "s2", "train", 15.0)])
    # Stage 2 redesign #2 (2a): expert / reachability 11830 (m 15)
    s2a = [(11830, "s2a", "train", 15.0)]
    # Stage 2 redesign #2, 2a attempt 1 (counterfactual paired scenes): test 11541 = 11540 + 1 (above); validation probe 21645
    # (s2a paired, train, m 15, groups 0-999)
    s2a_paired = [(21645, "s2a", "train", 15.0, "paired"),
                  (21655, "s2", "train", 15.0, "paired")]  # task 008: 2b sizes-round-2 paired probe (s2, groups 0-99)
    # Stage 3A / 3B re-plan (GATES, every stream at m 15): expert / reachability 11810 (s3a) / 11820 (s3b); validation
    # 21660 (s3a) / 21690 (s3b); test 11570 (+k, s3a) / 11590 (+k, s3b), k = 0..9.
    s3 = ([(11810, "s3a", "train", 15.0), (11820, "s3b", "train", 15.0),
           (21660, "s3a", "train", 15.0), (21690, "s3b", "train", 15.0)]
          + [(s + k, st, "train", 15.0) for s, st in [(11570, "s3a"), (11590, "s3b")] for k in range(10)])
    # task 009b (3A / 3B one cube size per scene, samplers s3a1 / s3b1, every stream at m 15): validation 21665 (s3a1) /
    # 21695 (s3b1); test 11580 (+k, s3a1) / 11600 (+k, s3b1), k = 0..9 (the s3a / s3b streams cannot be reused: other sampler).
    s3one = ([(21665, "s3a1", "train", 15.0), (21695, "s3b1", "train", 15.0)]
             + [(s + k, st, "train", 15.0) for s, st in [(11580, "s3a1"), (11600, "s3b1")] for k in range(10)])
    # 009b preconditions + probe: expert / reachability 11840 (s3a1) / 11850 (s3b1) (11830 is the 2a expert stream), m 15;
    # paired validation probes 21667 (s3a1) / 21697 (s3b1) (groups 0-99, train, m 15, as the s2 probes).
    s3one += [(11840, "s3a1", "train", 15.0), (11850, "s3b1", "train", 15.0),
              (21667, "s3a1", "train", 15.0, "paired"), (21697, "s3b1", "train", 15.0, "paired")]
    # Stage 4 re-plan (GATES e2cef02, every stream at m 15, samplers s2 / s3a / s3b per task, no attempt offsets): validation
    # 21720 / 21730 / 21740; tests 11610-11630 (a), 11640-11660 (b), 11670-11690 heldout_comp (c), 11700-11720 paraphrase,
    # 11730-11750 (b').
    s4 = [(21720 + 10 * j, st, "train", 15.0) for j, st in enumerate(S4_SAMPLERS)] + \
         [(11610 + 10 * (3 * g + j), st, sp, 15.0) for g, sp in enumerate(S4_TEST_SPLITS) for j, st in enumerate(S4_SAMPLERS)]
    return [(s + k, st, sp, 180.0) for s, st, sp in test for k in range(10)] + [(*v, 180.0) for v in val] + y2 + s2 + s2a + s2a_paired + s3 + s3one + s4


ASSIGNED_STREAMS = _streams()
# DATA streams (training sets; seeds in the data range 31000-39999): paired 2a attempt 1 = (32800, s2a, train, m 15, paired);
# 2b amended (paired s2 scenes + the contrastive loss) = (32900, s2, train, m 15, paired);
# task 007 (3A / 3B amended, paired place / stack scenes) = (33000, s3a, ...) / (33100, s3b, ...);
# task 009b (3A / 3B one cube size per scene, paired) = (33200, s3a1, ...) / (33300, s3b1, ...)
DATA_STREAMS: list[tuple] = [(32800, "s2a", "train", 15.0, "paired"), (32900, "s2", "train", 15.0, "paired"),
                             (33000, "s3a", "train", 15.0, "paired"), (33100, "s3b", "train", 15.0, "paired"),
                             (33200, "s3a1", "train", 15.0, "paired"), (33300, "s3b1", "train", 15.0, "paired")]
SCENE_STAGES = ("s2a", "s2", "s3a", "s3b", "s3a1", "s3b1")  # samplers whose scenes a paired data set shares


def regenerate(seed: int, i: int, stage: str, split: str, mix: str = "1,1,1",
               cube_yaw_max_deg: float = 180.0) -> tuple[TaskSpec, dict]:
    """vla_tabletop.rollout.make_stream's spec + layout for episode i of seed (same rng order, no torch)."""
    rng = np.random.default_rng([seed, i])
    spec = sample_spec(stage, split, rng, tuple(float(x) for x in mix.split(",")))
    return spec, sample_layout(spec, rng, cube_yaw_max_deg=cube_yaw_max_deg)[0]


def regenerate_paired(seed: int, layout_id: int, member: int, stage: str, split: str, cube_yaw_max_deg: float,
                      cache: dict = {}) -> tuple[TaskSpec, dict]:  # noqa: B006 - memo of the stream's first groups
    """(spec, layout) of member `member` of group `layout_id` (make_paired_stream; groups regenerated in blocks of 1000)."""
    blk = layout_id // 1000
    key = (seed, stage, split, cube_yaw_max_deg, blk)
    if key not in cache:
        eps, _ = make_paired_stream(seed, 0, (blk + 1) * 1000, stage, split, cube_yaw_max_deg)
        cache[key] = {(e.layout_id, e.member): (e.spec, e.layout) for e in eps}
    r = cache[key].get((layout_id, member))
    assert r is not None, f"paired stream {seed}: no episode for group {layout_id} member {member}"
    return r


def scene_key(spec: TaskSpec) -> str:
    return "scene:" + json.dumps(sorted(map(list, spec.cubes))) + json.dumps(spec.pads)


def same_layout(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        dyaw = abs(math.remainder(a[k][2] - b[k][2], 2 * math.pi))
        if abs(a[k][0] - b[k][0]) > 1e-3 or abs(a[k][1] - b[k][1]) > 1e-3 or dyaw > math.radians(1.0):
            return False
    return True


def stream_index(streams: list[tuple], n: int = 1000) -> dict[str, list[tuple]]:
    """streams: (seed, stage, split[, cube_yaw_max_deg = 180[, "paired"]]). Keyed by spec_json (every stream, episode i; for a
    paired stream every member of group i) and by scene_key (paired streams and every stream of a SCENE_STAGES sampler:
    the scene, target-free; the samplers whose scenes paired s2a / s2 / s3a / s3b data sets share)."""
    idx: dict[str, list[tuple]] = {}
    for seed, stage, split, *m in streams:
        yaw = m[0] if m else 180.0
        if "paired" in m[1:]:
            eps, _ = make_paired_stream(seed, 0, n, stage, split, yaw)
            for e in eps:
                idx.setdefault(e.spec.to_json(), []).append((seed, e.layout_id, e.layout))
                if e.member == 0:
                    idx.setdefault(scene_key(e.spec), []).append((seed, e.layout_id, e.layout))
            continue
        for i in range(n):
            spec, lay = regenerate(seed, i, stage, split, cube_yaw_max_deg=yaw)
            idx.setdefault(spec.to_json(), []).append((seed, i, lay))
            if stage in SCENE_STAGES:  # the stages a paired data set shares scenes with
                idx.setdefault(scene_key(spec), []).append((seed, i, lay))
    return idx


def read_exported(root: Path) -> tuple[list[dict], dict[int, dict]]:
    """(episodes_113.jsonl rows, per LeRobot episode: {'rows', 'tasks', 'labels', 'splits'} from the data parquet)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    tt = pq.read_table(root / "meta" / "tasks.parquet")
    tcol = "task" if "task" in tt.column_names else "__index_level_0__"
    tasks = dict(zip(tt.column("task_index").to_pylist(), tt.column(tcol).to_pylist()))
    cols: dict[int, dict] = {}
    for f in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "task_index", "labels", "split"])
        for ep in pc.unique(t.column("episode_index")).to_pylist():
            m = t.filter(pc.equal(t.column("episode_index"), ep))
            d = cols.setdefault(ep, {"rows": 0, "tasks": set(), "labels": set(), "splits": set()})
            d["rows"] += m.num_rows
            d["tasks"] |= {tasks[k] for k in set(m.column("task_index").to_pylist())}
            d["labels"] |= {tuple(x) for x in m.column("labels").to_pylist()}
            d["splits"] |= {x[0] if isinstance(x, list) else x for x in m.column("split").to_pylist()}
    return side, cols


def audit_dataset(root: Path, streams: list[tuple] = ASSIGNED_STREAMS) -> list[str]:
    side, cols = read_exported(root)
    bad = []
    for s in side:
        ep, tag = s["lerobot_episode_index"], f"ep {s['lerobot_episode_index']} (seed {s['seed']} i {s['episode_index']})"
        bad += [f"{tag}: {v}" for v in audit_episode(s)]
        c = cols.get(ep)
        if c is None:
            bad.append(f"{tag}: no rows in the data parquet")
            continue
        lab = tuple(json.loads(s["labels"]) if isinstance(s["labels"], str) else s["labels"])
        if c["rows"] != int(s["T"]):
            bad.append(f"{tag}: {c['rows']} rows != T {s['T']}")
        if c["tasks"] != {s["instruction"]}:
            bad.append(f"{tag}: task column {c['tasks']} != instruction {s['instruction']!r}")
        if c["labels"] != {lab}:
            bad.append(f"{tag}: labels column {c['labels']} != {lab}")
        if c["splits"] != {s["split"]}:
            bad.append(f"{tag}: split column {c['splits']} != {s['split']!r}")
    if len(cols) != len(side):
        bad.append(f"{len(cols)} episodes in the data, {len(side)} in episodes_113.jsonl")
    # 0C split rule per stream, and the recorded split file
    for key in sorted({(s["file_attrs"]["stage"], int(s["seed"])) for s in side}):
        ss = sorted((s for s in side if (s["file_attrs"]["stage"], int(s["seed"])) == key),
                    key=lambda s: (int(s["episode_index"])))
        if any(s.get("paired") for s in ss):  # paired: the unit is the group, rank k of the kept groups
            assert all(s.get("paired") for s in ss), f"stream {key}: paired and unpaired episodes mixed"
            order = sorted({int(s["layout_id"]) for s in ss})
            for s in ss:
                k = order.index(int(s["layout_id"]))
                want = "val" if k % VAL_EVERY == VAL_EVERY - 1 else "train"
                if s["split"] != want:
                    bad.append(f"stream {key}: kept group rank {k} (layout {s['layout_id']}, i {s['episode_index']}) "
                               f"split {s['split']} != {want}")
        else:
            for k, s in enumerate(ss):
                want = "val" if k % VAL_EVERY == VAL_EVERY - 1 else "train"
                if s["split"] != want:
                    bad.append(f"stream {key}: kept rank {k} (i {s['episode_index']}) split {s['split']} != {want}")
    spf = root / "meta" / "split_113.json"
    if not spf.exists():
        bad.append("meta/split_113.json missing")
    else:
        sp = json.loads(spf.read_text())
        for name in ("train", "val"):
            if sorted(sp[name]) != sorted(s["lerobot_episode_index"] for s in side if s["split"] == name):
                bad.append(f"split_113.json '{name}' list != the split column")
    # regeneration + test / validation stream collisions
    index = stream_index(streams)
    for s in side:
        fa = s["file_attrs"]
        paired = bool(s.get("paired"))
        if paired:
            spec, lay = regenerate_paired(int(s["seed"]), int(s["layout_id"]), int(s["member"]), fa["stage"],
                                          s.get("sampler_split", "train"), float(s.get("cube_yaw_max_deg", 180.0)))
        else:
            spec, lay = regenerate(int(s["seed"]), int(s["episode_index"]), fa["stage"], s.get("sampler_split", "train"),
                                   fa.get("mix", "1,1,1"), float(s.get("cube_yaw_max_deg", 180.0)))
        if spec.to_json() != s["spec_json"]:
            bad.append(f"ep {s['lerobot_episode_index']}: spec does not regenerate from (seed, "
                       f"{'layout_id, member' if paired else 'index'}, sampler)")
        for seed, i, other in index.get(scene_key(spec) if paired else spec.to_json(), []):
            if same_layout(lay, other):
                bad.append(f"ep {s['lerobot_episode_index']}: spec + layout equal to test/validation stream {seed} "
                           f"episode {i}")
    return bad


# --------------------------------------------------------------------------- synthetic exported dataset
SEED, STAGE = 39990, "s3a"


def make_synthetic(root: Path, n: int = 40) -> Path:
    """A minimal exported-dataset dir (the files the audit reads) of n kept s3a episodes of stream 39990."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vla_common.tasks import labels

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    side, rows, tasks = [], {"episode_index": [], "task_index": [], "labels": [], "split": []}, []
    for k in range(n):
        rng = np.random.default_rng([SEED, k])
        spec = sample_spec(STAGE, "train", rng)
        sample_layout(spec, rng)
        instr = instruction(spec, rng, "train")
        s = {"instruction": instr, "split": "val" if k % 20 == 19 else "train", "sampler_split": "train",
             "task": spec.task, "spec_json": spec.to_json(), "labels": json.dumps(labels(spec).tolist()), "seed": SEED,
             "episode_index": k, "T": 5, "stage": STAGE, "visual_mode": "nominal",
             "visual_json": json.dumps({"mode": "nominal"}), "lerobot_episode_index": k,
             "file_attrs": {"stage": STAGE, "split": "train", "seed": SEED, "mix": "1,1,1"}}
        side.append(s)
        if instr not in tasks:
            tasks.append(instr)
        for _ in range(5):
            rows["episode_index"].append(k)
            rows["task_index"].append(tasks.index(instr))
            rows["labels"].append(labels(spec).tolist())
            rows["split"].append(s["split"])
    write_synthetic(root, side, rows, tasks)
    pq.write_table(pa.table({"task_index": list(range(len(tasks))), "task": tasks}), root / "meta" / "tasks.parquet")
    return root


def write_synthetic(root: Path, side: list[dict], rows: dict, tasks: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "split_113.json").write_text(json.dumps(
        {name: [s["lerobot_episode_index"] for s in side if s["split"] == name] for name in ("train", "val")}))


def _mutate(root: Path, fn) -> None:
    """Apply fn(side rows) to the synthetic dataset's episodes_113.jsonl (attributes = what the HDF5 carried)."""
    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    fn(side)
    (root / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))


FEW = [(11210, "s3a", "train")]  # unit tests: one assigned stream (the full list takes ~1 min)


def test_synthetic_clean(tmp_path):
    assert audit_dataset(make_synthetic(tmp_path / "ds"), FEW) == []


def test_mutant_heldout_place_pair(tmp_path):
    """Coordinator mutant: one held-out place pair injected into training data is caught."""
    root = make_synthetic(tmp_path / "ds")
    (cube, pad) = HELD_OUT_PLACE_PAIRS[0]

    def inject(side):
        spec = TaskSpec("place", cube, pad, [cube, ("blue", "small")], list(COLORS))
        side[3]["spec_json"] = spec.to_json()
        side[3]["instruction"] = f"put the {cube[0]} cube on the {pad} pad"
    _mutate(root, inject)
    bad = audit_dataset(root, FEW)
    assert any("held-out place pair" in b for b in bad), bad


def test_mutant_heldout_target(tmp_path):
    root = make_synthetic(tmp_path / "ds")
    t = HELD_OUT_TARGETS[0]
    _mutate(root, lambda side: side[0].update(spec_json=TaskSpec("place", t, "red", [t], list(COLORS)).to_json()))
    assert any("held-out target" in b for b in audit_dataset(root, FEW))


def test_mutant_paraphrase_instruction(tmp_path):
    root = make_synthetic(tmp_path / "ds")

    def para(side):
        spec = TaskSpec.from_json(side[1]["spec_json"])
        side[1]["instruction"] = instruction(spec, np.random.default_rng(0), "paraphrase")
    _mutate(root, para)
    bad = audit_dataset(root, FEW)
    assert any("paraphrase template" in b for b in bad) and any("task column" in b for b in bad)


def test_mutant_heldout_band_visuals(tmp_path):
    root = make_synthetic(tmp_path / "ds")
    vis = {"mode": "train", "table_rgb": [0.25, 0.4, 0.35], "light": [-0.95, 0.0, -0.3]}  # hue 0.43, value 0.4

    _mutate(root, lambda side: side[2].update(visual_mode="train_rand", visual_json=json.dumps(vis)))
    assert any("held-out band" in b for b in audit_dataset(root, FEW))


def test_s4r_visuals_audit(tmp_path):
    """s4r palette: a train-mode grey table passes; held-out dark grey, held-out low light or a tinted table are flagged."""
    from export_dataset import visual_ok

    ok = {"mode": "s4r_train", "table_rgb": [0.5, 0.5, 0.5], "light": [-0.57735026, 0.57735026, -0.57735026]}
    assert visual_ok({"visual_mode": "train_rand", "visual_json": json.dumps(ok)}) is None
    assert visual_ok({"visual_mode": "nominal", "visual_json": json.dumps({"mode": "s4r_nominal"})}) is None
    for bad in ({"table_rgb": [0.12, 0.12, 0.12]}, {"light": [-0.92, 0.0, -0.39]}, {"table_rgb": [0.5, 0.3, 0.3]}):
        assert "s4r held-out band" in visual_ok({"visual_mode": "train_rand", "visual_json": json.dumps(ok | bad)}), bad
    root = make_synthetic(tmp_path / "ds")
    _mutate(root, lambda side: side[2].update(visual_mode="train_rand",
                                              visual_json=json.dumps(ok | {"table_rgb": [0.9, 0.9, 0.9]})))
    assert any("s4r held-out band" in b for b in audit_dataset(root, FEW))


def test_mutant_split_rule_and_collision(tmp_path):
    root = make_synthetic(tmp_path / "ds")
    _mutate(root, lambda side: side[0].update(split="val"))
    bad = audit_dataset(root, FEW)
    assert any("kept rank 0" in b for b in bad) and any("split column" in b for b in bad)
    # an episode copied from a test stream (spec + layout) is a collision
    root2 = make_synthetic(tmp_path / "ds2")
    spec, _ = regenerate(11210, 7, "s3a", "train")
    _mutate(root2, lambda side: side[4].update(seed=11210, episode_index=7, spec_json=spec.to_json()))
    bad = audit_dataset(root2, FEW)
    assert any("test/validation stream 11210 episode 7" in b for b in bad) and any("seed 11210" in b for b in bad)


def test_redesign2_streams_and_sidecar_yaw(tmp_path):
    """A data episode copied from a redesign #2 stream collides only when its sidecar cube_yaw_max_deg matches."""
    spec, _ = regenerate(11780, 7, "s1", "train", cube_yaw_max_deg=15.0)
    assert (11780, "s1", "train", 15.0) in ASSIGNED_STREAMS and (21790, "s1", "train", 0.0) in ASSIGNED_STREAMS
    streams = [(11780, "s1", "train", 15.0)]
    for m, hit in ((15.0, True), (None, False)):
        root = make_synthetic(tmp_path / f"ds{m}")
        _mutate(root, lambda side: side[4].update(seed=11780, episode_index=7, spec_json=spec.to_json(),
                                                  file_attrs={**side[4]["file_attrs"], "stage": "s1"},
                                                  **({} if m is None else {"cube_yaw_max_deg": m})))
        bad = audit_dataset(root, streams)
        assert any("test/validation stream 11780 episode 7" in b for b in bad) == hit, (m, bad)


def test_stage3_streams_assigned():
    """GATES 3A / 3B re-plan streams at m 15: expert 11810 / 11820, validation 21660 / 21690, test +0..+9."""
    want = ([(11810, "s3a"), (11820, "s3b"), (21660, "s3a"), (21690, "s3b")]
            + [(s + k, st) for s, st in ((11570, "s3a"), (11590, "s3b")) for k in range(10)])
    assert all((s, st, "train", 15.0) in ASSIGNED_STREAMS for s, st in want)
    spec, _ = regenerate(21660, 3, "s3a", "train", cube_yaw_max_deg=15.0)
    assert spec.task == "place"
    assert regenerate(21690, 3, "s3b", "train", cube_yaw_max_deg=15.0)[0].task == "stack"


def test_stage4_streams_assigned():
    """Stage 4 re-plan streams at m 15: validation 21720 / 21730 / 21740 (train), tests 11610-11750 per GATES."""
    want = {21720: ("s2", "train"), 21730: ("s3a", "train"), 21740: ("s3b", "train"),
            11610: ("s2", "train"), 11620: ("s3a", "train"), 11630: ("s3b", "train"),
            11640: ("s2", "train"), 11650: ("s3a", "train"), 11660: ("s3b", "train"),
            11670: ("s2", "heldout_comp"), 11680: ("s3a", "heldout_comp"), 11690: ("s3b", "heldout_comp"),
            11700: ("s2", "paraphrase"), 11710: ("s3a", "paraphrase"), 11720: ("s3b", "paraphrase"),
            11730: ("s2", "train"), 11740: ("s3a", "train"), 11750: ("s3b", "train")}
    for seed, (st, sp) in want.items():
        assert (seed, st, sp, 15.0) in ASSIGNED_STREAMS, seed
    assert regenerate(21740, 3, "s3b", "train", cube_yaw_max_deg=15.0)[0].task == "stack"
    assert regenerate(11700, 3, "s2", "paraphrase", cube_yaw_max_deg=15.0)[0].task == "pick"


def test_stage4_collision_flagged_at_m15(tmp_path):
    """A Stage 4 data episode equal to validation 21730 episode 7 (s3a, m 15) collides; at another m it does not."""
    spec, _ = regenerate(21730, 7, "s3a", "train", cube_yaw_max_deg=15.0)
    streams = [(21730, "s3a", "train", 15.0)]
    for m, hit in ((15.0, True), (None, False)):
        root = make_synthetic(tmp_path / f"s4_{m}")
        _mutate(root, lambda side: side[4].update(seed=21730, episode_index=7, spec_json=spec.to_json(),
                                                  **({} if m is None else {"cube_yaw_max_deg": m})))
        assert any("test/validation stream 21730 episode 7" in b for b in audit_dataset(root, streams)) == hit, m


def s4r_vis(**over) -> dict:
    return {"mode": "s4r_train", "table_rgb": [0.5, 0.5, 0.5], "light": [-0.57735026, 0.57735026, -0.57735026]} | over


def test_s4r_train_palette_exact_and_mutants(tmp_path):
    """Inside the s4r train palette (sat <= 0.06, value 0.35-0.75 or the nominal grey, light elevation 30-80 or default)
    passes; held-out-band parameters, an off-palette value / light and a 112-palette table in an s4r set are flagged."""
    from export_dataset import visual_ok

    def rec(vj, palette="s4r", mode="train_rand"):
        return {"visual_mode": mode, "visual_json": json.dumps(vj), "table_palette": palette}

    el = math.radians(60.0)
    light60 = [-math.cos(el), 0.0, -math.sin(el)]
    for good in (s4r_vis(), s4r_vis(table_rgb=[0.36, 0.36, 0.37], light=light60), s4r_vis(table_rgb=[0.74, 0.72, 0.74])):
        assert visual_ok(rec(good)) is None, good
    assert visual_ok(rec({"mode": "s4r_nominal"}, mode="nominal")) is None
    flagged = {"held-out dark": s4r_vis(table_rgb=[0.12, 0.12, 0.12]), "held-out light": s4r_vis(table_rgb=[0.9] * 3),
               "held-out low light": s4r_vis(light=[-0.92, 0.0, -0.39]),
               "off-palette value (between bands)": s4r_vis(table_rgb=[0.28, 0.28, 0.28]),
               "off-palette value (above)": s4r_vis(table_rgb=[0.78, 0.78, 0.78]),
               "off-palette light (steep)": s4r_vis(light=[-0.05, 0.0, -0.9987]),
               "112 table under s4r mode": s4r_vis(table_rgb=list(DEFAULT_112))}
    for name, vj in flagged.items():
        assert visual_ok(rec(vj)) is not None, name
    assert "table_palette" in visual_ok(rec({"mode": "train", "table_rgb": list(DEFAULT_112), "light": s4r_vis()["light"]}))
    assert "table_palette" in visual_ok(rec(s4r_vis(), palette="112"))
    # inside a dataset: the 112-table episode and the held-out-band episode are both reported
    root = make_synthetic(tmp_path / "ds")
    _mutate(root, lambda side: [side[i].update(visual_mode="train_rand", table_palette="s4r",
                                               visual_json=json.dumps(v)) for i, v in
                                ((1, s4r_vis(table_rgb=list(DEFAULT_112))), (2, s4r_vis(table_rgb=[0.9] * 3)),
                                 (3, s4r_vis()))])
    bad = audit_dataset(root, FEW)
    assert sum("s4r held-out band or off-palette" in b for b in bad) == 2 and not any(b.startswith("ep 3:") for b in bad), bad


DEFAULT_112 = (0.267, 0.467, 0.667)


def make_synthetic_paired(root: Path, seed: int = 32800, n_groups: int = 45, stage: str = "s2a") -> Path:
    """Exported-dataset dir of the first n_groups paired `stage` groups of `seed` (all members kept), split per group."""
    from vla_common.tasks import labels

    eps, _ = make_paired_stream(seed, 0, n_groups, stage, "train", 15.0)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    order = sorted({e.layout_id for e in eps})
    side, rows, tasks = [], {"episode_index": [], "task_index": [], "labels": [], "split": []}, []
    for k, e in enumerate(eps):
        sp = "val" if order.index(e.layout_id) % 20 == 19 else "train"
        side.append({"instruction": e.instruction, "split": sp, "sampler_split": "train", "task": e.spec.task,
                     "spec_json": e.spec.to_json(), "labels": json.dumps(labels(e.spec).tolist()), "seed": seed,
                     "episode_index": e.index, "T": 5, "stage": stage, "visual_mode": "nominal",
                     "visual_json": json.dumps({"mode": "nominal"}), "lerobot_episode_index": k, "paired": True,
                     "layout_id": e.layout_id, "member": e.member, "cube_yaw_max_deg": 15.0,
                     "file_attrs": {"stage": stage, "split": "train", "seed": seed, "mix": "1,1,1"}})
        if e.instruction not in tasks:
            tasks.append(e.instruction)
        for _ in range(5):
            rows["episode_index"].append(k)
            rows["task_index"].append(tasks.index(e.instruction))
            rows["labels"].append(labels(e.spec).tolist())
            rows["split"].append(sp)
    write_synthetic(root, side, rows, tasks)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"task_index": list(range(len(tasks))), "task": tasks}), root / "meta" / "tasks.parquet")
    return root


PROBE = [(21645, "s2a", "train", 15.0, "paired")]


def test_paired_streams_registered():
    assert (21645, "s2a", "train", 15.0, "paired") in ASSIGNED_STREAMS
    assert (21655, "s2", "train", 15.0, "paired") in ASSIGNED_STREAMS  # task 008 s2 probe
    assert (11541, "s2a", "train", 15.0) in ASSIGNED_STREAMS  # 2a attempt-1 test = 11540 + 1
    assert DATA_STREAMS == [(32800, "s2a", "train", 15.0, "paired"), (32900, "s2", "train", 15.0, "paired"),
                            (33000, "s3a", "train", 15.0, "paired"), (33100, "s3b", "train", 15.0, "paired"),
                            (33200, "s3a1", "train", 15.0, "paired"), (33300, "s3b1", "train", 15.0, "paired")]
    assert all(31000 <= d[0] <= 39999 for d in DATA_STREAMS)
    assert not {d[0] for d in DATA_STREAMS} & {a[0] for a in ASSIGNED_STREAMS}


def test_s3_one_size_expert_and_probe_streams_registered():
    for seed, st in ((11840, "s3a1"), (11850, "s3b1")):
        assert (seed, st, "train", 15.0) in ASSIGNED_STREAMS
    for seed, st in ((21667, "s3a1"), (21697, "s3b1")):
        assert (seed, st, "train", 15.0, "paired") in ASSIGNED_STREAMS
    # 11830 is the 2a expert stream (s2a), so the 009b expert streams are 11840 / 11850; each new seed is registered once
    for seed in (11840, 11850, 21667, 21697):
        assert sum(a[0] == seed for a in ASSIGNED_STREAMS) == 1, seed


def test_paired_s2_synthetic_clean_and_scene_collision_with_an_s2_stream(tmp_path):
    """A paired s2 set (data stream 32900) audits clean; registered as an s2 TEST stream, its scenes collide (any target)."""
    root = make_synthetic_paired(tmp_path / "p", seed=32900, n_groups=30, stage="s2")
    assert audit_dataset(root, PROBE) == []
    bad = audit_dataset(root, [(32900, "s2", "train", 15.0)])  # the unpaired s2 stream of the same seed: same scenes
    assert any("equal to test/validation stream 32900" in b for b in bad), bad


# task 007: the s3a / s3b test / validation streams a paired s3a / s3b training set must not share a scene with
S3_EVAL = [s for s in ASSIGNED_STREAMS if s[1] in ("s3a", "s3b") and len(s) > 3 and s[3] == 15.0
           and (11570 <= s[0] <= 11599 or s[0] in (21660, 21690))]


def test_s3_eval_streams_registered_and_scene_indexed():
    assert len(S3_EVAL) == 22 and {(21660, "s3a"), (21690, "s3b"), (11570, "s3a"), (11599, "s3b")} <= {s[:2] for s in S3_EVAL}
    idx = stream_index([(21660, "s3a", "train", 15.0), (21690, "s3b", "train", 15.0)], n=5)
    for seed, stage in ((21660, "s3a"), (21690, "s3b")):
        spec, lay = regenerate(seed, 2, stage, "train", cube_yaw_max_deg=15.0)
        assert any(h[:2] == (seed, 2) and h[2] == lay for h in idx[scene_key(spec)])


@pytest.mark.parametrize("stage,seed", [("s3a", 33000), ("s3b", 33100)])
def test_paired_s3_clean_vs_s3_eval_streams_and_scene_collision(tmp_path, monkeypatch, stage, seed):
    """A paired s3a / s3b set (data streams 33000 / 33100) audits clean against the s3a / s3b test + validation streams;
    registered as an unpaired s3 stream of the same seed, its scenes collide under every target; without s3a / s3b in the
    scene index (the pre-007 index) that collision is missed."""
    root = make_synthetic_paired(tmp_path / "p", seed=seed, n_groups=30, stage=stage)
    assert audit_dataset(root, S3_EVAL) == []
    same = [(seed, stage, "train", 15.0)]
    bad = audit_dataset(root, same)
    n_eps = len((root / "meta" / "episodes_113.jsonl").read_text().splitlines())
    assert sum(f"equal to test/validation stream {seed}" in b for b in bad) == n_eps, bad[:3]
    monkeypatch.setattr(sys.modules[__name__], "SCENE_STAGES", ("s2a", "s2"))
    assert audit_dataset(root, same) == []


def test_stream_index_paired_has_scene_keys():
    idx = stream_index(PROBE, n=40)
    spec, lay = regenerate_paired(21645, 3, 0, "s2a", "train", 15.0)
    assert any(h[:2] == (21645, 3) for h in idx[scene_key(spec)])
    assert any(h[:2] == (21645, 3) for h in idx[spec.to_json()])


def test_paired_synthetic_clean_and_group_split_rule(tmp_path):
    root = make_synthetic_paired(tmp_path / "p")
    assert audit_dataset(root, PROBE) == []


def test_mutant_paired_group_split_across_train_val(tmp_path):
    root = make_synthetic_paired(tmp_path / "p")
    _mutate(root, lambda side: side[1].update(split="val"))
    bad = audit_dataset(root, PROBE)
    assert any("kept group rank 0" in b for b in bad) and any("split column" in b for b in bad), bad


def test_mutant_paired_scene_collides_with_probe_under_any_target(tmp_path):
    """A paired data set whose scenes ARE the probe's (here: the same seed stream registered as validation) is flagged
    for every member; a different data seed is clean."""
    root = make_synthetic_paired(tmp_path / "p")
    bad = audit_dataset(root, [(32800, "s2a", "train", 15.0, "paired")])
    assert sum("test/validation stream 32800" in b for b in bad) >= 45 and any("episode 0" in b for b in bad)
    root2 = make_synthetic_paired(tmp_path / "q", seed=32801)
    assert audit_dataset(root2, [(32800, "s2a", "train", 15.0, "paired")]) == []
    # scene-level: the member's own spec is not in the index (only the first member of each probe group is), the scene is
    idx = stream_index([(32800, "s2a", "train", 15.0, "paired")], n=3)
    spec, _ = regenerate_paired(32800, 0, 1, "s2a", "train", 15.0)
    del idx[spec.to_json()]
    assert scene_key(spec) in idx


def test_unpaired_audit_unchanged_by_paired_support(tmp_path):
    assert audit_dataset(make_synthetic(tmp_path / "u"), FEW) == []


def test_split_rule_in_export_select():
    from export_dataset import select

    eps = [{"file": "f", "group": f"ep_{k}", "stage": "s2", "seed": 31800, "episode_index": 3 * k} for k in range(45)]
    ch = select(eps[::-1], 40)
    assert len(ch) == 40 and ("f", "ep_40") not in ch
    assert [g for (_, g), sp in sorted(ch.items(), key=lambda x: int(x[0][1][3:])) if sp == "val"] == ["ep_19", "ep_39"]


@pytest.mark.skipif(not os.environ.get("AUDIT_ROOT"), reason="set AUDIT_ROOT to exported dataset dir(s)")
@pytest.mark.parametrize("root", [p for p in os.environ.get("AUDIT_ROOT", "").split(os.pathsep) if p])
def test_real_dataset(root):
    bad = audit_dataset(Path(root))
    assert bad == [], bad[:20]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True, type=Path)
    a = ap.parse_args()
    fails = 0
    for r in a.root:
        b = audit_dataset(r)
        print(json.dumps({"dataset": str(r), "violations": len(b), "first": b[:20]}, indent=1))
        fails += bool(b)
    sys.exit(1 if fails else 0)
