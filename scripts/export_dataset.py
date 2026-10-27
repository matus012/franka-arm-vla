"""Collector HDF5 dirs -> ONE LeRobot v3.0 dataset (policy venv): held-out audit, kept-count selection, train / val split.

  python scripts/export_dataset.py --src /data/0c/s2/b* --name 0c_s2_pick --root /data/datasets --kept 2000 --workers 8

1. Inputs: every `*.h5` in the --src dirs (sorted by dir, then name). Each dir's MANIFEST.sha256 (scripts/collect.py)
   is verified first (--no-verify skips it).
2. Selection (GATES 0C): per data stream (stage, seed), the kept set = the first --kept successful episodes in episode
   index order (0 = all). Split: the stream's kept episode of rank k is `val` if k % 20 == 19 (5 %), else `train`;
   written to the per-episode `split` column (the sampler split, "train", moves to the `sampler_split` attribute of
   meta/episodes_113.jsonl) and listed in meta/split_113.json (LeRobot episode indices).
3. Audit (attributes only, before anything is written; any violation -> exit 1, nothing exported): `audit_episode`
   per episode — sampler split "train"; manipulated cube (pick / place target, stack top) not in HELD_OUT_TARGETS; no
   HELD_OUT_PLACE_PAIRS / HELD_OUT_STACK_PAIRS pair; the instruction fully matches a train template of its task and no
   paraphrase template (also after removing filler words); matches() / matches_dest() give exactly the spec's target /
   destination; labels attr = labels(spec); s1: one cube from SEEN_CUBES; seed in the data range 31000-39999;
   visual_mode nominal with nominal parameters, or train_rand with train-mode parameters outside the held-out band
   (either palette: modes train / s4r_train, nominal / s4r_nominal; visual_rand module doc).
   Result -> meta/audit_113.json. The dataset-level audit (exported columns, split rule, test / validation stream
   collisions) is tests/test_data_audit.py.
   --paired (counterfactual paired scenes, collect.py --paired-targets; default off = the rules above unchanged): the
   unit is the GROUP (layout_id) of stream `<stage>_paired_s<seed>` (s2a, s2, s3a, s3b, s3a1 or s3b1; a union of streams, task 007, is
   selected per stream with --kept each). Walk the groups in layout_id order; keep a group iff ALL
   its member episodes (the group's episode list in the src dirs' paired.json) succeeded and the kept total stays <=
   --kept (a group that would cross --kept, or leave a remainder of 1 that no group of >= 2 can fill, is skipped and the
   walk continues, so the total is exactly --kept; fewer reachable = error). Skipped groups and the reason are recorded (meta/split_113.json `paired`). Split per GROUP:
   kept-group rank k is `val` iff k % 20 == 19, so a layout's episodes are never split across train / val.
4. Export: vla_policy.export.export (control-rate rows, h264 4:4:4 crf 14). Sharding: with --workers P > 1 the files are
   split into P contiguous groups, each exported by its own process to <root>/<name>.shards/<name>_shardKK (LeRobot's
   writer is one process per dataset), then merged in shard order by LeRobot 0.6.1's
   `lerobot.datasets.aggregate.aggregate_datasets` (copies parquet / mp4 without re-encoding, re-aggregates stats);
   the per-episode sidecars (meta/episodes_113.jsonl) are concatenated with re-based `lerobot_episode_index`.
   The selection / split is applied by filtering `vla_policy.export.read_episodes` inside each worker (export() takes
   whole files; see the needed-change note in the 0C hand-off).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_common.tasks import (HELD_OUT_PLACE_PAIRS, HELD_OUT_STACK_PAIRS, HELD_OUT_TARGETS, SEEN_CUBES,  # noqa: E402
                              TEMPLATES, TaskSpec, labels, matches, matches_dest, skeleton)

_NP = r"the (?:(?:small|large) )?(?:red|green|blue|yellow) (?:cube|pad)"
DATA_SEEDS = (31000, 39999)  # GATES seed registry: data range
VAL_EVERY = 20  # GATES 0C: kept rank k is val iff k % 20 == 19


def template_regex(tpl: str) -> re.Pattern:
    parts = re.split(r"(\{[td]\})", tpl)
    return re.compile("".join(_NP if p in ("{t}", "{d}") else re.escape(p) for p in parts))


_RX = {split: {k: [template_regex(t) for t in ts] for k, ts in pool.items()} for split, pool in TEMPLATES.items()}
_PARA_SKEL = {skeleton(t) for ts in TEMPLATES["paraphrase"].values() for t in ts}


def template_split(instruction: str, spec: TaskSpec) -> set[str]:
    """Which template sets ('train', 'paraphrase') produce this instruction for this spec's task."""
    key = "s1" if spec.task == "pick" and len(spec.cubes) == 1 else spec.task
    return {split for split, pool in _RX.items() if any(rx.fullmatch(instruction) for rx in pool[key])}


def instruction_skeleton(instruction: str) -> tuple[str, ...]:
    return skeleton(re.sub(_NP, "{t}", instruction))


def visual_ok(attrs: dict) -> str | None:
    vj = attrs.get("visual_json")
    vj = json.loads(vj) if isinstance(vj, str) else vj
    pal = attrs.get("table_palette")  # Stage 4: "s4r" sets carry s4r_* modes only; absent / "112" sets never do
    mode = None if vj is None else vj.get("mode")
    if pal is not None and (pal == "s4r") != str(mode).startswith("s4r_") and mode is not None:
        return f"table_palette {pal} with visual mode {mode}"
    if attrs["visual_mode"] == "nominal":
        return None if vj is None or mode in ("nominal", "s4r_nominal") else f"nominal episode with visual {mode}"
    if vj is None or mode not in ("train", "s4r_train"):
        return f"visual_mode {attrs['visual_mode']} without train-mode parameters"
    import torch

    from vla_tabletop import visual_rand as VR

    h, s, v = VR.rgb_to_hsv(torch.tensor(vj["table_rgb"])).tolist()
    elev = VR.light_elevation_deg(vj["light"])
    if mode == "s4r_train":  # s4r palette (visual_rand module doc): inside train (sat cap, value band, light), outside held-out
        band, tr = VR.S4R_MODES["heldout"], VR.S4R_MODES["train"]
        nominal_grey = abs(v - VR.S4R_DEFAULT_TABLE[0]) < 1e-4 and s < 1e-4
        if s > VR.S4R_SAT_CAP + 1e-5 or any(lo <= v <= hi for lo, hi in band["values"]) or                 band["light_elev"][0] <= elev <= band["light_elev"][1] or                 not (nominal_grey or any(lo - 1e-4 <= v <= hi + 1e-4 for lo, hi in tr["values"])) or                 not (tr["light_elev"][0] - 1e-3 <= elev <= tr["light_elev"][1] + 1e-3):
            return f"visual parameters in the s4r held-out band or off-palette (sat {s:.3f}, value {v:.3f}, "                    f"light elevation {elev:.1f})"
        return None
    band = VR.MODES["heldout"]
    if any(lo <= h <= hi for lo, hi in band["hues"]) or band["value"][0] <= v <= band["value"][1] or             band["light_elev"][0] <= elev <= band["light_elev"][1]:
        return f"visual parameters in the held-out band (hue {h:.3f}, value {v:.3f}, light elevation {elev:.1f})"
    return None


def audit_episode(attrs: dict) -> list[str]:
    """Violations of the training-set rules for one episode's HDF5 attributes (empty list = clean)."""
    spec = TaskSpec.from_json(attrs["spec_json"])
    instr = attrs["instruction"]
    bad = []
    if attrs.get("sampler_split", attrs["split"]) != "train":
        bad.append(f"split={attrs.get('sampler_split', attrs['split'])}")
    if tuple(spec.target) in HELD_OUT_TARGETS:
        bad.append(f"held-out target {spec.target}")
    if spec.task == "place" and (tuple(spec.target), spec.dest) in HELD_OUT_PLACE_PAIRS:
        bad.append(f"held-out place pair {spec.target}->{spec.dest}")
    if spec.task == "stack" and (tuple(spec.target), tuple(spec.dest)) in HELD_OUT_STACK_PAIRS:
        bad.append(f"held-out stack pair {spec.target}->{spec.dest}")
    if attrs["task"] != spec.task:
        bad.append(f"task attr {attrs['task']} != spec {spec.task}")
    tsplit = template_split(instr, spec)
    if "paraphrase" in tsplit or instruction_skeleton(instr) in _PARA_SKEL:
        bad.append(f"paraphrase template: {instr!r}")
    if "train" not in tsplit:
        bad.append(f"no train template matches: {instr!r}")
    if len(spec.cubes) > 1 or spec.task != "pick":
        if matches(spec, instr) != [tuple(spec.target)]:
            bad.append(f"instruction names {matches(spec, instr)} not exactly {spec.target}")
        dest = {"place": [spec.dest], "stack": [tuple(spec.dest) if spec.dest else None]}.get(spec.task, [])
        if matches_dest(spec, instr) != dest:
            bad.append(f"destination {matches_dest(spec, instr)} != {dest}")
    lab = json.loads(attrs["labels"]) if isinstance(attrs["labels"], str) else list(attrs["labels"])
    if lab != labels(spec).tolist():
        bad.append(f"labels {lab} != labels(spec) {labels(spec).tolist()}")
    if attrs.get("stage") == "s1" and (len(spec.cubes) != 1 or tuple(spec.target) not in SEEN_CUBES):
        bad.append(f"s1 episode with cubes {spec.cubes}")
    if not DATA_SEEDS[0] <= int(attrs["seed"]) <= DATA_SEEDS[1]:
        bad.append(f"seed {attrs['seed']} outside the data range {DATA_SEEDS}")
    v = visual_ok(attrs)
    if v:
        bad.append(v)
    return bad


def episode_attrs(path: Path) -> list[dict]:
    from vla_common.episode import _py

    with h5py.File(path, "r") as f:
        fa = {k: _py(v) for k, v in f.attrs.items()}
        names = sorted((k for k in f if k.startswith("ep_")), key=lambda k: int(k[3:]))
        return [{"file": str(path), "group": n, "stage": fa["stage"], **{k: _py(v) for k, v in f[n].attrs.items()}}
                for n in names]


def select(eps: list[dict], kept: int) -> dict[tuple[str, str], str]:
    """(file, group) -> 'train' | 'val' for the selected episodes: per (stage, seed) stream the first `kept` (0 = all)
    in episode-index order; kept rank k is 'val' iff k % 20 == 19."""
    out = {}
    for key in sorted({(e["stage"], int(e["seed"])) for e in eps}):
        s = sorted((e for e in eps if (e["stage"], int(e["seed"])) == key), key=lambda e: int(e["episode_index"]))
        idx = [int(e["episode_index"]) for e in s]
        assert len(set(idx)) == len(idx), f"stream {key}: duplicate episode indices (overlapping collect ranges)"
        if kept:
            assert len(s) >= kept, f"stream {key}: only {len(s)} kept episodes < --kept {kept}"
            s = s[:kept]
        for k, e in enumerate(s):
            out[(e["file"], e["group"])] = "val" if k % VAL_EVERY == VAL_EVERY - 1 else "train"
    return out


def paired_expected(srcs: list[Path]) -> dict[tuple[int, int], list[int]]:
    """(seed, layout_id) -> the group's episode indices, from every src dir's paired.json."""
    out = {}
    for d in srcs:
        pj = json.loads((d / "paired.json").read_text())
        for g, v in pj["groups_map"].items():
            out[(int(pj["seed"]), int(g))] = [int(i) for i in v["episodes"]]
    return out


def select_paired(eps: list[dict], expected: dict[tuple[int, int], list[int]], kept_all: int,
                  kept_stream: dict[tuple[str, int], int] | None = None) -> tuple[dict[tuple[str, str], str], dict]:
    """--paired selection: ((file, group) -> 'train' | 'val', report). Per (stage, seed) stream, groups in layout_id
    order; a group is kept iff every expected member succeeded and it fits under that stream's kept count (`kept_stream`
    override, else `kept_all`; 0 = all complete groups); the kept-group rank k is 'val' iff k % 20 == 19 for all its
    episodes. (A stream whose groups all hold 3 episodes cannot reach a kept count that is not a multiple of 3, hence the
    per-stream override, task 007.)"""
    out, rep = {}, {}
    for key in sorted({(e["stage"], int(e["seed"])) for e in eps}):
        kept = (kept_stream or {}).get(key, kept_all)
        s = [e for e in eps if (e["stage"], int(e["seed"])) == key]
        assert all(e.get("paired") for e in s), f"stream {key}: --paired but an episode has no paired attrs"
        by: dict[int, list[dict]] = {}
        for e in s:
            by.setdefault(int(e["layout_id"]), []).append(e)
        assert len({int(e["episode_index"]) for e in s}) == len(s), f"stream {key}: duplicate episode indices"
        total, rank, skipped, kept_groups = 0, 0, [], []
        for g in sorted({g for sd, g in expected if sd == key[1]}):
            exp, got = sorted(expected[(key[1], g)]), sorted(int(e["episode_index"]) for e in by.get(g, []))
            if got != exp:
                skipped.append({"group": g, "reason": "member failed", "failed_episodes": sorted(set(exp) - set(got))})
            elif kept and total + len(exp) > kept:
                skipped.append({"group": g, "reason": "does not fit --kept", "size": len(exp), "total_before": total})
            elif kept and kept - (total + len(exp)) == 1:  # groups hold >= 2 episodes: a remainder of 1 can never be filled
                skipped.append({"group": g, "reason": "would leave remainder 1", "size": len(exp), "total_before": total})
            else:
                sp = "val" if rank % VAL_EVERY == VAL_EVERY - 1 else "train"
                for e in by[g]:
                    out[(e["file"], e["group"])] = sp
                kept_groups.append({"group": g, "rank": rank, "split": sp, "episodes": exp})
                total, rank = total + len(exp), rank + 1
            if kept and total == kept:
                break
        assert not kept or total == kept, f"stream {key}: only {total} episodes in whole groups < --kept {kept}"
        rep[f"{key[0]}_paired_s{key[1]}"] = {"kept_groups": kept_groups, "skipped_groups": skipped, "kept_episodes": total}
    return out, rep


def verify_manifest(d: Path) -> int:
    m = d / "MANIFEST.sha256"
    assert m.exists(), f"{d}: no MANIFEST.sha256 (unfinished collect?)"
    n = 0
    for line in m.read_text().splitlines():
        digest, name = line.split("  ", 1)
        h = hashlib.sha256()
        with open(d / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        assert h.hexdigest() == digest, f"{d / name}: sha256 mismatch with MANIFEST"
        n += 1
    return n


def _export_shard(files: list[str], name: str, root: str, chosen: dict, label: str = "action") -> str:
    import vla_policy.export as EX
    from s1_relabel import relabel

    read = EX.read_episodes

    def selected(path):
        for g, a, at in read(path):
            sp = chosen.get((str(path), g))
            if sp is not None:
                if label == "relabel_setpoint":  # Stage 1 fix (b): r_t = clip(a_t + (x_{t-1} - a_{t-1})), arm dims only
                    a = a | {"action": relabel(a["action"], a["action_exec"])}
                yield g, a, at | {"split": sp, "sampler_split": at["split"], "label": label}

    EX.read_episodes = selected
    EX.export([Path(f) for f in files], name, root=Path(root))
    return name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True, type=Path, help="collector output dirs")
    p.add_argument("--name", required=True)
    p.add_argument("--root", type=Path, default=ROOT / "datasets")
    p.add_argument("--kept", type=int, required=True, help="kept episodes per data stream (GATES plan count; 0 = all)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--kept-stream", action="append", default=None, metavar="STAGE:SEED=N",
                   help="--paired: kept count for one (stage, seed) stream, overriding --kept (repeatable)")
    p.add_argument("--paired", action="store_true",
                   help="paired scenes (collect.py --paired-targets): whole groups only, split per group (module docstring)")
    p.add_argument("--label", default="action", choices=("action", "relabel_setpoint"),
                   help="action = the stored clean label; relabel_setpoint = scripts/s1_relabel.relabel (HDF5 untouched)")
    args = p.parse_args()
    t0 = time.time()
    verified = 0 if args.no_verify else sum(verify_manifest(d) for d in args.src)
    eps = [e for d in args.src for f in sorted(d.glob("*.h5")) for e in episode_attrs(f)]
    assert eps, f"no episodes in {args.src}"
    if args.paired:
        ks = {}
        for spec in args.kept_stream or []:
            stream, n = spec.split("=")
            stage, seed = stream.split(":")
            ks[(stage, int(seed))] = int(n)
        chosen, prep = select_paired(eps, paired_expected(args.src), args.kept, ks)
    else:
        chosen, prep = select(eps, args.kept), None
    sel = [e for e in eps if (e["file"], e["group"]) in chosen]
    viol = [{"file": Path(e["file"]).name, "group": e["group"], "violation": v} for e in sel for v in audit_episode(e)]
    au = {"episodes": len(sel), "rows": sum(int(e["T"]) for e in sel), "violations": viol, "clean": not viol,
          "per_task": {t: sum(e["task"] == t for e in sel) for t in sorted({e["task"] for e in sel})},
          "streams": sorted({f"{e['stage']}_{'paired_' if e.get('paired') else ''}s{e['seed']}" for e in sel}), "rules": audit_episode.__doc__}
    print(f"[export] {len(eps)} episodes found, {len(sel)} selected, {au['rows']} rows, manifest-verified {verified} "
          f"files; audit {'CLEAN' if au['clean'] else 'VIOLATIONS'}", flush=True)
    if not au["clean"]:
        print(json.dumps(viol[:20], indent=1))
        sys.exit(1)
    out = args.root / args.name
    assert not out.exists(), f"{out} exists"
    files = sorted({e["file"] for e in sel})
    P = max(1, min(args.workers, len(files)))
    if P == 1:
        _export_shard(files, args.name, str(args.root), chosen, args.label)
    else:
        from lerobot.datasets.aggregate import aggregate_datasets

        groups = [g.tolist() for g in np.array_split(np.array(files, dtype=object), P)]
        sroot = args.root / f"{args.name}.shards"
        names = [f"{args.name}_shard{k:02d}" for k in range(P)]
        with ProcessPoolExecutor(P) as ex:
            list(ex.map(_export_shard, groups, names, [str(sroot)] * P, [chosen] * P, [args.label] * P))
        aggregate_datasets([f"local/{n}" for n in names], f"local/{args.name}", roots=[sroot / n for n in names],
                           aggr_root=out)
        side, off = [], 0
        for n in names:
            rows = [json.loads(x) for x in (sroot / n / "meta" / "episodes_113.jsonl").read_text().splitlines()]
            side += [r | {"lerobot_episode_index": r["lerobot_episode_index"] + off, "shard": n} for r in rows]
            off += len(rows)
        (out / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))
        shutil.rmtree(sroot)  # this run's intermediate shards
    from vla_policy.export import load

    ds = load(out)
    assert ds.meta.total_episodes == au["episodes"] and ds.meta.total_frames == au["rows"], \
        (ds.meta.total_episodes, ds.meta.total_frames, au["episodes"], au["rows"])
    side = [json.loads(x) for x in (out / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    sp = {"rule": f"per stream, kept rank k is val iff k % {VAL_EVERY} == {VAL_EVERY - 1}"
          + (" (rank = kept group; whole groups only)" if args.paired else ""), "kept_per_stream": args.kept, "kept_stream_overrides": args.kept_stream,
          **({"paired": prep} if args.paired else {}),
          "train": [s["lerobot_episode_index"] for s in side if s["split"] == "train"],
          "val": [s["lerobot_episode_index"] for s in side if s["split"] == "val"]}
    (out / "meta" / "split_113.json").write_text(json.dumps(sp, indent=1))
    (out / "meta" / "audit_113.json").write_text(json.dumps(au, indent=1))
    summary = {"name": args.name, "root": str(out), "sources": [str(d) for d in args.src], "files": len(files),
               "episodes": ds.meta.total_episodes, "rows": ds.meta.total_frames, "per_task": au["per_task"],
               "streams": au["streams"], "label": args.label, "workers": P, "manifest_verified_files": verified, "audit_clean": au["clean"],
               "train_episodes": len(sp["train"]), "val_episodes": len(sp["val"]),
               "disk_bytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file()),
               "wall_s": round(time.time() - t0, 1)}
    (out / "meta" / "export_113.json").write_text(json.dumps(summary, indent=1))
    print("EXPORT_SUMMARY=" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
