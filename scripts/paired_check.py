"""0C item "P" on an EXPORTED paired set (collect.py --paired-targets -> export_dataset.py --paired; PERUN or laptop,
policy venv; the unpaired items stay s0c_rl_check.py --fresh [paired rules inside] and tests/test_data_audit.py --root).
Implements GATES.md "2a attempt 1 - instrument amendment: item P(ii)" (the t=0 sha identity is unreachable: the render
depends on the env slot; results/paired_slot_control.json).

  python scripts/paired_check.py --dataset /data/datasets/s2a_paired --out /runs/paired_check

Gated items per kept group (sidecar `layout_id`):
 P_i_distinct_targets       every group has >= 2 episodes and all their targets (labels[1:3] = (colour, size)) are distinct;
 P1_layout_identical        every member's object poses equal member 0's within 1e-6 m / 1e-6 rad (layouts regenerated from
                            (seed, layout_id) by make_paired_stream, as the collector drew them) and every member's spec
                            equals member 0's except target (cubes, pads, task, dest);
 P2a_state_delta_vs_target  t=0 state (19) deltas to member 0 vs target index: per dimension with nonzero variance the
                            Pearson correlation p >= 0.01; zero-variance dimensions pass by construction (listed);
 P2b_state_classifier       target colour from the t=0 state: softmax regression, group-wise 5-fold CV (fold = layout_id % 5);
                            |accuracy - majority-class rate| <= 3 binomial SE of that rate (accuracy / chance reported);
 P2c_slot_independent       target colour x slot octile (slot // 16, 8 bins) chi-square independence p >= 0.01 (per-target
                            slot distribution and member x octile reported);
 P_iii_labels_instr_differ  the 4-int labels and the instruction are pairwise different between the members of a group;
 P_iv_group_in_one_split    every group's episodes share one split;
 P_same_scene_spec          members' cube sets and pad sets are equal (spec_json);
 P_regenerates              every episode's spec_json, instruction and stream index equal make_paired_stream(seed, group);
 P_kept_groups_rule / P_split_rule  the kept groups / splits are what export_dataset.select_paired (the export's own
                            function, remainder-1 skip included) returns for the source HDF5 + paired.json.
Target class of P2 (`target_class`): s2a the target colour (the colour names the target); s2 (paired s2 scenes, colours
AND sizes) the target cube (colour, size) as its catalog index, so two sizes of one colour are two classes; s3a / s3b
(task 007, two-noun targets) the target identity (cube, destination) = labels[1:4], so P_i's distinct targets are
distinct (cube, pad) / (top, bottom) pairs.
Union sets (task 007: a stage's paired set + the 2b paired s2 set): every item runs PER STREAM (stage, data seed) on that
stream's rows and its own source dirs (paired.json stage + seed); items are reported as "<stage>_paired_s<seed>/<item>"
and the set passes iff every stream passes. A single-stream set is reported exactly as before.
Reported for s3a / s3b streams (not gated): the relation-type split over every member pair of the kept groups
(same_target / same_dest / both_differ, rollout.relation) and whether each episode's recorded `relation` attr equals it.
Reported, not gated: per group and camera the t=0 frame sha equality, max abs pixel diff, differing-pixel count (decoded
from the exported dataset through the trainer's path: s0c_check.open_ds / decoded, torchcodec).
Writes <out>/paired_check_<name>.json; exit 1 if any gated item fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CAMS = ("side", "wrist")
POS_TOL, YAW_TOL = 1e-6, 1e-6  # GATES amendment P1 (m, rad)
P_MIN, SE_K, OCTILE, N_FOLDS = 0.01, 3.0, 16, 5  # GATES amendment P2


def groups_of(side: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for s in side:
        out.setdefault(int(s["layout_id"]), []).append(s)
    return {g: sorted(v, key=lambda s: int(s["member"])) for g, v in sorted(out.items())}


def _labels(s: dict) -> tuple:
    return tuple(json.loads(s["labels"]) if isinstance(s["labels"], str) else s["labels"])


TWO_NOUN = ("s3a", "s3b", "s3a1", "s3b1")  # s3a1 / s3b1 (task 009b): one cube size per scene


def target_key(s: dict) -> tuple:
    """P_i's target: (colour, size) = labels[1:3]; s3a / s3b (cube, destination) = labels[1:4]."""
    return _labels(s)[1:4] if s["file_attrs"]["stage"] in TWO_NOUN else _labels(s)[1:3]


def target_class(s: dict) -> int:
    """P2's target class: s2a the colour index (labels[1]); s2 the cube's catalog index (colour, size); s3a / s3b the
    (cube, destination) identity, cube index x 13 + destination label (0-12)."""
    from vla_common.catalog import SIZES

    lab = _labels(s)
    st = s["file_attrs"]["stage"]
    cube = lab[1] * len(SIZES) + lab[2]
    return lab[1] if st == "s2a" else cube * 13 + lab[3] if st in TWO_NOUN else cube


def frame_report(groups: dict[int, list[dict]], t0: dict[int, dict[str, np.ndarray]]) -> dict:
    """REPORTED ONLY. t0[lerobot_episode_index][cam] = HxWx3 uint8. Per group and camera: sha equal across members, max abs
    diff, differing-pixel count (max over members vs member 0)."""
    per, worst, npx, n_eq = {}, {c: 0 for c in CAMS}, {c: [] for c in CAMS}, 0
    for g, ms in groups.items():
        row = {}
        for c in CAMS:
            fr = [t0[int(s["lerobot_episode_index"])][c] for s in ms]
            sha = [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in fr]
            dd = [np.abs(f.astype(np.int16) - fr[0].astype(np.int16)) for f in fr]
            d, px = max(int(x.max()) for x in dd), max(int((x.max(-1) > 0).sum()) for x in dd)
            row[c] = {"sha_equal": len(set(sha)) == 1, "max_abs_diff": d, "differing_px": px}
            worst[c] = max(worst[c], d)
            npx[c].append(px)
        n_eq += all(r["sha_equal"] for r in row.values())
        per[str(g)] = row
    return {"per_group": per, "max_abs_diff": worst, "groups_all_cams_sha_equal": n_eq, "groups": len(groups),
            "differing_px_per_group": {c: {"median": float(np.median(v)), "max": int(max(v))} for c, v in npx.items()}}


# --------------------------------------------------------------------------- P1
def regenerated_layouts(side: list[dict]) -> dict[tuple[int, int], dict]:
    """(layout_id, member) -> layout dict, regenerated from the stream (make_paired_stream)."""
    from vla_tabletop.rollout import make_paired_stream

    keys = {(int(s["seed"]), float(s.get("cube_yaw_max_deg", 180.0)), s["file_attrs"]["stage"], s.get("sampler_split", "train"))
            for s in side}
    assert len(keys) == 1, f"mixed streams {keys}"
    seed, m, stage, split = next(iter(keys))
    ref, _ = make_paired_stream(seed, 0, max(int(s["layout_id"]) for s in side) + 1, stage, split, m)
    return {(e.layout_id, e.member): e.layout for e in ref}


def p1_bad_groups(groups: dict[int, list[dict]], layouts: dict[tuple[int, int], dict]) -> dict[str, list]:
    """Groups whose member layouts differ from member 0's beyond 1e-6 m / 1e-6 rad, or whose spec differs except target."""
    from vla_common.tasks import TaskSpec

    bad = {"layout": [], "spec": []}
    for g, ms in groups.items():
        l0 = layouts[(g, int(ms[0]["member"]))]
        for s in ms[1:]:
            lm = layouts[(g, int(s["member"]))]
            if lm.keys() != l0.keys() or any(abs(lm[k][0] - l0[k][0]) > POS_TOL or abs(lm[k][1] - l0[k][1]) > POS_TOL
                                             or abs(lm[k][2] - l0[k][2]) > YAW_TOL for k in l0):
                bad["layout"].append(g)
                break
        sp = [TaskSpec.from_json(s["spec_json"]) for s in ms]
        if any((x.task, [tuple(c) for c in x.cubes], x.pads) != (sp[0].task, [tuple(c) for c in sp[0].cubes], sp[0].pads)
               for x in sp):
            bad["spec"].append(g)
    return bad


# --------------------------------------------------------------------------- P2
def _softmax_cv_accuracy(X: np.ndarray, y: np.ndarray, fold: np.ndarray, n_classes: int) -> float:
    """Group-wise CV accuracy of an L2 softmax regression (numpy; 300 full-batch gradient steps, lr 0.5, lambda 1e-2)."""
    correct = 0
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        keep = sd > 1e-12
        if not keep.any():  # no usable feature: predict the training majority
            correct += int((y[te] == np.bincount(y[tr], minlength=n_classes).argmax()).sum())
            continue
        A, Ate = (X[tr][:, keep] - mu[keep]) / sd[keep], (X[te][:, keep] - mu[keep]) / sd[keep]
        W, b, Y = np.zeros((A.shape[1], n_classes)), np.zeros(n_classes), np.eye(n_classes)[y[tr]]
        for _ in range(300):
            Z = A @ W + b
            P = np.exp(Z - Z.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            G = (P - Y) / len(A)
            W -= 0.5 * (A.T @ G + 1e-2 * W)
            b -= 0.5 * G.sum(0)
        correct += int(((Ate @ W + b).argmax(1) == y[te]).sum())
    return correct / len(y)


def p2_items(side: list[dict], state0: dict[int, np.ndarray]) -> tuple[dict, dict]:
    """P2 a/b/c from the sidecar rows (labels, member, slot / env_id, layout_id) and the t=0 states (lerobot ep -> (19,))."""
    from scipy import stats

    groups = groups_of(side)
    rows = [s for ms in groups.values() for s in ms]
    tgt = np.array([target_class(s) for s in rows])  # s2a: target colour index; s2: target cube index
    D = np.stack([state0[int(s["lerobot_episode_index"])] - state0[int(ms[0]["lerobot_episode_index"])]
                  for ms in groups.values() for s in ms]).astype(np.float64)
    var = D.var(0)
    nz = [int(j) for j in np.flatnonzero(var > 0)]
    zero = [int(j) for j in np.flatnonzero(var == 0)]
    pv = {}
    for j in nz:
        r = stats.pearsonr(D[:, j], tgt)
        pv[str(j)] = {"r": float(r.statistic), "p": float(r.pvalue)} if tgt.std() > 0 else {"r": 0.0, "p": 1.0}
    # (b)
    S0 = np.stack([state0[int(s["lerobot_episode_index"])] for s in rows]).astype(np.float64)
    classes, y = np.unique(tgt, return_inverse=True)
    fold = np.array([int(s["layout_id"]) % N_FOLDS for s in rows])
    acc = _softmax_cv_accuracy(S0, y, fold, len(classes))
    chance = float(np.bincount(y).max() / len(y))
    se = float(np.sqrt(chance * (1 - chance) / len(y)))
    # (c)
    slot = np.array([int(s.get("slot", s.get("env_id"))) for s in rows]) // OCTILE
    tab = np.zeros((len(classes), 8), int)
    for a, b in zip(y, np.minimum(slot, 7)):
        tab[a, b] += 1
    t2 = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    chi_p = float(stats.chi2_contingency(t2).pvalue) if min(t2.shape) > 1 else 1.0
    mem = np.array([int(s["member"]) for s in rows])
    mtab = np.zeros((int(mem.max()) + 1, 8), int)
    for a, b in zip(mem, np.minimum(slot, 7)):
        mtab[a, b] += 1
    items = {"P2a_state_delta_vs_target": all(v["p"] >= P_MIN for v in pv.values()),
             "P2b_state_classifier": abs(acc - chance) <= SE_K * se,
             "P2c_slot_independent": chi_p >= P_MIN}
    nums = {"P2a": {"nonzero_variance_dims": nz, "zero_variance_dims": zero, "per_dim": pv,
                    "min_p": min((v["p"] for v in pv.values()), default=None)},
            "P2b": {"accuracy": acc, "majority_rate": chance, "se": se, "abs_diff_in_se": abs(acc - chance) / se if se else 0.0,
                    "classes": [int(c) for c in classes]},
            "P2c": {"chi2_p": chi_p, "target_colour_x_slot_octile": {str(int(c)): tab[i].tolist() for i, c in enumerate(classes)},
                    "member_x_slot_octile": {str(m): mtab[m].tolist() for m in range(len(mtab))}}}
    return items, nums


# --------------------------------------------------------------------------- all scene-level items
def p_items(side: list[dict], state0: dict[int, np.ndarray], layouts: dict[tuple[int, int], dict] | None = None,
            t0: dict[int, dict[str, np.ndarray]] | None = None) -> tuple[dict, dict]:
    """Gated scene-level items (module docstring) from the sidecar rows, the t=0 states and (default: regenerated) layouts;
    t0 frames only feed the reported numbers. Returns (items, numbers)."""
    groups = groups_of(side)
    p1 = p1_bad_groups(groups, layouts if layouts is not None else regenerated_layouts(side))
    p2, n2 = p2_items(side, state0)
    bad: dict[str, list] = {"distinct": [], "labels_instr": [], "split": [], "scene": []}
    for g, ms in groups.items():
        tg = [target_key(s) for s in ms]
        if len(ms) < 2 or len(set(tg)) != len(ms):
            bad["distinct"].append(g)
        if len({_labels(s) for s in ms}) != len(ms) or len({s["instruction"] for s in ms}) != len(ms):
            bad["labels_instr"].append(g)
        if len({s["split"] for s in ms}) != 1:
            bad["split"].append(g)
    bad["scene"] = p1["spec"]
    items = {"P_i_distinct_targets": not bad["distinct"], "P1_layout_identical": not p1["layout"] and not p1["spec"],
             **p2, "P_iii_labels_instr_differ": not bad["labels_instr"], "P_iv_group_in_one_split": not bad["split"],
             "P_same_scene_spec": not bad["scene"]}
    nums = {"groups": len(groups), "episodes": len(side), "bad_groups": {k: v[:20] for k, v in bad.items()} | {"p1_layout": p1["layout"][:20]},
            "group_sizes": {str(n): sum(len(v) == n for v in groups.values()) for n in sorted({len(v) for v in groups.values()})},
            **n2}
    if t0 is not None:
        nums["reported_t0_frames"] = frame_report(groups, t0)
    if side and side[0]["file_attrs"]["stage"] in TWO_NOUN:
        nums["relation_split"] = relation_split(groups)
    return items, nums


def relation_split(groups: dict[int, list[dict]]) -> dict:
    """REPORTED ONLY (s3a / s3b): relation-type counts over every member pair of every group (from the specs), and the
    episodes whose recorded `relation` attr differs from the spec-derived type vs member 0."""
    from vla_tabletop.rollout import RELATIONS, relation

    def pair(s):
        sp = json.loads(s["spec_json"])
        return tuple(sp["target"]), tuple(sp["dest"]) if isinstance(sp["dest"], list) else sp["dest"]

    cnt, bad = {r: 0 for r in (*RELATIONS, "duplicate")}, []  # duplicate: P_i fails on it; counted, not typed
    for ms in groups.values():
        pr = [pair(s) for s in ms]
        for i in range(len(pr)):
            for j in range(i + 1, len(pr)):
                cnt["duplicate" if pr[i] == pr[j] else relation(pr[i], pr[j])] += 1
        for s, p in zip(ms, pr):
            want = "anchor" if int(s["member"]) == 0 else "duplicate" if p == pr[0] else relation(pr[0], p)
            if s.get("relation") != want:
                bad.append(int(s["lerobot_episode_index"]))
    n = sum(cnt.values())
    return {"pairs": n, "counts": cnt, "fractions": {k: v / n for k, v in cnt.items()} if n else None,
            "recorded_relation_mismatch_episodes": bad[:20], "n_recorded_relation_mismatch": len(bad)}


def regenerates(side: list[dict]) -> list[int]:
    """Lerobot episode indices whose (spec, instruction, stream index) differ from the regenerated paired stream."""
    from vla_tabletop.rollout import make_paired_stream

    keys = {(int(s["seed"]), float(s.get("cube_yaw_max_deg", 180.0)), s["file_attrs"]["stage"], s.get("sampler_split", "train"))
            for s in side}
    assert len(keys) == 1, f"mixed streams {keys}"
    seed, m, stage, split = next(iter(keys))
    ref, _ = make_paired_stream(seed, 0, max(int(s["layout_id"]) for s in side) + 1, stage, split, m)
    by = {(e.layout_id, e.member): e for e in ref}
    bad = []
    for s in side:
        e = by.get((int(s["layout_id"]), int(s["member"])))
        if e is None or e.spec.to_json() != s["spec_json"] or e.instruction != s["instruction"] \
                or e.index != int(s["episode_index"]):
            bad.append(int(s["lerobot_episode_index"]))
    return bad


# --------------------------------------------------------------------------- kept groups / split: the export's own rule
def expected_selection(expected: dict[int, list[int]], ok: set[int], kept: int, seed: int) -> list[tuple[int, str]]:
    """export_dataset.select_paired applied to the successful episodes `ok` of the groups `expected` (layout_id -> member
    episode indices): [(layout_id, 'train' | 'val')] in kept order. No second implementation of the rule."""
    import export_dataset as X

    of = {i: g for g, v in expected.items() for i in v}
    rows = [{"file": "f", "group": f"ep_{i}", "stage": "s", "seed": seed, "episode_index": i, "paired": True, "layout_id": of[i]}
            for i in sorted(ok) if i in of]
    chosen, _ = X.select_paired(rows, {(seed, g): v for g, v in expected.items()}, kept)
    sp = {of[int(k[1][3:])]: v for k, v in chosen.items()}
    return sorted(sp.items())


def selection_items(side: list[dict], expected: dict[int, list[int]], ok: set[int], kept: int) -> tuple[dict, dict]:
    want = expected_selection(expected, ok, kept, int(side[0]["seed"]))
    got = [(g, ms[0]["split"]) for g, ms in groups_of(side).items()]
    items = {"P_kept_groups_rule": [g for g, _ in want] == [g for g, _ in got]
             and sum(len(expected[g]) for g, _ in want) == len(side) and (not kept or len(side) == kept)
             and all(sorted(int(s["episode_index"]) for s in ms) == sorted(expected[g]) for g, ms in groups_of(side).items()),
             "P_split_rule": want == got}
    return items, {"expected_kept_groups": len(want), "got_kept_groups": len(got),
                   "first_mismatch": next(([a, b] for a, b in zip(want, got) if a != b), None)}


def source_success(sources: list[Path], stream: tuple[str, int] | None = None) -> tuple[dict[int, list[int]], set[int]]:
    """({layout_id: member episode indices} from every source dir's paired.json, {successful episode indices in the HDF5});
    stream = (stage, seed): only the source dirs of that stream (union sets)."""
    import h5py

    expected, ok = {}, set()
    for d in sources:
        pj = json.loads((d / "paired.json").read_text())
        if stream is not None and (pj["stage"], int(pj["seed"])) != tuple(stream):
            continue
        expected |= {int(g): [int(i) for i in v["episodes"]] for g, v in pj["groups_map"].items()}
        for f in sorted(d.glob("*.h5")):
            with h5py.File(f, "r") as h:
                ok |= {int(h[n].attrs["episode_index"]) for n in h if n.startswith("ep_")}
    return expected, ok


def stream_rows(side: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Sidecar rows per stream (stage, data seed), in sidecar order."""
    out: dict[tuple[str, int], list[dict]] = {}
    for s in side:
        out.setdefault((s["file_attrs"]["stage"], int(s["seed"])), []).append(s)
    return out


def stream_items(side: list[dict], state0: dict[int, np.ndarray], frames: dict | None, sources: list[Path], kept: int,
                 stream: tuple[str, int] | None = None, layouts: dict | None = None) -> tuple[dict, dict, list[int], dict]:
    """Every gated item on ONE stream's rows: (items, numbers, regenerate mismatches, selection numbers). stream = (stage,
    seed) restricts the source dirs to that stream's (union sets); None = all source dirs (single-stream set)."""
    items, nums = p_items(side, state0, layouts=layouts, t0=frames)
    bad_regen = regenerates(side)
    items["P_regenerates"] = not bad_regen
    expected, ok = source_success(sources, stream)
    sel_items, sel_nums = selection_items(side, expected, ok, kept)
    return items | sel_items, nums, bad_regen, sel_nums


def stream_kept(spj: dict, stream: tuple[str, int]) -> int:
    """The kept count the export applied to `stream` (stage, seed): its `--kept-stream STAGE:SEED=N` override
    (split_113.json `kept_stream_overrides`) if any, else the uniform `kept_per_stream`."""
    for spec in spj.get("kept_stream_overrides") or []:
        key, n = spec.split("=")
        stage, seed = key.split(":")
        if (stage, int(seed)) == tuple(stream):
            return int(n)
    return int(spj["kept_per_stream"])


def combine(root: Path, per: dict[tuple[str, int], tuple]) -> dict:
    """The result JSON from stream_items per stream: one stream = the single-stream report as before; several = items
    keyed "<stage>_paired_s<seed>/<item>", numbers / mismatches / selection per stream name; pass iff every item passes."""
    if len(per) == 1:
        items, nums, bad_regen, sel_nums = next(iter(per.values()))
        return {"dataset": str(root), "name": root.name, "items": items, "pass": all(items.values()), "numbers": nums,
                "regenerate_mismatch_episodes": bad_regen[:20], "selection": sel_nums}
    name = {k: f"{k[0]}_paired_s{k[1]}" for k in per}
    items = {f"{name[k]}/{i}": v for k, r in per.items() for i, v in r[0].items()}
    return {"dataset": str(root), "name": root.name, "streams": list(name.values()), "items": items,
            "pass": all(items.values()), "numbers": {name[k]: r[1] for k, r in per.items()},
            "regenerate_mismatch_episodes": {name[k]: r[2][:20] for k, r in per.items()},
            "selection": {name[k]: r[3] for k, r in per.items()}}


def check(root: Path, out: Path) -> dict:
    import s0c_check as C

    t0w = time.time()
    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    exp = json.loads((root / "meta" / "export_113.json").read_text())
    spj = json.loads((root / "meta" / "split_113.json").read_text())
    assert all(s.get("paired") for s in side), "not a paired set (no paired attrs)"
    ds = C.open_ds(root)
    eps = ds.meta.episodes
    pq = C.read_parquet(root)
    ep, fi = pq["episode_index"].reshape(-1), pq["frame_index"].reshape(-1)
    state0 = {int(s["lerobot_episode_index"]): pq["observation.state"][(ep == int(s["lerobot_episode_index"])) & (fi == 0)][0]
              for s in side}
    frames = {int(s["lerobot_episode_index"]): C.decoded(ds, int(eps[int(s["lerobot_episode_index"])]["dataset_from_index"]), 0)
              for s in side}
    res = combine(root, {k: stream_items(v, state0, frames, [Path(x) for x in exp["sources"]], stream_kept(spj, k),
                                         k if len(stream_rows(side)) > 1 else None)
                         for k, v in stream_rows(side).items()})
    res["wall_s"] = round(time.time() - t0w, 1)
    (out / f"paired_check_{root.name}.json").write_text(json.dumps(res, indent=1))
    if "streams" in res:
        print("PAIRED_CHECK=" + json.dumps({"name": root.name, "pass": res["pass"], "items": res["items"], **{
            s_: {"groups": n["groups"], "episodes": n["episodes"], "group_sizes": n["group_sizes"], "P2b": n["P2b"],
                 "P2c_p": n["P2c"]["chi2_p"], **({"relation_split": n["relation_split"]} if "relation_split" in n else {})}
            for s_, n in res["numbers"].items()}}), flush=True)
        return res
    items, nums = res["items"], res["numbers"]
    rf = nums["reported_t0_frames"]
    print("PAIRED_CHECK=" + json.dumps({"name": root.name, "items": items, "groups": nums["groups"], "episodes": nums["episodes"],
                                        "group_sizes": nums["group_sizes"], "bad_groups": nums["bad_groups"],
                                        "P2a_zero_var_dims": nums["P2a"]["zero_variance_dims"], "P2a_min_p": nums["P2a"]["min_p"],
                                        "P2b": nums["P2b"], "P2c_p": nums["P2c"]["chi2_p"],
                                        "reported_frames": {k: rf[k] for k in ("max_abs_diff", "groups_all_cams_sha_equal", "groups",
                                                                               "differing_px_per_group")}}), flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.exit(0 if check(a.dataset, a.out)["pass"] else 1)


if __name__ == "__main__":
    main()
