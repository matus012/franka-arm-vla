"""Counterfactual paired stream (vla_tabletop.rollout.make_paired_stream): CPU unit tests + mutants m1 / m2."""

from __future__ import annotations

import numpy as np
import pytest

from vla_common.tasks import HELD_OUT_TARGETS, SEEN_CUBES, TaskSpec, instruction, matches, sample_spec
from vla_tabletop.layout import sample_layout
from vla_tabletop.rollout import Episode, make_paired_stream, make_stream, valid_targets

SEED, M = 32800, 15.0


def audit_groups(eps: list[Episode], split: str = "train") -> list[str]:
    """Violations of the paired-stream contract (the tests' oracle; mutants must make it non-empty)."""
    bad = []
    ok = HELD_OUT_TARGETS if split == "heldout_comp" else SEEN_CUBES
    for g in sorted({e.layout_id for e in eps}):
        ms = [e for e in eps if e.layout_id == g]
        if len(ms) < 2:
            bad.append(f"group {g}: {len(ms)} members")
        if any(e.layout != ms[0].layout for e in ms):
            bad.append(f"group {g}: layout differs between members")
        if any(sorted(e.spec.cubes) != sorted(ms[0].spec.cubes) or e.spec.pads != ms[0].spec.pads for e in ms):
            bad.append(f"group {g}: cube / pad set differs")
        tg = [tuple(e.spec.target) for e in ms]
        if len(set(tg)) != len(tg):
            bad.append(f"group {g}: duplicate targets {tg}")
        for e in ms:
            if tuple(e.spec.target) not in ok:
                bad.append(f"group {g}: target {e.spec.target} outside the split's set")
            if matches(e.spec, e.instruction) != [tuple(e.spec.target)]:
                bad.append(f"group {g} member {e.member}: instruction {e.instruction!r} does not name {e.spec.target}")
    return bad


def stream(n: int = 40, **kw):
    return make_paired_stream(SEED, 0, n, "s2a", "train", M, **kw)


def test_clean_stream_passes_oracle():
    eps, skipped = stream()
    assert eps and audit_groups(eps) == []
    assert {e.layout_id for e in eps} | {s["group"] for s in skipped} == set(range(40))


def test_group_contract_explicit():
    eps, _ = stream()
    for g in {e.layout_id for e in eps}:
        ms = [e for e in eps if e.layout_id == g]
        spec = sample_spec("s2a", "train", np.random.default_rng([SEED, g]))
        lay0 = sample_layout(spec, _after_spec(g), cube_yaw_max_deg=M)[0]
        assert all(e.layout == lay0 for e in ms)  # == the layout make_stream episode g would use
        assert [e.member for e in ms] == list(range(len(ms)))
        assert len({tuple(e.spec.target) for e in ms}) == len(ms) >= 2
        assert all(tuple(e.spec.target) in SEEN_CUBES for e in ms)
        assert {tuple(e.spec.target) for e in ms} == set(valid_targets(spec, "train"))
        assert [tuple(e.spec.target) for e in ms] == sorted((tuple(e.spec.target) for e in ms), key=_cat)


def _cat(c):
    from vla_common.catalog import ALL_CUBES
    return ALL_CUBES.index(c)


def _after_spec(g):
    rng = np.random.default_rng([SEED, g])
    sample_spec("s2a", "train", rng)
    return rng


def test_layout_equals_make_stream_episode_g():
    eps, _ = stream(10)
    ref = make_stream(SEED, 0, 10, lambda r: sample_spec("s2a", "train", r), cube_yaw_max_deg=M)
    for e in eps:
        assert e.layout == ref[e.layout_id].layout and e.spec.cubes == ref[e.layout_id].spec.cubes


def test_determinism_and_batch_restart():
    full, sk = stream(60)
    again, sk2 = stream(60)
    assert [(e.index, e.instruction, e.layout, e.spec.to_json()) for e in full] == \
        [(e.index, e.instruction, e.layout, e.spec.to_json()) for e in again] and sk == sk2
    assert [e.index for e in full] == list(range(len(full)))
    b2, skb = make_paired_stream(SEED, 25, 35, "s2a", "train", M)  # a later batch reproduces the full stream's tail
    tail = [e for e in full if e.layout_id >= 25]
    assert [(e.index, e.instruction, e.layout, e.member) for e in b2] == [(e.index, e.instruction, e.layout, e.member) for e in tail]
    assert skb == [s for s in sk if s["group"] >= 25]


def test_skipped_groups_have_fewer_than_two_valid():
    eps, skipped = stream(300)
    assert skipped, "300 groups must contain a skip (2 cubes with a held-out one)"
    for s in skipped:
        spec = sample_spec("s2a", "train", np.random.default_rng([SEED, s["group"]]))
        assert len(valid_targets(spec, "train")) == s["n_valid"] < 2
    assert not {s["group"] for s in skipped} & {e.layout_id for e in eps}


def test_heldout_comp_split_has_no_pairs_in_s2a():
    """The two held-out cubes differ in size and an s2a scene has one size: every heldout_comp group is skipped."""
    eps, skipped = make_paired_stream(SEED, 0, 100, "s2a", "heldout_comp", M)
    assert eps == [] and len(skipped) == 100
    assert all(tuple(c) in HELD_OUT_TARGETS for g in range(20) for c in valid_targets(
        sample_spec("s4", "heldout_comp", np.random.default_rng([1, g])), "heldout_comp"))


def test_instruction_uses_member_rng():
    eps, _ = stream(5)
    e = eps[1]
    assert e.instruction == instruction(e.spec, np.random.default_rng([SEED, e.layout_id, e.member]), "train")


# ---- mutants: the oracle must flag them
def test_mutant_m1_layout_resampled_per_member():
    eps, _ = stream(30)
    bad = []
    for e in eps:  # what a per-member re-sample would produce
        rng = np.random.default_rng([SEED, e.layout_id, e.member, 99])
        bad.append(Episode(e.index, e.spec, e.instruction, sample_layout(e.spec, rng, cube_yaw_max_deg=M)[0], {},
                           e.layout_id, e.member))
    flagged = audit_groups(bad)
    assert any("layout differs" in f for f in flagged)
    assert audit_groups(eps) == []


def test_mutant_m2_same_target_repeated():
    eps, _ = stream(30)
    first = {e.layout_id: e for e in eps if e.member == 0}
    bad = [Episode(e.index, first[e.layout_id].spec, first[e.layout_id].instruction, e.layout, {}, e.layout_id, e.member)
           for e in eps]
    assert any("duplicate targets" in f for f in audit_groups(bad))


def test_mutant_instruction_names_other_cube():
    eps, _ = stream(30)
    other = {e.layout_id: eps[[x.layout_id for x in eps].index(e.layout_id)] for e in eps}
    bad = [Episode(e.index, e.spec, other[e.layout_id].instruction, e.layout, {}, e.layout_id, e.member) for e in eps]
    assert any("does not name" in f for f in audit_groups(bad))


def test_mutant_target_outside_split_set():
    eps, _ = stream(30)
    e = eps[0]
    spec = TaskSpec("pick", HELD_OUT_TARGETS[0], None, [HELD_OUT_TARGETS[0], *e.spec.cubes[:1]], e.spec.pads)
    assert any("outside the split" in f for f in audit_groups([Episode(0, spec, "pick up the green cube", e.layout, {}, 0, 0)]))


# ---- collect.py --paired-targets (CPU parts)
def _collect():
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("collect", Path(__file__).resolve().parents[1] / "scripts" / "collect.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_collect_paired_manifest_and_prefix(tmp_path):
    C = _collect()
    assert C.prefix("s2a", "train", 32800) == "s2a_train_s32800"  # unchanged default
    assert C.prefix("s2a", "train", 32800, True) == "s2a_paired_train_s32800"
    eps, sk = make_paired_stream(SEED, 30, 40, "s2a", "train", M)
    plan = C.plan_chunks([e.spec.task for e in eps], 16, eps[0].index)
    m = C.paired_manifest(eps, sk, plan, SEED, 30, 40, "s2a", "train")
    assert m["skipped_groups"] == sk and m["n_episodes"] == len(eps)
    flat = sorted(i for g in m["groups_map"].values() for i in g["episodes"])
    assert flat == [e.index for e in eps] == sorted(int(k) for k in m["episode_slots"])
    assert all(len(g["episodes"]) >= 2 and len({tuple(t) for t in g["targets"]}) == len(g["targets"]) for g in m["groups_map"].values())
    assert all(0 <= v["slot"] < 16 for v in m["episode_slots"].values())
    assert m["episode_slots"][str(eps[0].index)] == {"chunk": plan[0]["chunk"], "slot": 0}
    # a batch split in two reproduces the full stream's global indices
    a, _ = make_paired_stream(SEED, 0, 30, "s2a", "train", M)
    assert eps[0].index == len(a)


def test_collect_run_config_legacy_and_paired(tmp_path):
    import json

    C = _collect()
    base = {k: 1 for k in C.RUN_KEYS if k not in C.RUN_LEGACY} | {"cube_yaw_max_deg": 180.0, "table_palette": "112"}
    (tmp_path / "RUN.json").write_text(json.dumps(base))  # written before the paired flag existed
    C.check_run_config(tmp_path, base)  # a normal resume still works
    with pytest.raises(RuntimeError):
        C.check_run_config(tmp_path, base | {"paired_targets": True})


# ---- export --paired selection
def _export():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import export_dataset
    return export_dataset


def _fake(n_groups: int = 80, fail: tuple = ()):
    """Episodes of the real paired stream as export-attr dicts; `fail` = episode indices that did not succeed."""
    eps, _ = make_paired_stream(SEED, 0, n_groups, "s2a", "train", M)
    exp = {}
    for e in eps:
        exp.setdefault((SEED, e.layout_id), []).append(e.index)
    rows = [{"file": f"f{e.index // 50}", "group": f"ep_{e.index}", "stage": "s2a", "seed": SEED, "episode_index": e.index,
             "paired": True, "layout_id": e.layout_id, "member": e.member} for e in eps if e.index not in fail]
    return rows, exp


def test_select_paired_whole_groups_exact_kept_and_group_split():
    X = _export()
    rows, exp = _fake()
    size = {g: len(v) for (_, g), v in exp.items()}
    fail = (exp[(SEED, 2)][1],)  # one member of group 2 failed -> the whole group is dropped
    rows, exp = _fake(fail=fail)
    ch, rep = X.select_paired(rows, exp, 100)
    r = rep[f"s2a_paired_s{SEED}"]
    assert r["kept_episodes"] == len(ch) == 100
    assert {"group": 2, "reason": "member failed", "failed_episodes": list(fail)} in r["skipped_groups"]
    kept = {kg["group"] for kg in r["kept_groups"]}
    assert 2 not in kept and sum(size[g] for g in kept) == 100
    by_layout = {}
    for e in rows:
        if (e["file"], e["group"]) in ch:
            by_layout.setdefault(e["layout_id"], set()).add(ch[(e["file"], e["group"])])
    assert all(len(v) == 1 for v in by_layout.values())  # a group never straddles train / val
    ranks = {kg["group"]: kg["rank"] for kg in r["kept_groups"]}
    assert all((next(iter(v)) == "val") == (ranks[g] % 20 == 19) for g, v in by_layout.items())
    assert "val" in {next(iter(v)) for v in by_layout.values()} or len(ranks) < 20
    # a group that would cross --kept is skipped, a later smaller one fills the gap
    assert any(s["reason"] == "does not fit --kept" for s in r["skipped_groups"]) or sum(
        size[g] for g in range(max(kept) + 1) if g in kept) == 100


def test_select_paired_crossing_group_is_skipped_and_walk_continues():
    X = _export()
    sizes, rows, exp, n = [3, 4, 2, 2], [], {}, 0
    for g, sz in enumerate(sizes):
        exp[(SEED, g)] = list(range(n, n + sz))
        rows += [{"file": "f", "group": f"ep_{i}", "stage": "s2a", "seed": SEED, "episode_index": i, "paired": True,
                  "layout_id": g, "member": i - n} for i in range(n, n + sz)]
        n += sz
    ch, rep = X.select_paired(rows, exp, 5)  # 3, then 4 would cross -> skipped, 2 fits -> exactly 5
    r = rep[f"s2a_paired_s{SEED}"]
    assert len(ch) == r["kept_episodes"] == 5 and [k["group"] for k in r["kept_groups"]] == [0, 2]
    assert r["skipped_groups"] == [{"group": 1, "reason": "does not fit --kept", "size": 4, "total_before": 3}]
    assert [k["group"] for k in X.select_paired(rows, exp, 4)[1][f"s2a_paired_s{SEED}"]["kept_groups"]] == [1]  # 3 would leave 1
    with pytest.raises(AssertionError, match="< --kept"):
        X.select_paired(rows, exp, 100)


def test_select_paired_never_leaves_remainder_one():
    """Old dead end: sizes [3, 2, 2, 3], kept 6 -> 3 + 2 = 5, remainder 1 unfillable, 3 crosses -> error. New rule skips
    both 2s ("would leave remainder 1") and takes the last 3."""
    X = _export()
    rows, exp, n = [], {}, 0
    for g, sz in enumerate([3, 2, 2, 3]):
        exp[(SEED, g)] = list(range(n, n + sz))
        rows += [{"file": "f", "group": f"ep_{i}", "stage": "s2a", "seed": SEED, "episode_index": i, "paired": True,
                  "layout_id": g, "member": i - n} for i in range(n, n + sz)]
        n += sz
    ch, rep = X.select_paired(rows, exp, 6)
    r = rep[f"s2a_paired_s{SEED}"]
    assert len(ch) == r["kept_episodes"] == 6 and [k["group"] for k in r["kept_groups"]] == [0, 3]
    assert [(s["group"], s["reason"]) for s in r["skipped_groups"]] == [(1, "would leave remainder 1"),
                                                                        (2, "would leave remainder 1")]


def test_select_paired_too_few_is_an_error():
    X = _export()
    rows, exp = _fake(20)
    with pytest.raises(AssertionError, match="< --kept"):
        X.select_paired(rows, exp, 10_000)


def test_select_paired_rejects_unpaired_rows():
    X = _export()
    rows, exp = _fake(10)
    rows[0].pop("paired")
    with pytest.raises(AssertionError, match="no paired attrs"):
        X.select_paired(rows, exp, 5)


def test_select_paired_per_stream_kept_override():
    """Task 007: a stream of 3-member groups cannot reach a kept count that is not a multiple of 3; --kept-stream sets
    that stream's count while the other stream keeps --kept (the pick-rehearsal union)."""
    X = _export()
    rows, exp, n = [], {}, 0
    for seed, stage, sizes in ((33000, "s3a", [3] * 5), (32900, "s2", [2, 3, 2, 3])):
        for g, sz in enumerate(sizes):
            exp[(seed, g)] = list(range(n, n + sz))
            rows += [{"file": "f", "group": f"ep_{i}", "stage": stage, "seed": seed, "episode_index": i, "paired": True,
                      "layout_id": g, "member": i - n} for i in range(n, n + sz)]
            n += sz
    with pytest.raises(AssertionError, match="stream \('s3a', 33000\)"):
        X.select_paired(rows, exp, 10)  # 3-member groups: 9, then a remainder of 1 -> unreachable
    ch, rep = X.select_paired(rows, exp, 10, {("s3a", 33000): 9})
    assert rep["s3a_paired_s33000"]["kept_episodes"] == 9 and rep["s2_paired_s32900"]["kept_episodes"] == 10
    assert len(ch) == 19
