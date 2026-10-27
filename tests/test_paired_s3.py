"""Task 007 (3A / 3B amended): counterfactual PAIRED s3a (place: cube + pad) / s3b (stack: top + bottom) scenes, the union
with the 2b paired s2 set (pick rehearsal), l' for two-noun rows, item P per stream, margin --stage. Both venvs (CPU);
the margin test needs lerobot (policy venv) and the HDF5 selection parts need h5py.

 stream     members = 2-3 DISTINCT valid train (cube, pad) / (top, bottom) pairs of the stage sampler's scene (held-out
            pairs / held-out manipulated cubes never members, held-out cube as a stack bottom allowed), member 0 = the
            sampler's own pair, the relation type of every member recorded, all three types present over the stream, same
            scene across members (= make_stream episode g), deterministic (batch restart, rng [seed, g, 1000]);
            s2a / s2 paired streams and the unpaired s3 samplers bit-identical to the pre-change implementation (digests);
 l'         s3 rows: another member's instruction, uniform over the others with the rng [seed, step, row] rule; the episode
            returned is that member's; check_others flags l == l', a foreign instruction, a paraphrase of the own pair;
 union      an s2 stream and an s3a stream with colliding layout ids stay separate groups (group = (stage, seed, layout));
            export_dataset.select_paired keeps --kept per stream for two streams;
 P          paired_check per stream on a synthetic s2 + s3a union passes; mutants flagged in their stream only; a source
            selection that is not per stream fails; relation split reported;
 margin     s2r_tools.margin --stage on a union export = the independent computation over that stage's groups.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paired_check as PC  # noqa: E402
from vla_common.catalog import ALL_CUBES, COLORS  # noqa: E402
from vla_common.tasks import (HELD_OUT_PLACE_PAIRS, HELD_OUT_STACK_PAIRS, HELD_OUT_TARGETS, SEEN_CUBES,  # noqa: E402
                              TaskSpec, instruction, labels, matches, matches_dest, sample_spec)
from vla_policy import contrastive as C  # noqa: E402
from vla_tabletop.rollout import (RELATIONS, make_paired_stream, make_stream, pair_members, relation,  # noqa: E402
                                  valid_pairs)

M, N = 15.0, 60
SEEDS = {"s3a": 33000, "s3b": 33100}  # data streams (tests/test_data_audit.py DATA_STREAMS)


@pytest.fixture(scope="module", params=["s3a", "s3b"])
def s3(request):
    st = request.param
    eps, sk = make_paired_stream(SEEDS[st], 0, N, st, "train", M)
    return st, eps, sk


def groups(eps):
    out = {}
    for e in eps:
        out.setdefault(e.layout_id, []).append(e)
    return out


def pair_of(e) -> tuple:
    return tuple(e.spec.target), tuple(e.spec.dest) if e.spec.task == "stack" else e.spec.dest


def valid_by_rules(spec: TaskSpec, t: tuple, d) -> bool:
    """The stage sampler's split-train rules, written out independently of rollout.valid_pairs."""
    cubes = [tuple(c) for c in spec.cubes]
    if t not in cubes or t not in SEEN_CUBES:
        return False
    if spec.task == "place":
        return d in COLORS and (t, d) not in HELD_OUT_PLACE_PAIRS
    return d in cubes and d != t and (t, d) not in HELD_OUT_STACK_PAIRS


# ---------------------------------------------------------------- stream
def test_members_distinct_valid_anchor_is_the_sampler_pair(s3):
    st, eps, skipped = s3
    gs = groups(eps)
    assert set(gs) | {s["group"] for s in skipped} == set(range(N)) and not set(gs) & {s["group"] for s in skipped}
    for g, ms in gs.items():
        spec = sample_spec(st, "train", np.random.default_rng([SEEDS[st], g]))
        prs = [pair_of(e) for e in ms]
        assert 2 <= len(prs) <= 3 and len(set(prs)) == len(prs)
        assert prs[0] == (tuple(spec.target), tuple(spec.dest) if st == "s3b" else spec.dest)
        assert [e.member for e in ms] == list(range(len(ms)))
        for e, (t, d) in zip(ms, prs):
            assert valid_by_rules(spec, t, d) and (t, d) in valid_pairs(spec)
            assert t not in HELD_OUT_TARGETS and (t, d) not in HELD_OUT_PLACE_PAIRS + HELD_OUT_STACK_PAIRS
            assert e.spec.task == spec.task and [tuple(c) for c in e.spec.cubes] == [tuple(c) for c in spec.cubes]
            assert e.spec.pads == spec.pads
            assert matches(e.spec, e.instruction) == [t] and matches_dest(e.spec, e.instruction) == [d]
    for s in skipped:
        spec = sample_spec(st, "train", np.random.default_rng([SEEDS[st], s["group"]]))
        assert len(pair_members(SEEDS[st], s["group"], spec)) == s["n_valid"] < 2


def test_instructions_pass_the_export_audit(s3):
    from test_paired import _export

    X = _export()
    st, eps, _ = s3
    for e in eps:
        at = {"spec_json": e.spec.to_json(), "instruction": e.instruction, "split": "train", "task": e.spec.task,
              "labels": labels(e.spec).tolist(), "seed": SEEDS[st], "stage": st, "visual_mode": "nominal",
              "visual_json": {"mode": "nominal"}}
        assert X.audit_episode(at) == [], (e.instruction, X.audit_episode(at))


def test_relation_types_recorded_and_all_three_occur(s3):
    st, eps, _ = s3
    seen = {r: 0 for r in RELATIONS}
    for ms in groups(eps).values():
        prs = [pair_of(e) for e in ms]
        assert ms[0].relation == "anchor"
        for e, p in zip(ms[1:], prs[1:]):
            t0, d0 = prs[0]
            want = "same_target" if p[0] == t0 else "same_dest" if p[1] == d0 else "both_differ"
            assert e.relation == want == relation(prs[0], p)
        for i in range(len(prs)):
            for j in range(i + 1, len(prs)):
                seen[relation(prs[i], prs[j])] += 1
        if len(ms) == 3 and ms[1].relation == "same_target" and ms[2].relation == "same_dest":
            assert relation(prs[1], prs[2]) == "both_differ"  # the selection rule's mix inside one group
    assert all(v > 0 for v in seen.values()), seen


def test_selection_rule_prefers_one_of_each_class():
    """pair_members: member 1 from the same-target class, member 2 from the same-destination class when both exist."""
    for st, seed in SEEDS.items():
        n3 = 0
        for g in range(80):
            spec = sample_spec(st, "train", np.random.default_rng([seed, g]))
            v = valid_pairs(spec)
            m = pair_members(seed, g, spec)
            a = [x for x in v if x[0] == m[0][0] and x[1] != m[0][1]]
            b = [x for x in v if x[0] != m[0][0] and x[1] == m[0][1]]
            assert m == pair_members(seed, g, spec)  # deterministic
            if a and b:
                assert m[1] in a and m[2] in b and len(m) == 3
                n3 += 1
            assert len(m) == min(3, len(v))
        assert n3 >= 20, (st, n3)


def test_held_out_pairs_and_targets_excluded_on_hand_scenes():
    cubes = [("red", "small"), ("blue", "large"), ("green", "large"), ("red", "large")]
    place = TaskSpec("place", ("red", "small"), "red", cubes, list(COLORS))
    vp = valid_pairs(place)
    assert (("red", "small"), "blue") not in vp and (("red", "large"), "green") not in vp
    assert (("blue", "large"), "yellow") not in vp and not any(t == ("green", "large") for t, _ in vp)
    assert len(vp) == 3 * 4 - 3
    stack = TaskSpec("stack", ("red", "small"), ("green", "large"), cubes, list(COLORS))
    vs = valid_pairs(stack)
    assert (("red", "small"), ("blue", "large")) not in vs and not any(t == b for t, b in vs)
    assert (("red", "small"), ("green", "large")) in vs  # a held-out cube may be the bottom
    assert not any(t == ("green", "large") for t, _ in vs) and len(vs) == 3 * 3 - 1


def test_scene_is_make_stream_episode_g_and_identical_across_members(s3):
    st, eps, _ = s3
    ref = {e.index: e for e in make_stream(SEEDS[st], 0, N, lambda rng: sample_spec(st, "train", rng), "train", M)}
    for g, ms in groups(eps).items():
        assert all(e.layout == ref[g].layout and e.layout_info == ref[g].layout_info for e in ms)
        assert (ms[0].spec.target, ms[0].spec.dest) == (ref[g].spec.target, ref[g].spec.dest)


def test_deterministic_in_seed_group_member_and_batch_restart(s3):
    st, eps, skipped = s3
    a, sa = make_paired_stream(SEEDS[st], 0, 23, st, "train", M)
    b, sb = make_paired_stream(SEEDS[st], 23, N - 23, st, "train", M)
    key = lambda xs: [(e.index, e.layout_id, e.member, e.relation, e.spec.to_json(), e.instruction, e.layout)  # noqa: E731
                      for e in xs]
    assert key(a + b) == key(eps) and sa + sb == skipped
    assert [e.index for e in eps] == list(range(len(eps)))
    for e in eps[:30]:
        assert instruction(e.spec, np.random.default_rng([SEEDS[st], e.layout_id, e.member]), "train") == e.instruction
    with pytest.raises(AssertionError):
        make_paired_stream(SEEDS[st], 0, 2, st, "heldout_comp", M)


def _digest(eps, sk) -> str:  # tests/test_paired_s2.py _digest
    blob = json.dumps([[e.index, e.layout_id, e.member, e.spec.to_json(), e.instruction,
                        sorted((k, list(v)) for k, v in e.layout.items()), sorted(e.layout_info.items())] for e in eps]
                      + [sk], default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@pytest.mark.parametrize("seed,stage,start,n,n_eps,digest", [
    (32800, "s2a", 0, 40, 85, "078349f0a79ed2c5bdc6eab971502fcce330e56baa83c4ceaff00b02be0f0fd0"),
    (21645, "s2a", 7, 13, 24, "857da9dfd2a4db3bd7bb3b6a1161c3b4d3c512f7682091453ac8d4ab98b826a8"),
    (32900, "s2", 0, 40, 97, "b8e750501f8bef7a835af07df9fdce5bdaf7c2caa84f614892d6e9ccd16343dc"),
    (32900, "s2", 11, 17, 36, "a99da82cd05548890899c6e136aae2908f58f6abf112ade75b96305e4cb98733")])
def test_s2a_s2_paired_streams_bit_identical_to_pre_change(seed, stage, start, n, n_eps, digest):
    """Digests recorded from the implementation before the s3a / s3b extension (2026-10-26, task 007)."""
    eps, sk = make_paired_stream(seed, start, n, stage, "train", M)
    assert len(eps) == n_eps and _digest(eps, sk) == digest and all(e.relation is None for e in eps)


@pytest.mark.parametrize("stage,digest", [("s3a", "82e1d367501f7becafdd9c0437c084336a7249e008c46a82199ec42dc6685af8"),
                                          ("s3b", "b0a455260daf4c356bbaeba5f996ee77c77ea5aa27dd9eb90f465140f1bfe1f4")])
def test_unpaired_s3_streams_unchanged(stage, digest):
    eps = make_stream(21660, 0, 30, lambda rng: sample_spec(stage, "train", rng), "train", M)
    assert _digest(eps, []) == digest


def test_collect_accepts_s3_and_manifest_records_pairs(s3):
    from test_paired import _collect

    K = _collect()
    st, eps, sk = s3
    assert K.prefix(st, "train", SEEDS[st], True) == f"{st}_paired_train_s{SEEDS[st]}"
    plan = K.plan_chunks([e.spec.task for e in eps], 16, eps[0].index)
    assert {c["task"] for c in plan} == {"place" if st == "s3a" else "stack"}
    m = K.paired_manifest(eps, sk, plan, SEEDS[st], 0, N, st, "train")
    gs = groups(eps)
    for g, v in m["groups_map"].items():
        ms = gs[int(g)]
        assert v["dests"] == [list(e.spec.dest) if st == "s3b" else e.spec.dest for e in ms]
        prs = [pair_of(e) for e in ms]
        assert v["pairs"] == [[i, j, relation(prs[i], prs[j])] for i in range(len(ms)) for j in range(i + 1, len(ms))]
    s2, _ = make_paired_stream(32900, 0, 5, "s2", "train", M)
    m2 = K.paired_manifest(s2, [], K.plan_chunks(["pick"] * len(s2), 16, 0), 32900, 0, 5, "s2", "train")
    assert all(set(v) == {"episodes", "members", "targets"} for v in m2["groups_map"].values())  # s2: unchanged
    src = (Path(__file__).resolve().parents[1] / "scripts" / "collect.py").read_text()
    assert 'args.stage in ("s3a", "s3b")' in src and '**({"relation": e.relation} if e.relation is not None else {})' in src


# ---------------------------------------------------------------- l' (contrastive) on s3 rows and on a union
def _rows(eps, stage: str, seed: int, k0: int = 0) -> list[dict]:
    return [{"lerobot_episode_index": k0 + k, "layout_id": e.layout_id, "member": e.member, "paired": True,
             "instruction": e.instruction, "spec_json": e.spec.to_json(), "seed": seed, "relation": e.relation,
             "file_attrs": {"stage": stage, "seed": seed}} for k, e in enumerate(eps)]


def test_s3_table_records(s3):
    st, eps, _ = s3
    tab = C.layout_table(_rows(eps, st, SEEDS[st]))
    for k, e in enumerate(eps):
        rec = tab[k]
        ms = [x for x in eps if x.layout_id == e.layout_id]
        assert rec["target"] == pair_of(e) and rec["group"] == (st, SEEDS[st], e.layout_id) and rec["layout"] == e.layout_id
        assert rec["other_instructions"] == [x.instruction for x in ms if x is not e]
        assert [tab[o]["instruction"] for o in rec["other_eps"]] == rec["other_instructions"]
        assert rec["others"] == [pair_of(x) for x in ms if x is not e]


def test_pick_other_s3_rule_uniform_and_episode(s3):
    st, eps, _ = s3
    tab = C.layout_table(_rows(eps, st, SEEDS[st]))
    E = sorted(tab)[:45]
    tasks = [tab[e]["instruction"] for e in E]
    counts = {e: {} for e in E}
    S = 300
    for step in range(S):
        out, oeps = C.pick_other_eps(tab, E, tasks, [7 * e for e in E], seed=2, step=step)
        assert out == C.pick_other(tab, E, tasks, [7 * e for e in E], seed=2, step=step)
        for e, x, o in zip(E, out, oeps):
            rec = tab[e]
            k = int(np.random.default_rng([2, step, 7 * e]).integers(len(rec["other_instructions"])))
            assert x == rec["other_instructions"][k] != rec["instruction"]  # the same rng [seed, step, row] rule
            assert o == rec["other_eps"][k] and tab[o]["instruction"] == x and tab[o]["group"] == rec["group"]
            counts[e][o] = counts[e].get(o, 0) + 1
    for e in E:
        k = len(tab[e]["others"])
        assert len(counts[e]) == k
        assert all(abs(c - S / k) <= 4 * np.sqrt(S * (1 / k) * (1 - 1 / k)) for c in counts[e].values()), counts[e]


def test_check_others_s3_mutants(s3):
    st, eps, _ = s3
    tab = C.layout_table(_rows(eps, st, SEEDS[st]))
    e = 0
    t = tab[e]["instruction"]
    with pytest.raises(AssertionError, match="contrast is void"):
        C.check_others(tab, [e], [t], [t])
    foreign = next(r["instruction"] for r in tab.values() if r["group"] != tab[e]["group"])
    with pytest.raises(AssertionError, match="does not name exactly one other"):
        C.check_others(tab, [e], [t], [foreign])
    para = "please " + t if not t.startswith("please") else t[len("please "):]  # the own pair in another template
    with pytest.raises(AssertionError, match="does not name exactly one other"):
        C.check_others(tab, [e], [t], [para])
    C.check_others(tab, [e], [t], [tab[e]["other_instructions"][-1]])


def test_s3_table_rejects_wrong_destination_and_duplicate_pair(s3):
    st, eps, _ = s3
    g = next(e.layout_id for e in eps if len(e.spec.cubes) >= 3)
    rows = _rows([x for x in eps if x.layout_id == g], st, SEEDS[st])
    bad = copy.deepcopy(rows)  # the spec's destination moved, the instruction still names the old one
    sp = json.loads(bad[0]["spec_json"])
    used = [json.loads(r["spec_json"])["dest"] for r in rows]
    pool = COLORS if st == "s3a" else [list(c) for c in sp["cubes"] if list(c) != sp["target"]]
    new = next((d for d in pool if d not in used), None)
    if new is None:
        pytest.skip("every destination of this scene is used by a member")
    bad[0]["spec_json"] = json.dumps(sp | {"dest": new})
    with pytest.raises(ValueError, match="does not name exactly its target"):
        C.layout_table(bad)
    dup = copy.deepcopy(rows)
    dup[1]["spec_json"], dup[1]["instruction"] = dup[0]["spec_json"], dup[0]["instruction"]
    with pytest.raises(ValueError, match="distinct targets"):
        C.layout_table(dup)


def test_union_layout_ids_collide_but_groups_stay_separate():
    s2, _ = make_paired_stream(32900, 0, 12, "s2", "train", M)
    s3a, _ = make_paired_stream(33000, 0, 12, "s3a", "train", M)
    rows = _rows(s2, "s2", 32900) + _rows(s3a, "s3a", 33000, k0=len(s2))
    assert {e.layout_id for e in s2} & {e.layout_id for e in s3a}  # the layout ids collide
    tab = C.layout_table(rows)
    assert len({r["group"] for r in tab.values()}) == len({e.layout_id for e in s2}) + len({e.layout_id for e in s3a})
    for e, r in tab.items():
        mates = [o for o, q in tab.items() if q["group"] == r["group"] and o != e]
        assert len(mates) == len(r["others"]) and all(tab[o]["group"][0] == r["group"][0] for o in mates)
        assert ("cubes" in r) == (r["group"][0] == "s2") and ("other_instructions" in r) == (r["group"][0] == "s3a")
    E = sorted(tab)
    tasks = [tab[e]["instruction"] for e in E]
    for step in range(20):
        out, oeps = C.pick_other_eps(tab, E, tasks, E, seed=0, step=step)
        assert all(tab[o]["group"] == tab[e]["group"] and tab[o]["instruction"] != tab[e]["instruction"]
                   for e, o in zip(E, oeps))
    clash = copy.deepcopy(rows)  # one (seed, layout_id) holding two stages is refused
    for r in clash[len(s2):]:
        r["seed"] = 32900
    with pytest.raises(ValueError, match="different stages"):
        C.layout_table(clash)


def test_export_select_paired_two_streams_kept_per_stream():
    from test_paired import _export

    X = _export()
    rows, exp = [], {}
    for st, seed, n in (("s2", 32900, 40), ("s3a", 33000, 40)):
        eps, _ = make_paired_stream(seed, 0, n, st, "train", M)
        for e in eps:
            exp.setdefault((seed, e.layout_id), []).append(e.index)
            rows.append({"file": f"f{st}", "group": f"ep_{e.index}", "stage": st, "seed": seed, "episode_index": e.index,
                         "paired": True, "layout_id": e.layout_id, "member": e.member})
    chosen, rep = X.select_paired(rows, exp, 60)
    assert set(rep) == {"s2_paired_s32900", "s3a_paired_s33000"}
    for name, (st, f) in {"s2_paired_s32900": ("s2", "fs2"), "s3a_paired_s33000": ("s3a", "fs3a")}.items():
        assert rep[name]["kept_episodes"] == 60 == sum(k[0] == f for k in chosen)
        r = rep[name]["kept_groups"]
        assert [g["split"] for g in r] == ["val" if g["rank"] % 20 == 19 else "train" for g in r]


# ---------------------------------------------------------------- item P per stream on a synthetic s2 + s3a union
def synth_union(tmp_path: Path, n2: int = 30, n3: int = 40):
    """Sidecar rows + t=0 states of an s2 (32900) + s3a (33000) paired union (all groups kept, split per group rank) and
    one source dir per stream (paired.json from collect.paired_manifest + an HDF5 holding the episode indices)."""
    h5py = pytest.importorskip("h5py")
    from test_paired import _collect

    K = _collect()
    side, state0, dirs, k = [], {}, [], 0
    rng = np.random.default_rng(5)
    for st, seed, n in (("s2", 32900, n2), ("s3a", 33000, n3)):
        eps, sk = make_paired_stream(seed, 0, n, st, "train", M)
        d = tmp_path / st
        d.mkdir()
        plan = K.plan_chunks([e.spec.task for e in eps], 128, 0)
        (d / "paired.json").write_text(json.dumps(K.paired_manifest(eps, sk, plan, seed, 0, n, st, "train")))
        with h5py.File(d / "x.h5", "w") as h:
            for e in eps:
                h.create_group(f"ep_{e.index}").attrs["episode_index"] = e.index
        dirs.append(d)
        ranks = {g: r for r, g in enumerate(sorted({e.layout_id for e in eps}))}
        for e in eps:
            slot = int(rng.integers(0, 128))
            side.append({"layout_id": e.layout_id, "member": e.member, "lerobot_episode_index": k, "paired": True,
                         "labels": json.dumps(labels(e.spec).tolist()), "instruction": e.instruction, "slot": slot,
                         "env_id": slot, "split": "val" if ranks[e.layout_id] % 20 == 19 else "train",
                         "spec_json": e.spec.to_json(), "seed": seed, "episode_index": e.index, "cube_yaw_max_deg": M,
                         **({"relation": e.relation} if e.relation else {}), "file_attrs": {"stage": st, "seed": seed}})
            state0[k] = np.arange(19, dtype=np.float32) * 0.01
            k += 1
    return side, state0, dirs


def run_union(side, state0, dirs, per_stream: bool = True, layouts: dict | None = None):
    streams = PC.stream_rows(side)
    per = {key: PC.stream_items(v, state0, None, dirs, 0, key if per_stream else None,
                                (layouts or {}).get(key)) for key, v in streams.items()}
    return PC.combine(Path("union"), per)


def test_paired_check_union_passes_per_stream(tmp_path):
    side, state0, dirs = synth_union(tmp_path)
    res = run_union(side, state0, dirs)
    assert res["pass"], {k: v for k, v in res["items"].items() if not v}
    assert res["streams"] == ["s2_paired_s32900", "s3a_paired_s33000"]
    assert {k.split("/")[1] for k in res["items"]} >= {"P_i_distinct_targets", "P1_layout_identical", "P2b_state_classifier",
                                                         "P_regenerates", "P_kept_groups_rule", "P_split_rule"}
    n3 = res["numbers"]["s3a_paired_s33000"]
    rs = n3["relation_split"]
    eps, _ = make_paired_stream(33000, 0, 40, "s3a", "train", M)
    want = {r: 0 for r in (*RELATIONS, "duplicate")}
    for ms in groups(eps).values():
        prs = [pair_of(e) for e in ms]
        for i in range(len(prs)):
            for j in range(i + 1, len(prs)):
                want[relation(prs[i], prs[j])] += 1
    assert rs["counts"] == want and rs["n_recorded_relation_mismatch"] == 0 and all(want[r] > 0 for r in RELATIONS)
    assert "relation_split" not in res["numbers"]["s2_paired_s32900"]
    s3rows = [s for s in side if s["file_attrs"]["stage"] == "s3a"]  # P2 classes = (cube, pad) identity
    assert n3["P2b"]["classes"] == sorted({PC.target_class(s) for s in s3rows})
    assert len(n3["P2b"]["classes"]) == len({(tuple(json.loads(s["spec_json"])["target"]), json.loads(s["spec_json"])["dest"])
                                             for s in s3rows})


def test_paired_check_union_mutants_flagged_in_their_stream(tmp_path):
    side, state0, dirs = synth_union(tmp_path)
    s3 = [i for i, s in enumerate(side) if s["file_attrs"]["stage"] == "s3a"]
    # duplicate (cube, pad) target in an s3a group
    d = copy.deepcopy(side)
    a = next(i for i in s3 if d[i]["member"] == 0)
    for key in ("labels", "instruction", "spec_json"):
        d[a + 1][key] = d[a][key]
    res = run_union(d, state0, dirs)
    assert not res["items"]["s3a_paired_s33000/P_i_distinct_targets"] and res["items"]["s2_paired_s32900/P_i_distinct_targets"]
    # same cube, other pad is a DIFFERENT two-noun target (the clean union passes P_i with such pairs)
    same = next(i for i in s3 if side[i].get("relation") == "same_target")
    m0 = next(j for j in s3 if side[j]["layout_id"] == side[same]["layout_id"] and side[j]["member"] == 0)
    assert PC.target_key(side[same])[:2] == PC.target_key(side[m0])[:2] and PC.target_key(side[same]) != PC.target_key(side[m0])
    # re-sampled layout for one s3a member
    from vla_tabletop.layout import sample_layout

    lay = PC.regenerated_layouts([side[i] for i in s3])
    g = side[a]["layout_id"]
    bad = dict(lay)
    bad[(g, 1)] = sample_layout(TaskSpec.from_json(side[a]["spec_json"]), np.random.default_rng([1, g, 99]), cube_yaw_max_deg=M)[0]
    res = run_union(side, state0, dirs, layouts={("s3a", 33000): bad})
    assert not res["items"]["s3a_paired_s33000/P1_layout_identical"] and res["items"]["s2_paired_s32900/P1_layout_identical"]
    # split straddle in s3a
    s = copy.deepcopy(side)
    s[a + 1]["split"] = "val" if s[a + 1]["split"] == "train" else "train"
    res = run_union(s, state0, dirs)
    assert not res["items"]["s3a_paired_s33000/P_iv_group_in_one_split"] and not res["pass"]
    # target-dependent t=0 state in s3a only
    st = {k: v.copy() for k, v in state0.items()}
    for i in s3:
        st[side[i]["lerobot_episode_index"]] += np.float32(0.002) * np.float32(PC.target_class(side[i]))
    res = run_union(side, st, dirs)
    assert not res["items"]["s3a_paired_s33000/P2b_state_classifier"] and res["items"]["s2_paired_s32900/P2b_state_classifier"]
    # selection from the merged source dirs (not per stream): layout ids of the two streams collide -> flagged
    res = run_union(side, state0, dirs, per_stream=False)
    assert not res["pass"] and not all(res["items"][f"{n}/P_kept_groups_rule"] for n in res["streams"])
    # recorded relation attr wrong -> reported
    r = copy.deepcopy(side)
    r[a + 1]["relation"] = "both_differ" if r[a + 1]["relation"] != "both_differ" else "same_dest"
    assert run_union(r, state0, dirs)["numbers"]["s3a_paired_s33000"]["relation_split"]["n_recorded_relation_mismatch"] == 1


def test_paired_check_single_stream_report_unchanged_shape(tmp_path):
    side, state0, dirs = synth_union(tmp_path)
    one = [s for s in side if s["file_attrs"]["stage"] == "s2"]
    res = PC.combine(Path("x"), {("s2", 32900): PC.stream_items(one, state0, None, dirs[:1], 0)})
    assert list(res) == ["dataset", "name", "items", "pass", "numbers", "regenerate_mismatch_episodes", "selection"]
    assert res["pass"] and "P_i_distinct_targets" in res["items"]


# ---------------------------------------------------------------- margin --stage on a union export
def test_margin_stage_on_union_equals_independent(tmp_path):
    pytest.importorskip("lerobot")
    torch = pytest.importorskip("torch")
    import s2r_tools as S
    from synth_episodes import LATENCY, PERIOD, episode_arrays

    from vla_common.episode import EpisodeWriter
    from vla_policy import train_smolvla as T
    from vla_policy.export import export

    rng = np.random.default_rng(3)
    files, members, k = [], {}, 0
    for st, seed in (("s2", 32900), ("s3a", 33000)):
        eps, _ = make_paired_stream(seed, 0, 4, st, "train", M)
        f = tmp_path / f"{st}.h5"
        with EpisodeWriter(f, {"git_sha": "test", "created": "test", "stage": st}) as w:
            for e in eps:
                n = 52 + (7 * k) % 11
                at = {"instruction": e.instruction, "split": "train", "task": e.spec.task, "spec_json": e.spec.to_json(),
                      "labels": labels(e.spec).tolist(), "seed": seed, "env_id": k, "success": True, "metrics_json": {},
                      "period": PERIOD, "latency": LATENCY, "image_size": 48, "visual_mode": "nominal",
                      "action_noise_sigma": 0.1, "expert_version": "synthetic", "episode_index": e.index, "T": n,
                      "n_action_steps": 10, "detector_version": "none", "paired": True, "layout_id": e.layout_id,
                      "member": e.member}
                w.add(**episode_arrays(n, k, rng), attrs=at)
                members.setdefault((st, e.layout_id), []).append(k)
                k += 1
        files.append(f)
    export(files, "u", root=tmp_path / "ds")
    root = tmp_path / "ds" / "u"
    ds = T.open_dataset(root, "pyav")
    is_val, train_eps, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    mean, std = (torch.as_tensor(stats["action"][q], dtype=torch.float64) for q in ("mean", "std"))
    ep = T.column(ds, "episode_index").reshape(-1)
    first = {int(e): int(np.flatnonzero(ep == e)[0]) for e in train_eps}
    chunk = {e: (torch.as_tensor(np.asarray(ds[first[e]]["action"]), dtype=torch.float64) - mean) / (std + 1e-8)
             for e in train_eps}
    mates = {i: [j for j in v if j != i] for v in members.values() for i in v}

    def want(stage=None):
        es = [e for e in train_eps if stage is None or any(e in v for (s_, _), v in members.items() if s_ == stage)]
        return np.mean([np.mean([float(((chunk[e] - chunk[o]) ** 2).mean()) for o in mates[e]]) for e in es]), es

    for stage in (None, "s3a", "s2"):
        r = S.margin(root, stage=stage)
        w, es = want(stage)
        assert r["expected_sq_diff"] == pytest.approx(w, rel=1e-9) and r["margin"] == pytest.approx(0.5 * w, rel=1e-9)
        assert r["train_episodes"] == len(es) and r["task_column_checked"]
        assert r["ordered_pairs"] == sum(len(mates[e]) for e in es)
        assert r.get("stage") == stage
    assert {lid for s_, lid in members if s_ == "s2"} & {lid for s_, lid in members if s_ == "s3a"}  # ids collide
    assert S.margin(root, stage=None)["train_layouts"] == len(members)  # ... and stay separate groups
    with pytest.raises(AssertionError, match="no train episodes"):
        S.margin(root, stage="s3b")
    assert S.main(["margin", "--dataset", str(root), "--stage", "s3a", "--out", str(tmp_path / "m.json")]) == 0
    assert json.loads((tmp_path / "m.json").read_text())["stage"] == "s3a"


def test_s3_union_check_split_rule_is_per_group_for_paired_streams():
    import s3_union_check as U

    rows = []
    for st, seed, n in (("s2", 32900, 45), ("s3a", 33000, 45)):
        eps, _ = make_paired_stream(seed, 0, n, st, "train", M)
        ranks = {g: r for r, g in enumerate(sorted({e.layout_id for e in eps}))}
        rows += [{"file_attrs": {"stage": st}, "seed": seed, "episode_index": e.index, "paired": True,
                  "layout_id": e.layout_id, "split": "val" if ranks[e.layout_id] % 20 == 19 else "train"} for e in eps]
    assert U.split_violations(rows) == []
    bad = copy.deepcopy(rows)
    i = next(k for k, r in enumerate(bad) if r["file_attrs"]["stage"] == "s3a" and r["split"] == "val")
    bad[i]["split"] = "train"
    assert U.split_violations(bad) == [(["s3a", 33000], bad[i]["episode_index"])]
    unp = [{k: v for k, v in r.items() if k != "paired"} for r in rows]  # unpaired reading: rank by episode
    assert U.split_violations(unp)  # the per-episode rule would reject a per-group split
