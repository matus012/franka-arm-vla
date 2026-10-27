"""Task 009b: one cube size per scene for place / stack (samplers s3a1 / s3b1; s3a1 : s3a = s3b1 : s3b = s2a : s2).

 sampler   every scene is ONE size with distinct colours (2-4 cubes), target in SEEN_CUBES, destination on the table
           (place: one of the 4 pads; stack: another cube of the scene); the drawn (cube, dest) set over a long run is
           exactly the s3a / s3b split-train valid pairs of one size, written out independently (held-out pairs never;
           held-out manipulated cubes never; held-out cubes may be stack bottoms); size U{small, large}; given the size,
           the pair is uniform over that size's valid pairs (as s3a / s3b are over theirs); deterministic in the rng;
           heldout_comp refused;
 instr.    colour alone resolves the target (and the stack bottom) to exactly one table cube; a size word, when the
           shared template draws one (P_SIZE_UNFORCED, as s2a), is the scene's size; train templates pass the export audit;
 paired    make_paired_stream s3a1 / s3b1 follows the 007 member rule (member 0 = sampler pair, 1 = same cube / other
           destination, 2 = other cube / same destination when the class exists), relation recorded, same scene across
           members (= make_stream episode g), batch restart deterministic; collect / eval / export / paired_check (item P) /
           contrastive layout table (group (stage, seed, layout)) / l' (another member's instruction) / probe_s3 accept them;
 audit     data streams 33200 / 33300 registered, validation 21665 / 21695 and tests 11580+k / 11600+k registered and
           scene-indexed, a paired s3a1 / s3b1 set audits clean against them and collides with its own seed's stream;
 frozen    s2a / s2 / s3a / s3b / s4 streams bit-identical to the pre-change implementation (digests recorded before
           the edit, 2026-10-26).
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import paired_check as PC  # noqa: E402
import test_data_audit as A  # noqa: E402
from test_paired import _collect, _export  # noqa: E402
from test_paired_s3 import _digest, groups, pair_of, run_union  # noqa: E402
from vla_common.catalog import ALL_CUBES, COLORS, SIZES  # noqa: E402
from vla_common.tasks import (HELD_OUT_PLACE_PAIRS, HELD_OUT_STACK_PAIRS, HELD_OUT_TARGETS, SEEN_CUBES,  # noqa: E402
                              STACK_ALLOWED, TaskSpec, _parse, instruction, labels, matches, matches_dest, sample_spec)
from vla_policy import contrastive as C  # noqa: E402
from vla_tabletop.rollout import RELATIONS, make_paired_stream, make_stream, relation, valid_pairs  # noqa: E402

M, N = 15.0, 60
TASK = {"s3a1": "place", "s3b1": "stack"}
SEEDS = {"s3a1": 33200, "s3b1": 33300}  # data streams (tests/test_data_audit.py DATA_STREAMS)
VAL = {"s3a1": 21665, "s3b1": 21695}
TEST = {"s3a1": 11580, "s3b1": 11600}


def expected_pairs(stage: str, size: str) -> set:
    """s3a / s3b split-train rules on a one-size table, written out from the catalog (not rollout.valid_pairs)."""
    tops = [c for c in ALL_CUBES if c[1] == size and c not in HELD_OUT_TARGETS]
    if TASK[stage] == "place":
        return {(t, p) for t in tops for p in COLORS if (t, p) not in HELD_OUT_PLACE_PAIRS}
    return {(t, b) for t in tops for b in ALL_CUBES if b[1] == size and b != t and (t, b) not in HELD_OUT_STACK_PAIRS}


def dest_of(spec: TaskSpec):
    return tuple(spec.dest) if spec.task == "stack" else spec.dest


@pytest.fixture(scope="module", params=["s3a1", "s3b1"])
def one(request):
    st = request.param
    eps, sk = make_paired_stream(SEEDS[st], 0, N, st, "train", M)
    return st, eps, sk


# ---------------------------------------------------------------- sampler
@pytest.mark.parametrize("stage", ["s3a1", "s3b1"])
@pytest.mark.parametrize("split", ["train", "paraphrase"])
def test_one_size_distinct_colours_target_and_dest_on_table(stage, split):
    rng = np.random.default_rng(9)
    ns = set()
    for _ in range(3000):
        s = sample_spec(stage, split, rng)
        cubes = [tuple(c) for c in s.cubes]
        assert s.task == TASK[stage] and 2 <= len(cubes) <= 4 and s.pads == list(COLORS)
        assert len({c[1] for c in cubes}) == 1 and len({c[0] for c in cubes}) == len(cubes)
        assert tuple(s.target) in cubes and tuple(s.target) in SEEN_CUBES
        if s.task == "place":
            assert s.dest in s.pads and (tuple(s.target), s.dest) not in HELD_OUT_PLACE_PAIRS
        else:
            assert tuple(s.dest) in cubes and tuple(s.dest) != tuple(s.target)
            assert (tuple(s.target), tuple(s.dest)) not in HELD_OUT_STACK_PAIRS
        ns.add(len(cubes))
    assert ns == {2, 3, 4}


@pytest.mark.parametrize("stage", ["s3a1", "s3b1"])
def test_drawn_pairs_are_exactly_the_one_size_valid_pairs_uniform_given_size(stage):
    """Held-out mapping: a held-out pair whose cubes share the scene's size is excluded, every other pair is reachable."""
    rng = np.random.default_rng(3)
    S = 12000
    cnt: dict = {}
    for _ in range(S):
        s = sample_spec(stage, "train", rng)
        k = (tuple(s.target), dest_of(s))
        cnt[k] = cnt.get(k, 0) + 1
    want = {sz: expected_pairs(stage, sz) for sz in SIZES}
    assert set(cnt) == want["small"] | want["large"]
    n_small = sum(v for (t, _), v in cnt.items() if t[1] == "small")
    assert abs(n_small - S / 2) <= 4 * np.sqrt(S / 4), n_small  # size U{small, large}
    for sz, ps in want.items():
        n = sum(cnt[p] for p in ps)
        for p in ps:  # uniform over the size's valid pairs (the s3a / s3b rule restricted to the size)
            assert abs(cnt[p] - n / len(ps)) <= 4.5 * np.sqrt(n / len(ps)), (p, cnt[p], n / len(ps))
    if stage == "s3a1":  # every held-out place pair is one-size-feasible -> all three must be excluded
        assert {len(want[sz]) for sz in SIZES} == {11, 10} and not set(HELD_OUT_PLACE_PAIRS) & set(cnt)
    else:  # every held-out stack pair mixes sizes -> none can occur; held-out cubes appear as bottoms only
        assert all(t[1] != b[1] for t, b in HELD_OUT_STACK_PAIRS) and len(want["small"]) == len(want["large"]) == 9
        assert {b for _, b in cnt} & set(HELD_OUT_TARGETS) == set(HELD_OUT_TARGETS)
        assert not {t for t, _ in cnt} & set(HELD_OUT_TARGETS)


def test_same_size_stacks_allowed_and_present():
    assert all(STACK_ALLOWED(a, b) for a, b in itertools.permutations(ALL_CUBES, 2))
    rng = np.random.default_rng(1)
    specs = [sample_spec("s3b1", "train", rng) for _ in range(200)]
    assert all(s.target[1] == s.dest[1] for s in specs) and {s.target[1] for s in specs} == set(SIZES)


@pytest.mark.parametrize("stage", ["s3a1", "s3b1"])
def test_deterministic_in_seed_and_heldout_comp_refused(stage):
    a = [sample_spec(stage, "train", np.random.default_rng([41, i])).to_json() for i in range(50)]
    b = [sample_spec(stage, "train", np.random.default_rng([41, i])).to_json() for i in range(50)]
    c = [sample_spec(stage, "train", np.random.default_rng([42, i])).to_json() for i in range(50)]
    assert a == b and a != c
    with pytest.raises(AssertionError, match="one size"):
        sample_spec(stage, "heldout_comp", np.random.default_rng(0))


# ---------------------------------------------------------------- instructions
@pytest.mark.parametrize("stage", ["s3a1", "s3b1"])
@pytest.mark.parametrize("split", ["train", "paraphrase"])
def test_colour_alone_names_target_and_dest(stage, split):
    rng = np.random.default_rng(5)
    sized = 0
    for _ in range(1500):
        s = sample_spec(stage, split, rng)
        ins = instruction(s, rng, split)
        nps = _parse(ins)
        assert len(nps) == 2 and nps[0][2] == "cube" and nps[0][1] == s.target[0], ins
        assert matches(s, ins) == [tuple(s.target)] and matches_dest(s, ins) == [dest_of(s)]
        colour_only = [c for c in s.cubes if c[0] == nps[0][1]]
        assert colour_only == [tuple(s.target)]  # the size word is never needed
        if s.task == "stack":
            assert nps[1][2] == "cube" and [c for c in s.cubes if c[0] == nps[1][1]] == [tuple(s.dest)]
        else:
            assert nps[1] == ("", s.dest, "pad")
        for size, _, kind in nps:
            assert size in ("", s.target[1]) and (kind == "cube" or size == "")
            sized += size != ""
    assert sized > 0  # the shared template's optional size word (P_SIZE_UNFORCED), as in s2a


@pytest.mark.parametrize("stage", ["s3a1", "s3b1"])
def test_unpaired_train_instructions_pass_the_export_audit(stage):
    X = _export()
    for e in make_stream(VAL[stage], 0, 80, lambda rng: sample_spec(stage, "train", rng), "train", M):
        at = {"spec_json": e.spec.to_json(), "instruction": e.instruction, "split": "train", "task": e.spec.task,
              "labels": labels(e.spec).tolist(), "seed": SEEDS[stage], "stage": stage, "visual_mode": "nominal",
              "visual_json": {"mode": "nominal"}}
        assert X.audit_episode(at) == [], (e.instruction, X.audit_episode(at))


# ---------------------------------------------------------------- paired stream (007 member rule)
def test_paired_members_follow_the_007_rule(one):
    st, eps, skipped = one
    gs = groups(eps)
    assert set(gs) | {s["group"] for s in skipped} == set(range(N))
    n3 = 0
    for g, ms in gs.items():
        spec = sample_spec(st, "train", np.random.default_rng([SEEDS[st], g]))
        v = valid_pairs(spec)
        prs = [pair_of(e) for e in ms]
        assert prs[0] == (tuple(spec.target), dest_of(spec)) and [e.member for e in ms] == list(range(len(ms)))
        assert 2 <= len(prs) <= 3 and len(set(prs)) == len(prs) and set(prs) <= set(v)
        assert set(v) == {p for p in expected_pairs(st, spec.target[1]) if p[0] in spec.cubes
                          and (st == "s3a1" or p[1] in spec.cubes)}
        a = [x for x in v if x[0] == prs[0][0] and x[1] != prs[0][1]]
        b = [x for x in v if x[0] != prs[0][0] and x[1] == prs[0][1]]
        if a and b:
            assert prs[1] in a and prs[2] in b and ms[1].relation == "same_target" and ms[2].relation == "same_dest"
            n3 += 1
        assert ms[0].relation == "anchor" and all(e.relation == relation(prs[0], p) for e, p in zip(ms[1:], prs[1:]))
        for e, (t, d) in zip(ms, prs):
            assert len({c[1] for c in e.spec.cubes}) == 1 and e.spec.cubes == spec.cubes and e.spec.pads == spec.pads
            assert t not in HELD_OUT_TARGETS and (t, d) not in HELD_OUT_PLACE_PAIRS + HELD_OUT_STACK_PAIRS
            assert matches(e.spec, e.instruction) == [t] and matches_dest(e.spec, e.instruction) == [d]
    assert n3 >= 20, n3
    seen = {relation(p, q) for ms in gs.values() for p, q in itertools.combinations([pair_of(e) for e in ms], 2)}
    assert seen == set(RELATIONS)


def test_paired_scene_is_make_stream_episode_g_and_restart_deterministic(one):
    st, eps, skipped = one
    ref = {e.index: e for e in make_stream(SEEDS[st], 0, N, lambda rng: sample_spec(st, "train", rng), "train", M)}
    for g, ms in groups(eps).items():
        assert all(e.layout == ref[g].layout and e.layout_info == ref[g].layout_info for e in ms)
    a, sa = make_paired_stream(SEEDS[st], 0, 17, st, "train", M)
    b, sb = make_paired_stream(SEEDS[st], 17, N - 17, st, "train", M)
    assert _digest(a + b, sa + sb) == _digest(eps, skipped) and [e.index for e in eps] == list(range(len(eps)))
    with pytest.raises(AssertionError):
        make_paired_stream(SEEDS[st], 0, 2, st, "heldout_comp", M)


def test_collect_eval_report_accept_the_new_stages(one, tmp_path):
    st, eps, sk = one
    K = _collect()
    assert st in K.STAGES and K.prefix(st, "train", SEEDS[st], True) == f"{st}_paired_train_s{SEEDS[st]}"
    plan = K.plan_chunks([e.spec.task for e in eps], 16, eps[0].index)
    m = K.paired_manifest(eps, sk, plan, SEEDS[st], 0, N, st, "train")
    for g, v in m["groups_map"].items():
        prs = [pair_of(e) for e in groups(eps)[int(g)]]
        assert v["pairs"] == [[i, j, relation(prs[i], prs[j])] for i, j in itertools.combinations(range(len(prs)), 2)]
    src = (ROOT / "scripts" / "collect.py").read_text()
    assert 'or args.stage in ("s3a1", "s3b1")) and args.groups > 0' in src
    ev_src = (ROOT / "scripts" / "eval_policy.py").read_text()
    assert 'and args.stage in ("s2a", "s2", "s3a", "s3b", "s3a1", "s3b1")), \\' in ev_src
    assert 'choices=("s1", "s2", "s2a", "s3a", "s3b", "s3a1", "s3b1", "s4")' in ev_src
    # s1_report probe_s3: a perfect policy on an s3a1 / s3b1 paired eval JSON reads 100 %
    spec_ = importlib.util.spec_from_file_location("s1_report", ROOT / "scripts" / "s1_report.py")
    S = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(S)
    recs = [{"index": e.index, "task": e.spec.task, "layout_id": e.layout_id, "member": e.member, "relation": e.relation,
             "target": "_".join(e.spec.target), "n_cubes": len(e.spec.cubes), "instruction": e.instruction,
             "spec": e.spec.to_json(), "success": True, "wrong_grasp": False, "grasp_confirmed": True,
             "rest_dest": COLORS.index(e.spec.dest) if e.spec.task == "place" else ALL_CUBES.index(tuple(e.spec.dest)),
             "wrong_grasp_cube": -1} for e in eps]
    f = tmp_path / "e.json"
    f.write_text(json.dumps({"config": {"paired_targets": True, "stage": st, "seed": 1}, "ckpt": "x", "ckpt_sha": "y",
                             "episodes": recs}))
    S.probe_s3(str(f), "--out", str(tmp_path / "o.json"))
    assert json.loads((tmp_path / "o.json").read_text())["all"]["both_right_of_all"]["rate"] == 1.0


# ---------------------------------------------------------------- export / item P / contrastive
def _rows(eps, stage: str, seed: int, k0: int = 0) -> list[dict]:
    return [{"lerobot_episode_index": k0 + k, "layout_id": e.layout_id, "member": e.member, "paired": True,
             "instruction": e.instruction, "spec_json": e.spec.to_json(), "seed": seed, "relation": e.relation,
             "file_attrs": {"stage": stage, "seed": seed}} for k, e in enumerate(eps)]


def test_contrastive_table_and_l_prime(one):
    st, eps, _ = one
    tab = C.layout_table(_rows(eps, st, SEEDS[st]))
    for k, e in enumerate(eps):
        ms = [x for x in eps if x.layout_id == e.layout_id]
        assert tab[k]["group"] == (st, SEEDS[st], e.layout_id) and tab[k]["target"] == pair_of(e)
        assert tab[k]["other_instructions"] == [x.instruction for x in ms if x is not e]
    E = sorted(tab)
    tasks = [tab[e]["instruction"] for e in E]
    for step in range(10):
        out, oeps = C.pick_other_eps(tab, E, tasks, E, seed=0, step=step)
        assert all(o in tab[e]["other_instructions"] and tab[oe]["instruction"] == o and tab[oe]["group"] == tab[e]["group"]
                   for e, o, oe in zip(E, out, oeps))
    # union with the s3a stream: colliding layout ids stay separate groups (group key (stage, seed, layout))
    s3a, _ = make_paired_stream(33000, 0, 12, "s3a", "train", M)
    u = C.layout_table(_rows(eps[:30], st, SEEDS[st]) + _rows(s3a, "s3a", 33000, k0=30))
    assert len({r["group"] for r in u.values()}) == len({e.layout_id for e in eps[:30]}) + len({e.layout_id for e in s3a})


def test_export_select_paired_and_item_p_per_stream(one, tmp_path):
    st, eps, sk = one
    X = _export()
    rows = [{"file": "f", "group": f"ep_{e.index}", "stage": st, "seed": SEEDS[st], "episode_index": e.index,
             "paired": True, "layout_id": e.layout_id, "member": e.member} for e in eps]
    exp: dict = {}
    for e in eps:
        exp.setdefault((SEEDS[st], e.layout_id), []).append(e.index)
    chosen, rep = X.select_paired(rows, exp, 60)
    assert list(rep) == [f"{st}_paired_s{SEEDS[st]}"] and len(chosen) == 60
    # item P on a synthetic s2 + <stage> union (all groups kept, split per group rank)
    h5py = pytest.importorskip("h5py")
    K = _collect()
    side, state0, dirs, k = [], {}, [], 0
    rng = np.random.default_rng(5)
    n = 30
    for s_, seed, (es, skp) in (("s2", 32900, make_paired_stream(32900, 0, n, "s2", "train", M)),
                                (st, SEEDS[st], ([e for e in eps if e.layout_id < n], [x for x in sk if x["group"] < n]))):
        d = tmp_path / s_
        d.mkdir()
        (d / "paired.json").write_text(json.dumps(K.paired_manifest(
            es, skp, K.plan_chunks([e.spec.task for e in es], 128, 0), seed, 0, n, s_, "train")))
        with h5py.File(d / "x.h5", "w") as h:
            for e in es:
                h.create_group(f"ep_{e.index}").attrs["episode_index"] = e.index
        dirs.append(d)
        ranks = {g: r for r, g in enumerate(sorted({e.layout_id for e in es}))}
        for e in es:
            slot = int(rng.integers(0, 128))
            side.append({"layout_id": e.layout_id, "member": e.member, "lerobot_episode_index": k, "paired": True,
                         "labels": json.dumps(labels(e.spec).tolist()), "instruction": e.instruction, "slot": slot,
                         "env_id": slot, "split": "val" if ranks[e.layout_id] % 20 == 19 else "train",
                         "spec_json": e.spec.to_json(), "seed": seed, "episode_index": e.index, "cube_yaw_max_deg": M,
                         **({"relation": e.relation} if e.relation else {}), "file_attrs": {"stage": s_, "seed": seed}})
            state0[k] = np.arange(19, dtype=np.float32) * 0.01
            k += 1
    res = run_union(side, state0, dirs)
    name = f"{st}_paired_s{SEEDS[st]}"
    assert res["pass"], {k: v for k, v in res["items"].items() if not v}
    assert res["streams"] == sorted(["s2_paired_s32900", name]) and "relation_split" in res["numbers"][name]
    rows3 = [s for s in side if s["file_attrs"]["stage"] == st]
    assert len({PC.target_class(s) for s in rows3}) == len({PC.target_key(s) for s in rows3})  # (cube, dest) classes


# ---------------------------------------------------------------- data audit registration
def test_streams_registered_and_seeds_unique():
    assert (SEEDS["s3a1"], "s3a1", "train", 15.0, "paired") in A.DATA_STREAMS
    assert (SEEDS["s3b1"], "s3b1", "train", 15.0, "paired") in A.DATA_STREAMS
    new = [(VAL[st], st) for st in TASK] + [(TEST[st] + k, st) for st in TASK for k in range(10)]
    assert all((s, st, "train", 15.0) in A.ASSIGNED_STREAMS for s, st in new)
    for s, st in new:  # a new seed carries only its new stage
        assert {a[1] for a in A.ASSIGNED_STREAMS if a[0] == s} == {st}, s
    assert {"s3a1", "s3b1"} <= set(A.SCENE_STAGES)
    assert not {d[0] for d in A.DATA_STREAMS} & {a[0] for a in A.ASSIGNED_STREAMS}


@pytest.mark.parametrize("stage", ["s3a1", "s3b1"])
def test_paired_one_size_set_audits_clean_and_collides_with_its_own_seed(tmp_path, stage):
    root = A.make_synthetic_paired(tmp_path / "p", seed=SEEDS[stage], n_groups=25, stage=stage)
    ev = [(VAL[stage], stage, "train", 15.0), (TEST[stage], stage, "train", 15.0)]
    assert A.audit_dataset(root, ev) == []
    bad = A.audit_dataset(root, [(SEEDS[stage], stage, "train", 15.0)])
    n_eps = len((root / "meta" / "episodes_113.jsonl").read_text().splitlines())
    assert sum(f"equal to test/validation stream {SEEDS[stage]}" in b for b in bad) == n_eps, bad[:3]
    idx = A.stream_index([(VAL[stage], stage, "train", 15.0)], n=5)
    spec, lay = A.regenerate(VAL[stage], 2, stage, "train", cube_yaw_max_deg=15.0)
    assert any(h[:2] == (VAL[stage], 2) and h[2] == lay for h in idx[A.scene_key(spec)])


# ---------------------------------------------------------------- existing streams unchanged (digests before the edit)
@pytest.mark.parametrize("seed,stage,start,n,n_eps,digest", [
    (33000, "s3a", 0, 40, 120, "3c82520bf88043e2d2fd7d0169edb7ece332b235e0b85789b69b82febef8cd68"),
    (33100, "s3b", 0, 40, 100, "02bd3eeb17b83b038f6085dc3473a8b723147dfaaa77a8042a2bab73141af09a"),
    (33000, "s3a", 9, 13, 39, "7aa84dba4c53a024040ed93ea22a5b87e8ab6149bb535109846bc222f66b4cd1"),
    (33100, "s3b", 9, 13, 28, "1da50ca5819d71c301a18f682c8fded9e289019d44e84c5bc07545df1d7d8ddd")])
def test_paired_s3_streams_unchanged(seed, stage, start, n, n_eps, digest):
    eps, sk = make_paired_stream(seed, start, n, stage, "train", M)
    assert len(eps) == n_eps and _digest(eps, sk) == digest


@pytest.mark.parametrize("seed,stage,split,digest", [
    (21640, "s2a", "train", "9d20a10a8f0f2244e24143d9bd8684069010cf09febb1466a7a9c64189c9091f"),
    (21650, "s2", "train", "f4d75d1ff53adaed3db9941503ce7a2e4e2aab0b0b3749dfc1e59387dcb77a87"),
    (21660, "s3a", "train", "492d95c7527b256bba2caecaaa37ac44d21aeb9741bd4312394eb122d9071f66"),
    (21690, "s3b", "train", "b19ebdadb2fe94e24b669cc762e78098df3548fb399da137c7b6a8f85bc49200"),
    (21215, "s3a", "heldout_comp", "6ab4aba7cc82e7dc28c91fe737c647909c3bc63e6a4b122a6abac93a61bc5270"),
    (21225, "s3b", "heldout_comp", "a3eccad284f14c24e6da2691a2cf21b1061ee15379754c3088ed7d9647364f77"),
    (11710, "s3a", "paraphrase", "accea32b0a4603b5c8814e9aba7c07cd6c1b80bac67428ba2ccedf56021b1366"),
    (21720, "s4", "train", "6c07ba0e03ea003653cfb600097a315b7a3f47fe74de027c73e1ee1747a58282")])
def test_unpaired_streams_unchanged(seed, stage, split, digest):
    eps = make_stream(seed, 0, 40, lambda rng: sample_spec(stage, split, rng), split, M)
    assert _digest(eps, []) == digest
