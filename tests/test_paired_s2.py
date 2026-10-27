"""2b amended (consultant ruling): counterfactual PAIRED s2 scenes (colours AND sizes, 2-5 cubes, the s2 sampler) and the
instruction-contrastive loss on them. Runs in both venvs (CPU): stream / collect / paired_check / l' parts need no
lerobot; the margin test (scripts/s2r_tools.py) is skipped without lerobot (policy venv only).

 stream     every member is a valid train target on the table, targets distinct as (colour, size), a same-colour pair of
            sizes gives two members whose instructions state the size, no held-out target, one scene per group,
            deterministic in (seed, g, member); s2a output bit-identical to the pre-change implementation (digests);
 l'         never l, names exactly one other member target, the size-word rule (both cases), uniform over the others,
            pick_other_eps returns the named member's episode; the guard catches a foreign / held-out / wrong-size /
            ambiguous l';
 P          paired_check on a synthetic s2 set passes; duplicate target, re-sampled layout, split straddle are flagged;
 margin     s2r_tools.margin on an exported s2 paired set equals the independent LeRobot chunk-loader computation.
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
from vla_common.catalog import ALL_CUBES, SIZES  # noqa: E402
from vla_common.guidance import named_cubes, other_instruction  # noqa: E402
from vla_common.tasks import HELD_OUT_TARGETS, SEEN_CUBES, labels, matches, sample_spec  # noqa: E402
from vla_policy import contrastive as C  # noqa: E402
from vla_tabletop.layout import sample_layout  # noqa: E402
from vla_tabletop.rollout import make_paired_stream, make_stream, valid_targets  # noqa: E402

SEED, M, N = 32900, 15.0, 60  # data stream (32900, s2, train, m 15, paired), tests/test_data_audit.py DATA_STREAMS


@pytest.fixture(scope="module")
def s2():
    return make_paired_stream(SEED, 0, N, "s2", "train", M)


def groups(eps):
    out = {}
    for e in eps:
        out.setdefault(e.layout_id, []).append(e)
    return out


# ---------------------------------------------------------------- stream
def test_members_are_the_valid_train_targets_distinct_cubes_no_heldout(s2):
    eps, skipped = s2
    gs = groups(eps)
    assert set(gs) | {s["group"] for s in skipped} == set(range(N)) and not set(gs) & {s["group"] for s in skipped}
    for g, ms in gs.items():
        spec = sample_spec("s2", "train", np.random.default_rng([SEED, g]))
        cubes = [tuple(c) for c in spec.cubes]
        tg = [tuple(e.spec.target) for e in ms]
        assert tg == valid_targets(spec, "train") == [c for c in ALL_CUBES if c in cubes and c in SEEN_CUBES]
        assert len(set(tg)) == len(tg) >= 2 and not set(tg) & set(HELD_OUT_TARGETS)
        assert [e.member for e in ms] == list(range(len(ms)))
        for e in ms:
            assert [tuple(c) for c in e.spec.cubes] == cubes and e.spec.pads == spec.pads and e.spec.task == "pick"
            assert matches(e.spec, e.instruction) == [tuple(e.spec.target)]  # names exactly its target
            assert named_cubes(e.instruction, cubes) == [[tuple(e.spec.target)]]
    for s in skipped:
        assert len(valid_targets(sample_spec("s2", "train", np.random.default_rng([SEED, s["group"]])), "train")) == s["n_valid"] < 2


def test_same_colour_two_sizes_are_two_members_with_size_words(s2):
    eps, _ = s2
    hits = 0
    for g, ms in groups(eps).items():
        tg = [tuple(e.spec.target) for e in ms]
        for col in {c for c, _ in tg}:
            pair = [e for e in ms if e.spec.target[0] == col]
            if len(pair) == 2:
                hits += 1
                assert {e.spec.target[1] for e in pair} == set(SIZES)
                assert all(f"the {e.spec.target[1]} {col} cube" in e.instruction for e in pair)
        for e in ms:  # a colour on the table twice (a held-out distractor included) forces the size word
            if sum(c[0] == e.spec.target[0] for c in e.spec.cubes) > 1:
                assert f"the {e.spec.target[1]} {e.spec.target[0]} cube" in e.instruction
    assert hits >= 5, hits  # the fixture's 60 groups hold several same-colour pairs


def test_scene_is_make_stream_episode_g_and_identical_across_members(s2):
    eps, _ = s2
    gs = groups(eps)
    ref = {e.index: e for e in make_stream(SEED, 0, N, lambda rng: sample_spec("s2", "train", rng), "train", M)}
    for g, ms in gs.items():
        assert all(e.layout == ref[g].layout and e.layout_info == ref[g].layout_info for e in ms)
        assert sorted(ms[0].spec.cubes) == sorted(ref[g].spec.cubes)
    g = next(iter(gs))
    spec = sample_spec("s2", "train", rng := np.random.default_rng([SEED, g]))
    assert gs[g][0].layout == sample_layout(spec, rng, cube_yaw_max_deg=M)[0]


def test_deterministic_in_seed_group_member_and_batch_restart(s2):
    eps, skipped = s2
    a, sa = make_paired_stream(SEED, 0, 25, "s2", "train", M)
    b, sb = make_paired_stream(SEED, 25, N - 25, "s2", "train", M)
    key = lambda xs: [(e.index, e.layout_id, e.member, e.spec.to_json(), e.instruction, e.layout) for e in xs]  # noqa: E731
    assert key(a + b) == key(eps) and sa + sb == skipped
    assert [e.index for e in eps] == list(range(len(eps)))
    from vla_common.tasks import TaskSpec, instruction

    for e in eps[:30]:  # member m's instruction is a function of rng([seed, g, m]) only
        ms = TaskSpec("pick", e.spec.target, None, list(e.spec.cubes), list(e.spec.pads))
        assert instruction(ms, np.random.default_rng([SEED, e.layout_id, e.member]), "train") == e.instruction


def _digest(eps, sk) -> str:
    blob = json.dumps([[e.index, e.layout_id, e.member, e.spec.to_json(), e.instruction,
                        sorted((k, list(v)) for k, v in e.layout.items()), sorted(e.layout_info.items())] for e in eps]
                      + [sk], default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@pytest.mark.parametrize("seed,start,n,n_eps,digest", [
    (32800, 0, 40, 85, "078349f0a79ed2c5bdc6eab971502fcce330e56baa83c4ceaff00b02be0f0fd0"),
    (21645, 7, 13, 24, "857da9dfd2a4db3bd7bb3b6a1161c3b4d3c512f7682091453ac8d4ab98b826a8")])
def test_s2a_stream_bit_identical_to_pre_change(seed, start, n, n_eps, digest):
    """Digests recorded from the implementation before the s2 extension (2026-10-26, same function as _digest)."""
    eps, sk = make_paired_stream(seed, start, n, "s2a", "train", M)
    assert len(eps) == n_eps and _digest(eps, sk) == digest


def test_collect_prefix_and_manifest_for_s2(s2):
    from test_paired import _collect

    K = _collect()
    assert K.prefix("s2", "train", SEED, True) == "s2_paired_train_s32900"
    assert K.prefix("s2a", "train", 32800, True) == "s2a_paired_train_s32800"
    eps, sk = s2
    plan = K.plan_chunks([e.spec.task for e in eps], 16, eps[0].index)
    m = K.paired_manifest(eps, sk, plan, SEED, 0, N, "s2", "train")
    assert all(len({tuple(t) for t in g["targets"]}) == len(g["targets"]) >= 2 for g in m["groups_map"].values())
    src = (Path(__file__).resolve().parents[1] / "scripts" / "collect.py").read_text()
    assert 'args.stage in ("s2a", "s2")' in src


def test_export_select_paired_keys_s2_stream(s2):
    from test_paired import _export

    X = _export()
    eps, _ = s2
    exp = {}
    for e in eps:
        exp.setdefault((SEED, e.layout_id), []).append(e.index)
    fail = {eps[3].index}
    rows = [{"file": "f", "group": f"ep_{e.index}", "stage": "s2", "seed": SEED, "episode_index": e.index, "paired": True,
             "layout_id": e.layout_id, "member": e.member} for e in eps if e.index not in fail]
    chosen, rep = X.select_paired(rows, exp, 0)
    r = rep["s2_paired_s32900"]
    assert r["skipped_groups"][0]["group"] == eps[3].layout_id and r["skipped_groups"][0]["reason"] == "member failed"
    assert r["kept_episodes"] == len(eps) - len(exp[(SEED, eps[3].layout_id)]) == len(chosen)


# ---------------------------------------------------------------- l' (vla_policy.contrastive, s2 records)
def _rows(eps, stage: str = "s2") -> list[dict]:
    return [{"lerobot_episode_index": k, "layout_id": e.layout_id, "member": e.member, "paired": True,
             "instruction": e.instruction, "spec_json": e.spec.to_json(), "file_attrs": {"stage": stage}}
            for k, e in enumerate(eps)]


def _hand_rows():
    """One layout: red small + red large + blue small + yellow large (all seen) + green large (held-out distractor)."""
    cubes = [["red", "small"], ["blue", "small"], ["red", "large"], ["yellow", "large"], ["green", "large"]]
    ins = {("red", "small"): "pick up the small red cube", ("red", "large"): "pick the large red cube up",
           ("blue", "small"): "pick up the blue cube", ("yellow", "large"): "please pick up the large yellow cube"}
    return [{"lerobot_episode_index": 10 + m, "layout_id": 7, "member": m, "paired": True, "instruction": ins[t],
             "spec_json": json.dumps({"task": "pick", "target": list(t), "dest": None, "cubes": cubes,
                                      "pads": ["red", "green", "blue", "yellow"]}),
             "file_attrs": {"stage": "s2"}} for m, t in enumerate(ins)]


def test_s2_table_records_and_s2a_records_unchanged(s2):
    tab = C.layout_table(_hand_rows())
    assert tab[12]["target"] == ("blue", "small") and tab[12]["layout"] == 7
    assert tab[12]["others"] == [("red", "small"), ("red", "large"), ("yellow", "large")]
    assert ("green", "large") in tab[12]["cubes"] and ("green", "large") not in tab[12]["others"]
    eps, _ = make_paired_stream(32800, 0, 5, "s2a", "train", M)  # s2a rows: the legacy record (colours, no "cubes")
    for rows in (_rows(eps, "s2a"), [{k: v for k, v in r.items() if k != "file_attrs"} for r in _rows(eps, "s2a")]):
        t = C.layout_table(rows)
        assert set(t[0]) == {"instruction", "target", "layout", "others", "group"} and t[0]["target"] == eps[0].spec.target[0]
        assert t[0]["group"] == ("s2a", None, t[0]["layout"])  # task 007 group key (stage, seed, layout_id); no seed here


def test_size_word_rule_both_cases():
    tab = C.layout_table(_hand_rows())
    blue, yellow = tab[12], tab[13]
    # original without a size word: the other colour in two sizes -> size word; a unique colour -> none
    assert other_instruction(blue["instruction"], blue["target"], ("red", "large"), blue["cubes"]) == "pick up the large red cube"
    assert other_instruction(blue["instruction"], blue["target"], ("yellow", "large"), blue["cubes"]) == "pick up the yellow cube"
    # original with a size word: kept (other colour unique), the rest of the template unchanged
    assert other_instruction(yellow["instruction"], yellow["target"], ("blue", "small"), yellow["cubes"]) == \
        "please pick up the small blue cube"
    red = tab[11]  # "pick the large red cube up" -> the small red (same colour, other size)
    assert other_instruction(red["instruction"], red["target"], ("red", "small"), red["cubes"]) == "pick the small red cube up"
    with pytest.raises(ValueError):
        other_instruction(blue["instruction"], blue["target"], ("blue", "small"), blue["cubes"])  # the target itself


def _check_lprime(tab, eps_, tasks, out, oeps):
    for e, t, x, o in zip(eps_, tasks, out, oeps):
        rec = tab[e]
        assert x != t
        (named,) = named_cubes(x, rec["cubes"])
        assert len(named) == 1 and named[0] in rec["others"]
        assert tab[o]["layout"] == rec["layout"] and tab[o]["target"] == named[0]  # the member l' names
        assert x == other_instruction(t, rec["target"], named[0], rec["cubes"])
        assert x.replace(" small", "").replace(" large", "").split()[0] == t.split()[0]  # template kept


def test_pick_other_s2_never_l_names_one_member_uniform_and_episode(s2):
    eps, _ = s2
    tab = C.layout_table(_rows(eps))
    E = sorted(tab)[:40]
    tasks = [tab[e]["instruction"] for e in E]
    counts = {e: {} for e in E}
    S = 300
    for step in range(S):
        out, oeps = C.pick_other_eps(tab, E, tasks, [3 * e for e in E], seed=1, step=step)
        assert out == C.pick_other(tab, E, tasks, [3 * e for e in E], seed=1, step=step)
        _check_lprime(tab, E, tasks, out, oeps)
        for e, o in zip(E, oeps):
            counts[e][o] = counts[e].get(o, 0) + 1
    for e in E:
        k = len(tab[e]["others"])
        assert len(counts[e]) == k  # every other member drawn
        assert all(abs(c - S / k) <= 4 * np.sqrt(S * (1 / k) * (1 - 1 / k)) for c in counts[e].values()), counts[e]
    hand = C.layout_table(_hand_rows())  # every size-word case appears among the drawn l'
    seen = set()
    for step in range(60):
        seen |= set(C.pick_other(hand, [12, 13], [hand[12]["instruction"], hand[13]["instruction"]], [0, 1], 0, step))
    assert {"pick up the large red cube", "pick up the small red cube", "pick up the yellow cube"} <= seen


def test_s2a_pick_other_bit_identical_rule():
    """s2a l' is still other_instructions' colour swap with the same draw (the pre-change code path)."""
    from vla_common.guidance import other_instructions

    eps, _ = make_paired_stream(32800, 0, 20, "s2a", "train", M)
    tab = C.layout_table(_rows(eps, "s2a"))
    E = sorted(tab)
    tasks = [tab[e]["instruction"] for e in E]
    for step in (1, 7, 99):
        got = C.pick_other(tab, E, tasks, E, seed=3, step=step)
        for e, x in zip(E, got):
            c = other_instructions(tab[e]["instruction"], tab[e]["target"], [tab[e]["target"], *tab[e]["others"]])
            assert x == c[int(np.random.default_rng([3, step, e]).integers(len(c)))]


def test_guard_catches_foreign_heldout_wrong_size_and_ambiguous():
    tab = C.layout_table(_hand_rows())
    t = tab[12]["instruction"]  # blue small
    for bad in ("pick up the large green cube",  # held-out distractor on the table (not a member)
                "pick up the small green cube",  # not on the table
                "pick up the large blue cube",  # same colour, wrong size (absent)
                "pick up the red cube"):  # ambiguous: two reds
        with pytest.raises(AssertionError, match="does not name exactly one other cube"):
            C.check_others(tab, [12], [t], [bad])
    with pytest.raises(AssertionError, match="does not name exactly one other cube"):  # same colour as the target, own size
        C.check_others(tab, [10], [tab[10]["instruction"]], ["pick the small red cube up"])
    with pytest.raises(AssertionError, match="contrast is void"):
        C.check_others(tab, [12], [t], [t])
    C.check_others(tab, [10], [tab[10]["instruction"]], ["pick up the large red cube"])  # the other red: fine


def test_s2_table_rejects_ambiguous_instruction_and_mixed_stages():
    rows = _hand_rows()
    rows[0]["instruction"] = "pick up the red cube"  # two reds on the table
    with pytest.raises(ValueError, match="does not name exactly its target"):
        C.layout_table(rows)
    rows = _hand_rows()
    rows[1]["file_attrs"] = {"stage": "s2a"}
    with pytest.raises(ValueError, match="different stages"):
        C.layout_table(rows)
    rows = _hand_rows()
    rows[1]["spec_json"] = rows[0]["spec_json"]  # duplicate (colour, size) target
    with pytest.raises(ValueError, match="distinct targets"):
        C.layout_table(rows)


# ---------------------------------------------------------------- item P (scripts/paired_check.py) on s2
def synth_s2(n_groups: int = N):
    eps, _ = make_paired_stream(SEED, 0, n_groups, "s2", "train", M)
    ranks = {g: k for k, g in enumerate(sorted({e.layout_id for e in eps}))}
    rng = np.random.default_rng(3)
    side, state0 = [], {}
    for k, e in enumerate(eps):
        slot = int(rng.integers(0, 128))
        side.append({"layout_id": e.layout_id, "member": e.member, "lerobot_episode_index": k, "paired": True,
                     "labels": json.dumps(labels(e.spec).tolist()), "instruction": e.instruction, "slot": slot,
                     "env_id": slot, "split": "val" if ranks[e.layout_id] % 20 == 19 else "train",
                     "spec_json": e.spec.to_json(), "seed": SEED, "episode_index": e.index, "cube_yaw_max_deg": M,
                     "file_attrs": {"stage": "s2"}})
        state0[k] = np.arange(19, dtype=np.float32) * 0.01
    return side, state0


def run_p(side, state0, layouts=None):
    items, nums = PC.p_items(side, state0, layouts=layouts)
    items["P_regenerates"] = not PC.regenerates(side)
    return items, nums


def test_paired_check_s2_clean_passes_with_cube_classes():
    side, state0 = synth_s2()
    items, nums = run_p(side, state0)
    assert all(items.values()), (items, nums["bad_groups"])
    want = sorted({ALL_CUBES.index(tuple(json.loads(s["spec_json"])["target"])) for s in side})
    assert nums["P2b"]["classes"] == want and len(want) == len(SEEN_CUBES)  # (colour, size) classes, both reds
    assert all(PC.target_class(s) == ALL_CUBES.index(tuple(json.loads(s["spec_json"])["target"])) for s in side)


def test_paired_check_s2_target_dependent_state_fails_P2():
    side, state0 = synth_s2()
    for k, s in enumerate(side):  # the robot start shifts with the target cube (colour, size)
        state0[k] = state0[k] + np.float32(0.02) * np.float32(PC.target_class(s))
    items, nums = run_p(side, state0)
    assert not items["P2a_state_delta_vs_target"] and not items["P2b_state_classifier"], nums["P2b"]
    assert items["P1_layout_identical"] and items["P2c_slot_independent"]


def test_paired_check_s2_mutants_flagged():
    side, state0 = synth_s2()
    gs = PC.groups_of(side)
    # duplicate target: member 1 a copy of member 0
    d = copy.deepcopy(side)
    a = next(i for i, s in enumerate(d) if s["member"] == 0)
    for key in ("labels", "instruction", "spec_json"):
        d[a + 1][key] = d[a][key]
    items, nums = run_p(d, state0)
    assert not items["P_i_distinct_targets"] and nums["bad_groups"]["distinct"] == [d[a]["layout_id"]]
    # re-sampled layout for member 1 of one group
    from vla_common.tasks import TaskSpec

    lay = PC.regenerated_layouts(side)
    g = next(iter(gs))
    bad = dict(lay)
    bad[(g, 1)] = sample_layout(TaskSpec.from_json(gs[g][0]["spec_json"]), np.random.default_rng([SEED, g, 1, 99]),
                                cube_yaw_max_deg=M)[0]
    items, nums = run_p(side, state0, layouts=bad)
    assert not items["P1_layout_identical"] and nums["bad_groups"]["p1_layout"] == [g]
    # split straddle
    s = copy.deepcopy(side)
    s[1]["split"] = "val" if s[1]["split"] == "train" else "train"
    items, nums = run_p(s, state0)
    assert not items["P_iv_group_in_one_split"] and nums["bad_groups"]["split"] == [s[1]["layout_id"]]


def test_s2a_paired_check_target_class_is_the_colour():
    eps, _ = make_paired_stream(32800, 0, 40, "s2a", "train", M)
    side = [{"labels": labels(e.spec).tolist(), "file_attrs": {"stage": "s2a"}} for e in eps]
    assert all(PC.target_class(s) == PC._labels(s)[1] for s in side) and any(PC._labels(s)[2] for s in side)


# ---------------------------------------------------------------- margin (scripts/s2r_tools.py) on an s2 paired set
def test_margin_s2_equals_lerobot_chunk_loader(tmp_path):
    pytest.importorskip("lerobot")
    torch = pytest.importorskip("torch")
    import s2r_tools as S
    from synth_episodes import LATENCY, PERIOD, episode_arrays

    from vla_common.episode import EpisodeWriter
    from vla_policy import train_smolvla as T
    from vla_policy.export import export

    eps, _ = make_paired_stream(SEED, 0, 4, "s2", "train", M)
    gs = groups(eps)
    lengths = {e.index: 52 + (7 * e.index) % 11 for e in eps}
    rng = np.random.default_rng(3)
    with EpisodeWriter(tmp_path / "p.h5", {"git_sha": "test", "created": "test", "stage": "s2"}) as w:
        for k, e in enumerate(eps):
            n = lengths[e.index]
            at = {"instruction": e.instruction, "split": "train", "task": "pick", "spec_json": e.spec.to_json(),
                  "labels": labels(e.spec).tolist(), "seed": SEED, "env_id": k, "success": True, "metrics_json": {},
                  "period": PERIOD, "latency": LATENCY, "image_size": 48, "visual_mode": "nominal",
                  "action_noise_sigma": 0.1, "expert_version": "synthetic", "episode_index": e.index, "T": n,
                  "n_action_steps": 10, "detector_version": "none", "paired": True, "layout_id": e.layout_id,
                  "member": e.member}
            w.add(**episode_arrays(n, k, rng), attrs=at)
    export([tmp_path / "p.h5"], "s2p", root=tmp_path / "ds")
    root = tmp_path / "ds" / "s2p"
    tab = C.read_table(root)
    assert all("cubes" in r and isinstance(r["target"], tuple) for r in tab.values())  # the s2 records
    r = S.margin(root)
    ds = T.open_dataset(root, "pyav")
    is_val, train_eps, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    mean, std = (torch.as_tensor(stats["action"][k], dtype=torch.float64) for k in ("mean", "std"))
    ep = T.column(ds, "episode_index").reshape(-1)
    first_row = {int(e): int(np.flatnonzero(ep == e)[0]) for e in train_eps}
    chunk = {e: ((torch.as_tensor(np.asarray(ds[first_row[e]]["action"]), dtype=torch.float64) - mean) / (std + 1e-8))
             for e in train_eps}
    members, k = {}, 0
    for g, ms in gs.items():  # lerobot episode k = stream order (export keeps file order)
        idx = list(range(k, k + len(ms)))
        members.update({i: [j for j in idx if j != i] for i in idx})
        k += len(ms)
    want = np.mean([np.mean([float(((chunk[e] - chunk[o]) ** 2).mean()) for o in members[e]]) for e in train_eps])
    assert r["expected_sq_diff"] == pytest.approx(want, rel=1e-9) and r["margin"] == pytest.approx(0.5 * want, rel=1e-9)
    assert r["ordered_pairs"] == sum(len(ms) * (len(ms) - 1) for ms in gs.values()) and r["train_layouts"] == len(gs)
    assert r["task_column_checked"]
