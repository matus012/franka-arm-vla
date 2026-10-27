"""Held-out disjointness (compositions) and paraphrase template disjointness."""

import numpy as np

from vla_common.catalog import ALL_CUBES
from vla_common.tasks import (HELD_OUT_PLACE_PAIRS, HELD_OUT_STACK_PAIRS, HELD_OUT_TARGETS, TEMPLATES, instruction,
                              sample_spec, skeleton)

N = 10000
STAGES = ("s1", "s2", "s2a", "s3a", "s3b", "s4")


def _specs(stage, split, seed=0):
    rng = np.random.default_rng(seed)
    return [sample_spec(stage, split, rng) for _ in range(N)]


def test_held_out_definition():
    assert len(HELD_OUT_TARGETS) == 2 and len(set(HELD_OUT_TARGETS)) == 2
    assert HELD_OUT_TARGETS[0][0] != HELD_OUT_TARGETS[1][0] and HELD_OUT_TARGETS[0][1] != HELD_OUT_TARGETS[1][1]
    assert all(c in ALL_CUBES for c in HELD_OUT_TARGETS)
    assert all(t not in HELD_OUT_TARGETS for t, _ in HELD_OUT_PLACE_PAIRS + HELD_OUT_STACK_PAIRS)
    assert 2 <= len(HELD_OUT_PLACE_PAIRS) <= 3 and 2 <= len(HELD_OUT_STACK_PAIRS) <= 3


def test_train_never_manipulates_held_out():
    for stage in STAGES:
        for s in _specs(stage, "train") + _specs(stage, "paraphrase", 1):
            assert s.target not in HELD_OUT_TARGETS, s
            if s.task == "place":
                assert (s.target, s.dest) not in HELD_OUT_PLACE_PAIRS, s
            if s.task == "stack":
                assert (s.target, s.dest) not in HELD_OUT_STACK_PAIRS, s


def test_heldout_comp_is_held_out():
    for stage in STAGES:
        for s in _specs(stage, "heldout_comp"):
            ok = s.target in HELD_OUT_TARGETS
            ok |= s.task == "place" and (s.target, s.dest) in HELD_OUT_PLACE_PAIRS
            ok |= s.task == "stack" and (s.target, s.dest) in HELD_OUT_STACK_PAIRS
            assert ok, s
    assert {(s.target, s.dest) for s in _specs("s3a", "heldout_comp")} >= set(HELD_OUT_PLACE_PAIRS)
    assert {(s.target, s.dest) for s in _specs("s3b", "heldout_comp")} >= set(HELD_OUT_STACK_PAIRS)
    assert {s.target for s in _specs("s2", "heldout_comp")} == set(HELD_OUT_TARGETS)


def test_heldout_cubes_appear_in_train_as_distractors_and_bottoms():
    for stage in ("s2", "s3a", "s3b"):
        count = {c: 0 for c in HELD_OUT_TARGETS}
        for s in _specs(stage, "train"):
            for c in HELD_OUT_TARGETS:
                count[c] += c in s.cubes
        assert all(v > 500 for v in count.values()), (stage, count)
    bottoms = {c: 0 for c in HELD_OUT_TARGETS}
    for s in _specs("s3b", "train"):
        if s.dest in bottoms:
            bottoms[s.dest] += 1
    assert all(v > 100 for v in bottoms.values()), bottoms


def test_stage1_train_cubes_are_exactly_the_seen_six():
    assert {s.target for s in _specs("s1", "train")} == set(ALL_CUBES) - set(HELD_OUT_TARGETS)


def test_paraphrase_templates_disjoint_as_strings_and_skeletons():
    tr = [t for v in TEMPLATES["train"].values() for t in v]
    pa = [t for v in TEMPLATES["paraphrase"].values() for t in v]
    assert not set(tr) & set(pa)
    assert not {skeleton(t) for t in tr} & {skeleton(t) for t in pa}


def test_emitted_instructions_are_disjoint_between_splits():
    rng = np.random.default_rng(9)
    tr, pa = set(), set()
    for stage in ("s1", "s2", "s3a", "s3b"):
        for _ in range(2000):
            s = sample_spec(stage, "train", rng)
            tr.add(instruction(s, rng, "train"))
            pa.add(instruction(s, rng, "paraphrase"))
    assert not tr & pa
