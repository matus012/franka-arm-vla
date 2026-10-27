"""Every sampled instruction resolves to exactly one cube and one destination; size stated when colours collide."""

import numpy as np
import pytest

from vla_common.catalog import COLORS
from vla_common.tasks import TEMPLATES, TaskSpec, instruction, labels, matches, matches_dest, sample_spec

N = 10000  # GATES 0B.4 item 3: >= 10,000 sampled specs per stage and split


@pytest.mark.parametrize("stage", ["s1", "s2", "s2a", "s3a", "s3b", "s4"])
@pytest.mark.parametrize("split", ["train", "heldout_comp", "paraphrase"])
def test_instructions_resolve_uniquely(stage, split):
    rng = np.random.default_rng(1)
    for _ in range(N):
        spec = sample_spec(stage, split, rng)
        ins = instruction(spec, rng, "paraphrase" if split == "paraphrase" else "train")
        assert matches(spec, ins) == [spec.target], (spec, ins)
        assert len(set(spec.cubes)) == len(spec.cubes) and spec.target in spec.cubes
        assert len(spec.cubes) == 1 if stage == "s1" else 2 <= len(spec.cubes) <= 5
        if spec.task == "pick":
            assert matches_dest(spec, ins) == []
            continue
        assert matches_dest(spec, ins) == [spec.dest], (spec, ins)
        assert spec.dest != spec.target
        if spec.task == "place":
            assert spec.dest in spec.pads and len(spec.pads) == 4
        else:
            assert spec.dest in spec.cubes


def test_size_stated_when_colour_collides():
    rng = np.random.default_rng(2)
    collisions = 0
    for _ in range(N):
        spec = sample_spec("s4", "train", rng)
        ins = instruction(spec, rng)
        for c in [spec.target] + ([spec.dest] if spec.task == "stack" else []):
            if sum(x[0] == c[0] for x in spec.cubes) > 1:
                collisions += 1
                assert f"{c[1]} {c[0]} cube" in ins, (spec, ins)
    assert collisions > 500


def test_labels_and_json_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(500):
        spec = sample_spec("s4", "train", rng)
        assert TaskSpec.from_json(spec.to_json()) == spec
        lab = labels(spec)
        assert lab.dtype == np.int64 and lab.shape == (4,) and 0 <= lab[3] <= 12
        assert (lab[3] == 12) == (spec.task == "pick") and (4 <= lab[3] <= 11) == (spec.task == "stack")


def test_deterministic_given_rng():
    def run() -> list[str]:
        rng = np.random.default_rng(4)
        return [instruction(sample_spec("s4", "train", rng), rng) for _ in range(50)]

    assert run() == run()


def test_every_template_is_emitted():
    rng = np.random.default_rng(6)
    for key in ("train", "paraphrase"):
        seen = set()
        for stage in ("s1", "s2", "s3a", "s3b"):
            for _ in range(1500):
                seen.add(instruction(sample_spec(stage, key, rng), rng, key).split()[0])
        assert len(seen) >= 6, seen
        assert sum(len(v) for v in TEMPLATES[key].values()) >= 12


@pytest.mark.parametrize("stage", ["s1", "s2", "s2a", "s3a", "s3b", "s4"])
@pytest.mark.parametrize("split", ["train", "heldout_comp", "paraphrase"])
def test_all_pads_in_every_scene(stage, split):
    rng = np.random.default_rng(7)
    for _ in range(1000):
        spec = sample_spec(stage, split, rng)
        assert sorted(spec.pads) == sorted(COLORS), spec  # same pad set whatever the task


@pytest.mark.parametrize("split", ["train", "heldout_comp", "paraphrase"])
def test_s2a_one_size_distinct_colours(split):
    rng = np.random.default_rng(8)
    for _ in range(N):
        spec = sample_spec("s2a", split, rng)
        assert spec.task == "pick" and 2 <= len(spec.cubes) <= 4
        assert len({c[1] for c in spec.cubes}) == 1
        assert len({c[0] for c in spec.cubes}) == len(spec.cubes)
