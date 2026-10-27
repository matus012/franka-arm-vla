"""Task 003 grounding diagnosis helpers (scripts/g3_probe.py, scripts/g3_dump_t0.py), CPU only: probe on separable /
unlearnable synthetic sets, group folds, readings / verdict rule as written in GATES 003, divergence arithmetic, and
the sample plan (S_p whole groups, |S_p| + |S_u| = 2,000)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("g3_probe")


def _synthetic(n: int = 600, d: int = 40, signal: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    X = rng.normal(size=(n, d))
    X[:, :4] += signal * np.eye(4)[y]
    groups = np.arange(n) // 3  # groups of 3, like a paired layout
    return X, y, groups


def test_probe_separable_is_decodable():
    X, y, g = _synthetic(signal=6.0)
    acc = (G.cv_predict(X, y, G.group_folds(g)) == y).mean()
    assert acc > 0.98


def test_probe_unlearnable_is_chance():
    X, y, g = _synthetic(signal=0.0)
    acc = (G.cv_predict(X, y, G.group_folds(g)) == y).mean()
    assert abs(acc - 0.25) < 0.08


def test_probe_cannot_memorise_groups():
    """Label constant within a group, features = group identity only: CV by group must give chance, not memorisation."""
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(200), 3)
    y = rng.integers(0, 4, 200)[g]
    X = np.eye(200)[g] + 0.01 * rng.normal(size=(600, 200))
    acc = (G.cv_predict(X, y, G.group_folds(g)) == y).mean()
    assert acc < 0.40


def test_group_folds_disjoint_and_balanced():
    g = np.repeat(np.arange(101), 2)
    f = G.group_folds(g)
    for k in range(5):
        assert not set(g[f == k]) & set(g[f != k])
    counts = np.bincount(f)
    assert counts.min() >= 2 * 20 and counts.max() <= 2 * 21


@pytest.mark.parametrize("acc,shuf,want", [(0.95, 0.40, "decodable"), (0.90, 0.50, "decodable"),
                                           (0.89, 0.30, "partial"), (0.95, 0.56, "partial"),
                                           (0.45, 0.35, "not decodable"), (0.99, 0.95, "not decodable"),
                                           (0.46, 0.35, "partial")])
def test_reading_thresholds(acc, shuf, want):
    assert G.reading(acc, shuf) == want


def test_family_and_verdict():
    D, N, P = "decodable", "not decodable", "partial"
    assert G.family([N, N, D, N]) == D and G.family([N] * 4) == N and G.family([N, P, N, N]) == P
    assert G.verdict(N, N, True) == "R1 text"
    assert G.verdict(D, N, True) == "R2 mixing"
    assert G.verdict(D, D, True) == "R3 data/loss"
    assert G.verdict(D, D, False) == "none of R1-R3 at the first query"
    assert G.verdict(P, D, True) == "mixed" and G.verdict(D, P, True) == "mixed" and G.verdict(P, P, False) == "mixed"
    assert G.verdict(N, P, True) == "mixed" and G.verdict(P, N, False) == "mixed"


def test_divergence_arithmetic():
    a = np.zeros((50, 4))
    ci = [np.stack([a, a + 1.0, a + 2.0])]  # pairwise Frobenius: sqrt(200), sqrt(200), sqrt(800)
    npair = [(a, a + 0.5)]  # sqrt(50)
    d = G.divergence(ci, npair)
    want = (2 * np.sqrt(200) + np.sqrt(800)) / 3
    assert d["instruction_effect"] == pytest.approx(want) and d["noise_effect"] == pytest.approx(np.sqrt(50))
    assert d["ratio"] == pytest.approx(want / np.sqrt(50)) and not d["ignores_language"]
    same = G.divergence([np.stack([a, a + 0.01])], npair)
    assert same["ignores_language"]
    assert G.divergence(ci, npair, 10)["noise_effect"] == pytest.approx(np.sqrt(10 * 4 * 0.25))


def test_shuffle_is_a_fixed_permutation():
    instr = np.array([f"pick up {i}" for i in range(2000)])
    s = G.shuffled_instructions(instr)
    assert sorted(s.tolist()) == sorted(instr.tolist()) and (s != instr).mean() > 0.99
    assert (G.shuffled_instructions(instr) == s).all()


def test_sample_plan_whole_groups_total_2000():
    D = _load("g3_dump_t0")
    p = D.paired_samples(21645, 1000)
    u = D.unpaired_samples(21640, 2000 - len(p))
    assert len(p) >= 1000 and len(p) + len(u) == 2000
    last = p[-1].layout_id
    from vla_tabletop.rollout import make_paired_stream

    full, _ = make_paired_stream(21645, 0, last + 1, "s2a", "train", 15.0)
    assert [(e.index, e.instruction) for e in full] == [(e.index, e.instruction) for e in p]  # whole groups, in order
    m = D.meta(p, 21645, True)
    assert (np.unique(m["group"]) >= D.GROUP_OFFSET).all() and set(m["label"].tolist()) <= {0, 1, 2, 3}
    mu = D.meta(u, 21640, False)
    assert len(set(mu["group"].tolist())) == len(u) and not set(mu["group"].tolist()) & set(m["group"].tolist())
