"""Instruction-swap donors (instrument checks only): a fixed permutation of episodes so that nobody keeps its own
instruction text, preferring a donor whose target differs (and, with `cubes`, whose target is not in the scene)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

W_SAME_INSTR, W_SAME_TARGET, W_IN_SCENE = 1e6, 1e3, 1e2


def donors(instr: list[str], target: list, cubes: list[list] | None = None, seed: int = 0) -> np.ndarray:
    """donor[i] = the episode whose instruction episode i receives; `donor` is a permutation, donor[i] != i.
    Cost per pair (i, j): same instruction text 1e6 > same target 1e3 > target j present among cubes of i 1e2, plus
    a seeded random tie-break (< 1)."""
    n = len(instr)
    assert n >= 2 and len(target) == n, n
    tg = [tuple(t) if isinstance(t, (list, tuple)) else t for t in target]
    c = np.random.default_rng(seed).random((n, n))
    for i in range(n):
        for j in range(n):
            c[i, j] += (W_SAME_INSTR * (instr[i] == instr[j] or i == j) + W_SAME_TARGET * (tg[i] == tg[j])
                        + (W_IN_SCENE * (tg[j] in [tuple(x) for x in cubes[i]]) if cubes is not None else 0))
    r, col = linear_sum_assignment(c)
    assert (r == np.arange(n)).all()
    return col


def swap_report(instr: list[str], target: list, d: np.ndarray, cubes: list[list] | None = None) -> dict:
    tg = [tuple(t) if isinstance(t, (list, tuple)) else t for t in target]
    out = {"episodes": len(d), "frac_different_target": float(np.mean([tg[i] != tg[j] for i, j in enumerate(d)])),
           "frac_same_instruction": float(np.mean([instr[i] == instr[j] for i, j in enumerate(d)]))}
    if cubes is not None:
        out["frac_donor_target_absent_from_scene"] = float(np.mean(
            [tg[j] not in [tuple(x) for x in cubes[i]] for i, j in enumerate(d)]))
    return out
