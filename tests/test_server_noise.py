"""Stateless server noise (contract v2 b): an env's flow-matching start noise depends only on (reset seed, query index
since reset) — never on batch size, batch order or the other envs. Same seeds + different batch splits -> identical
noise per env (the model is deterministic given its noise and inputs; the end-to-end check with the real SmolVLA at
B = 1 vs B = 16 is GATES 0B.2 item 2, run by scripts/s0b_bridge.py --invariance)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vla_policy.server import NoiseBank
from vla_tabletop.bridge import episode_seed

SHAPE = (50, 32)
IDS = np.arange(6, dtype=np.int32)
SEEDS = np.array([episode_seed(31500, i) for i in range(6)], np.int64)


def _queries(splits: list[list[int]], n_queries: int = 3) -> dict[tuple[int, int], torch.Tensor]:
    bank = NoiseBank(SHAPE)
    bank.reset(IDS, SEEDS)
    out = {}
    for q in range(n_queries):
        for part in splits:
            ids = np.asarray(part, np.int32)
            for i, z in zip(part, bank.draw(ids)):
                out[(i, q)] = z
    return out


def test_batch_split_and_order_do_not_matter():
    whole = _queries([[0, 1, 2, 3, 4, 5]])
    for splits in ([[3, 1], [0, 5, 2], [4]], [[5], [4], [3], [2], [1], [0]], [[2, 0, 4, 1, 3, 5]]):
        other = _queries(splits)
        assert whole.keys() == other.keys()
        for k in whole:
            assert torch.equal(whole[k], other[k]), (splits, k)


def test_queries_and_envs_differ_and_reset_replays():
    z = _queries([[0, 1, 2, 3, 4, 5]])
    assert not torch.equal(z[(0, 0)], z[(0, 1)]) and not torch.equal(z[(0, 0)], z[(1, 0)])
    bank = NoiseBank(SHAPE)
    bank.reset(IDS[:1], SEEDS[:1])
    assert torch.equal(bank.draw(IDS[:1])[0], z[(0, 0)])
    bank.reset(IDS[:1], SEEDS[:1])  # reset with the same seed replays the stream
    assert torch.equal(bank.draw(IDS[:1])[0], z[(0, 0)])


def test_act_for_an_env_never_reset_is_refused():
    bank = NoiseBank(SHAPE)
    bank.reset(IDS[:2], SEEDS[:2])
    with pytest.raises(ValueError, match="never reset"):
        bank.draw(np.array([0, 7], np.int32))
    assert torch.equal(bank.draw(IDS[:1])[0], _queries([[0]], 1)[(0, 0)])  # the refused call consumed nothing
