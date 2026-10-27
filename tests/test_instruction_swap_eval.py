"""Eval-side instruction instruments (no simulator, no model): the swap plan on a real task stream, the config
legacy default, the debug-token round trip bridge -> handler -> bridge."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("eval_policy", ROOT / "scripts" / "eval_policy.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def _stream(n=60):
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import make_stream

    return make_stream(77, 0, n, lambda rng: sample_spec("s2", "train", rng), split="train")


def test_swap_none_is_empty_and_legacy_key():
    assert E.plan_instruction_swap([], "none") == ({}, None)
    assert "instruction_swap" in E.CONFIG_KEYS and E.CONFIG_LEGACY["instruction_swap"] == "none"


def test_swap_shuffle_on_a_stream():
    eps = _stream()
    sent, rep = E.plan_instruction_swap(eps, "shuffle")
    assert sorted(sent) == [e.index for e in eps]
    assert all(sent[e.index] != e.instruction for e in eps)          # nobody keeps their own text
    assert rep["frac_same_instruction"] == 0.0 and rep["frac_different_target"] > 0.9
    assert rep["frac_donor_target_absent_from_scene"] > 0.5
    donors = rep["donor_of"]
    assert sorted(donors.values()) == [e.index for e in eps]         # a permutation
    assert E.plan_instruction_swap(eps, "shuffle")[0] == sent        # fixed


def test_debug_tokens_roundtrip():
    from vla_policy.server import make_handler
    from vla_tabletop.bridge import ChunkExecutor, PolicyBridge

    class Srv:
        debug, last_tokens = False, None

        def act(self, msg):
            if self.debug:
                self.debug, self.last_tokens = False, [t + "\n" for t in msg["instruction"]]
            return np.zeros((len(msg["instruction"]), 2, 5), np.float32)

        def reset(self, msg):
            pass

    srv = Srv()
    h = make_handler(srv, 3, 1)

    class Cli:
        def call(self, msg):
            return h(msg)

    br = PolicyBridge.__new__(PolicyBridge)
    br.client, br.info = Cli(), {"n_action_steps": 2}
    ex = ChunkExecutor(br, n=2, period=3, latency=1, instructions=["a b", "c"])
    ex.log_first = True
    ex.reset(np.zeros(2, np.int64))
    img = np.zeros((2, 8, 8, 3), np.uint8)
    for t in range(4):
        ex.step(t, lambda: (img, img), np.zeros((2, 19), np.float32))
    assert ex.first_query == {"sent": ["a b", "c"], "tokens": ["a b\n", "c\n"]}
    ex2 = ChunkExecutor(br, n=2, period=3, latency=1, instructions=["x"])  # default: nothing recorded, no extra op
    ex2.reset(np.zeros(1, np.int64))
    ex2.step(0, lambda: (img[:1], img[:1]), np.zeros((1, 19), np.float32))
    assert ex2.first_query is None and not srv.debug
