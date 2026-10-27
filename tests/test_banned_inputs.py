"""Banned inputs, protocol level (brief section 6; GATES 0B.4 item 1, bridge part). An `act` request may carry only
env_ids, side, wrist, state (B, 19), instruction (+ labels for the baseline). Any other key, or a state width != 19, is
rejected by the client before sending AND by the server before the policy handler runs. The sim client sends only the
first 19 student-obs dims (joint pos 9 + joint vel 9 + gripper width 1; target and prev_action never), and the policy
server reads nothing else from a request. (The sim-side obs-group / teleport checks live with the scene.)"""

from __future__ import annotations

import re
import socket
import threading
from multiprocessing.connection import Client as RawClient
from pathlib import Path

import numpy as np
import pytest

from vla_common import protocol as P
from vla_tabletop.bridge import ChunkExecutor

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("cube_pos", "cube_pose", "object_pose", "target", "prev_action", "phase", "contact_force", "pad_force",
          "teacher_arm", "privileged", "proprio")


def _act(B: int = 2, state_dim: int = 19, **extra) -> dict:
    return {"op": "act", "env_ids": np.arange(B, dtype=np.int32), "side": np.zeros((B, 8, 8, 3), np.uint8),
            "wrist": np.zeros((B, 8, 8, 3), np.uint8), "state": np.zeros((B, state_dim), np.float32),
            "instruction": ["pick up the cube"] * B, **extra}


def test_whitelist_is_exactly_the_contract():
    assert P.ACT_REQUIRED == ("op", "env_ids", "side", "wrist", "state", "instruction")
    assert P.ACT_OPTIONAL == ("labels",) and P.STATE_DIM == 19
    assert P.check_act(_act()) == 2
    assert P.check_act(_act(labels=np.zeros((2, 4), np.int64))) == 2


@pytest.mark.parametrize("key", BANNED)
def test_extra_key_rejected(key):
    with pytest.raises(ValueError, match="banned"):
        P.check_act(_act(**{key: np.zeros((2, 3), np.float32)}))


@pytest.mark.parametrize("dim", (0, 18, 20, 22, 27, 48))
def test_state_width_rejected(dim):
    with pytest.raises(ValueError, match="state shape"):
        P.check_act(_act(state_dim=dim))


def test_wrong_types_rejected():
    for bad in (_act() | {"state": np.zeros((2, 19), np.float64)}, _act() | {"side": np.zeros((2, 8, 8, 3), np.float32)},
                _act() | {"instruction": ["only one"]}, _act() | {"labels": np.zeros((2, 5), np.int64)},
                {k: v for k, v in _act().items() if k != "wrist"}):
        with pytest.raises(ValueError):
            P.check_act(bad)


def _server():
    with socket.socket() as s:
        s.bind((P.HOST, 0))
        port = s.getsockname()[1]
    calls: list[dict] = []
    ready = threading.Event()
    th = threading.Thread(target=P.serve, args=(lambda m: calls.append(m) or {"ok": True}, port, b"k", ready.set),
                          daemon=True)
    th.start()
    assert ready.wait(10)
    return port, calls


def test_server_rejects_before_the_handler():
    port, calls = _server()
    conn = RawClient((P.HOST, port), authkey=b"k")  # bypasses the client-side check
    for bad in (_act(cube_pos=np.zeros((2, 3), np.float32)), _act(state_dim=27), _act(target=[0.5, 0.0, 0.2])):
        P.send(conn, bad)
        out = P.recv(conn)
        assert "error" in out and "ValueError" in out["error"]
    assert calls == []
    P.send(conn, _act())
    assert P.recv(conn) == {"ok": True} and len(calls) == 1
    P.send(conn, {"op": "close"})
    conn.close()


def test_client_rejects_before_sending():
    port, calls = _server()
    c = P.Client(port, b"k")
    with pytest.raises(ValueError):
        c.call(_act(prev_action=np.zeros((2, 5), np.float32)))
    with pytest.raises(ValueError):
        c.call(_act(state_dim=27))
    assert c.call(_act()) == {"ok": True}
    assert len(calls) == 1
    c.close()


def test_sim_client_sends_only_the_first_19_student_dims():
    class Fake:
        info = {"n_action_steps": 2}

        def __init__(self):
            self.sent = []

        def act(self, ids, side, wrist, state, instr):
            self.sent.append(state)
            return np.zeros((len(ids), 2, 5), np.float32), 0.0, 0.0

        def reset(self, ids, seeds):
            pass

    fake = Fake()
    ex = ChunkExecutor(fake, n=2, period=3, latency=1, instructions=["x", "y"])
    ex.reset(np.zeros(2, np.int64))
    student = np.arange(2 * 27, dtype=np.float32).reshape(2, 27)  # 112 student obs: 19 + target 3 + prev_action 5
    img = np.zeros((2, 8, 8, 3), np.uint8)
    for t in range(4):
        ex.step(t, lambda: (img, img), student)
    assert len(fake.sent) == 2
    for s in fake.sent:
        np.testing.assert_array_equal(s, student[:, :19])


def test_garbage_in_privileged_fields_leaves_actions_bit_identical():
    """GATES 0B.4 item 1: the executed actions are a function of the whitelisted request only; garbage (nan, inf,
    huge, random) in every observation dim past the 19 whitelisted ones changes nothing (fake policy = f(request))."""

    class Policy:
        info = {"n_action_steps": 3}

        def act(self, ids, side, wrist, state, instr):
            v = np.tanh(state.sum(1) * 1e-3 + side.reshape(len(ids), -1).mean(1) + wrist.reshape(len(ids), -1).mean(1))
            return np.repeat(v[:, None, None], 3 * 5, 1).reshape(len(ids), 3, 5).astype(np.float32), 0.0, 0.0

        def reset(self, ids, seeds):
            pass

    def run(extra: np.ndarray) -> np.ndarray:
        ex = ChunkExecutor(Policy(), n=3, period=3, latency=1, instructions=["x", "y"])
        ex.reset(np.zeros(2, np.int64))
        out = []
        for t in range(9):
            obs = np.concatenate([np.full((2, 19), t, np.float32), extra], 1)
            img = np.full((2, 8, 8, 3), t, np.uint8)
            out.append(ex.step(t, lambda: (img, img), obs))
        return np.stack(out)

    ref = run(np.zeros((2, 8), np.float32))
    for g in (np.full((2, 8), np.nan, np.float32), np.full((2, 8), np.inf, np.float32),
              np.random.default_rng(0).normal(0, 1e6, (2, 8)).astype(np.float32)):
        assert run(g).tobytes() == ref.tobytes()


def test_policy_server_reads_only_whitelisted_request_keys():
    src = (ROOT / "vla_policy" / "server.py").read_text(encoding="utf-8")
    keys = set(re.findall(r'msg\["([a-z_]+)"\]', src))
    # GATES 004 (guidance probe, validation only): op `guide` carries the scene's other instructions (privileged,
    # disclosed). Allowed ONLY inside SmolVLAServer.guide, which refuses unless the server runs with --guidance-w.
    assert keys <= set(P.ACT_REQUIRED + P.ACT_OPTIONAL + ("seeds", "others")), keys
    guide = src[src.index("    def guide(self, msg: dict)"):]
    guide = guide[:guide.index("\n    def ", 1)]
    assert src.count('msg["others"]') == guide.count('msg["others"]') == 1
    assert 'if self.guidance_w is None:\n            raise ValueError' in guide


def test_labels_banned_for_vlas_required_for_baseline():
    lab = np.zeros((2, 4), np.int64)
    with pytest.raises(ValueError, match="banned"):
        P.check_act(_act(labels=lab), labels=False)  # SmolVLA / pi0.5 get the text, never the labels
    with pytest.raises(ValueError, match="needs labels"):
        P.check_act(_act(), labels=True)
    assert P.check_act(_act(labels=lab), labels=True) == 2 and P.check_act(_act(), labels=False) == 2


@pytest.mark.parametrize("policy_labels", (False, True))
def test_server_enforces_the_labels_rule_before_the_handler(policy_labels):
    with socket.socket() as s:
        s.bind((P.HOST, 0))
        port = s.getsockname()[1]
    calls: list[dict] = []
    ready = threading.Event()
    th = threading.Thread(target=P.serve, args=(lambda m: calls.append(m) or {"ok": True}, port, b"k", ready.set,
                                                policy_labels), daemon=True)
    th.start()
    assert ready.wait(10)
    conn = RawClient((P.HOST, port), authkey=b"k")
    bad = _act() if policy_labels else _act(labels=np.zeros((2, 4), np.int64))
    P.send(conn, bad)
    assert "ValueError" in P.recv(conn)["error"] and calls == []
    P.send(conn, {"op": "close"})
    conn.close()


def test_smolvla_server_bans_labels():
    src = (ROOT / "vla_policy" / "server.py").read_text(encoding="utf-8")
    assert re.search(r"class SmolVLAServer:\s+LABELS = False", src) and re.search(r"class BaselineServer:[^$]*?LABELS = True", src)
    assert "labels=srv.LABELS" in src
