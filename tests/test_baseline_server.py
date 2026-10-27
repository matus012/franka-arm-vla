"""`--policy baseline` in the policy server: a tiny random BaselinePolicy checkpoint served on CPU, in-process (handler
+ protocol) and as a spawned server process through the sim-side PolicyBridge (policy_python argument). Labels (B, 4)
are required by the baseline (and banned for SmolVLA: tests/test_banned_inputs.py)."""

from __future__ import annotations

import socket
import sys
import threading

import numpy as np
import pytest

from vla_common import protocol as P
from vla_policy.baseline import BaselineConfig, BaselinePolicy
from vla_policy.server import BaselineServer, build, make_handler
from vla_tabletop.bridge import ChunkExecutor, PolicyBridge

N, H, B = 4, 32, 3
EXPECT = {"policy": "baseline", "n_action_steps": N, "period": 3, "latency": 1, "image_size": H, "state_dim": 19,
          "action_dim": 5}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("bl") / "epoch_000"
    import torch

    torch.manual_seed(0)
    BaselinePolicy(BaselineConfig(n_action_steps=N)).save(d)
    return d


def _req(labels: bool = True) -> dict:
    rng = np.random.default_rng(0)
    m = {"op": "act", "env_ids": np.arange(B, dtype=np.int32),
         "side": rng.integers(0, 256, (B, H, H, 3), dtype=np.uint8),
         "wrist": rng.integers(0, 256, (B, H, H, 3), dtype=np.uint8),
         "state": rng.normal(size=(B, 19)).astype(np.float32), "instruction": ["pick up the cube"] * B}
    if labels:
        m["labels"] = np.array([[0, 1, 0, 12], [1, 2, 1, 3], [2, 3, 0, 7]], np.int64)
    return m


def _port() -> int:
    with socket.socket() as s:
        s.bind((P.HOST, 0))
        return s.getsockname()[1]


def test_n_action_steps_must_match(ckpt):
    with pytest.raises(ValueError, match="n_action_steps"):
        BaselineServer(str(ckpt), N + 1, "cpu", H)
    with pytest.raises(ValueError, match="image-size"):
        build("baseline", str(ckpt), N, "cpu", None, None)
    with pytest.raises(NotImplementedError):
        build("pi05", str(ckpt), N, "cpu", None, None)


def test_in_process_round_trip(ckpt):
    srv = BaselineServer(str(ckpt), N, "cpu", H)
    port, ready = _port(), threading.Event()
    th = threading.Thread(target=P.serve, args=(make_handler(srv, 3, 1), port, b"k", ready.set, srv.LABELS),
                          daemon=True)
    th.start()
    assert ready.wait(10)
    c = P.Client(port, b"k")
    info = c.call({"op": "info"})
    P.check_info(info, EXPECT)
    assert set(P.INFO_KEYS) <= set(info) and len(info["ckpt_sha"]) == 64
    c.call({"op": "reset", "env_ids": np.arange(B, dtype=np.int32), "seeds": np.zeros(B, np.int64)})
    m = _req()
    out = c.call(m)
    a = out["actions"]
    assert a.shape == (B, N, 5) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.abs(a[..., :4]).max() <= 1.0 and set(np.unique(a[..., 4]).tolist()) <= {0.0, 1.0}
    ref = BaselinePolicy.load(ckpt, "cpu").predict(m["side"], m["wrist"], m["state"], m["labels"])
    np.testing.assert_array_equal(a, ref)
    np.testing.assert_array_equal(c.call(_req())["actions"], a)  # deterministic
    with pytest.raises(RuntimeError, match="needs labels"):  # server-side refusal (client does not know yet)
        c.call(_req(labels=False))
    c.close()
    th.join(10)


def test_spawned_server_through_the_bridge(ckpt, tmp_path):
    args = ["--policy", "baseline", "--ckpt", str(ckpt), "--n-action-steps", str(N), "--period", "3", "--latency", "1",
            "--device", "cpu", "--image-size", str(H)]
    br = PolicyBridge(EXPECT, args, _port(), log_path=tmp_path / "server.log", timeout_s=120,
                      policy_python=sys.executable)
    try:
        assert br.client.labels is True
        with pytest.raises(ValueError, match="needs labels"):  # client-side refusal once info is known
            br.act(np.arange(B, dtype=np.int32), *(_req()[k] for k in ("side", "wrist", "state")), ["x"] * B)
        m = _req()
        ex = ChunkExecutor(br, N, 3, 1, m["instruction"], labels=m["labels"])
        ex.reset(np.zeros(B, np.int64))
        acts = [ex.step(t, lambda: (m["side"], m["wrist"]), m["state"]) for t in range(2 * N)]
        assert len(ex.queries) == 2 and all(x.shape == (B, 5) for x in acts)
    finally:
        br.close()


def _eval_policy_module():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    sp = importlib.util.spec_from_file_location("eval_policy_bl", root / "scripts" / "eval_policy.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def test_eval_policy_server_args_and_expected_info_match_the_real_baseline_server(ckpt, tmp_path):
    """The two closed-loop blockers: eval_policy must pass --image-size and expect chunk_size = N for the baseline."""
    E = _eval_policy_module()
    c = dict(policy="baseline", ckpt=str(ckpt), n=N, image_size=H, dataset="")
    srv = [("cpu" if x == "cuda" else x) for x in E.server_args(c)]
    br = PolicyBridge(E.expected_info(c), srv, _port(), log_path=tmp_path / "ok.log", timeout_s=120, policy_python=sys.executable)
    try:
        assert br.info["chunk_size"] == N and br.info["image_size"] == H
        # instrument B-I1 channel: the labels the server RECEIVED come back through last_labels
        m = _req()
        ex = ChunkExecutor(br, N, 3, 1, m["instruction"], labels=m["labels"])
        ex.log_first = True
        ex.reset(np.zeros(B, np.int64))
        ex.step(0, lambda: (m["side"], m["wrist"]), m["state"])
        np.testing.assert_array_equal(ex.first_query["labels"], m["labels"])
        assert "tokens" not in ex.first_query
    finally:
        br.close()
    # mutant 1: expected chunk_size 50 (the old value) is refused by check_info
    with pytest.raises(RuntimeError, match="mismatch"):
        PolicyBridge(E.expected_info(c) | {"chunk_size": 50}, srv, _port(), log_path=tmp_path / "m1.log", timeout_s=120,
                     policy_python=sys.executable)
    # mutant 2: no --image-size -> the baseline server refuses to start
    i = srv.index("--image-size")
    with pytest.raises(RuntimeError, match="exited"):
        PolicyBridge(E.expected_info(c), srv[:i] + srv[i + 2:], _port(), log_path=tmp_path / "m2.log", timeout_s=120,
                     policy_python=sys.executable)
