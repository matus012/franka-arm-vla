"""Bridge wire protocol (vla_common.protocol): encode / decode round trip of every dtype and shape (incl. strings and
empty batches), refusal of pickled numpy / any class, transport over a real authenticated connection. Runs in both
venvs (stdlib + numpy only)."""

from __future__ import annotations

import pickle
import socket
import threading
from multiprocessing import AuthenticationError

import numpy as np
import pytest

from vla_common import protocol as P

DTYPES = ["?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16", ">f4", ">i8", "<U7",
          "S5"]
SHAPES = [(), (0,), (0, 19), (3,), (2, 3, 4), (1, 128, 128, 3), (0, 128, 128, 3)]


def _array(dt: str, shape: tuple, rng: np.random.Generator) -> np.ndarray:
    n = int(np.prod(shape))
    d = np.dtype(dt)
    if d.kind == "U":
        flat = np.array([f"s{i}é"[:7] for i in range(n)], dtype=d)
    elif d.kind == "S":
        flat = np.array([f"b{i}".encode() for i in range(n)], dtype=d)
    elif d.kind == "b":
        flat = rng.random(n) > 0.5
    elif d.kind == "c":
        flat = rng.normal(size=n) + 1j * rng.normal(size=n)
    else:
        flat = rng.normal(size=n) * 50
    return np.asarray(flat).astype(d).reshape(shape)


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_array_round_trip(dt, shape):
    a = _array(dt, shape, np.random.default_rng(0))
    b = P.loads(P.dumps({"a": a}))["a"]
    assert b.dtype == a.dtype and b.shape == a.shape
    np.testing.assert_array_equal(b, a)
    assert b.flags.writeable


def test_non_contiguous_and_nested():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    msg = {"op": "act", "x": a, "nest": {"l": [a[0], 1, 2.5, "str", None, True, b"raw"], "t": (np.int64(3),)},
           "instruction": ["pick up the cube", "", "ünïcode"], "scalar": np.float32(1.5)}
    out = P.loads(P.dumps(msg))
    np.testing.assert_array_equal(out["x"], a)
    np.testing.assert_array_equal(out["nest"]["l"][0], a[0])
    assert out["nest"]["l"][1:] == [1, 2.5, "str", None, True, b"raw"]
    assert out["nest"]["t"] == [3] and out["instruction"] == msg["instruction"] and out["scalar"] == 1.5


def test_empty_act_batch_passes_check():
    msg = {"op": "act", "env_ids": np.zeros(0, np.int32), "side": np.zeros((0, 8, 8, 3), np.uint8),
           "wrist": np.zeros((0, 8, 8, 3), np.uint8), "state": np.zeros((0, 19), np.float32), "instruction": []}
    out = P.loads(P.dumps(msg))
    assert P.check_act(out) == 0 and out["state"].shape == (0, 19)


def test_refuses_pickled_numpy_and_classes():
    for bad in ({"x": np.zeros(3)}, {"x": np.float32(1.0)}, {"x": np.dtype("f4")}, {"x": complex(1, 2)}):
        with pytest.raises(pickle.UnpicklingError):
            P.loads(pickle.dumps(bad))
    with pytest.raises(TypeError):
        P.dumps({"x": np.array([object()], dtype=object)})
    with pytest.raises(TypeError):
        P.dumps({"x": {1, 2}})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((P.HOST, 0))
        return s.getsockname()[1]


def test_transport_echo_and_errors():
    port, key = _free_port(), b"test-key"
    ready = threading.Event()

    def handler(msg):
        if msg["op"] == "boom":
            raise RuntimeError("handler failed")
        return {"echo": msg}

    th = threading.Thread(target=P.serve, args=(handler, port, key, ready.set), daemon=True)
    th.start()
    assert ready.wait(10)
    c = P.Client(port, key)
    a = np.arange(6, dtype=np.int64).reshape(2, 3)
    out = c.call({"op": "x", "a": a, "s": ["a", "b"]})
    np.testing.assert_array_equal(out["echo"]["a"], a)
    assert out["echo"]["s"] == ["a", "b"]
    with pytest.raises(RuntimeError, match="handler failed"):
        c.call({"op": "boom"})
    assert c.call({"op": "x"})["echo"] == {"op": "x"}  # the server keeps serving after an error
    c.close()
    th.join(10)
    assert not th.is_alive()


def test_wrong_authkey_is_refused():
    port = _free_port()
    ready = threading.Event()
    seen = []

    def server():
        try:
            P.serve(lambda m: m, port, b"right", ready.set)
        except AuthenticationError as e:
            seen.append(e)

    th = threading.Thread(target=server, daemon=True)
    th.start()
    assert ready.wait(10)
    with pytest.raises(AuthenticationError):
        P.Client(port, b"wrong")
    th.join(10)
    assert len(seen) == 1  # the server refused the connection too
