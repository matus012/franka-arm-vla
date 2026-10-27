"""Sim <-> policy bridge wire protocol (docs/CONTRACT.md "Bridge"). stdlib + numpy only.

Transport: `multiprocessing.connection` on 127.0.0.1 with an authkey. A message is a dict of str / int / float / bool /
None / bytes / list / dict; numpy arrays travel as `{"__nd__": dtype, "shape": [...], "data": bytes}`. Frames are sent
with `send_bytes` and read back with an unpickler that refuses every global, so no pickled numpy (or any other class)
can cross between the two venvs.

Requests (contract v2): `info` -> INFO_KEYS; `reset` {env_ids int32 (B,), seeds int64 (B,)} seeds each env's policy
noise; `act` {env_ids, side, wrist, state, instruction (+ labels)} -> {actions (B, N, 5), server_s}, one per query step.
`act` requests are checked against the input whitelist on BOTH sides (`check_act`): exactly env_ids, side, wrist,
state (B, 19), instruction (+ labels for the baseline); any other key or a state width != 19 is rejected.
"""

from __future__ import annotations

import io
import os
import pickle
from multiprocessing.connection import Client as _Client
from multiprocessing.connection import Connection, Listener
from typing import Any, Callable

import numpy as np

HOST = "127.0.0.1"
DEFAULT_PORT = 50731
AUTHKEY_ENV = "VLA_AUTHKEY"
STATE_DIM = 19
ACTION_DIM = 5
N_LABELS = 4
ACT_REQUIRED = ("op", "env_ids", "side", "wrist", "state", "instruction")
ACT_OPTIONAL = ("labels",)
INFO_KEYS = ("policy", "n_action_steps", "chunk_size", "image_size", "state_dim", "action_dim", "period", "latency",
             "ckpt_sha")


def encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError("object arrays cannot cross the bridge")
        return {"__nd__": obj.dtype.str, "shape": list(obj.shape), "data": obj.tobytes()}  # tobytes: C order
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    raise TypeError(f"type {type(obj).__name__} cannot cross the bridge")


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__nd__" in obj:
            dt = np.dtype(obj["__nd__"])
            if dt.hasobject:
                raise TypeError("object arrays cannot cross the bridge")
            return np.frombuffer(obj["data"], dtype=dt).reshape(obj["shape"]).copy()
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refused global {module}.{name} (only builtin containers cross the bridge)")


def dumps(msg: dict) -> bytes:
    return pickle.dumps(encode(msg), protocol=5)


def loads(buf: bytes) -> dict:
    return decode(_NoGlobals(io.BytesIO(buf)).load())


def send(conn: Connection, msg: dict) -> None:
    conn.send_bytes(dumps(msg))


def recv(conn: Connection) -> dict:
    return loads(conn.recv_bytes())


def authkey() -> bytes:
    key = os.environ.get(AUTHKEY_ENV)
    if not key:
        raise RuntimeError(f"set {AUTHKEY_ENV} (shared secret of sim client and policy server)")
    return key.encode()


def check_act(msg: dict, labels: bool | None = None) -> int:
    """Whitelist + shape check of an `act` request; returns the batch size B. Raises ValueError on any violation.
    labels: True = required (label baseline), False = banned (VLAs: they get the instruction text only), None = either
    (a client that has not seen the server's `info` yet)."""
    keys = set(msg)
    extra = keys - set(ACT_REQUIRED) - set(ACT_OPTIONAL)
    if extra:
        raise ValueError(f"banned keys in act request: {sorted(extra)}")
    missing = set(ACT_REQUIRED) - keys
    if missing:
        raise ValueError(f"act request misses {sorted(missing)}")
    if labels is True and "labels" not in msg:
        raise ValueError("this policy needs labels")
    if labels is False and "labels" in msg:
        raise ValueError("banned keys in act request: ['labels'] (this policy gets the instruction, not labels)")
    ids, side, wrist, state, instr = (msg[k] for k in ACT_REQUIRED[1:])
    for name, a, dt in (("env_ids", ids, np.int32), ("side", side, np.uint8), ("wrist", wrist, np.uint8),
                        ("state", state, np.float32)):
        if not isinstance(a, np.ndarray) or a.dtype != dt:
            raise ValueError(f"{name} must be a {np.dtype(dt).name} array")
    B = ids.shape[0]
    if ids.shape != (B,):
        raise ValueError(f"env_ids shape {ids.shape}")
    if state.shape != (B, STATE_DIM):
        raise ValueError(f"state shape {state.shape} != ({B}, {STATE_DIM})")
    for name, a in (("side", side), ("wrist", wrist)):
        if a.ndim != 4 or a.shape[0] != B or a.shape[3] != 3 or a.shape[1] != a.shape[2]:
            raise ValueError(f"{name} shape {a.shape} != ({B}, H, H, 3)")
    if not isinstance(instr, list) or len(instr) != B or not all(isinstance(s, str) for s in instr):
        raise ValueError("instruction must be a list of B strings")
    if "labels" in msg:
        lab = msg["labels"]
        if not isinstance(lab, np.ndarray) or lab.dtype != np.int64 or lab.shape != (B, N_LABELS):
            raise ValueError(f"labels must be int64 ({B}, {N_LABELS})")
    return B


def check_reset(msg: dict) -> int:
    ids, seeds = msg.get("env_ids"), msg.get("seeds")
    if set(msg) != {"op", "env_ids", "seeds"}:
        raise ValueError(f"reset request keys {sorted(msg)} != ['env_ids', 'op', 'seeds']")
    if not (isinstance(ids, np.ndarray) and ids.dtype == np.int32 and ids.ndim == 1):
        raise ValueError("reset env_ids must be int32 (B,)")
    if not (isinstance(seeds, np.ndarray) and seeds.dtype == np.int64 and seeds.shape == ids.shape):
        raise ValueError("reset seeds must be int64 (B,)")
    return ids.shape[0]


def check_info(info: dict, expected: dict) -> None:
    """Client side: fail fast when the server's policy does not match the sim's timing / image / dims config."""
    missing = set(INFO_KEYS) - set(info)
    if missing:
        raise RuntimeError(f"server info misses {sorted(missing)}")
    bad = {k: (info[k], v) for k, v in expected.items() if info[k] != v}
    if bad:
        raise RuntimeError(f"server/sim mismatch (server, sim): {bad}")


class Client:
    """Sim-side connection: `call(msg)` sends one request and returns the decoded response."""

    def __init__(self, port: int = DEFAULT_PORT, key: bytes | None = None):
        self.conn = _Client((HOST, port), authkey=key or authkey())
        self.labels: bool | None = None  # set from the server's `info` (policy == "baseline")

    def call(self, msg: dict) -> dict:
        if msg.get("op") == "act":
            check_act(msg, self.labels)
        elif msg.get("op") == "reset":
            check_reset(msg)
        send(self.conn, msg)
        out = recv(self.conn)
        if "error" in out:
            raise RuntimeError(f"policy server: {out['error']}")
        return out

    def close(self) -> None:
        send(self.conn, {"op": "close"})
        self.conn.close()


def serve(handler: Callable[[dict], dict], port: int = DEFAULT_PORT, key: bytes | None = None,
          on_listen: Callable[[], None] | None = None, labels: bool | None = None) -> None:
    """Accept one client and answer its requests until it sends `close` (or disconnects). A failing request is
    answered with {"error": message} and the loop continues; banned inputs are therefore refused, not fatal.
    labels: as in `check_act` (the policy server passes True for the baseline, False for the VLAs)."""
    with Listener((HOST, port), authkey=key or authkey()) as lst:
        if on_listen is not None:
            on_listen()
        with lst.accept() as conn:
            while True:
                try:
                    msg = recv(conn)
                except EOFError:
                    return
                if msg.get("op") == "close":
                    return
                try:
                    if msg.get("op") == "act":
                        check_act(msg, labels)
                    elif msg.get("op") == "reset":
                        check_reset(msg)
                    out = handler(msg)
                except Exception as e:  # noqa: BLE001 - production path: report to the client, keep serving
                    out = {"error": f"{type(e).__name__}: {e}"}
                send(conn, out)
