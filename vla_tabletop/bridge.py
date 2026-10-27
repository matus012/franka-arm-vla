"""Sim-side bridge client (docs/CONTRACT.md "Bridge", v2 b/c/i). Sim venv; imports no lerobot / torch.

`PolicyBridge` spawns the policy server with the policy-venv python (or connects to a running one), checks its `info`
against the sim's config, and sends `reset` / `act`. `ChunkExecutor` runs chunked open-loop execution for K envs in
lockstep from a common reset: at query steps t % N == 0 it sends ONE `act` for all envs with the frame the policy sees
at t (vision_lift.frames.DelayedFrames, PERIOD / LATENCY passed explicitly) and the 19-dim state, then executes the
returned (K, N, 5) chunk one action per control step (arm clipped to [-1, 1], claw > 0.5).
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from vision_lift.frames import DelayedFrames, frame_index
from vla_common import protocol as P

ROOT = Path(__file__).resolve().parents[1]
POLICY_PY = ROOT / "policy" / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def episode_seed(stream_seed: int, episode: int) -> int:
    """Noise seed of episode `episode` of the stream `stream_seed` (GATES conventions: per-env generator seeded from
    (episode seed, episode index, query index); the server adds the query index)."""
    return int(np.random.SeedSequence([stream_seed, episode]).generate_state(1, np.uint64)[0] >> 1)


class PolicyBridge:
    def __init__(self, expected_info: dict, server_args: list[str] | None = None, port: int = P.DEFAULT_PORT,
                 log_path: Path | None = None, timeout_s: float = 900.0, policy_python: str | Path | None = None):
        """server_args given: spawn `<policy_python> -m vla_policy.server <server_args> --port <port>` with a fresh
        authkey (policy_python default: POLICY_PY, the repo's policy venv; the container passes /opt/policy/bin/python);
        else connect to a running server (authkey from VLA_AUTHKEY)."""
        self.proc = None
        key = None
        if server_args is not None:
            key = secrets.token_hex(16).encode()
            env = os.environ | {P.AUTHKEY_ENV: key.decode(), "PYTHONPATH": str(ROOT)}
            out = open(log_path, "w") if log_path else subprocess.DEVNULL
            self.proc = subprocess.Popen([str(policy_python or POLICY_PY), "-u", "-m", "vla_policy.server", *server_args,
                                          "--port", str(port)], cwd=ROOT, env=env, stdout=out,
                                         stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            try:
                self.client = P.Client(port, key)
                break
            except ConnectionRefusedError:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(f"policy server exited with {self.proc.returncode} (log: {log_path})")
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"no policy server on port {port} after {timeout_s} s")
                time.sleep(1.0)
        self.load_s = time.time() - t0
        self.info = self.client.call({"op": "info"})
        P.check_info(self.info, expected_info)
        self.client.labels = self.info["policy"] == "baseline"  # required for the baseline, banned for the VLAs

    def reset(self, env_ids: np.ndarray, seeds: np.ndarray) -> None:
        self.client.call({"op": "reset", "env_ids": env_ids.astype(np.int32), "seeds": seeds.astype(np.int64)})

    def act(self, env_ids: np.ndarray, side: np.ndarray, wrist: np.ndarray, state: np.ndarray,
            instruction: list[str], labels: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
        """Returns (actions (B, N, 5) float32, server_s, client round trip s). labels (B, 4) int64: baseline only."""
        t = time.perf_counter()
        msg = {"op": "act", "env_ids": env_ids.astype(np.int32), "side": side, "wrist": wrist,
               "state": state.astype(np.float32), "instruction": list(instruction)}
        if labels is not None:
            msg["labels"] = labels.astype(np.int64)
        out = self.client.call(msg)
        rtt = time.perf_counter() - t
        a = out["actions"]
        shape = (len(env_ids), self.info["n_action_steps"], P.ACTION_DIM)
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise RuntimeError(f"bad actions from server: {a.dtype} {a.shape} != {shape} or non-finite")
        return a, float(out["server_s"]), rtt

    def guide(self, env_ids: np.ndarray, others: list[list[str]]) -> None:
        """Task 004: the envs' "other" instructions for a server started with --guidance-w (after `reset`)."""
        self.client.call({"op": "guide", "env_ids": env_ids.astype(np.int32), "others": [list(o) for o in others]})

    def set_debug(self) -> None:
        """Instrument: the server keeps the decoded tokens of its latest `act` (smolvla only)."""
        self.client.call({"op": "debug"})

    def last_tokens(self) -> list[str]:
        return self.client.call({"op": "last_tokens"})["tokens"]

    def last_labels(self) -> np.ndarray:
        """Instrument (baseline): the labels the server received in its latest `act` after `set_debug`."""
        return self.client.call({"op": "last_labels"})["labels"]

    def close(self) -> None:
        self.client.close()
        if self.proc is not None:
            self.proc.wait(timeout=120)


class ChunkExecutor:
    """K envs in lockstep from a common reset; call `step(t, ...)` every control step t = 0, 1, ..."""

    def __init__(self, bridge: PolicyBridge, n: int, period: int, latency: int, instructions: list[str],
                 labels: np.ndarray | None = None, others: list[list[str]] | None = None):
        """labels (K, 4) int64: the baseline's task labels (vla_common.tasks.labels), sent instead of being parsed.
        others (task 004, guidance server only): each env's "other" instructions, sent with op `guide` after reset."""
        self.bridge, self.n, self.period, self.latency = bridge, n, period, latency
        self.instructions, self.labels, self.others = instructions, labels, others
        self.ids = np.arange(len(instructions), dtype=np.int32)
        self.frames = DelayedFrames(period, latency)
        self.chunk: np.ndarray | None = None
        self.queries: list[dict] = []  # per query: t, frame index sent, expected index, timings
        self.log_first, self.first_query = False, None  # instrument: instructions sent + server's decoded tokens

    def reset(self, seeds: np.ndarray) -> None:
        self.frames.reset()
        self.chunk = None
        self.bridge.reset(self.ids, seeds)
        if self.others is not None:
            self.bridge.guide(self.ids, self.others)
        if self.log_first:
            self.bridge.set_debug()  # one-shot: the server keeps the decoded tokens of the next act

    def step(self, t: int, capture: Callable[[], tuple[np.ndarray, np.ndarray]], state: np.ndarray) -> np.ndarray:
        """capture() -> (side, wrist) uint8 (K, H, H, 3), called only on camera ticks; state (K, >= 19): only the
        first 19 dims (joint pos 9 + joint vel 9 + gripper width 1) are sent. Returns the (K, 5) action for step t."""
        if self.frames.is_tick(t):
            self.frames.capture(t, capture())
        if t % self.n == 0:
            k, (side, wrist) = self.frames.current()
            extra = {} if self.labels is None else {"labels": self.labels}
            self.chunk, server_s, rtt = self.bridge.act(self.ids, side, wrist, np.ascontiguousarray(state[:, :19]),
                                                        self.instructions, **extra)
            self.queries.append({"t": t, "frame_index": k,
                                 "expected_frame_index": frame_index(t, self.period, self.latency),
                                 "server_s": server_s, "rtt_s": rtt})
            if self.log_first and self.first_query is None:
                self.first_query = {"sent": list(self.instructions)} | (
                    {"labels": self.bridge.last_labels()} if self.labels is not None else {"tokens": self.bridge.last_tokens()})
        a = self.chunk[:, t % self.n].copy()
        a[:, :4] = np.clip(a[:, :4], -1.0, 1.0)
        a[:, 4] = (a[:, 4] > 0.5).astype(np.float32)
        return a
