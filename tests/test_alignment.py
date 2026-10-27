"""Frame / action alignment (GATES 0B.4 item 2) with synthetic frames that encode (frame index, env, camera).

A fake K-env sim renders, on every camera tick, frames encoding the tick's frame index and the env id; per step it
produces a state and an executed action that encode (env, t). The real sim-side runtime pieces drive it: the bridge
client's `ChunkExecutor` (DelayedFrames with PERIOD / LATENCY passed explicitly, a query every N steps) against a fake
server that records exactly what it was sent. Each env's episode is written with `vla_common.episode`, exported with
`vla_policy.export`, and read back through `LeRobotDataset` (pyav). For every (env, step):
  frame the policy was fed at runtime == frame the dataset returns for row t == the timing rule 3 * max(t // 3 - 1, 0),
  from the right env, and the row's state / action are those of step t; at query steps the server received that frame.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")

from synth_episodes import attrs, decode_frame, make_frame  # noqa: E402
from vision_lift.frames import frame_index  # noqa: E402
from vla_common.episode import EpisodeWriter  # noqa: E402
from vla_policy.export import export, load  # noqa: E402
from vla_tabletop.bridge import ChunkExecutor  # noqa: E402

K, T, N, PERIOD, LATENCY, H = 3, 40, 10, 3, 1, 48


class FakeServer:
    info = {"n_action_steps": N}

    def __init__(self):
        self.received: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def reset(self, ids, seeds):
        pass

    def act(self, ids, side, wrist, state, instr):
        self.received.append((side.copy(), wrist.copy(), state.copy()))
        return np.zeros((len(ids), N, 5), np.float32), 0.0, 0.0


def test_timing_rule():
    assert [3 * max(t // 3 - 1, 0) for t in range(10)] == [0, 0, 0, 0, 0, 0, 3, 3, 3, 6]
    assert [frame_index(t, PERIOD, LATENCY) * PERIOD for t in range(10)] == [0, 0, 0, 0, 0, 0, 3, 3, 3, 6]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("align")
    srv = FakeServer()
    ex = ChunkExecutor(srv, N, PERIOD, LATENCY, ["pick up the cube"] * K)
    ex.reset(np.zeros(K, np.int64))
    frames = {"side": [], "wrist": []}
    fed, states, actions, fidx = [], [], [], []

    def capture():
        k = len(frames["side"])
        side = np.stack([make_frame(k, e, 0, H) for e in range(K)])
        wrist = np.stack([make_frame(k, e, 1, H) for e in range(K)])
        frames["side"].append(side)
        frames["wrist"].append(wrist)
        return side, wrist

    for t in range(T):
        state = np.zeros((K, 27), np.float32)
        state[:, 0], state[:, 1] = t, np.arange(K)
        ex.step(t, capture, state)
        k, (side, _) = ex.frames.current()
        fed.append([decode_frame(side[e])[:2] for e in range(K)])  # (frame index, env) the policy sees at t
        fidx.append(k)
        a = np.zeros((K, 5), np.float32)
        a[:, 0], a[:, 1] = t, np.arange(K)  # the executed action at step t encodes (t, env)
        states.append(state[:, :19].copy())
        actions.append(a)
    path = d / "ep.h5"
    with EpisodeWriter(path, {"git_sha": "test", "created": "test", "stage": "0B"}) as w:
        for e in range(K):
            act = np.stack([a[e] for a in actions])
            w.add(np.stack([f[e] for f in frames["side"]]), np.stack([f[e] for f in frames["wrist"]]),
                  np.stack([s[e] for s in states]), act, act.copy(), np.asarray(fidx, np.int32),
                  attrs(e, T, H, env_id=e))
    export([path], "align", root=d / "ds")
    return srv, fed, fidx, load(d / "ds" / "align", return_uint8=True)


def test_runtime_feed_follows_the_rule(run):
    _, fed, fidx, _ = run
    for t in range(T):
        assert fidx[t] == frame_index(t, PERIOD, LATENCY)
        assert fed[t] == [(frame_index(t, PERIOD, LATENCY), e) for e in range(K)], t


def test_server_got_the_delayed_frame_at_query_steps_only(run):
    srv, _, _, _ = run
    qs = list(range(0, T, N))
    assert len(srv.received) == len(qs)
    for (side, wrist, state), t in zip(srv.received, qs):
        for e in range(K):
            assert decode_frame(side[e])[:3] == (frame_index(t, PERIOD, LATENCY), e, 0)
            assert decode_frame(wrist[e])[:3] == (frame_index(t, PERIOD, LATENCY), e, 1)
        np.testing.assert_array_equal(state[:, 0], t)
        assert state.shape == (K, 19)


def test_dataset_row_equals_runtime_feed(run):
    _, fed, _, ds = run
    assert len(ds) == K * T
    for i in range(len(ds)):
        r = ds[i]
        e, t = int(r["episode_index"]), int(r["frame_index"])
        assert i == e * T + t
        for c, cam in enumerate(("side", "wrist")):
            k, env, cc, _ = decode_frame(r[f"observation.images.{cam}"].permute(1, 2, 0).numpy())
            assert (k, env, cc) == (fed[t][e][0], e, c), (e, t, cam)
        assert int(r["frame_idx"]) == fed[t][e][0]
        assert float(r["action"][0]) == t and float(r["action"][1]) == e
        assert float(r["observation.state"][0]) == t and float(r["observation.state"][1]) == e
