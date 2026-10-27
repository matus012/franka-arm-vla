"""Contract episode files (vla_common.episode, v2 schema): write / read round trip and schema enforcement. Runs in
both venvs (stdlib + numpy + h5py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from synth_episodes import attrs, episode_arrays
from vla_common.episode import ARRAYS, EP_ATTRS, EpisodeWriter, file_attrs, read_episodes

FA = {"git_sha": "abc", "created": "now", "stage": "0B"}


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    src = [(episode_arrays(T, e, rng), attrs(e, T, 48, env_id=e)) for e, T in enumerate((7, 30, 1))]
    with EpisodeWriter(tmp_path / "x.h5", FA) as w:
        for a, at in src:
            w.add(**a, attrs=at)
    assert file_attrs(tmp_path / "x.h5") == FA
    got = list(read_episodes(tmp_path / "x.h5"))
    assert [g for g, _, _ in got] == ["ep_0", "ep_1", "ep_2"]
    for (_, a, at), (a0, at0) in zip(got, src):
        assert set(a) == set(ARRAYS)
        for k in ARRAYS:
            assert a[k].dtype == a0[k].dtype
            np.testing.assert_array_equal(a[k], a0[k])
        assert set(at) == set(EP_ATTRS)
        assert json.loads(at["spec_json"]) == at0["spec_json"] and json.loads(at["labels"]) == at0["labels"]
        assert at["instruction"] == at0["instruction"] and at["T"] == at0["T"] and at["success"] is True


@pytest.mark.parametrize("bad", ["state19", "action_exec", "frame_range", "frame_order", "attr", "T"])
def test_schema_enforced(tmp_path, bad):
    rng = np.random.default_rng(1)
    a, at = episode_arrays(12, 0, rng), attrs(0, 12, 48)
    if bad == "state19":
        a["state"] = np.zeros((12, 27), np.float32)
    elif bad == "action_exec":
        a["action_exec"] = a["action_exec"].astype(np.float64)
    elif bad == "frame_range":
        a["frame_idx"][-1] = a["side_rgb"].shape[0]
    elif bad == "frame_order":
        a["frame_idx"][5] = 3
    elif bad == "attr":
        del at["detector_version"]
    else:
        at["T"] = 11
    with EpisodeWriter(tmp_path / "x.h5", FA) as w, pytest.raises(AssertionError):
        w.add(**a, attrs=at)
