"""train_baseline --max-train-episodes: pure addition (absent = unchanged sampler / S), K keeps the first K train episodes."""
import json
import sys

import numpy as np
import pytest

pytest.importorskip("lerobot")
from test_train_smolvla import runs, split_ds  # noqa: E402,F401  (fixtures: train episodes 0 and 2, val episode 1)

from vla_policy import train_baseline as TB  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402

N = 3


@pytest.fixture(scope="module")
def ds(split_ds):
    meta = TB.make_ds(split_ds[0], None, N, "pyav")
    return TB.make_ds(split_ds[0], None, N, "pyav", meta.fps)


def sampler(ds, keep):
    return T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), 0, keep=keep)


def test_flag_absent_is_unchanged(ds):
    is_val, train_eps, _ = T.split_episodes(ds)
    keep, kept = TB.train_keep(ds, is_val, train_eps, None)
    assert np.array_equal(keep, ~is_val) and kept == train_eps == [0, 2]
    a, b = sampler(ds, keep), sampler(ds, ~is_val)
    assert len(a) == len(b) and all(np.array_equal(a.order(e), b.order(e)) for e in range(3))


def test_k_keeps_first_k_train_episodes_only(ds, split_ds):
    is_val, train_eps, _ = T.split_episodes(ds)
    ep = T.column(ds, "episode_index").reshape(-1)
    full = len(sampler(ds, ~is_val))
    keep, kept = TB.train_keep(ds, is_val, train_eps, 1)
    smp = sampler(ds, keep)
    assert kept == [0] and set(ep[smp.order(0)]) == {0} and 0 < len(smp) < full
    assert len(smp) == len(set(T.column(ds, "frame_idx").reshape(-1)[ep == 0]))
    assert not keep[is_val].any()
    assert TB.train_keep(ds, is_val, train_eps, 5)[1] == [0, 2]  # K above the train count keeps all
    ids = TB.kept_ids(split_ds[0], kept)
    assert ids == [{"lerobot_episode_index": 0, "seed": 0, "episode_index": 0}]
    assert TB.kept_ids(split_ds[0].parent, [3]) == [{"lerobot_episode_index": 3}]  # no sidecar


def test_main_records_kept_episodes(split_ds, tmp_path, monkeypatch):
    out = tmp_path / "o"
    argv = ["train_baseline", "--root", str(split_ds[0]), "--out", str(out), "--epochs", "1", "--n-action-steps", str(N),
            "--batch-size", "4", "--device", "cpu", "--max-train-episodes", "1"]
    monkeypatch.setattr(sys, "argv", argv + ["--expect-windows", "6"])
    TB.main()
    s = json.loads((out / "train_summary.json").read_text())
    assert s["S"] == 6 and s["max_train_episodes"] == 1 and s["n_train_episodes"] == 2
    assert s["kept_train_episodes"] == [{"lerobot_episode_index": 0, "seed": 0, "episode_index": 0}]
    seen = {i for r in map(json.loads, (out / "steps.jsonl").read_text().splitlines()) for i in r["index"]}
    ep = T.column(TB.make_ds(split_ds[0], None, N, "pyav"), "episode_index").reshape(-1)
    assert set(ep[sorted(seen)]) == {0}
    monkeypatch.setattr(sys, "argv", argv + ["--expect-windows", "7"])
    with pytest.raises(SystemExit):
        TB.main()
    monkeypatch.setattr(sys, "argv", argv[:-2] + ["--out", str(tmp_path / "p")])
    TB.main()
    s2 = json.loads((tmp_path / "p" / "train_summary.json").read_text())
    assert s2["max_train_episodes"] is None and s2["kept_train_episodes"] is None and s2["S"] == 11
