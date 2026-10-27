"""0B: contract HDF5 -> vla_policy.export -> LeRobotDataset (pyav) round trip on synthetic episodes (policy venv).

Exact: state, action label (not action_exec), task, labels, split, frame_idx, rows per episode, episode boundaries,
the 50-step action chunk and its padding mask. Images: lossy AV1 (libsvtav1, crf 30, yuv420p). Tolerance: PSNR >= 35 dB
per frame (GATES 0B.1) and every interior stripe pixel within 3 grey levels of its source (< STEP / 2 = 4, the margin
the index decoder needs), plus frame identity (nearest source frame by MSE). Writes the decoded-episode contact sheet
results/frames/s0b_roundtrip/decoded_ep0.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from synth_episodes import H, decode_frame, write_file  # noqa: E402
from vla_common.episode import read_episodes  # noqa: E402
from vla_policy import roundtrip  # noqa: E402
from vla_policy.export import FrameWindowSampler, export, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = ([20, 31], [17])  # two files; lengths not multiples of PERIOD; one episode shorter than the chunk


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("rt")
    files = [d / "a.h5", d / "b.h5"]
    write_file(files[0], LENGTHS[0], seed=0, ep0=0)
    write_file(files[1], LENGTHS[1], seed=1, ep0=2)
    export(files, "rt", root=d / "ds")
    return files, d / "ds" / "rt"


def test_roundtrip_columns_chunks_and_psnr(exported):
    files, dsdir = exported
    out = roundtrip.check(files, dsdir, keep_frames=6)
    assert out["episodes"] == out["dataset_episodes"] == 3
    assert out["rows"] == sum(sum(x) for x in LENGTHS)
    assert out["column_mismatch"] == [] and out["rows_per_episode_mismatch"] == []
    assert out["chunk_mismatch"] == 0 and out["pad_mismatch"] == 0
    assert out["identity_fail"] == 0
    assert out["psnr_min"] >= 35.0, out["psnr_min"]
    roundtrip.sheet(out["sheet"], ROOT / "results" / "frames" / "s0b_roundtrip" / "decoded_ep0.png",
                    f"synthetic stripes H={H}: source vs decoded (pyav); psnr_min={out['psnr_min']:.1f} dB")


def test_label_not_executed_action_and_boundaries(exported):
    files, dsdir = exported
    ds = load(dsdir, return_uint8=True)
    src = [a for f in files for _, a, _ in read_episodes(f)]
    eps = ds.meta.episodes
    start = 0
    for e, a in enumerate(src):
        T = a["state"].shape[0]
        assert (int(eps[e]["dataset_from_index"]), int(eps[e]["dataset_to_index"])) == (start, start + T)
        assert not np.array_equal(a["action"], a["action_exec"])
        last = ds[start + T - 1]
        assert int(last["episode_index"]) == e and int(last["frame_index"]) == T - 1
        np.testing.assert_array_equal(last["action"].numpy(), a["action"][T - 1])
        start += T


def test_decoded_stripes_within_tolerance(exported):
    files, dsdir = exported
    ds = load(dsdir, return_uint8=True)
    src = [a for f in files for _, a, _ in read_episodes(f)]
    i = 0
    for e, a in enumerate(src):
        for t in range(a["state"].shape[0]):
            r = ds[i]
            for c, cam in enumerate(("side", "wrist")):
                k, ep, cc, err = decode_frame(r[f"observation.images.{cam}"].permute(1, 2, 0).numpy())
                assert (k, ep, cc) == (int(a["frame_idx"][t]), e, c), (e, t, cam)
                assert err <= 3, (e, t, cam, err)
            i += 1


def test_frame_window_sampler(exported):
    """Row option B: per epoch exactly one row per (episode, frame_idx) window; over epochs every row is drawn."""
    _, dsdir = exported
    ds = load(dsdir)
    ep = np.asarray(ds.hf_dataset["episode_index"]).reshape(-1)
    fi = np.asarray(ds.hf_dataset["frame_idx"]).reshape(-1)
    windows = {(int(e), int(k)) for e, k in zip(ep, fi)}
    s = FrameWindowSampler(ds, seed=0)
    assert len(s) == len(windows)
    seen = set()
    for _ in range(40):
        rows = list(iter(s))
        assert len(rows) == len(windows) == len({(int(ep[r]), int(fi[r])) for r in rows})
        seen |= set(rows)
    assert seen == set(range(len(ds)))  # windows hold <= 6 rows; 40 epochs draw each with p > 0.999
    assert list(iter(FrameWindowSampler(ds, seed=0))) == list(iter(FrameWindowSampler(ds, seed=0)))


def test_export_select_filters_and_overrides(exported, tmp_path):
    files, _ = exported
    seen = []

    def select(f, group, at):
        seen.append((f.name, group))
        return None if (f.name, group) == ("a.h5", "ep_0") else at | {"split": "val"}

    ds = export(files, "sel", root=tmp_path, select=select)
    assert seen == [("a.h5", "ep_0"), ("a.h5", "ep_1"), ("b.h5", "ep_0")]
    assert ds.meta.total_episodes == 2 and ds.meta.total_frames == 31 + 17
    d = load(tmp_path / "sel")
    assert set(d.hf_dataset["split"]) == {"val"}
    assert float(d[0]["action"][0]) == 1000.0  # first kept episode = synthetic episode 1 (action[0] = t + 1000 ep)
