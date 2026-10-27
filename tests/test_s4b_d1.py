"""Instrument D1 (scripts/s4b_d1_compare.py) on the synthetic split dataset (48 px) + a random-weight baseline."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")
from test_train_smolvla import runs, split_ds  # noqa: E402,F401  (fixtures)

from vla_policy import train_baseline as TB  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402
from vla_policy.baseline import BaselineConfig, BaselinePolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("s4b_d1_compare", ROOT / "scripts" / "s4b_d1_compare.py")
M = importlib.util.module_from_spec(sp)
sp.loader.exec_module(M)
N, H = 3, 48


class Coloured:
    """The synthetic frames are grey (a BGR flip is invisible on them): same rows, random coloured decoded frames."""

    def __init__(self, ds):
        self.ds = ds

    def __getattr__(self, k):
        return getattr(self.ds, k)

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        it = dict(self.ds[i])
        for k in ("observation.images.side", "observation.images.wrist"):
            it[k] = (torch.rand(3, H, H, generator=g) * 255).round() / 255
        return it


@pytest.fixture(scope="module")
def setup(split_ds, tmp_path_factory):
    dsdir, _ = split_ds
    torch.manual_seed(0)
    m = BaselinePolicy(BaselineConfig(n_action_steps=N))
    m.set_norm(torch.randn(19), torch.rand(19) + 0.5, torch.randn(4) * 0.1, torch.rand(4) + 0.5)
    ck = tmp_path_factory.mktemp("d1ck") / "ckpt_000001"
    m.save(ck)
    meta = TB.make_ds(dsdir, None, N, "pyav")
    ds = Coloured(TB.make_ds(dsdir, None, N, "pyav", meta.fps))
    return ds, ck, m


def make_dump(ds, m, eps=(0, 1, 2), steps=15, seed=0) -> dict:
    """What eval_policy --dump-slots would write if the closed loop reproduced the dataset exactly."""
    ep = T.column(ds, "episode_index").reshape(-1)
    st, ac = T.column(ds, "observation.state"), T.column(ds, "action")
    out = {k: [] for k in ("side0", "wrist0", "state0", "labels0", "chunk0", "actions", "states")}
    for e in eps:
        rows = np.flatnonzero(ep == e)
        it = ds[int(rows[0])]
        out["side0"].append(M.u8(it["observation.images.side"].numpy()))
        out["wrist0"].append(M.u8(it["observation.images.wrist"].numpy()))
        out["state0"].append(st[rows[0]])
        out["labels0"].append(TB.episode_labels(it))
        out["chunk0"].append(m.predict(it["observation.images.side"].numpy()[None], it["observation.images.wrist"].numpy()[None],
                                       st[rows[0]][None], out["labels0"][-1][None])[0])
        out["actions"].append(ac[rows[:steps]])
        out["states"].append(st[rows[:steps]])
    d = {k: np.stack(v) for k, v in out.items()}
    return d | {"seed": np.int64(seed), "episode_index": np.array(eps, np.int64)}


def test_clean_dump_passes_and_reports_everything(setup):
    ds, ck, m = setup
    r = M.compare(make_dump(ds, m), ds, ck)
    a = r["aggregate"]
    assert a["pass"] and all(a["items"].values()) and a["n_matched"] == 3 and a["unmatched_episode_index"] == []
    assert a["state0_max_abs_diff"] == 0 and a["labels_equal_all"] and a["psnr_side"]["min"] == float("inf")
    assert a["ds_pred_vs_server"]["arm_max_abs"] <= 1e-5 and a["ds_pred_vs_server"]["claw_equal_slots"] == 3
    assert a["vs_dataset_action"]["server"]["claw_agreement_mean"] == a["vs_dataset_action"]["ds_pred"]["claw_agreement_mean"]
    assert a["state_diff_at_step"]["0"]["max_dims_max_slots"] == 0 and "10" in a["state_diff_at_step"]
    assert "30" not in a["state_diff_at_step"]  # only 15 recorded steps
    s0 = r["slots"][0]
    assert len(s0["state_diff_mean_dims"]) == 15 and s0["first_close"]["executed"] == s0["first_close"]["dataset"]
    json.dumps(r)


def test_bgr_swapped_frames_fail_the_psnr_item(setup):
    ds, ck, m = setup
    d = make_dump(ds, m)
    d["side0"] = np.ascontiguousarray(d["side0"][..., ::-1])
    a = M.compare(d, ds, ck)["aggregate"]
    assert not a["pass"] and not a["items"]["psnr_side"] and a["items"]["psnr_wrist"] and a["items"]["state0"]
    assert a["psnr_side"]["min"] < 35.0


def test_state_offset_fails_the_state_item(setup):
    ds, ck, m = setup
    d = make_dump(ds, m)
    d["state0"] = d["state0"] + np.float32(1e-3)
    a = M.compare(d, ds, ck)["aggregate"]
    assert not a["pass"] and not a["items"]["state0"] and a["items"]["psnr_side"] and a["items"]["labels"]


def test_labels_mismatch_and_unmatched_slot(setup):
    ds, ck, m = setup
    d = make_dump(ds, m)
    d["labels0"][1, 0] = (d["labels0"][1, 0] + 1) % 3
    a = M.compare(d, ds, ck)["aggregate"]
    assert not a["items"]["labels"] and not a["pass"]
    d = make_dump(ds, m)
    d["episode_index"] = np.array([0, 1, 99], np.int64)  # stream episode 99 has no kept demo
    a = M.compare(d, ds, ck)["aggregate"]
    assert a["n_matched"] == 2 and a["unmatched_episode_index"] == [99]
    d = make_dump(ds, m, seed=5)  # other stream seed: nothing matches
    assert M.compare(d, ds, ck)["aggregate"]["pass"] is False


def test_policy_divergence_is_reported_not_hidden(setup):
    ds, ck, m = setup
    d = make_dump(ds, m)
    d["chunk0"] = d["chunk0"] + np.float32(0.25)
    a = M.compare(d, ds, ck)["aggregate"]
    assert a["ds_pred_vs_server"]["arm_max_abs"] > 0.2 and a["pass"]  # inputs identical: the t=0 item set still passes


def test_cli_writes_json(split_ds, setup, tmp_path, monkeypatch):
    ds, ck, m = setup
    d = make_dump(ds, m)
    # the CLI reads the plain (grey) dataset: build the dump from it
    plain = ds.ds
    d = make_dump(plain, m)
    np.savez(tmp_path / "x_dump.npz", **d)
    out = tmp_path / "c.json"
    monkeypatch.setattr(sys, "argv", ["s4b_d1_compare", "--dump", str(tmp_path / "x_dump.npz"), "--dataset",
                                      str(split_ds[0]), "--ckpt", str(ck), "--out", str(out)])
    M.main()
    assert json.loads(out.read_text())["aggregate"]["pass"] is True
