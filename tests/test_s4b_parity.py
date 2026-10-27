"""Instrument B-I2 (scripts/s4b_parity.py) on the synthetic split dataset (48 px) + a random-weight baseline."""
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
sp = importlib.util.spec_from_file_location("s4b_parity", ROOT / "scripts" / "s4b_parity.py")
M = importlib.util.module_from_spec(sp)
sp.loader.exec_module(M)
N, H = 3, 48


@pytest.fixture(scope="module")
def setup(split_ds, tmp_path_factory):
    dsdir, _ = split_ds
    torch.manual_seed(0)
    m = BaselinePolicy(BaselineConfig(n_action_steps=N))
    m.set_norm(torch.randn(19), torch.rand(19) + 0.5, torch.randn(4) * 0.1, torch.rand(4) + 0.5)
    ck = tmp_path_factory.mktemp("pck") / "ckpt_000001"
    m.save(ck)
    meta = TB.make_ds(dsdir, None, N, "pyav")
    ds = TB.make_ds(dsdir, None, N, "pyav", meta.fps)
    return ds, T.split_episodes(ds)[0], ck


class Coloured:
    """The synthetic frames are grey (a BGR flip is invisible on them): same rows, random coloured decoded frames."""

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        it = dict(self.ds[i])
        for k in ("observation.images.side", "observation.images.wrist"):
            it[k] = (torch.rand(3, H, H, generator=g) * 255).round() / 255
        return it


def test_parity_passes_and_mutants_are_flagged(setup):
    ds, is_val, ck = setup
    ds = Coloured(ds)
    ok = M.check(ds, is_val, ck, N, H, rows=8, seed=0)
    assert ok["pass"] and ok["max_abs_arm_diff"] <= 1e-5 and ok["claw_mismatches"] == 0 and ok["info_ok"] is True
    assert ok["chunk_nonconstant"] and ok["uint8_roundtrip_max_err"] <= 1e-6 and set(ok["rows"]) <= set(np.flatnonzero(is_val))
    assert ok["info"]["chunk_size"] == N and ok["info"]["image_size"] == H
    for mut in ("bgr", "unnormalised", "swapped_cams"):
        r = M.check(ds, is_val, ck, N, H, rows=8, seed=0, mutant=mut)
        assert not r["pass"] and (r["max_abs_arm_diff"] > 1e-5 or r["claw_mismatches"] > 0), (mut, r)


def test_wrong_info_is_flagged(setup):
    ds, is_val, ck = setup
    with pytest.raises(AssertionError):
        M.check(ds, is_val, ck, N, H + 16, rows=8, seed=0)  # dataset frames are 48 px, claimed 64
    with pytest.raises(ValueError, match="n_action_steps"):
        M.check(ds, is_val, ck, N + 1, H, rows=8, seed=0)  # server refuses a chunk size != the checkpoint's


def test_cli_writes_json(split_ds, setup, tmp_path, monkeypatch):
    ds, is_val, ck = setup
    out = tmp_path / "p.json"
    monkeypatch.setattr(sys, "argv", ["s4b_parity", "--dataset", str(split_ds[0]), "--ckpt", str(ck), "--out", str(out),
                                      "--n", str(N), "--image-size", str(H), "--rows", "6"])
    M.main()
    assert json.loads(out.read_text())["pass"] is True
