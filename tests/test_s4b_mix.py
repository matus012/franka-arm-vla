"""Instrument B-I4 (scripts/s4b_mix.py): pure check on synthetic streams + the CLI on a real tiny baseline run."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vla_policy.train_smolvla import WindowSampler

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("s4b_mix", ROOT / "scripts" / "s4b_mix.py")
M = importlib.util.module_from_spec(sp)
sp.loader.exec_module(M)

BS, EPOCHS, SEED = 8, 2, 3


def world():
    """4 streams x 6 episodes (1 val each), 10-14 frames per episode, frame_idx = t // 2 (windows of 2 rows)."""
    ep, fi, stream, val = [], [], [], []
    e = 0
    for s in range(4):
        for k in range(6):
            n = 10 + (e % 5)
            ep += [e] * n
            fi += [t // 2 for t in range(n)]
            stream += [s] * n
            val += [k == 5] * n
            e += 1
    return tuple(np.array(x) for x in (ep, fi, stream, val))


def steps_of(ep, fi, val, seed=SEED, bs=BS, epochs=EPOCHS):
    sam = WindowSampler(ep, fi, seed, keep=~val)
    spe = len(sam) // bs
    out = []
    for e in range(epochs):
        o = sam.order(e)
        out += [{"step": e * spe + k + 1, "epoch": e, "index": o[k * bs:(k + 1) * bs].tolist()} for k in range(spe)]
    return out


def test_pass_and_mutants_fail():
    ep, fi, stream, val = world()
    steps = steps_of(ep, fi, val)
    r = M.check(steps, ep, fi, val, stream, SEED, BS, EPOCHS)
    assert r["pass"], r
    assert all(sum(v["draws_per_stream"].values()) == r["steps_per_epoch"] * BS for v in r["per_epoch"].values())
    assert r["S"] % BS != 0 and all(v["dropped_tail"] == r["S"] % BS for v in r["per_epoch"].values())  # drop_last tail
    for mut in ("drop-stream", "seed1"):
        m = M.check(steps, ep, fi, val, stream, SEED, BS, EPOCHS, mutant=mut)
        assert not m["pass"], mut
    assert M.check(steps, ep, fi, val, stream, SEED, BS, EPOCHS, mutant="seed1")["steps_with_sampler_mismatch"] > 0
    # a val row in a step, a repeated step, a trained-with-the-wrong-seed run
    vi = int(np.flatnonzero(val)[0])
    bad = [dict(s) for s in steps]
    bad[2]["index"] = bad[2]["index"][:-1] + [vi]
    assert not M.check(bad, ep, fi, val, stream, SEED, BS, EPOCHS)["pass"]
    assert not M.check(steps_of(ep, fi, val, seed=SEED + 1), ep, fi, val, stream, SEED, BS, EPOCHS)["pass"]
    assert not M.check(steps[:-1], ep, fi, val, stream, SEED, BS, EPOCHS)["pass"]  # truncated run


def test_cli_on_a_real_tiny_baseline_run(split_ds, tmp_path, monkeypatch):
    from test_train_smolvla import run_baseline  # noqa: F401

    dsdir, _ = split_ds
    out = tmp_path / "run"
    run_baseline(dsdir, out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["s4b_mix", "--dataset", str(dsdir), "--run", str(out), "--out", str(tmp_path / "m.json")])
    M.main()
    j = json.loads((tmp_path / "m.json").read_text())
    assert j["pass"] and j["steps_with_sampler_mismatch"] == 0 and len(j["streams"]) >= 1
    monkeypatch.setattr(sys, "argv", ["s4b_mix", "--dataset", str(dsdir), "--run", str(out), "--out", str(tmp_path / "m2.json"),
                                      "--mutant", "seed1"])
    M.main()
    assert json.loads((tmp_path / "m2.json").read_text())["pass"] is False


from test_train_smolvla import runs, split_ds  # noqa: E402,F401  (fixtures)
