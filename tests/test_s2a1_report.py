"""s1_report 2a attempt 1 modes (val_s2a1 / final_s2a1): the s2a fixtures under tag s2a1, test 11541, + mutants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_s2_report as T

S = T.S
V15 = "c3dcef76f53e00e5954469970c25c38980a396723f7bbbf2b85919e6cc6688f5"
NORM = {"observation.state": {"mean": [1.0], "std": [0.5]}, "action": {"mean": [0.25], "std": [0.75]}}
VSEED, TSEED = 21640, 11541


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    return tmp_path


def ev2(seed: int, eps: list[dict], step: str, run: str = "s2a1_VT", sha: str = "4dd2") -> dict:
    d = T.ev(seed, eps, step, stage="s2a")
    d["ckpt"] = f"/runs/{run}/checkpoints/{step}/pretrained_model"
    d["stats_sha"] = sha
    return d


def val_files(R: Path, ok=(62, 83, 90), run: str = "s2a1_VT") -> None:
    for (c, step), k in zip(S.CKPTS_S2.items(), ok):
        T.put(R / "s2a1" / f"s2a1_VT_val_{c}_s{VSEED}.json", ev2(VSEED, T.eps_n(300, k), step, run))
    for step, v in (("008610", 0.0264), ("017220", 0.0208)):
        T.put(R / "s2a1" / f"vloss_{step}.json", {"val_loss": v})


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def test_val_s2a1(R):
    val_files(R)
    S.val_s2a1()
    v = load(R / "s2a1_val.json")
    assert v["pick"] == "c100" and v["pick_step"] == "034440" and v["attempt"] == 1 and v["init_sha256"] == V15
    assert v["early_abort_c"]["abort"] is False and v["early_abort_d"]["abort"] is False
    assert not (R / "s2a_val.json").exists()


def test_val_s2a1_rejects_attempt0_checkpoints(R):
    val_files(R, run="s2a_VT")  # attempt 0's run read as attempt 1 must fail
    with pytest.raises(AssertionError):
        S.val_s2a1()


def final_setup(R: Path, ok: int, wg: int) -> None:
    val_files(R)
    T.put(R / "s2a1_val.json", {"pick": "c100", "pick_step": "034440"})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 3,
                     others=[T.GS, T.BL, T.YS][: 1 + i % 3]) for i in range(1000)]
    T.put(R / f"s2a1_test_s{TSEED}.json", ev2(TSEED, eps, "034440"))
    T.put(R / "s2a1" / "train_summary_VT.json", {"init": "/runs/s1y15_U/checkpoints/041328/pretrained_model",
                                                 "init_sha256": V15, "argv": {}, "norm_stats": NORM})


@pytest.mark.parametrize("ok,wg,gate", [(850, 49, True), (849, 49, False), (850, 50, False)])
def test_final_s2a1_gate(R, ok, wg, gate):
    final_setup(R, ok, wg)
    S.final_s2a1()
    f = load(R / "s2a1.json")
    assert f["gate_pass"] is gate and f["seed"] == TSEED and f["init_check"]["is_stage1_v15_041328"] is True
    assert not (R / "s2a.json").exists()


def test_final_s2a1_rejects_attempt0_test_seed(R):
    final_setup(R, 900, 0)
    t = load(R / f"s2a1_test_s{TSEED}.json")
    t["config"]["seed"] = 11540
    T.put(R / f"s2a1_test_s{TSEED}.json", t)
    with pytest.raises(AssertionError):
        S.final_s2a1()
