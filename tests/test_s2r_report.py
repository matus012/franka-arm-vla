"""s1_report Stage 2R modes (val_s2r / final_s2r, task 005): tag s2r, run s2r_VT, test 11542, init = 2a attempt 1's
034440; the bars add slam < 5 % and jam < 2 %; mutants (2a attempt 1's checkpoints / init / test seed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_s2_report as T

S = T.S
A1 = "07e87cc6f5b2533ab4b931e356e7cf944e6b53d49bbf2cc0109ab416290cb3ff"
V15 = "c3dcef76f53e00e5954469970c25c38980a396723f7bbbf2b85919e6cc6688f5"
NORM = {"observation.state": {"mean": [1.0], "std": [0.5]}, "action": {"mean": [0.25], "std": [0.75]}}
VSEED, TSEED = 21640, 11542


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    return tmp_path


def ev2(seed: int, eps: list[dict], step: str, run: str = "s2r_VT", sha: str = "4dd2") -> dict:
    d = T.ev(seed, eps, step, stage="s2a")
    d["ckpt"] = f"/runs/{run}/checkpoints/{step}/pretrained_model"
    d["stats_sha"] = sha
    return d


def val_files(R: Path, ok=(62, 83, 90), run: str = "s2r_VT") -> None:
    for (c, step), k in zip(S.CKPTS_S2.items(), ok):
        T.put(R / "s2r" / f"s2r_VT_val_{c}_s{VSEED}.json", ev2(VSEED, T.eps_n(300, k), step, run))
    for step, v in (("008610", 0.0264), ("017220", 0.0208)):
        T.put(R / "s2r" / f"vloss_{step}.json", {"val_loss": v})


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def test_val_s2r(R):
    val_files(R)
    S.val_s2r()
    v = load(R / "s2r_val.json")
    assert v["pick"] == "c100" and v["pick_step"] == "034440" and v["attempt"] == "2R" and v["init_sha256"] == A1
    assert v["early_abort_d"]["abort"] is False
    assert not (R / "s2a1_val.json").exists()


def test_val_s2r_early_abort_d(R):
    val_files(R, ok=(15, 83, 90))  # s_50 = 15 / 300 = 5.0 % <= 5 % -> abort
    S.val_s2r()
    assert load(R / "s2r_val.json")["early_abort_d"]["abort"] is True


def test_val_s2r_rejects_attempt1_checkpoints(R):
    val_files(R, run="s2a1_VT")
    with pytest.raises(AssertionError):
        S.val_s2r()


def final_setup(R: Path, ok: int, wg: int, slam: int = 0, jam: int = 0, init_sha: str = A1) -> None:
    val_files(R)
    T.put(R / "s2r_val.json", {"pick": "c100", "pick_step": "034440"})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 3,
                     others=[T.GS, T.BL, T.YS][: 1 + i % 3]) for i in range(1000)]
    for e in eps[ok:ok + slam]:
        e["slam"] = True
    for e in eps[ok + slam:ok + slam + jam]:
        e["jam"] = True
    T.put(R / f"s2r_test_s{TSEED}.json", ev2(TSEED, eps, "034440"))
    T.put(R / "s2r" / "train_summary_VT.json", {"init": "/runs/s2a1_VT/checkpoints/034440/pretrained_model",
                                                "init_sha256": init_sha, "argv": {}, "norm_stats": NORM})


@pytest.mark.parametrize("ok,wg,slam,jam,gate", [(850, 49, 49, 19, True), (849, 49, 0, 0, False),
                                                 (850, 50, 0, 0, False), (850, 0, 50, 0, False),
                                                 (850, 0, 0, 20, False)])
def test_final_s2r_gate(R, ok, wg, slam, jam, gate):
    final_setup(R, ok, wg, slam, jam)
    S.final_s2r()
    f = load(R / "s2r.json")
    assert f["gate_pass"] is gate and f["seed"] == TSEED and f["init_check"]["is_2a1_034440"] is True


def test_final_s2r_rejects_v15_init(R):
    final_setup(R, 900, 0, init_sha=V15)
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s2r()


def test_final_s2r_rejects_attempt1_test_seed(R):
    final_setup(R, 900, 0)
    t = load(R / f"s2r_test_s{TSEED}.json")
    t["config"]["seed"] = 11541
    T.put(R / f"s2r_test_s{TSEED}.json", t)
    with pytest.raises(AssertionError):
        S.final_s2r()


R2 = "4dbe9303848db666d341937580425106e48d6dedc5d81a53741a00bc8e3dfcc1"


def fix_setup(R: Path, ok: int, wg: int, slam: int = 0, init_sha: str = R2, seed: int = 11543) -> None:
    for (c, step), k in zip(S.CKPTS_S2.items(), (62, 83, 90)):
        T.put(R / "s2r1" / f"s2r1_VT_val_{c}_s{VSEED}.json", ev2(VSEED, T.eps_n(300, k), step, "s2r1_VT"))
    T.put(R / "s2r1_val.json", {"pick": "c100", "pick_step": "034440"})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 3,
                     others=[T.GS, T.BL, T.YS][: 1 + i % 3]) for i in range(1000)]
    for e in eps[ok:ok + slam]:
        e["slam"] = True
    T.put(R / f"s2r1_test_s{seed}.json", ev2(seed, eps, "034440", "s2r1_VT"))
    T.put(R / "s2r1" / "train_summary_VT.json", {"init": "/runs/s2r_VT/checkpoints/034440/pretrained_model",
                                                 "init_sha256": init_sha, "argv": {}, "norm_stats": NORM})


@pytest.mark.parametrize("ok,wg,slam,gate", [(850, 49, 49, True), (849, 0, 0, False), (900, 50, 0, False),
                                             (900, 0, 50, False)])
def test_final_s2r1_gate_and_init(R, ok, wg, slam, gate):
    fix_setup(R, ok, wg, slam)
    S.final_s2r1()
    f = load(R / "s2r1.json")
    assert f["gate_pass"] is gate and f["seed"] == 11543 and f["init_check"]["is_2r_034440"] is True


def test_final_s2r1_rejects_2a1_init_and_attempt_seed(R):
    fix_setup(R, 900, 0, init_sha=A1)
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s2r1()


def test_val_s2r1_reads_its_own_run(R):
    fix_setup(R, 900, 0)
    for step, v in (("008610", 0.02), ("017220", 0.019)):
        T.put(R / "s2r1" / f"vloss_{step}.json", {"val_loss": v})
    S.val_s2r1()
    assert load(R / "s2r1_val.json")["pick_step"] == "034440"


R1 = "bc880e236677284c360078794a8c11ca22e43f753c276a8f0c60fa1c40e2f777"
CK_M = {"c050": "034440", "c075": "051660", "c100": "068880"}


def m_setup(R: Path, ok: int, wg: int, init_sha: str = R1, run: str = "s2m_VT", ok_val=(220, 230, 240)) -> None:
    for (c, step), k in zip(CK_M.items(), ok_val):
        T.put(R / "s2m" / f"s2m_VT_val_{c}_s{VSEED}.json", ev2(VSEED, T.eps_n(300, k), step, run))
    for step, v in (("017220", 0.02), ("034440", 0.019)):
        T.put(R / "s2m" / f"vloss_{step}.json", {"val_loss": v})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 3,
                     others=[T.GS, T.BL, T.YS][: 1 + i % 3]) for i in range(1000)]
    T.put(R / "s2m_test_s11544.json", ev2(11544, eps, "068880", run))
    T.put(R / "s2m" / "train_summary_VT.json", {"init": "/runs/s2r1_VT/checkpoints/034440/pretrained_model",
                                               "init_sha256": init_sha, "argv": {}, "norm_stats": NORM})


def test_val_s2m_uses_its_68880_step_checkpoints(R):
    m_setup(R, 900, 0)
    S.val_s2m()
    v = load(R / "s2m_val.json")
    assert v["pick_step"] == "068880" and v["per_ckpt"]["c050"]["step"] == "034440"
    assert v["early_abort_c"]["loss_25"] == 0.02 and v["early_abort_c"]["loss_50"] == 0.019


def test_val_s2m_rejects_34440_step_names(R):
    for (c, step), k in zip(S.CKPTS_S2.items(), (220, 230, 240)):  # the default saves read as 2M must fail
        T.put(R / "s2m" / f"s2m_VT_val_{c}_s{VSEED}.json", ev2(VSEED, T.eps_n(300, k), step, "s2m_VT"))
    with pytest.raises(AssertionError):
        S.val_s2m()


@pytest.mark.parametrize("ok,wg,gate", [(850, 49, True), (849, 0, False), (900, 50, False)])
def test_final_s2m_gate_init_and_stats(R, ok, wg, gate):
    m_setup(R, ok, wg)
    S.val_s2m()
    S.final_s2m()
    f = load(R / "s2m.json")
    assert f["gate_pass"] is gate and f["seed"] == 11544 and f["init_check"]["is_2r1_034440"] is True
    assert f["stats_sha"]["equal_across_evals"] is True


def test_final_s2m_rejects_another_init(R):
    m_setup(R, 900, 0, init_sha=R2)
    S.val_s2m()
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s2m()


M_PICK = "ab" * 32


def bp_setup(R: Path, monkeypatch, ok: int, wg: int, init_sha: str = M_PICK) -> None:
    monkeypatch.setitem(S.S2_RUNS["s2bp"]["extra"], "init_sha256", M_PICK)
    for (c, step), k in zip(S.CKPTS_S2.items(), (200, 240, 255)):
        d = T.ev(21650, T.eps_n(300, k), step, stage="s2")
        d["ckpt"], d["stats_sha"] = f"/runs/s2bp_VT/checkpoints/{step}/pretrained_model", "77aa"
        T.put(R / "s2bp" / f"s2bp_VT_val_{c}_s21650.json", d)
    for step, v in (("008610", 0.02), ("017220", 0.019)):
        T.put(R / "s2bp" / f"vloss_{step}.json", {"val_loss": v})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 3,
                     others=[T.GS, T.BL, T.YS][: 1 + i % 3]) for i in range(1000)]
    t = T.ev(11550, eps, "034440", stage="s2")
    t["ckpt"], t["stats_sha"] = "/runs/s2bp_VT/checkpoints/034440/pretrained_model", "77aa"
    T.put(R / "s2bp_test_s11550.json", t)
    h = T.ev(11560, T.eps_n(300, 150), "034440", split="heldout_comp", stage="s2")
    h["ckpt"], h["stats_sha"] = t["ckpt"], "77aa"
    T.put(R / "s2bp_heldout_s11560.json", h)
    T.put(R / "s2bp" / "train_summary_VT.json", {"init": "/runs/s2m_VT/checkpoints/068880/pretrained_model",
                                                 "init_sha256": init_sha, "argv": {}, "norm_stats": NORM})


@pytest.mark.parametrize("ok,wg,gate", [(850, 49, True), (849, 0, False), (900, 50, False)])
def test_s2bp_val_final_gate_heldout(R, monkeypatch, ok, wg, gate):
    bp_setup(R, monkeypatch, ok, wg)
    S.val_s2bp()
    v = load(R / "s2bp_val.json")
    assert v["pick_step"] == "034440" and v["early_abort_d"]["abort"] is False
    S.final_s2bp()
    f = load(R / "s2bp.json")
    assert f["gate_pass"] is gate and f["seed"] == 11550 and f["init_check"]["is_2m_pick"] is True
    assert f["heldout"]["n"] == 300


def test_s2bp_rejects_another_init(R, monkeypatch):
    bp_setup(R, monkeypatch, 900, 0, init_sha=R1)
    S.val_s2bp()
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s2bp()


def test_s2bp1_reads_its_own_run_and_fix_seeds(R, monkeypatch):
    monkeypatch.setitem(S.S2_RUNS["s2bp1"]["extra"], "init_sha256", M_PICK)
    for (c, step), k in zip(S.CKPTS_S2.items(), (200, 240, 255)):
        d = T.ev(21650, T.eps_n(300, k), step, stage="s2")
        d["ckpt"], d["stats_sha"] = f"/runs/s2bp1_VT/checkpoints/{step}/pretrained_model", "77aa"
        T.put(R / "s2bp1" / f"s2bp1_VT_val_{c}_s21650.json", d)
    S.val_s2bp1()
    assert load(R / "s2bp1_val.json")["pick_step"] == "034440"
    assert S.S2_RUNS["s2bp1"]["tseed"] == 11551 and S.S2_RUNS["s2bp1"]["hseed"] == 11561


B1 = "e87fd97bdd74dbf2625dd402c90fac734e2afc56414716d2270faf97abd1e503"


@pytest.mark.parametrize("ok,slam,gate", [(850, 49, True), (849, 0, False), (900, 50, False)])
def test_s2bp2_68880_ckpts_bars_with_slam(R, ok, slam, gate):
    for (c, step), k in zip(CK_M.items(), (200, 230, 260)):
        d = T.ev(21650, T.eps_n(300, k), step, stage="s2")
        d["ckpt"], d["stats_sha"] = f"/runs/s2bp2_VT/checkpoints/{step}/pretrained_model", "88bb"
        T.put(R / "s2bp2" / f"s2bp2_VT_val_{c}_s21650.json", d)
    for step, v in (("017220", 0.02), ("034440", 0.019)):
        T.put(R / "s2bp2" / f"vloss_{step}.json", {"val_loss": v})
    S.val_s2bp2()
    assert load(R / "s2bp2_val.json")["pick_step"] == "068880"
    eps = [T.episode(i, ok=i < ok, n_cubes=2 + i % 3, others=[T.GS, T.BL, T.YS][: 1 + i % 3]) for i in range(1000)]
    for e in eps[ok:ok + slam]:
        e["slam"] = True
    t = T.ev(11552, eps, "068880", stage="s2")
    t["ckpt"], t["stats_sha"] = "/runs/s2bp2_VT/checkpoints/068880/pretrained_model", "88bb"
    T.put(R / "s2bp2_test_s11552.json", t)
    T.put(R / "s2bp2" / "train_summary_VT.json", {"init": "/runs/s2bp1_VT/checkpoints/034440/pretrained_model",
                                                  "init_sha256": B1, "argv": {}, "norm_stats": NORM})
    S.final_s2bp2()
    f = load(R / "s2bp2.json")
    assert f["gate_pass"] is gate and f["seed"] == 11552 and f["init_check"]["is_2b_fix_pick"] is True
