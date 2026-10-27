"""s1_report Stage 2 2b modes (val_s2b / final_s2b / lang_s2b): synthetic fixtures + mutants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_s2_report as T

S = T.S
NORM = {"observation.state": {"mean": [1.0], "std": [0.5]}, "action": {"mean": [0.25], "std": [0.75]}}
INIT = "a" * 64  # 2a's gated checkpoint sha
A0 = "4dd2"  # results/s2_test_s11520.json stats_sha
VSEED, TSEED, HSEED = 21650, 11550, 11560


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    return tmp_path


def ev2(seed: int, eps: list[dict], step: str, sha: str = A0, split: str = "train", **kw) -> dict:
    d = T.ev(seed, eps, step, split=split, stage="s2", **kw)  # 2b is the full s2 task
    d["ckpt"] = f"/runs/s2b_VT/checkpoints/{step}/pretrained_model"
    d["stats_sha"] = sha
    return d


def val_files(R: Path, ok=(150, 180, 210), shas=(A0, A0, A0)) -> None:
    for (c, step), k, sha in zip(S.CKPTS_S2.items(), ok, shas):
        T.put(R / "s2b" / f"s2b_VT_val_{c}_s{VSEED}.json", ev2(VSEED, T.eps_n(300, k), step, sha))
    for step, v in (("008610", 1.0), ("017220", 1.5), ("034440", 1.2)):
        T.put(R / "s2b" / f"vloss_{step}.json", {"val_loss": v})


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def test_val_s2b(R):
    val_files(R)
    S.val_s2b()
    v = load(R / "s2b_val.json")
    assert v["arm"] == "VT" and v["seed"] == VSEED and v["pick"] == "c100" and v["pick_step"] == "034440"
    assert v["early_abort_c"] == {"loss_25": 1.0, "loss_50": 1.5, "abort": False}
    assert v["per_ckpt"]["c050"]["ckpt"].endswith("/s2b_VT/checkpoints/017220/pretrained_model")
    val_files(R, ok=(15, 180, 210))
    S.val_s2b()
    assert load(R / "s2b_val.json")["early_abort_d"]["abort"] is True
    assert not any((R / f).exists() for f in ("s2_val.json", "s2a_val.json", "s2base_val.json"))


@pytest.mark.parametrize("what", ["ckpt_step", "ckpt_dir", "seed_s2a", "stage_s2a", "split"])
def test_val_s2b_mutants_fire(R, what):
    val_files(R)
    p = R / "s2b" / f"s2b_VT_val_c075_s{VSEED}.json"
    d = load(p)
    if what == "ckpt_step":
        d["ckpt"] = "/runs/s2b_VT/checkpoints/017220/pretrained_model"
    elif what == "ckpt_dir":
        d["ckpt"] = "/runs/s2a_VT/checkpoints/025830/pretrained_model"
    elif what == "seed_s2a":
        d["config"]["seed"] = 21640
    elif what == "stage_s2a":
        d["config"]["stage"] = "s2a"
    else:
        d["config"]["split"] = "heldout_comp"
    T.put(p, d)
    with pytest.raises(AssertionError):
        S.val_s2b()


def final_setup(R: Path, ok: int = 900, wg: int = 10, heldout: bool = False, init_sha: str = INIT, a0_sha: str = A0) -> None:
    val_files(R)
    T.put(R / "s2b_val.json", {"pick": "c100", "pick_step": "034440"})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 4,
                     others=[T.GS, T.BL, T.YS, T.YL][: 1 + i % 4]) for i in range(1000)]
    T.put(R / f"s2b_test_s{TSEED}.json", ev2(TSEED, eps, "034440"))
    T.put(R / "s2b" / "train_summary_VT.json", {"init": "/runs/s2a_VT/checkpoints/034440/pretrained_model",
                                                "init_sha256": init_sha, "argv": {}, "norm_stats": NORM})
    T.put(R / "s2" / "train_summary_U.json", {"norm_stats": NORM})
    T.put(R / "s2_test_s11520.json", ev2(11520, eps, "034440", sha=a0_sha))
    T.put(R / "s2a.json", {"ckpt_sha": INIT})
    if heldout:
        he = [T.episode(i, ok=i < 240, target=T.GL, others=[T.BL, T.YS]) for i in range(300)]
        T.put(R / f"s2b_heldout_s{HSEED}.json", ev2(HSEED, he, "034440", split="heldout_comp"))


@pytest.mark.parametrize("ok,wg,gate", [(850, 49, True), (849, 49, False), (850, 50, False), (1000, 0, True)])
def test_final_s2b_gate_and_instruments(R, ok, wg, gate):
    final_setup(R, ok, wg, heldout=True)
    S.final_s2b()  # --init-sha omitted: results/s2a.json ckpt_sha
    f = load(R / "s2b.json")
    assert f["gate_pass"] is gate and f["success"]["count"] == ok and f["wrong_grasp"]["count"] == wg
    assert f["seed"] == TSEED and f["arm"] == "VT" and f["n"] == 1000
    assert f["i1_init"]["is_2a_gated_checkpoint"] is True and f["i1_init"]["expected_sha256"] == INIT
    assert f["i3_norm"]["norm_stats_equal_attempt0"] is True and f["i3_norm"]["test_stats_sha_equal_attempt0"] is True
    assert f["i3_norm"]["heldout_stats_sha_equal_attempt0"] is True
    assert f["heldout"]["n"] == 300 and f["stats_check"] == "not run (checkpoint not synced)"
    assert not (R / "s2.json").exists()


def test_final_s2b_init_sha_flag_and_no_heldout(R):
    final_setup(R)
    T.put(R / "s2a.json", {"ckpt_sha": "b" * 64})  # the default would now fail ...
    with pytest.raises(AssertionError):
        S.final_s2b()
    S.final_s2b("--init-sha", INIT)  # ... the flag wins
    assert "heldout" not in load(R / "s2b.json")


@pytest.mark.parametrize("what", ["init_sha", "norm", "test_sha", "val_sha", "heldout_sha", "seed", "ckpt_dir", "ckpt_step",
                                  "heldout_seed", "heldout_ckpt"])
def test_final_s2b_mutants_fire(R, what):
    final_setup(R, heldout=True)
    tp, hp = R / f"s2b_test_s{TSEED}.json", R / f"s2b_heldout_s{HSEED}.json"
    t, h = load(tp), load(hp)
    if what == "init_sha":
        final_setup(R, heldout=True, init_sha="c" * 64)
    elif what == "norm":
        d = load(R / "s2b" / "train_summary_VT.json")
        d["norm_stats"]["action"]["std"] = [0.7000001]
        T.put(R / "s2b" / "train_summary_VT.json", d)
    elif what == "test_sha":
        t["stats_sha"] = "beef"
        T.put(tp, t)
    elif what == "val_sha":
        val_files(R, shas=(A0, "beef", A0))
    elif what == "heldout_sha":
        h["stats_sha"] = "beef"
        T.put(hp, h)
    elif what == "seed":
        t["config"]["seed"] = 11540
        T.put(tp, t)
    elif what == "ckpt_dir":
        t["ckpt"] = "/runs/s2a_VT/checkpoints/034440/pretrained_model"
        T.put(tp, t)
    elif what == "ckpt_step":
        t["ckpt"] = "/runs/s2b_VT/checkpoints/025830/pretrained_model"
        T.put(tp, t)
    elif what == "heldout_seed":
        h["config"]["seed"] = 11530
        T.put(hp, h)
    else:
        h["ckpt"] = "/runs/s2b_VT/checkpoints/025830/pretrained_model"
        T.put(hp, h)
    with pytest.raises(AssertionError):
        S.final_s2b()


# ------------------------------------------------------------------------------------------------ language check
def lang_files(R: Path, ck: str, true_ok: set, shuf_ok: set, **mut) -> None:
    step = S.CKPTS_S2[ck]
    for mode, name, ok in (("none", "true", true_ok), ("shuffle", "shuffle", shuf_ok)):
        eps = []
        for i in range(100):
            e = T.episode(i, ok=i in ok)
            e["instruction"] = f"pick up cube {i}"
            if mode == "shuffle":
                e["instruction_sent"] = f"pick up cube {(i + 1) % 100}"
            eps.append(e)
        d = ev2(VSEED, eps, step)
        d["config"]["instruction_swap"] = mode
        if mode == "shuffle":
            d["instruction_swap_report"] = {"donor_of": {str(i): (i + 1) % 100 for i in range(100)}}
        T.put(R / "s2b_lang" / f"s2b_VT_{ck}_{name}_s{VSEED}.json", d)
    for f, fn in mut.items():
        p = R / "s2b_lang" / f
        d = load(p)
        fn(d)
        T.put(p, d)


def test_lang_s2b_numbers(R):
    lang_files(R, "c100", set(range(14)), set(range(9)) | {50, 51})
    S.lang_s2b("c100")
    r = load(R / "s2b_lang_c100.json")
    p = r["pair"]
    assert (p["true_success"], p["shuffled_success"], p["b_true_only"], p["c_shuffled_only"]) == (14, 11, 5, 2)
    assert r["seed"] == VSEED and r["step"] == "034440" and p["ckpt"].endswith("/s2b_VT/checkpoints/034440/pretrained_model")
    assert "attempt0" not in r and "gate_pass" not in r
    with pytest.raises(AssertionError):
        S.lang_s2b("c075")


@pytest.mark.parametrize("what", ["seed", "stage", "ckpt_dir", "swap", "n99"])
def test_lang_s2b_mutants_fire(R, what):
    tr = f"s2b_VT_c050_true_s{VSEED}.json"
    mut = {"seed": lambda d: d["config"].update(seed=21640), "stage": lambda d: d["config"].update(stage="s2a"),
           "ckpt_dir": lambda d: d.update(ckpt="/runs/s2a_VT/checkpoints/017220/pretrained_model"),
           "swap": lambda d: d["config"].update(instruction_swap="shuffle"),
           "n99": lambda d: d.update(episodes=d["episodes"][:99])}[what]
    lang_files(R, "c050", set(range(14)), set(range(11)), **{tr: mut})
    with pytest.raises(AssertionError):
        S.lang_s2b("c050")
