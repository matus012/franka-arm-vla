"""s1_report Stage 2 redesign (a) modes (val_s2base / final_s2base / lang_s2base / init_disc): synthetic fixtures + mutants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import test_s2_report as T

S = T.S
SCRIPT = Path(S.__file__)
BASE = "lerobot/smolvla_base"
NORM = {"observation.state": {"mean": [1.0], "std": [0.5]}, "action": {"mean": [0.1], "std": [0.7]}}


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    return tmp_path


def evb(seed: int, eps: list[dict], step: str, tag: str = "s2base", sha: str = "4dd2", **kw) -> dict:
    d = T.ev(seed, eps, step, **kw)
    d["ckpt"] = f"/runs/{tag}_U/checkpoints/{step}/pretrained_model"
    d["stats_sha"] = sha
    return d


def val_files(R: Path, ok=(150, 180, 210), tag: str = "s2base") -> None:
    for (c, step), k in zip(S.CKPTS_S2.items(), ok):
        T.put(R / tag / f"{tag}_U_val_{c}_s21620.json", evb(21620, T.eps_n(300, k), step, tag))
    for step, v in (("008610", 1.0), ("017220", 1.5), ("034440", 1.2)):
        T.put(R / tag / f"vloss_{step}.json", {"val_loss": v})
    T.put(R / tag / "offline_val_034440.json", {"mae_arm_norm": 0.1, "claw_acc": 0.9, "per_episode": [1, 2]})


def test_val_s2base(R):
    val_files(R)
    S.val_s2base()
    v = json.loads((R / "s2base_val.json").read_text())
    assert v["init"] == BASE and v["attempt"] == 1 and v["pick"] == "c100" and v["pick_step"] == "034440"
    assert v["early_abort_c"] == {"loss_25": 1.0, "loss_50": 1.5, "abort": False} and v["offline_val_100"]["episodes"] == 2
    assert v["per_ckpt"]["c050"]["ckpt"].endswith("/s2base_U/checkpoints/017220/pretrained_model")
    T.put(R / "s2base" / "vloss_017220.json", {"val_loss": 2.01})
    S.val_s2base()
    assert json.loads((R / "s2base_val.json").read_text())["early_abort_c"]["abort"] is True
    val_files(R, ok=(15, 180, 210))
    S.val_s2base()
    assert json.loads((R / "s2base_val.json").read_text())["early_abort_d"]["abort"] is True
    assert not (R / "s2_val.json").exists()  # attempt 0's output is never touched


def test_val_s2base_wrong_ckpt_fires(R):
    val_files(R)
    T.put(R / "s2base" / "s2base_U_val_c075_s21620.json", evb(21620, T.eps_n(300, 180), "017220"))  # c075 = 025830
    with pytest.raises(AssertionError):
        S.val_s2base()
    val_files(R)
    T.put(R / "s2base" / "s2base_U_val_c050_s21620.json", evb(21620, T.eps_n(300, 150), "017220", tag="s2"))  # attempt 0's run dir
    with pytest.raises(AssertionError):
        S.val_s2base()


def final_setup(R: Path, ok: int = 900, wg: int = 10, heldout: bool = False, a0_sha: str = "4dd2") -> None:
    T.put(R / "s2base_val.json", {"pick": "c100", "pick_step": "034440"})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 4,
                     others=[T.GS, T.BL, T.YS, T.YL][: 1 + i % 4]) for i in range(1000)]
    T.put(R / "s2base_test_s11521.json", evb(11521, eps, "034440"))
    T.put(R / "s2base" / "train_summary_U.json", {"argv": {"init": BASE}, "norm_stats": NORM})
    T.put(R / "s2" / "train_summary_U.json", {"argv": {"init": "/runs/s1y15_U/x"}, "norm_stats": NORM})
    T.put(R / "s2_test_s11520.json", evb(11520, eps, "034440", tag="s2", sha=a0_sha))
    if heldout:
        he = [T.episode(i, ok=i < 240, target=T.GL, others=[T.BL, T.YS]) for i in range(300)]
        T.put(R / "s2base_heldout_s11531.json", evb(11531, he, "034440", split="heldout_comp"))


def load(p: Path) -> dict:
    return json.loads(p.read_text())


@pytest.mark.parametrize("ok,wg,gate", [(850, 49, True), (849, 49, False), (850, 50, False), (1000, 0, True)])
def test_final_s2base_gate_and_instruments(R, ok, wg, gate):
    final_setup(R, ok, wg, heldout=True)
    S.final_s2base()
    f = load(R / "s2base.json")
    assert f["gate_pass"] is gate and f["success"]["count"] == ok and f["wrong_grasp"]["count"] == wg
    assert f["seed"] == 11521 and f["init"] == BASE and f["attempt"] == 1
    assert f["i1_init"] == {"init": BASE, "is_smolvla_base": True}
    assert f["i3_norm"]["norm_stats_equal_attempt0"] is True and f["i3_norm"]["test_stats_sha_equal_attempt0"] is True
    assert f["i3_norm"]["heldout_stats_sha_equal_attempt0"] is True
    assert f["heldout"]["n"] == 300 and f["stats_check"] == "not run (checkpoint not synced)"
    assert not (R / "s2.json").exists()


@pytest.mark.parametrize("what", ["norm", "test_sha", "heldout_sha", "init", "seed_attempt0", "ckpt_dir", "ckpt_step", "heldout_seed"])
def test_final_s2base_mutants_fire(R, what):
    final_setup(R, heldout=True)
    t, h = load(R / "s2base_test_s11521.json"), load(R / "s2base_heldout_s11531.json")
    if what == "norm":
        d = load(R / "s2base" / "train_summary_U.json")
        d["norm_stats"]["action"]["std"] = [0.7000001]
        T.put(R / "s2base" / "train_summary_U.json", d)
    elif what == "test_sha":
        t["stats_sha"] = "beef"
        T.put(R / "s2base_test_s11521.json", t)
    elif what == "heldout_sha":
        h["stats_sha"] = "beef"
        T.put(R / "s2base_heldout_s11531.json", h)
    elif what == "init":
        T.put(R / "s2base" / "train_summary_U.json", {"argv": {"init": "/runs/s1y15_U/x"}, "norm_stats": NORM})
    elif what == "seed_attempt0":
        t["config"]["seed"] = 11520
        T.put(R / "s2base_test_s11521.json", t)
    elif what == "ckpt_dir":
        t["ckpt"] = "/runs/s2_U/checkpoints/034440/pretrained_model"
        T.put(R / "s2base_test_s11521.json", t)
    elif what == "ckpt_step":
        t["ckpt"] = "/runs/s2base_U/checkpoints/025830/pretrained_model"
        T.put(R / "s2base_test_s11521.json", t)
    else:
        h["config"]["seed"] = 11530
        T.put(R / "s2base_heldout_s11531.json", h)
    with pytest.raises(AssertionError):
        S.final_s2base()


def test_final_s2_attempt0_unchanged_by_run_config(R):
    """final_s2 default run still reads/writes attempt 0's names and carries no redesign fields."""
    T.put(R / "s2_val.json", {"pick": "c100", "pick_step": "034440"})
    eps = [T.episode(i, ok=i < 900) for i in range(1000)]
    T.put(R / "s2_test_s11520.json", T.ev(11520, eps, "034440"))
    T.put(R / "s2" / "train_summary_U.json", {"norm_stats": {}})
    S.final_s2()
    f = load(R / "s2.json")
    assert f["seed"] == 11520 and "init" not in f and "i3_norm" not in f and f["design"] == "Stage 2 (s2 / train, cube yaw +-15 deg)"


# ------------------------------------------------------------------------------------------------ language check
def lang_ep(i: int, ok: bool, wg: bool, shuffled: bool) -> dict:
    e = T.episode(i, ok=ok, wg=wg, wcube=T.GS if wg else None)
    e["instruction"] = f"pick up cube {i}"
    if shuffled:
        e["instruction_sent"] = f"pick up cube {(i + 1) % 100}"
    return e


def lang_files(R: Path, run: str, ck: str, true_ok: set, shuf_ok: set, wg_t: int = 0, wg_s: int = 0, **mut) -> None:
    tag = "s2" if run == "a0" else "s2base"
    step = S.CKPTS_S2[ck]
    for mode, name, ok, wg in (("none", "true", true_ok, wg_t), ("shuffle", "shuffle", shuf_ok, wg_s)):
        eps = [lang_ep(i, i in ok, i < wg, mode == "shuffle") for i in range(100)]
        d = evb(21620, eps, step, tag=tag, stage="s2")
        d["config"]["instruction_swap"] = mode
        if mode == "shuffle":
            d["instruction_swap_report"] = {"donor_of": {str(i): (i + 1) % 100 for i in range(100)}}
        T.put(R / f"{tag}_lang" / f"{tag}_U_{ck}_{name}_s21620.json", d)
    for f, fn in mut.items():  # mutate a written file: {(run-agnostic) filename: fn(d)}
        p = R / f"{tag}_lang" / f
        d = load(p)
        fn(d)
        T.put(p, d)


def both(R: Path, ck: str = "c050", **mut):
    t1, s1 = set(range(14)), set(range(9)) | {50, 51}  # 14 vs 11: b = 5 (9..13), c = 2 (50, 51)
    t0, s0 = set(range(28)), set(range(25)) | {60, 61, 62}  # 28 vs 28: b = 3, c = 3
    lang_files(R, "a1", ck, t1, s1, wg_t=23, wg_s=25, **mut)
    lang_files(R, "a0", ck, t0, s0, wg_t=43, wg_s=42)


def test_lang_s2base_numbers(R):
    both(R)
    S.lang_s2base("c050")
    r = load(R / "s2base_lang_c050.json")
    a1, a0 = r["attempt1"], r["attempt0"]
    assert (a1["true_success"], a1["shuffled_success"], a1["b_true_only"], a1["c_shuffled_only"]) == (14, 11, 5, 2)
    assert a1["mcnemar_z"] == pytest.approx(3 / 7 ** 0.5) and (a1["wrong_grasp_true"], a1["wrong_grasp_shuffled"]) == (23, 25)
    assert (a0["true_success"], a0["shuffled_success"], a0["b_true_only"], a0["c_shuffled_only"], a0["mcnemar_z"]) == (28, 28, 3, 3, 0.0)
    assert r["same_donor_permutation_as_attempt0"] is True and "gate_pass" not in r
    assert a1["ckpt"].endswith("/s2base_U/checkpoints/017220/pretrained_model") and a0["ckpt"].endswith("/s2_U/checkpoints/017220/pretrained_model")


def test_lang_z_none_when_no_discordant(R):
    same = set(range(10))
    lang_files(R, "a1", "c100", same, same)
    lang_files(R, "a0", "c100", same, same)
    S.lang_s2base("c100")
    assert load(R / "s2base_lang_c100.json")["attempt1"]["mcnemar_z"] is None


@pytest.mark.parametrize("what", ["order", "swap_true", "swap_shuffle", "ckpt", "ckpt_attempt0_dir", "n99", "seed", "own_instruction",
                                  "donor", "sent", "report_missing"])
def test_lang_mutants_fire(R, what):
    sh, tr = "s2base_U_c050_shuffle_s21620.json", "s2base_U_c050_true_s21620.json"
    mut = {
        "order": {sh: lambda d: d["episodes"].reverse()},
        "swap_true": {tr: lambda d: d["config"].update(instruction_swap="shuffle")},
        "swap_shuffle": {sh: lambda d: d["config"].update(instruction_swap="none")},
        "ckpt": {tr: lambda d: d.update(ckpt="/runs/s2base_U/checkpoints/034440/pretrained_model")},
        "ckpt_attempt0_dir": {sh: lambda d: d.update(ckpt="/runs/s2_U/checkpoints/017220/pretrained_model")},
        "n99": {tr: lambda d: d.update(episodes=d["episodes"][:99])},
        "seed": {tr: lambda d: d["config"].update(seed=21621)},
        "own_instruction": {sh: lambda d: d["episodes"][3].update(instruction_sent=d["episodes"][3]["instruction"])},
        "donor": {sh: lambda d: d["instruction_swap_report"]["donor_of"].update({"0": 7})},
        "sent": {sh: lambda d: d["episodes"][3].update(instruction_sent="pick up cube 99")},
        "report_missing": {sh: lambda d: d.pop("instruction_swap_report")},
    }[what]
    both(R, **mut)
    with pytest.raises(AssertionError):
        S.lang_s2base("c050")


def test_lang_attempt0_committed_numbers():
    """The committed attempt-0 files reproduce the numbers in GATES (c050 14 vs 11, 5/2, z 1.13; c100 28 vs 28, 3/3, z 0.0)."""
    for ck, k, b, c, z in (("c050", (14, 11), 5, 2, 1.13), ("c100", (28, 28), 3, 3, 0.0)):
        p = S.lang_pair("s2", ck)
        assert (p["true_success"], p["shuffled_success"], p["b_true_only"], p["c_shuffled_only"]) == (*k, b, c)
        assert round(p["mcnemar_z"], 2) == z


# ------------------------------------------------------------------------------------------------ I2
def steps_file(p: Path, mean: float, rows: int = 50, start: int = 1) -> Path:
    p.write_text("\n".join(json.dumps({"step": start + i, "loss": mean + (0.1 if i % 2 else -0.1)}) for i in range(rows)) + "\n"
                 + json.dumps({"step": 51, "loss": 99.0}) + "\n")  # step 51 must not count
    return p


def run_disc(p: Path, *a: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "init_disc", str(p), *a], capture_output=True, text=True)


@pytest.mark.parametrize("mean,code,word", [(0.731, 0, "base init confirmed"), (0.241, 1, "INSTRUMENT FINDING"),
                                            (0.5001, 0, "base init confirmed")])
def test_init_disc(tmp_path, mean, code, word):
    r = run_disc(steps_file(tmp_path / "steps.jsonl", mean))
    assert r.returncode == code and word in r.stdout and f"{mean:.4f}" in r.stdout, (r.stdout, r.stderr)
    assert S.init_disc(str(tmp_path / "steps.jsonl")) == code


def test_init_disc_threshold_and_bad_file(tmp_path):
    p = steps_file(tmp_path / "steps.jsonl", 0.731)
    assert run_disc(p, "--ref-threshold", "0.8").returncode == 1
    assert run_disc(steps_file(tmp_path / "short.jsonl", 0.731, rows=49)).returncode != 0  # not 50 rows with step 1..50
    assert run_disc(steps_file(tmp_path / "late.jsonl", 0.731, start=2)).returncode != 0
