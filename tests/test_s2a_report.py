"""s1_report Stage 2 redesign #2 (2a) modes (val_s2a / final_s2a / lang_s2a / init_disc --expect le): synthetic fixtures + mutants."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import test_s2_report as T

S = T.S
SCRIPT = Path(S.__file__)
V15 = "c3dcef76f53e00e5954469970c25c38980a396723f7bbbf2b85919e6cc6688f5"
NORM = {"observation.state": {"mean": [1.0], "std": [0.5]}, "action": {"mean": [0.25], "std": [0.75]}}  # float32-exact
VSEED, TSEED = 21640, 11540


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    return tmp_path


def ev2(seed: int, eps: list[dict], step: str, tag: str = "s2a_VT", sha: str = "4dd2", stage: str = "s2a", **kw) -> dict:
    d = T.ev(seed, eps, step, stage=stage, **kw)
    d["ckpt"] = f"/runs/{tag}/checkpoints/{step}/pretrained_model"
    d["stats_sha"] = sha
    return d


def val_files(R: Path, ok=(150, 180, 210), shas=("4dd2", "4dd2", "4dd2")) -> None:
    for (c, step), k, sha in zip(S.CKPTS_S2.items(), ok, shas):
        T.put(R / "s2a" / f"s2a_VT_val_{c}_s{VSEED}.json", ev2(VSEED, T.eps_n(300, k), step, sha=sha))
    for step, v in (("008610", 1.0), ("017220", 1.5), ("034440", 1.2)):
        T.put(R / "s2a" / f"vloss_{step}.json", {"val_loss": v})
    T.put(R / "s2a" / "offline_val_034440.json", {"mae_arm_norm": 0.1, "claw_acc": 0.9, "per_episode": [1, 2]})


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def test_val_s2a(R):
    val_files(R)
    S.val_s2a()
    v = load(R / "s2a_val.json")
    assert v["arm"] == "VT" and v["seed"] == VSEED and v["pick"] == "c100" and v["pick_step"] == "034440"
    assert v["init_sha256"] == V15 and v["stats_sha"] == "4dd2" and "attempt" not in v
    assert v["early_abort_c"] == {"loss_25": 1.0, "loss_50": 1.5, "abort": False} and v["offline_val_100"]["episodes"] == 2
    assert v["per_ckpt"]["c050"]["ckpt"].endswith("/s2a_VT/checkpoints/017220/pretrained_model")
    T.put(R / "s2a" / "vloss_017220.json", {"val_loss": 2.01})
    S.val_s2a()
    assert load(R / "s2a_val.json")["early_abort_c"]["abort"] is True
    val_files(R, ok=(15, 180, 210))
    S.val_s2a()
    assert load(R / "s2a_val.json")["early_abort_d"]["abort"] is True
    assert not (R / "s2_val.json").exists() and not (R / "s2base_val.json").exists()


@pytest.mark.parametrize("what", ["stats_sha", "ckpt_step", "ckpt_dir", "seed", "stage", "yaw"])
def test_val_s2a_mutants_fire(R, what):
    val_files(R)
    p = R / "s2a" / f"s2a_VT_val_c075_s{VSEED}.json"
    d = load(p)
    if what == "stats_sha":
        d["stats_sha"] = "beef"
    elif what == "ckpt_step":
        d["ckpt"] = "/runs/s2a_VT/checkpoints/017220/pretrained_model"  # c075 = 025830
    elif what == "ckpt_dir":
        d["ckpt"] = "/runs/s2_U/checkpoints/025830/pretrained_model"
    elif what == "seed":
        d["config"]["seed"] = 21620
    elif what == "stage":
        d["config"]["stage"] = "s2"
    else:
        d["config"]["cube_yaw_max_deg"] = 180.0
    T.put(p, d)
    with pytest.raises(AssertionError):
        S.val_s2a()


def final_setup(R: Path, ok: int = 900, wg: int = 10, sha: str = "4dd2", init_sha: str = V15) -> None:
    val_files(R)
    T.put(R / "s2a_val.json", {"pick": "c100", "pick_step": "034440"})
    eps = [T.episode(i, ok=i < ok, wg=(i >= 1000 - wg), wcube=T.GS if i >= 1000 - wg else None, n_cubes=2 + i % 3,
                     others=[T.GS, T.BL, T.YS][: 1 + i % 3]) for i in range(1000)]
    T.put(R / f"s2a_test_s{TSEED}.json", ev2(TSEED, eps, "034440", sha=sha))
    T.put(R / "s2a" / "train_summary_VT.json", {"init": "/runs/s1y15_U/checkpoints/041328/pretrained_model",
                                                "init_sha256": init_sha, "argv": {}, "norm_stats": NORM})


@pytest.mark.parametrize("ok,wg,gate", [(850, 49, True), (849, 49, False), (850, 50, False), (1000, 0, True)])
def test_final_s2a_gate_and_instruments(R, ok, wg, gate):
    final_setup(R, ok, wg)
    S.final_s2a()
    f = load(R / "s2a.json")
    assert f["gate_pass"] is gate and f["success"]["count"] == ok and f["wrong_grasp"]["count"] == wg
    assert f["seed"] == TSEED and f["arm"] == "VT" and f["n"] == 1000 and f["pick_step"] == "034440"
    assert f["init_check"] == {"init": "/runs/s1y15_U/checkpoints/041328/pretrained_model", "init_sha256": V15,
                               "is_stage1_v15_041328": True}
    assert f["stats_sha"]["equal_across_evals"] is True and f["stats_sha"]["train_run"].startswith("not run")
    assert f["stats_check"] == "not run (checkpoint not synced)" and "heldout" not in f and "i3_norm" not in f
    assert set(f["by_n_cubes"]) == {"2", "3", "4"} and f["wrong_cube_confusion"]["wrong_grasp_total"] == wg
    assert not (R / "s2.json").exists() and not (R / "s2base.json").exists()


def normalizer_files(d: Path) -> str:
    """A checkpoint dir with the trainer's normalizer files holding NORM; returns eval_policy.stats_sha's value."""
    safetensors = pytest.importorskip("safetensors.numpy")
    import numpy as np
    d.mkdir(parents=True)
    f = lambda m: np.array(m, np.float32)  # noqa: E731
    pre = {f"{k}.{s}": f(NORM[k][s]) for k in NORM for s in ("mean", "std")}
    safetensors.save_file(pre, str(d / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    safetensors.save_file({f"action.{s}": f(NORM["action"][s]) for s in ("mean", "std")},
                          str(d / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    h = hashlib.sha256()
    for p in sorted(d.glob("*normalizer*.safetensors")):  # the same glob the eval harness uses
        h.update(p.read_bytes())
    return h.hexdigest()


def test_final_s2a_stats_sha_against_the_training_run(R, tmp_path):
    ck = tmp_path / "ck"
    sha = normalizer_files(ck)
    final_setup(R)
    val_files(R, shas=(sha,) * 3)
    t = load(R / f"s2a_test_s{TSEED}.json")
    t["stats_sha"] = sha
    T.put(R / f"s2a_test_s{TSEED}.json", t)
    S.final_s2a("--ckpt-dir", str(ck))
    f = load(R / "s2a.json")
    assert f["stats_sha"]["train_run"] == sha and f["stats_sha"]["equal_train_run"] is True
    assert f["stats_check"]["equal"] is True
    t["stats_sha"] = "beef"  # evals agree with each other but not with the training run
    T.put(R / f"s2a_test_s{TSEED}.json", t)
    val_files(R, shas=("beef",) * 3)
    with pytest.raises(AssertionError, match="training run"):
        S.final_s2a("--ckpt-dir", str(ck))


@pytest.mark.parametrize("what", ["init_sha", "test_sha", "val_sha", "seed_other_arm", "ckpt_dir", "ckpt_step", "stage", "norm_ckpt"])
def test_final_s2a_mutants_fire(R, tmp_path, what):
    final_setup(R)
    tp = R / f"s2a_test_s{TSEED}.json"
    t = load(tp)
    args: tuple = ()
    if what == "init_sha":
        final_setup(R, init_sha=hashlib.sha256(b"smolvla_base").hexdigest())
    elif what == "test_sha":
        t["stats_sha"] = "beef"
        T.put(tp, t)
    elif what == "val_sha":
        val_files(R, shas=("4dd2", "beef", "4dd2"))
    elif what == "seed_other_arm":
        t["config"]["seed"] = 11520
        T.put(tp, t)
    elif what == "ckpt_dir":
        t["ckpt"] = "/runs/s2base_U/checkpoints/034440/pretrained_model"
        T.put(tp, t)
    elif what == "ckpt_step":
        t["ckpt"] = "/runs/s2a_VT/checkpoints/025830/pretrained_model"
        T.put(tp, t)
    elif what == "stage":
        t["config"]["stage"] = "s2"
        T.put(tp, t)
    else:  # the checkpoint's normalizer stats differ from the training summary's
        ck = tmp_path / "ck"
        normalizer_files(ck)
        d = load(R / "s2a" / "train_summary_VT.json")
        d["norm_stats"]["action"]["std"] = [0.5]
        T.put(R / "s2a" / "train_summary_VT.json", d)
        args = ("--ckpt-dir", str(ck))
    with pytest.raises(AssertionError):
        S.final_s2a(*args)


# ------------------------------------------------------------------------------------------------ language check
def lang_ep(i: int, ok: bool, wg: bool, shuffled: bool) -> dict:
    e = T.episode(i, ok=ok, wg=wg, wcube=T.GS if wg else None)
    e["instruction"] = f"pick up cube {i}"
    if shuffled:
        e["instruction_sent"] = f"pick up cube {(i + 1) % 100}"
    return e


def lang_files(R: Path, ck: str, true_ok: set, shuf_ok: set, wg_t: int = 0, wg_s: int = 0, **mut) -> None:
    step = S.CKPTS_S2[ck]
    for mode, name, ok, wg in (("none", "true", true_ok, wg_t), ("shuffle", "shuffle", shuf_ok, wg_s)):
        eps = [lang_ep(i, i in ok, i < wg, mode == "shuffle") for i in range(100)]
        d = ev2(VSEED, eps, step)
        d["config"]["instruction_swap"] = mode
        if mode == "shuffle":
            d["instruction_swap_report"] = {"donor_of": {str(i): (i + 1) % 100 for i in range(100)}}
        T.put(R / "s2a_lang" / f"s2a_VT_{ck}_{name}_s{VSEED}.json", d)
    for f, fn in mut.items():
        p = R / "s2a_lang" / f
        d = load(p)
        fn(d)
        T.put(p, d)


def test_lang_s2a_numbers(R):
    lang_files(R, "c050", set(range(14)), set(range(9)) | {50, 51}, wg_t=23, wg_s=25)  # 14 vs 11: b = 5, c = 2
    S.lang_s2a("c050")
    r = load(R / "s2a_lang_c050.json")
    p = r["pair"]
    assert (p["true_success"], p["shuffled_success"], p["b_true_only"], p["c_shuffled_only"]) == (14, 11, 5, 2)
    assert p["mcnemar_z"] == pytest.approx(3 / 7 ** 0.5) and (p["wrong_grasp_true"], p["wrong_grasp_shuffled"]) == (23, 25)
    assert r["seed"] == VSEED and r["step"] == "017220" and p["ckpt"].endswith("/s2a_VT/checkpoints/017220/pretrained_model")
    assert "attempt0" not in r and "gate_pass" not in r
    lang_files(R, "c100", set(range(10)), set(range(10)))
    S.lang_s2a("c100")
    assert load(R / "s2a_lang_c100.json")["pair"]["mcnemar_z"] is None
    with pytest.raises(AssertionError):
        S.lang_s2a("c075")


@pytest.mark.parametrize("what", ["order", "swap_true", "swap_shuffle", "ckpt", "ckpt_other_dir", "n99", "seed", "stage",
                                  "own_instruction", "report_missing"])
def test_lang_s2a_mutants_fire(R, what):
    sh, tr = f"s2a_VT_c050_shuffle_s{VSEED}.json", f"s2a_VT_c050_true_s{VSEED}.json"
    mut = {
        "order": {sh: lambda d: d["episodes"].reverse()},
        "swap_true": {tr: lambda d: d["config"].update(instruction_swap="shuffle")},
        "swap_shuffle": {sh: lambda d: d["config"].update(instruction_swap="none")},
        "ckpt": {tr: lambda d: d.update(ckpt="/runs/s2a_VT/checkpoints/034440/pretrained_model")},
        "ckpt_other_dir": {sh: lambda d: d.update(ckpt="/runs/s2_U/checkpoints/017220/pretrained_model")},
        "n99": {tr: lambda d: d.update(episodes=d["episodes"][:99])},
        "seed": {tr: lambda d: d["config"].update(seed=21620)},
        "stage": {tr: lambda d: d["config"].update(stage="s2")},
        "own_instruction": {sh: lambda d: d["episodes"][3].update(instruction_sent=d["episodes"][3]["instruction"])},
        "report_missing": {sh: lambda d: d.pop("instruction_swap_report")},
    }[what]
    lang_files(R, "c050", set(range(14)), set(range(11)), **mut)
    with pytest.raises(AssertionError):
        S.lang_s2a("c050")


# ------------------------------------------------------------------------------------------------ I2 --expect
def steps_file(p: Path, mean: float, rows: int = 50) -> Path:
    p.write_text("\n".join(json.dumps({"step": 1 + i, "loss": mean + (0.1 if i % 2 else -0.1)}) for i in range(rows)) + "\n"
                 + json.dumps({"step": 51, "loss": 99.0}) + "\n")  # step 51 must not count
    return p


def run_disc(p: Path, *a: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "init_disc", str(p), *a], capture_output=True, text=True)


@pytest.mark.parametrize("mean,expect,code,word", [
    (0.241, "le", 0, "Stage-1-checkpoint init confirmed"), (0.5, "le", 0, "Stage-1-checkpoint init confirmed"),  # <= is inclusive
    (0.5001, "le", 1, "INSTRUMENT FINDING"), (0.731, "le", 1, "INSTRUMENT FINDING"),
    (0.731, "gt", 0, "base init confirmed"), (0.241, "gt", 1, "INSTRUMENT FINDING")])
def test_init_disc_expect(tmp_path, mean, expect, code, word):
    p = steps_file(tmp_path / "steps.jsonl", mean)
    r = run_disc(p, "--expect", expect)
    assert r.returncode == code and word in r.stdout and f"{mean:.4f}" in r.stdout, (r.stdout, r.stderr)
    assert S.init_disc(str(p), "--expect", expect) == code


def test_init_disc_default_is_gt_and_threshold_and_bad_expect(tmp_path):
    p = steps_file(tmp_path / "steps.jsonl", 0.731)
    assert run_disc(p).returncode == 0 and "base init confirmed" in run_disc(p).stdout  # default unchanged
    assert run_disc(p, "--expect", "le", "--ref-threshold", "0.8").returncode == 0
    assert run_disc(p, "--expect", "lt").returncode != 0
