"""s1_report Stage 2 modes (val_s2 / final_s2): bookkeeping assertions, confusion mapping, cube-count breakdown, gate edges."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vla_common.catalog import ALL_CUBES

SPEC = importlib.util.spec_from_file_location("s1_report", Path(__file__).resolve().parents[1] / "scripts" / "s1_report.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

RS, RL, GS, GL, BS, BL, YS, YL = range(8)  # ALL_CUBES indices
PICK = "s2_U"


def spec(target: int, others: list[int]) -> str:
    cubes = [list(ALL_CUBES[c]) for c in others + [target]]  # deliberately NOT in catalog order, target last
    return json.dumps({"task": "pick", "target": list(ALL_CUBES[target]), "dest": None, "cubes": cubes,
                       "pads": ["red", "green", "blue", "yellow"]})


def episode(i: int, ok: bool = True, wg: bool = False, wcube: int | None = None, n_cubes: int = 3, target: int = RS,
            others: list[int] | None = None, yaw: float = 3.0) -> dict:
    others = others if others is not None else [GS, BL, YS, YL][: n_cubes - 1]
    e = {"index": i, "target": "_".join(ALL_CUBES[target]), "n_cubes": n_cubes, "spec": spec(target, others),
         "success": ok, "wrong_grasp": wg, "slam": False, "jam": False, "drop": False, "early_close": False,
         "spawn_yaw_fold_deg": yaw, "grasp_confirmed": True,
         **{k: False for k in S.FAILS if k not in ("success", "wrong_grasp", "slam", "jam", "drop", "early_close")}}
    if wcube is not None:
        e["wrong_grasp_cube"] = wcube
    return e


def ev(seed: int, eps: list[dict], step: str, split: str = "train", stage: str = "s2", m: float = 15.0,
       sampling: str = "default", h: int = 224) -> dict:
    return {"config": {"n": 10, "image_size": h, "seed": seed, "stage": stage, "split": split, "cube_yaw_max_deg": m},
            "server_info": {"sampling": sampling}, "ckpt": f"/runs/{PICK}/checkpoints/{step}/pretrained_model",
            "ckpt_sha": "abc", "episodes": eps, "smart_length": 1000, "bars": [], "latency": {"server_s_p50": 2.0},
            "gpu_h": 0.1, "aggregate": {"pick": {k: 0.0 for k in ("success", "slam", "jam", "drop", "early_close")}}}


def eps_n(n: int, ok: int) -> list[dict]:
    return [episode(i, ok=i < ok) for i in range(n)]


def put(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    monkeypatch.setattr(S, "RAW_S2", tmp_path / "s2")
    return tmp_path


def val_files(raw: Path, ok=(150, 180, 210)) -> None:
    for (c, step), k in zip(S.CKPTS_S2.items(), ok):
        put(raw / "s2" / f"s2_U_val_{c}_s21620.json", ev(21620, eps_n(300, k), step))
    put(raw / "s2" / "vloss_008610.json", {"val_loss": 1.0})
    put(raw / "s2" / "vloss_017220.json", {"val_loss": 1.5})
    put(raw / "s2" / "vloss_034440.json", {"val_loss": 1.2})


def test_val_s2_pick_and_abort_rules(raw):
    val_files(raw)
    S.val_s2()
    v = json.loads((raw / "s2_val.json").read_text())
    assert v["pick"] == "c100" and v["pick_step"] == "034440" and v["cube_yaw_max_deg"] == 15.0
    assert v["early_abort_c"] == {"loss_25": 1.0, "loss_50": 1.5, "abort": False} and v["early_abort_d"]["abort"] is False
    assert v["per_ckpt"]["c050"]["wrong_grasp"]["count"] == 0 and v["per_ckpt"]["c100"]["step"] == "034440"
    put(raw / "s2" / "vloss_017220.json", {"val_loss": 2.01})  # (c): > 2 x the 25 % loss
    S.val_s2()
    assert json.loads((raw / "s2_val.json").read_text())["early_abort_c"]["abort"] is True
    val_files(raw, ok=(15, 180, 210))  # (d): s_50 <= 5 %
    S.val_s2()
    assert json.loads((raw / "s2_val.json").read_text())["early_abort_d"]["abort"] is True
    val_files(raw, ok=(250, 270, 150))  # collapse: s_100 < max - 10 points -> the later of 50 / 75 within 5 points
    S.val_s2()
    v = json.loads((raw / "s2_val.json").read_text())
    assert v["pick"] == "c075" and v["pick_300"]["collapsed"] is True


@pytest.mark.parametrize("kw,what", [
    (dict(seed=21621), "seed"), (dict(seed=21620, m=0.0), "m"), (dict(seed=21620, stage="s1"), "stage"),
    (dict(seed=21620, split="heldout_comp"), "split"), (dict(seed=21620, sampling="mean"), "sampling"),
    (dict(seed=21620, h=128), "H")])
def test_val_s2_assertions_fire(raw, kw, what):
    val_files(raw)
    put(raw / "s2" / "s2_U_val_c050_s21620.json", ev(kw.pop("seed"), eps_n(300, 150), S.CKPTS_S2["c050"], **kw))
    with pytest.raises(AssertionError):
        S.val_s2()


def test_val_s2_wrong_ckpt_and_yaw_fire(raw):
    val_files(raw)
    put(raw / "s2" / "s2_U_val_c075_s21620.json", ev(21620, eps_n(300, 180), "017220"))  # c075 = 025830
    with pytest.raises(AssertionError):
        S.val_s2()
    val_files(raw)
    eps = eps_n(300, 150)
    eps[7]["spawn_yaw_fold_deg"] = 15.6  # outside +-15.5
    put(raw / "s2" / "s2_U_val_c050_s21620.json", ev(21620, eps, "017220"))
    with pytest.raises(AssertionError):
        S.val_s2()


def test_confusion_mapping_uses_catalog_index_not_spec_order():
    ct = S.confusion_type
    assert ct(["red", "small"], GS, json.loads(spec(RS, [GS, BL]))["cubes"]) == "other_colour_same_size"
    assert ct(["red", "small"], RL, json.loads(spec(RS, [RL, BL]))["cubes"]) == "same_colour_other_size"
    assert ct(["red", "small"], BL, json.loads(spec(RS, [GS, BL]))["cubes"]) == "other"
    assert ct(["blue", "large"], BS, json.loads(spec(BL, [BS]))["cubes"]) == "same_colour_other_size"
    assert ct(["blue", "large"], YL, json.loads(spec(BL, [YL]))["cubes"]) == "other_colour_same_size"
    assert ct(["blue", "large"], RS, json.loads(spec(BL, [RS]))["cubes"]) == "other"
    assert ct(["red", "small"], None, json.loads(spec(RS, [GS]))["cubes"]) == "unknown"
    with pytest.raises(AssertionError):  # a cube that is not in the episode's scene, or the target itself
        ct(["red", "small"], YS, json.loads(spec(RS, [GS]))["cubes"])
    with pytest.raises(AssertionError):
        ct(["red", "small"], RS, json.loads(spec(RS, [GS]))["cubes"])


def test_confusion_counts_and_legacy_unknown():
    eps = [episode(0, ok=False, wg=True, wcube=RL, others=[RL, GS]), episode(1, ok=False, wg=True, wcube=GS, others=[GS]),
           episode(2, ok=False, wg=True, wcube=BL, others=[BL]), episode(3), episode(4, ok=False, wg=True, others=[GS])]
    c = S.confusion(eps)
    assert c == {"wrong_grasp_total": 4, "same_colour_other_size": 1, "other_colour_same_size": 1, "other": 1,
                 "unknown": 1}
    with pytest.raises(AssertionError):  # a cube recorded without a wrong grasp
        S.confusion([episode(0, wg=False, wcube=GS)])


def test_by_n_cubes():
    eps = [episode(i, ok=i % 2 == 0, n_cubes=2 + i % 4) for i in range(40)]
    b = S.by_n_cubes(eps)
    assert list(b) == ["2", "3", "4", "5"] and all(v["success"]["n"] == 10 for v in b.values())
    assert b["2"]["success"]["count"] == 10 and b["3"]["success"]["count"] == 0  # i % 4 == 0 even, == 1 odd


def test_stats_check(tmp_path):
    from safetensors.numpy import save_file
    st = {"observation.state": {"mean": [1.0, 2.0], "std": [0.5, 0.25]}, "action": {"mean": [0.1], "std": [0.7]}}

    def write(mean_a=0.1):
        d = tmp_path / "ck"
        d.mkdir(exist_ok=True)
        f32 = lambda x: np.array(x, np.float32)  # noqa: E731
        pre = {"observation.state.mean": f32([1, 2]), "observation.state.std": f32([0.5, 0.25]),
               "action.mean": f32([mean_a]), "action.std": f32([0.7])}
        post = {"action.mean": f32([0.1]), "action.std": f32([0.7])}
        save_file(pre, str(d / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
        save_file(post, str(d / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        return str(d)

    st32 = json.loads(json.dumps(st))
    st32["action"]["mean"] = np.array([0.1], np.float32).tolist()
    st32["action"]["std"] = np.array([0.7], np.float32).tolist()
    assert S.stats_check(None, st32) == "not run (checkpoint not synced)"
    assert S.stats_check(write(), st32)["equal"] is True
    with pytest.raises(AssertionError):
        S.stats_check(write(mean_a=0.1000001), st32)


def final_setup(raw: Path, ok: int, wg: int, step: str = "034440", n: int = 1000) -> None:
    put(raw / "s2_val.json", {"pick": "c100", "pick_step": step})
    eps = [episode(i, ok=i < ok, wg=(i >= n - wg), wcube=GS if i >= n - wg else None, n_cubes=2 + i % 4,
                   others=[GS, BL, YS, YL][: 1 + i % 4]) for i in range(n)]
    put(raw / "s2_test_s11520.json", ev(11520, eps, step))
    put(raw / "s2" / "train_summary_U.json", {"norm_stats": {}})


@pytest.mark.parametrize("ok,wg,gate", [(850, 49, True), (849, 49, False), (850, 50, False), (851, 0, True),
                                        (1000, 51, False), (849, 50, False), (850, 0, True)])
def test_final_s2_gate_edges(raw, ok, wg, gate):
    final_setup(raw, ok, wg)
    S.final_s2()
    f = json.loads((raw / "s2.json").read_text())
    assert f["gate_pass"] is gate and f["success"]["count"] == ok and f["wrong_grasp"]["count"] == wg
    assert f["stats_check"] == "not run (checkpoint not synced)"
    assert f["wrong_cube_confusion"]["other_colour_same_size"] == wg  # GS is other colour, same size as the red_small target
    assert set(f["by_n_cubes"]) == {"2", "3", "4", "5"} and sum(v["success"]["n"] for v in f["by_n_cubes"].values()) == 1000
    assert set(f["by"]) == {"colour", "size", "cube"} and "heldout" not in f


def test_final_s2_assertions_fire(raw):
    final_setup(raw, 900, 10)
    good = json.loads((raw / "s2_test_s11520.json").read_text())
    for mutate in (lambda d: d["config"].update(seed=11521), lambda d: d["config"].update(stage="s1"),
                   lambda d: d["config"].update(split="heldout_comp"), lambda d: d["config"].update(cube_yaw_max_deg=0.0),
                   lambda d: d.update(episodes=d["episodes"][:999]),
                   lambda d: d.update(ckpt="/runs/s2_U/checkpoints/025830/pretrained_model"),  # not the val pick
                   lambda d: d["episodes"][3].update(spawn_yaw_fold_deg=20.0)):
        d = json.loads(json.dumps(good))
        mutate(d)
        put(raw / "s2_test_s11520.json", d)
        with pytest.raises(AssertionError):
            S.final_s2()


def test_final_s2_heldout(raw):
    final_setup(raw, 900, 10)
    he = [episode(i, ok=i < 240, wg=i >= 290, wcube=BL if i >= 290 else None, target=GL, others=[BL, YS]) for i in range(300)]
    put(raw / "s2_heldout_s11530.json", ev(11530, he, "034440", split="heldout_comp"))
    S.final_s2()
    h = json.loads((raw / "s2.json").read_text())["heldout"]
    assert h["n"] == 300 and h["success"]["count"] == 240 and h["wrong_grasp"]["count"] == 10
    assert h["wrong_cube_confusion"]["same_colour_other_size"] == 0 and h["wrong_cube_confusion"]["other"] == 0 \
        and h["wrong_cube_confusion"]["other_colour_same_size"] == 10  # BL vs green_large target: other colour, same size
    put(raw / "s2_heldout_s11530.json", ev(11530, he, "034440", split="train"))
    with pytest.raises(AssertionError):
        S.final_s2()
    put(raw / "s2_heldout_s11530.json", ev(11531, he, "034440", split="heldout_comp"))
    with pytest.raises(AssertionError):
        S.final_s2()
    put(raw / "s2_heldout_s11530.json", ev(11530, he[:299], "034440", split="heldout_comp"))
    with pytest.raises(AssertionError):
        S.final_s2()
