"""s1_report task 009b modes (one cube size per scene, runs a1p / b1p; expert_s3 precondition): synthetic JSONs, no simulator.
Helpers (episode / ev / eps_k / put) come from test_s3_report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_s3_report import S, ev, eps_k, episode, put

INIT_2M = "e4741cf66cbbb57686df5a2ae7948e46a3a35ac65bc8c79bb48a598a32ca0f64"
ONE = ("a1p", "b1p")


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    return tmp_path


def test_one_size_runs_registered():
    for w, st, task, vs, ts, bar, wrong in (("a1p", "s3a1", "place", 21665, 11580, 80, "wrong_dest"),
                                            ("b1p", "s3b1", "stack", 21695, 11600, 70, "wrong_object")):
        sp = S.S3[w]
        assert (sp["tag"], sp["stage"], sp["task"], sp["vseed"], sp["tseed"], sp["bar"], sp["wrong"], sp["arm"], sp["s2ref"]) \
            == (st + "p", st, task, vs, ts, bar, wrong, "VT", "s2m")
        assert (sp["pick_stage"], sp["pick_vseed"], sp["pick_tseed"], sp["init_sha256"]) == ("s2a", 21640, 11544, INIT_2M)
    assert S.S3["ap"]["s2ref"] == "s2bp" and "pick_vseed" not in S.S3["ap"] and "guards" not in S.S3["bp"]  # 007 entries unchanged


def val_files_one(R: Path, which: str, main=(180, 200, 220), pick=(250, 255, 260), wrong50: int = 0, slam75: int = 0,
                  pick_seed: int = 21640, pick_stage: str = "s2a") -> None:
    sp = S.S3[which]
    tag, task, st = sp["tag"], sp["task"], sp["stage"]
    for (c, step), kp, kk in zip(S.CKPTS_S3.items(), main, pick):
        eps = eps_k(task, 300, kp)
        for e in eps[kp:kp + (wrong50 if c == "c050" else 0)]:
            e.update(wrong_dest=which == "a1p", wrong_grasp=which == "b1p", wrong_object=True)
        for e in eps[:slam75 if c == "c075" else 0]:
            e["slam"] = True
        put(R / tag / f"{tag}_VT_val_{task}_{c}_s{sp['vseed']}.json", ev(task, sp["vseed"], st, eps, step, tag, arm="VT") | {"stats_sha": "S"})
        put(R / tag / f"{tag}_VT_val_pick_{c}_s{pick_seed}.json",
            ev("pick", pick_seed, pick_stage, eps_k("pick", 300, kk), step, tag, arm="VT") | {"stats_sha": "S"})
    put(R / tag / "vloss_008877.json", {"val_loss": 1.0})
    put(R / tag / "vloss_017754.json", {"val_loss": 1.4})


def run_val_one(R: Path, which: str) -> dict:
    S.val_s3(which)
    return json.loads((R / f"{S.S3[which]['tag']}_val.json").read_text())


@pytest.mark.parametrize("which", ONE)
def test_one_size_val_guards_at_c050_and_c075(R, which):
    bar = S.S3[which]["bar"]
    line = (bar - 15) * 3  # successes of 300 at bar - 15 points: 195 (place) / 165 (stack)
    val_files_one(R, which, main=(150, line, 220))
    v = run_val_one(R, which)
    assert v["pick"] == "c100" and v["seeds"]["pick"] == 21640 and "s2a pick s21640" in v["design"] and "one cube size" in v["design"]
    assert "b1" not in v["guards_c050"] and v["guards_c050"]["b2"]["fires"] is False  # no c050 success guard here
    assert v["guards_c075"]["success"]["fires"] is False and v["guards_c075"]["slam"]["fires"] is False  # exactly bar - 15 / 0 slam
    val_files_one(R, which, main=(150, line - 1, 220))
    assert run_val_one(R, which)["guards_c075"]["success"]["fires"] is True
    val_files_one(R, which, main=(150, line, 220), slam75=30)  # 10.0 % is not > 10 %
    g = run_val_one(R, which)["guards_c075"]["slam"]
    assert g["count"] == 30 and g["fires"] is False
    val_files_one(R, which, main=(150, line, 220), slam75=31)
    assert run_val_one(R, which)["guards_c075"]["slam"]["fires"] is True
    val_files_one(R, which, wrong50=30)
    assert run_val_one(R, which)["guards_c050"]["b2"]["fires"] is False
    val_files_one(R, which, wrong50=31)
    assert run_val_one(R, which)["guards_c050"]["b2"]["fires"] is True


@pytest.mark.parametrize("which", ONE)
def test_one_size_val_pick_stream_is_s2a_21640_and_collapse_rule(R, which):
    val_files_one(R, which, pick_seed=21650, pick_stage="s2")  # the 007 pick stream does not satisfy 009b
    with pytest.raises(FileNotFoundError):
        run_val_one(R, which)
    val_files_one(R, which, main=(210, 220, 150))  # task skill collapses at c100: c075 is within 5 points of the max
    v = run_val_one(R, which)
    assert v["pick_300"]["collapsed_skills"] == [S.S3[which]["task"]] and v["pick"] == "c075"
    val_files_one(R, which, pick=(270, 280, 200))  # the pick skill collapses too
    assert run_val_one(R, which)["pick_300"]["skills"]["pick"]["collapsed"] is True
    val_files_one(R, which)
    tag = S.S3[which]["tag"]
    p = R / tag / f"{tag}_VT_val_pick_c100_s21640.json"
    e = json.loads(p.read_text())
    e["config"]["stage"] = "s2"  # pick eval on the wrong sampler
    put(p, e)
    with pytest.raises(AssertionError):
        run_val_one(R, which)


def final_files_one(R: Path, which: str, k_task: int = 800, wrong: int = 0, c3: int = 917, c2: int = 967, init: str = INIT_2M,
                    ref_sha: str = INIT_2M, ref_seed: int = 11544) -> None:
    sp = S.S3[which]
    tag, task, st, step = sp["tag"], sp["task"], sp["stage"], "035508"
    put(R / f"{tag}_val.json", {"pick_step": step, "arm_tag": "VT"})
    te = [episode(i, task, ok=i < k_task, wd=(k_task <= i < k_task + wrong) and which == "a1p",
                  wg=(k_task <= i < k_task + wrong) and which == "b1p") for i in range(1000)]
    put(R / f"{tag}_test_{task}_s{sp['tseed']}.json", ev(task, sp["tseed"], st, te, step, tag, arm="VT") | {"stats_sha": "S"})
    put(R / f"{tag}_test_pick_s11544.json",
        ev("pick", 11544, "s2a", [episode(i, "pick", ok=i < c3) for i in range(1000)], step, tag, arm="VT") | {"stats_sha": "S"})
    put(R / "s2m_test_s11544.json", ev("pick", 11544, "s2a", [episode(i, "pick", ok=i < c2) for i in range(1000)], "068880", "s2m"))
    put(R / "s2m.json", {"seed": ref_seed, "n": 1000, "success": {"count": c2}, "gate_pass": True, "ckpt_sha": ref_sha})
    put(R / tag / "train_summary_VT.json", {"init": "/runs/s2m_VT/checkpoints/068880/pretrained_model", "init_sha256": init})
    for sk, seed in ((task, sp["vseed"]), ("pick", 21640)):
        for c in S.CKPTS_S3:
            put(R / tag / f"{tag}_VT_val_{sk}_{c}_s{seed}.json", {"stats_sha": "S"})


def run_final_one(R: Path, which: str) -> dict:
    S.final_s3(which)
    return json.loads((R / f"{S.S3[which]['tag']}.json").read_text())


def test_one_size_final_gate_edges_and_2m_reference(R):
    final_files_one(R, "a1p", k_task=800, wrong=49, c3=917, c2=967)  # exactly 80 %, 4.9 %, pick exactly 5.0 below 2M (96.7 -> 91.7)
    r = run_final_one(R, "a1p")
    assert r["gate"] == {"place_success_ge_80": True, "wrong_dest_lt_5": True, "pick_ge_stage2_minus_5_points": True}
    assert r["gate_pass"] and r["init_check"]["is_2m_pick"] and "is_2b_pick" not in r["init_check"]
    assert r["pick"]["stage2_ref"] == "results/s2m.json (Stage 2M test pick, seed 11544)" and "one cube size" in r["design"]
    final_files_one(R, "a1p", c3=916)
    assert not run_final_one(R, "a1p")["gate"]["pick_ge_stage2_minus_5_points"]
    final_files_one(R, "a1p", k_task=799)
    assert not run_final_one(R, "a1p")["gate"]["place_success_ge_80"]
    final_files_one(R, "b1p", k_task=700, wrong=49)
    r = run_final_one(R, "b1p")
    assert r["gate"] == {"stack_success_ge_70": True, "wrong_object_lt_5": True, "pick_ge_stage2_minus_5_points": True}
    assert "stack_stability" in r["stack"]
    final_files_one(R, "b1p", k_task=700, wrong=50)
    assert not run_final_one(R, "b1p")["gate"]["wrong_object_lt_5"]


@pytest.mark.parametrize("which", ONE)
def test_one_size_final_init_and_reference_mutants(R, which):
    final_files_one(R, which, init="0" * 64)
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s3(which)
    final_files_one(R, which, ref_sha="1" * 64)
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s3(which)
    final_files_one(R, which, ref_seed=11550)  # results/s2m.json is not the 11544 test result
    with pytest.raises(AssertionError):
        S.final_s3(which)
    final_files_one(R, which)
    assert run_final_one(R, which)["gate_pass"]


def expert_files(R: Path, which: str, kf: int, kc: int, n: int = 1000) -> None:
    sp = S.EXPERT_S3[which]
    st, task, seed = sp["stage"], sp["task"], sp["seed"]
    for k, kk in (("fresh", kf), ("N10", kc)):
        e = ev(task, seed, st, eps_k(task, n, kk), "", st)
        e["config"] |= {"policy": "expert", "expert_fresh": k == "fresh"}
        put(R / f"{st}p_pre" / f"{st}_expert_{k}_s{seed}.json", e)


@pytest.mark.parametrize("which,fresh,chunk", [("a1", 90, 80), ("b1", 85, 70)])
def test_expert_s3_gate_edges(R, which, fresh, chunk):
    def run(kf, kc):
        expert_files(R, which, kf, kc)
        S.expert_s3(which)
        return json.loads((R / f"{S.EXPERT_S3[which]['stage']}p_pre.json").read_text())
    r = run(10 * fresh, 10 * fresh - 30)  # fresh exactly at its bar, chunked exactly 3.0 points below it (inclusive)
    assert r["gate"][f"expert_fresh_ge_{fresh}"] and r["gate"]["chunked_within_3_points"]
    assert r["gate"][f"chunked_ge_{chunk}"] is (10 * fresh - 30 >= 10 * chunk) and r["diff_points"] == 3.0
    assert run(10 * fresh + 30, 10 * fresh - 1 + 0)["gate"]["chunked_within_3_points"] is False  # 3.1 points
    assert run(10 * fresh - 1, 10 * fresh - 1)["gate"][f"expert_fresh_ge_{fresh}"] is False
    r = run(10 * fresh + 80, 10 * chunk)  # chunked exactly at its bar but 8 or more points below the expert
    assert r["gate"][f"chunked_ge_{chunk}"] and r["gate"]["chunked_within_3_points"] is False and not r["gate_pass"]
    assert run(10 * chunk + 100, 10 * chunk - 1)["gate"][f"chunked_ge_{chunk}"] is False
    assert run(10 * fresh, 10 * fresh + 20)["gate_pass"]  # chunked above the expert is fine
    assert run(10 * fresh, 10 * fresh)["gate_pass"] and run(10 * fresh, 10 * fresh)["fresh"]["count"] == 10 * fresh


def test_expert_s3_asserts_stream_and_length(R):
    expert_files(R, "a1", 950, 930, n=300)
    with pytest.raises(AssertionError):
        S.expert_s3("a1")
    expert_files(R, "a1", 950, 930)
    p = R / "s3a1p_pre" / "s3a1_expert_N10_s11840.json"
    e = json.loads(p.read_text())
    e["config"]["expert_fresh"] = True  # a chunked-slot file that ran the fresh expert
    put(p, e)
    with pytest.raises(AssertionError):
        S.expert_s3("a1")


# ------------------------------------------------------------------ task 009 3A one diagnosed fix (two saves)
def val_files_fix(R: Path, place=(220, 230), pick=(265, 268)) -> None:
    sp = S.S3["a1pf"]
    tag = sp["tag"]
    for (c, step), kp, kk in zip(sp["ckpts"].items(), place, pick):
        put(R / tag / f"{tag}_VT_val_place_{c}_s21665.json", ev("place", 21665, "s3a1", eps_k("place", 300, kp), step, tag, arm="VT") | {"stats_sha": "S"})
        put(R / tag / f"{tag}_VT_val_pick_{c}_s21640.json", ev("pick", 21640, "s2a", eps_k("pick", 300, kk), step, tag, arm="VT") | {"stats_sha": "S"})


def test_fix_registered():
    sp = S.S3["a1pf"]
    assert (sp["tag"], sp["stage"], sp["vseed"], sp["tseed"], sp["pick_tseed"], sp["guards"]) == ("s3a1pf", "s3a1", 21665, 11580, 11544, "fix")
    assert sp["ckpts"] == {"c050": "008877", "c100": "017754"} and sp["init_sha256"].startswith("f6adef88") and sp["init_is_s2ref"] is False


def test_fix_two_save_rule_and_guard(R):
    val_files_fix(R)
    v = run_val_one(R, "a1pf")
    assert v["pick"] == "c100" and v["pick_step"] == "017754" and v["guards_c050"]["place_lt_70"]["fires"] is False
    assert set(v["pick_300"]["skills"]["place"]["within5_of_max"]) == {"c050"}
    val_files_fix(R, place=(210, 230))  # 70.0 % exactly does not fire
    assert run_val_one(R, "a1pf")["guards_c050"]["place_lt_70"]["fires"] is False
    val_files_fix(R, place=(209, 230))
    assert run_val_one(R, "a1pf")["guards_c050"]["place_lt_70"]["fires"] is True
    val_files_fix(R, place=(240, 190))  # c100 collapsed on place (63.3 < 80.0 - 10, 6.7 from the line) -> the earlier save
    v = run_val_one(R, "a1pf")
    assert v["pick"] == "c050" and v["pick_step"] == "008877" and v["pick_300"]["collapsed_skills"] == ["place"]
    val_files_fix(R, place=(240, 213))  # 71.0 within 5 points of the 70.0 line -> extend
    assert run_val_one(R, "a1pf")["pick"] is None


def test_two_save_pick_rule_unit():
    r = S.pick_rule2({"place": {"c050": 200, "c100": 240}, "pick": {"c050": 270, "c100": 200}}, {"place": 300, "pick": 300}, "place")
    assert r["collapsed_skills"] == ["pick"] and r["pick"] == "c050"
