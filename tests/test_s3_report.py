"""s1_report Stage 3A / 3B modes (val_s3 / final_s3): the two-skill pick rule, bookkeeping assertions, gate edges,
paired bootstrap (synthetic JSONs, no simulator)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("s1_report", Path(__file__).resolve().parents[1] / "scripts" / "s1_report.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

FLAGS = ("slam", "slam_10N", "slam_50N", "jam", "drop", "early_close", "distractor_disturbed", "invalid",
         "release_failure", "release_fail_not_released", "release_fail_off_destination", "release_fail_not_clear",
         "release_fail_unstable")
CUBES = [["red", "small"], ["green", "large"], ["blue", "small"]]


def spec_json(task: str, i: int) -> str:
    dest = {"pick": None, "place": ["red", "green", "blue", "yellow"][i % 4],
            "stack": ["green", ["small", "large"][i % 2]]}[task]
    return json.dumps({"task": task, "target": CUBES[0], "dest": dest, "cubes": CUBES, "pads": ["red", "green", "blue", "yellow"]})


def episode(i: int, task: str, ok: bool = True, wg: bool = False, wd: bool = False, yaw: float = 3.0) -> dict:
    e = {"index": i, "task": task, "target": "red_small", "n_cubes": 2 + i % 2, "spec": spec_json(task, i),
         "instruction": f"i{i}", "success": ok, "wrong_grasp": wg, "wrong_dest": wd, "wrong_object": wg or wd,
         "grasp_confirmed": True, "stack_release_rest_seen": ok and task == "stack", "stack_stable_after_release": ok,
         "spawn_yaw_fold_deg": yaw, **{f: False for f in FLAGS}}
    if wg:
        e["wrong_grasp_cube"] = 3  # ALL_CUBES[3] = green large? (index in catalog order), in CUBES below via spec
    return e


def ev(task: str, seed: int, stage: str, eps: list[dict], step: str, tag: str = "s3a", sha: str = "abc", split: str = "train",
       m: float = 15.0, sampling: str = "default", n: int = 10, h: int = 224, arm: str = "U") -> dict:
    return {"config": {"n": n, "image_size": h, "seed": seed, "stage": stage, "split": split, "cube_yaw_max_deg": m,
                       "mix": "1,1,1", "visual": "nominal"},
            "server_info": {"sampling": sampling}, "ckpt": f"/runs/{tag}_{arm}/checkpoints/{step}/pretrained_model",
            "ckpt_sha": sha, "episodes": eps, "smart_length": None, "bars": [], "latency": {"server_s_p50": 2.0}, "gpu_h": 0.1,
            "aggregate": {task: {k: 0.0 for k in ("slam", "jam", "drop", "early_close")}}}


def eps_k(task: str, n: int, k: int, **kw) -> list[dict]:
    return [episode(i, task, ok=i < k, **kw) for i in range(n)]


def put(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    return tmp_path


def val_files(R: Path, which: str = "a", place=(180, 200, 220), pick=(250, 255, 260), n: int = 300, arm: str = "U") -> None:
    sp = S.S3[which]
    tag, task = sp["stage"], sp["task"]
    for (c, step), kp, kk in zip(S.CKPTS_S3.items(), place, pick):
        put(R / tag / f"{tag}_{arm}_val_{task}_{c}_s{sp['vseed']}.json", ev(task, sp["vseed"], tag, eps_k(task, n, kp), step, tag, arm=arm))
        put(R / tag / f"{tag}_{arm}_val_pick_{c}_s21650.json", ev("pick", 21650, "s2", eps_k("pick", n, kk), step, tag, arm=arm))
    put(R / tag / "vloss_008877.json", {"val_loss": 1.0})
    put(R / tag / "vloss_017754.json", {"val_loss": 1.4})


def run_val(R: Path, which: str = "a", *a: str) -> dict:
    S.val_s3(which, *a)
    return json.loads((R / f"{S.S3[which]['stage']}_val.json").read_text())


# ------------------------------------------------------------------------------------------ pick rule
def rule(place, pick, n=1000, main="place"):
    ks = lambda t: dict(zip(("c050", "c075", "c100"), t))  # noqa: E731
    return S.pick_rule2({"place": ks(place), "pick": ks(pick)}, {"place": n, "pick": n}, main)


def test_rule_no_collapse_and_exact_edge():
    assert rule((600, 700, 650), (900, 900, 900))["pick"] == "c100"
    r = rule((600, 700, 600), (900, 900, 900))  # exactly 10.0 points below max (n 1000): not a collapse
    assert r["pick"] == "c100" and r["collapsed_skills"] == []
    r = rule((600, 700, 599), (900, 900, 900))  # 10.1 points
    assert r["collapsed_skills"] == ["place"] and r["pick"] == "c075"


def test_rule_collapse_on_one_skill_each():
    assert rule((600, 700, 500), (900, 900, 900))["pick"] == "c075"
    assert rule((700, 640, 500), (900, 900, 900))["pick"] == "c050"  # c075 60 below the max > 50 -> the earlier
    r = rule((600, 650, 650), (900, 800, 700))  # only pick collapsed (700 < 900 - 100): c075 100 below, c050 = max
    assert r["collapsed_skills"] == ["pick"] and r["pick"] == "c050"


def test_rule_both_collapsed_conflict_takes_higher_main_skill():
    # place: only c075 within 5 of its max; pick: only c050 within 5 of its max -> neither on every collapsed skill
    r = rule((600, 700, 500), (900, 800, 700))
    assert r["collapsed_skills"] == ["pick", "place"] and r["pick"] == "c075" and "higher place" in r["decision"]
    r = rule((700, 600, 500), (800, 900, 700))  # place max at c050 now -> higher place success = c050
    assert r["pick"] == "c050"
    r = rule((700, 700, 500), (800, 900, 700))  # c050 within 5 on place but not on pick; c075 on both -> the later
    assert r["pick"] == "c075"


def test_rule_smart_extension_per_skill():
    r = rule((180, 200, 175), (270, 270, 270), n=300)  # place line 170, s_100 175 within 5 points (15 episodes) -> extend
    assert r["extend_skills"] == ["place"] and r["pick"] is None and r["decision"].startswith("extend")
    r = rule((180, 200, 185), (270, 270, 258), n=300)  # pick line 240: 258 is 18 above (> 15); place 185 vs 170: 15 -> inclusive
    assert r["extend_skills"] == ["place"]
    r = rule((180, 200, 186), (270, 270, 270), n=300)  # 16 above the line: outside 5 points
    assert r["extend_skills"] == [] and r["pick"] == "c100"
    r = rule((180, 200, 170), (270, 270, 270), n=300)  # exactly on the line: not collapsed, but within 5 points -> extend
    assert r["extend_skills"] == ["place"] and r["skills"]["place"]["collapsed"] is False
    r = rule((180, 200, 100), (270, 270, 270), n=300)  # far below the line: collapse decided at 300
    assert r["extend_skills"] == [] and r["collapsed_skills"] == ["place"] and r["pick"] == "c075"


# ------------------------------------------------------------------------------------------ val_s3
@pytest.mark.parametrize("which", ["a", "b"])
def test_val_s3_pick_and_aborts(R, which):
    val_files(R, which)
    v = run_val(R, which)
    assert v["pick"] == "c100" and v["pick_step"] == "035508" and v["cube_yaw_max_deg"] == 15.0
    assert v["early_abort_c"] == {"loss_25": 1.0, "loss_50": 1.4, "abort": False} and v["early_abort_d"]["abort"] is False
    assert v["per_ckpt"][S.S3[which]["task"]]["c050"]["step"] == "017754" and v["per_ckpt"]["pick"]["c100"]["n"] == 300
    put(R / S.S3[which]["stage"] / "vloss_017754.json", {"val_loss": 2.01})
    assert run_val(R, which)["early_abort_c"]["abort"] is True
    val_files(R, which, place=(15, 200, 220))  # (d): place / stack s_50 <= 5 %
    assert run_val(R, which)["early_abort_d"] == {"skill": S.S3[which]["task"], "s_50": 0.05, "abort": True}
    val_files(R, which, place=(16, 200, 220))
    assert run_val(R, which)["early_abort_d"]["abort"] is False
    val_files(R, which, pick=(10, 250, 255))  # (d) is place / stack only
    assert run_val(R, which)["early_abort_d"]["abort"] is False


def test_val_s3_collapse_on_pick_skill_and_extension_read(R):
    val_files(R, "a", place=(180, 200, 220), pick=(270, 280, 150))  # pick collapsed, place not
    v = run_val(R)
    assert v["pick_300"]["collapsed_skills"] == ["pick"] and v["pick"] == "c075"
    val_files(R, "a", place=(180, 200, 175))  # place within 5 of its line -> extend
    v = run_val(R)
    assert v["pick"] is None and v["pick_step"] is None and v["pick_300"]["extend_skills"] == ["place"]
    val_files(R, "a", place=(600, 700, 650), pick=(900, 900, 900), n=1000)  # extended to 1,000 episodes on both skills
    v = run_val(R)
    assert v["pick"] == "c100" and v["pick_extended"]["n_per_skill"] == {"place": 1000, "pick": 1000}


@pytest.mark.parametrize("what", ["seed", "stage", "split", "m", "sampling", "n", "H", "ckpt", "yaw", "stage_pick", "seed_pick", "sha", "task"])
def test_val_s3_assertions_fire(R, what):
    val_files(R, "a")
    kw: dict = {}
    sk, seed, stage, step = "place", 21660, "s3a", "017754"
    if what == "seed": seed = 21661
    if what == "stage": stage = "s3b"
    if what == "split": kw["split"] = "heldout_comp"
    if what == "m": kw["m"] = 0.0
    if what == "sampling": kw["sampling"] = "mean"
    if what == "n": kw["n"] = 20
    if what == "H": kw["h"] = 128
    if what == "ckpt": step = "026631"
    eps = eps_k("place", 300, 180)
    if what == "yaw": eps[3]["spawn_yaw_fold_deg"] = 15.6
    if what == "task": eps[3]["task"] = "pick"
    if what in ("stage_pick", "seed_pick", "sha"):
        sk, seed, stage = "pick", 21650, "s2"
        if what == "stage_pick": stage = "s3a"
        if what == "seed_pick": seed = 21621
        e = ev("pick", seed, stage, eps_k("pick", 300, 250), step, sha="zzz" if what == "sha" else "abc")
        put(R / "s3a" / "s3a_U_val_pick_c050_s21650.json", e)
    else:
        put(R / "s3a" / "s3a_U_val_place_c050_s21660.json", ev("place", seed, stage, eps, step, **kw))
    with pytest.raises(AssertionError):
        S.val_s3("a")


# ------------------------------------------------------------------------------------------ final_s3
def final_files(R: Path, which: str = "a", k_task: int = 800, wrong: int = 0, c3: int = 900, c2: int = 950,
                pick_wg: int = 0, pick_spec_shift: bool = False, arm: str = "U") -> None:
    sp = S.S3[which]
    tag, task, step = sp["stage"], sp["task"], "035508"
    put(R / f"{tag}_val.json", {"pick_step": step} | ({"arm_tag": arm} if arm != "U" else {}))
    te = [episode(i, task, ok=i < k_task, wd=(k_task <= i < k_task + wrong) if which == "a" else False,
                  wg=(k_task <= i < k_task + wrong) if which == "b" else False) for i in range(1000)]
    put(R / f"{tag}_test_{task}_s{sp['tseed']}.json", ev(task, sp["tseed"], tag, te, step, tag, sha="abc", arm=arm))
    p3 = [episode(i, "pick", ok=i < c3) for i in range(1000)]
    # Stage 2 successes: the first c2 episodes -> 3A-only / S2-only discordance appears when c3 != c2
    p2 = [episode(i, "pick", ok=i < c2) for i in range(1000)]
    if pick_spec_shift:
        p2[5]["spec"] = spec_json("pick", 1)
        p2[5]["instruction"] = "different"
    put(R / f"{tag}_test_pick_s11550.json", ev("pick", 11550, "s2", p3, step, tag, sha="abc", arm=arm))
    put(R / "s2b_test_s11550.json", ev("pick", 11550, "s2", p2, "034440", "s2b"))
    put(R / "s2b.json", {"seed": 11550, "n": 1000, "success": {"count": c2}, "gate_pass": True})


def run_final(R: Path, which: str = "a", *a: str) -> dict:
    S.final_s3(which, *a)
    return json.loads((R / f"{S.S3[which]['stage']}.json").read_text())


def test_final_gate_edges_3a(R):
    final_files(R, "a", k_task=800, wrong=49, c3=900, c2=950)  # place exactly 80 %, wrong-pad 4.9 %, pick exactly 5.0 below
    r = run_final(R)
    assert r["gate"] == {"place_success_ge_80": True, "wrong_dest_lt_5": True, "pick_ge_stage2_minus_5_points": True}
    assert r["gate_pass"] and r["place"]["wrong_dest"]["count"] == 49 and r["pick"]["stage2_success_count"] == 950
    final_files(R, "a", k_task=799)
    assert not run_final(R)["gate"]["place_success_ge_80"]
    final_files(R, "a", k_task=800, wrong=50)  # 5.0 % is not < 5 %
    assert not run_final(R)["gate"]["wrong_dest_lt_5"]
    final_files(R, "a", c3=899, c2=950)  # 5.1 below
    r = run_final(R)
    assert not r["gate"]["pick_ge_stage2_minus_5_points"] and not r["gate_pass"]
    final_files(R, "a", c3=980, c2=950)  # above Stage 2
    assert run_final(R)["gate"]["pick_ge_stage2_minus_5_points"]


def test_final_gate_edges_3b_and_extra_fields(R):
    final_files(R, "b", k_task=700, wrong=49)
    r = run_final(R, "b")
    assert r["gate"] == {"stack_success_ge_70": True, "wrong_object_lt_5": True, "pick_ge_stage2_minus_5_points": True}
    assert r["stack"]["stack_stability"]["released_resting_on_bottom"] == 700 and "by_top_bottom_size" in r["stack"]
    assert set(r["stack"]["by_top_bottom_size"]) == {"top_small__bottom_small", "top_small__bottom_large"}
    final_files(R, "b", k_task=699)
    assert not run_final(R, "b")["gate"]["stack_success_ge_70"]
    final_files(R, "b", k_task=700, wrong=50)
    assert not run_final(R, "b")["gate"]["wrong_object_lt_5"]
    final_files(R, "a")
    r = run_final(R, "a")
    assert set(r["place"]["by_destination"]) == {"red", "green", "blue", "yellow"} and "red_small" in r["place"]["success_by_target"]
    assert set(r["place"]["by_n_cubes"]) == {"2", "3"}


def test_final_paired_numbers_and_bootstrap_determinism(R):
    final_files(R, "a", c3=900, c2=950)
    r = run_final(R)["pick"]["paired_vs_stage2"]
    assert (r["stage3_only"], r["stage2_only"], r["both"], r["neither"]) == (0, 50, 900, 50)
    assert r["diff_points"] == pytest.approx(-5.0) and r["ci95_points"][0] < -5.0 < r["ci95_points"][1] and r["resamples"] == 10_000
    assert run_final(R)["pick"]["paired_vs_stage2"] == r  # deterministic (default_rng(0))
    same = S.paired_counts([True, False] * 500, [True, False] * 500)
    assert same["ci95_points"] == [0.0, 0.0] and same["stage3_only"] == same["stage2_only"] == 0


def test_final_wrong_stream_and_inconsistency_assert(R):
    final_files(R, "a", pick_spec_shift=True)  # the two pick files differ in one episode's spec
    with pytest.raises(AssertionError):
        S.final_s3("a")
    final_files(R, "a", c2=950)
    put(R / "s2b.json", {"seed": 11550, "n": 1000, "success": {"count": 949}})  # != the per-episode file
    with pytest.raises(AssertionError):
        S.final_s3("a")
    final_files(R, "a")
    put(R / "s2b_test_s11550.json", ev("pick", 11520, "s2", [episode(i, "pick") for i in range(1000)], "034440", "s2b"))
    with pytest.raises(AssertionError):  # Stage 2 reference on another stream (attempt 0's 11520)
        S.final_s3("a")
    final_files(R, "a")
    put(R / "s2b.json", {"seed": 11520, "n": 1000, "success": {"count": 950}})  # attempt 0's result as the reference
    with pytest.raises(AssertionError):
        S.final_s3("a")
    final_files(R, "a")
    put(R / "s3a_test_pick_s11550.json", ev("pick", 11520, "s2", [episode(i, "pick", ok=i < 900) for i in range(1000)], "035508", "s3a"))
    with pytest.raises(AssertionError):  # 3A pick test not on the Stage 2 stream
        S.final_s3("a")
    final_files(R, "a")
    put(R / "s3a_test_place_s11570.json", ev("place", 11571, "s3a", [episode(i, "place") for i in range(1000)], "035508", "s3a"))
    with pytest.raises(AssertionError):  # place test on the wrong seed
        S.final_s3("a")
    final_files(R, "a")
    put(R / "s3a_test_place_s11570.json", ev("place", 11570, "s3a", [episode(i, "place") for i in range(300)], "035508", "s3a"))
    with pytest.raises(AssertionError):  # not 1,000 episodes
        S.final_s3("a")
    final_files(R, "a")
    put(R / "s3a_test_place_s11570.json", ev("place", 11570, "s3a", [episode(i, "place") for i in range(1000)], "017754", "s3a"))
    with pytest.raises(AssertionError):  # a checkpoint other than the picked one
        S.final_s3("a")
    final_files(R, "b")
    put(R / "s3b_test_stack_s11590.json", ev("stack", 11590, "s3a", [episode(i, "stack") for i in range(1000)], "035508", "s3b"))
    with pytest.raises(AssertionError):  # 3B reading a 3A-stage file
        S.final_s3("b")


# ------------------------------------------------------------------------------------------ arm VT (3A / 3B after Stage 2b)
def test_pick_reference_is_stage_2b():
    assert (S.PICK_VSEED, S.PICK_TSEED) == (21650, 11550)


@pytest.mark.parametrize("which", ["a", "b"])
def test_arm_vt_val_and_final_read_the_vt_files(R, which):
    val_files(R, which, arm="VT")
    v = run_val(R, which, "--arm", "VT")
    assert v["arm_tag"] == "VT" and v["per_ckpt"]["pick"]["c100"]["ckpt"].endswith(f"/{S.S3[which]['stage']}_VT/checkpoints/035508/pretrained_model")
    with pytest.raises(FileNotFoundError):  # default arm U finds no U files
        run_val(R, which)
    final_files(R, which, arm="VT")
    f = run_final(R, which)  # arm read from s3x_val.json
    assert f["arm_tag"] == "VT" and f["pick"]["stage2_ref"].startswith("results/s2b.json")
    assert run_final(R, which, "--arm", "VT")["arm_tag"] == "VT"
    with pytest.raises(AssertionError):  # --arm U against VT checkpoints
        run_final(R, which, "--arm", "U")


def test_final_s3_default_arm_unchanged_and_vt_mutant(R):
    final_files(R, "a")
    assert run_final(R)["arm_tag"] == "U"
    final_files(R, "a", arm="VT")
    put(R / "s3a_val.json", {"pick_step": "035508"})  # val.json without arm_tag -> U -> VT ckpts rejected
    with pytest.raises(AssertionError):
        S.final_s3("a")


# ------------------------------------------------------------------------------------------ task 007: amended 3A / 3B (s3ap / s3bp)
INIT_2BP = "f" * 64  # the 2b amended gated checkpoint's sha in these synthetic files


def test_amended_runs_registered():
    for w, st, task, vs, ts, bar, wrong in (("ap", "s3a", "place", 21660, 11570, 80, "wrong_dest"),
                                            ("bp", "s3b", "stack", 21690, 11590, 70, "wrong_object")):
        sp = S.S3[w]
        assert (sp["tag"], sp["stage"], sp["task"], sp["vseed"], sp["tseed"], sp["bar"], sp["wrong"], sp["arm"], sp["s2ref"]) \
            == (st + "p", st, task, vs, ts, bar, wrong, "VT", "s2bp")
        assert sp["init_sha256"] is None  # placeholder until the 2b PASS
    assert S.CKPTS_S3 == {"c050": "017754", "c075": "026631", "c100": "035508"} and S.STEP25_S3 == "008877"
    assert "tag" not in S.S3["a"] and "tag" not in S.S3["b"]  # 3A / 3B entries unchanged


def val_files_p(R: Path, which: str = "ap", main=(180, 200, 220), pick=(250, 255, 260), wrong50: int = 0,
                shas: dict | None = None, n: int = 300) -> None:
    sp = S.S3[which]
    tag, task, st = sp["tag"], sp["task"], sp["stage"]
    for (c, step), kp, kk in zip(S.CKPTS_S3.items(), main, pick):
        eps = eps_k(task, n, kp)
        if c == "c050":
            for e in eps[kp:kp + wrong50]:  # failures that are wrong-pad (3A) / wrong-cube (3B)
                e.update(wrong_dest=which == "ap", wrong_grasp=which == "bp", wrong_object=True)
        put(R / tag / f"{tag}_VT_val_{task}_{c}_s{sp['vseed']}.json",
            ev(task, sp["vseed"], st, eps, step, tag, arm="VT") | {"stats_sha": (shas or {}).get(f"{task}_{c}", "S")})
        put(R / tag / f"{tag}_VT_val_pick_{c}_s21650.json",
            ev("pick", 21650, "s2", eps_k("pick", n, kk), step, tag, arm="VT") | {"stats_sha": (shas or {}).get(f"pick_{c}", "S")})
    put(R / tag / "vloss_008877.json", {"val_loss": 1.0})
    put(R / tag / "vloss_017754.json", {"val_loss": 1.4})


def run_val_p(R: Path, which: str = "ap") -> dict:
    S.val_s3(which)
    return json.loads((R / f"{S.S3[which]['tag']}_val.json").read_text())


@pytest.mark.parametrize("which", ["ap", "bp"])
def test_val_amended_pick_guards_and_stats_sha(R, which):
    bar = S.S3[which]["bar"]
    line = (bar - 20) * 3  # successes of 300 at bar - 20 points: 180 (3A) / 150 (3B)
    val_files_p(R, which, main=(line, 200, 220))
    v = run_val_p(R, which)
    assert v["pick"] == "c100" and v["arm_tag"] == "VT" and v["stats_sha"] == "S"
    assert v["design"].startswith(f"Stage 3{which[0].upper()} (")
    assert v["per_ckpt"]["pick"]["c100"]["ckpt"].endswith(f"/{S.S3[which]['tag']}_VT/checkpoints/035508/pretrained_model")
    assert v["guards_c050"]["b1"]["fires"] is False and v["guards_c050"]["b2"]["fires"] is False  # exactly bar - 20: no stop
    val_files_p(R, which, main=(line - 1, 200, 220))
    assert run_val_p(R, which)["guards_c050"]["b1"]["fires"] is True
    val_files_p(R, which, wrong50=30)  # 10.0 % is not > 10 %
    g = run_val_p(R, which)["guards_c050"]["b2"]
    assert g["count"] == 30 and g["fires"] is False
    val_files_p(R, which, wrong50=31)
    assert run_val_p(R, which)["guards_c050"]["b2"]["fires"] is True
    val_files_p(R, which, shas={"pick_c075": "T"})
    with pytest.raises(AssertionError, match="stats_sha"):
        run_val_p(R, which)


def test_val_amended_reads_its_own_run_not_3a(R):
    val_files(R, "a", arm="VT")  # the 3A VT files (tag s3a) do not satisfy s3ap
    with pytest.raises(FileNotFoundError):
        run_val_p(R, "ap")


def final_files_p(R: Path, which: str = "ap", k_task: int = 800, wrong: int = 0, c3: int = 900, c2: int = 950,
                  init: str = INIT_2BP, ref_sha: str = INIT_2BP, test_sha: str = "S") -> None:
    sp = S.S3[which]
    tag, task, st, step = sp["tag"], sp["task"], sp["stage"], "035508"
    put(R / f"{tag}_val.json", {"pick_step": step, "arm_tag": "VT"})
    te = [episode(i, task, ok=i < k_task, wd=(k_task <= i < k_task + wrong) and which == "ap",
                  wg=(k_task <= i < k_task + wrong) and which == "bp") for i in range(1000)]
    put(R / f"{tag}_test_{task}_s{sp['tseed']}.json", ev(task, sp["tseed"], st, te, step, tag, arm="VT") | {"stats_sha": test_sha})
    put(R / f"{tag}_test_pick_s11550.json",
        ev("pick", 11550, "s2", [episode(i, "pick", ok=i < c3) for i in range(1000)], step, tag, arm="VT") | {"stats_sha": "S"})
    put(R / "s2bp_test_s11550.json", ev("pick", 11550, "s2", [episode(i, "pick", ok=i < c2) for i in range(1000)], "034440", "s2bp"))
    put(R / "s2bp.json", {"seed": 11550, "n": 1000, "success": {"count": c2}, "gate_pass": True, "ckpt_sha": ref_sha})
    put(R / tag / "train_summary_VT.json", {"init": "/runs/s2bp_VT/checkpoints/034440/pretrained_model", "init_sha256": init})
    for sk, seed in ((task, sp["vseed"]), ("pick", 21650)):
        for c in S.CKPTS_S3:
            put(R / tag / f"{tag}_VT_val_{sk}_{c}_s{seed}.json", {"stats_sha": "S"})


def run_final_p(R: Path, which: str = "ap") -> dict:
    S.final_s3(which)
    return json.loads((R / f"{S.S3[which]['tag']}.json").read_text())


@pytest.fixture
def filled(monkeypatch):
    """The init sha placeholder filled (as the orchestrator does after the 2b PASS)."""
    for w in ("ap", "bp"):
        monkeypatch.setitem(S.S3, w, S.S3[w] | {"init_sha256": INIT_2BP})


def test_final_amended_gate_edges_3a(R, filled):
    final_files_p(R, "ap", k_task=800, wrong=49, c3=900, c2=950)  # exactly 80 %, 4.9 %, pick exactly 5.0 below s2bp
    r = run_final_p(R)
    assert r["gate"] == {"place_success_ge_80": True, "wrong_dest_lt_5": True, "pick_ge_stage2_minus_5_points": True}
    assert r["gate_pass"] and r["pick"]["stage2_ref"].startswith("results/s2bp.json") and r["arm_tag"] == "VT"
    assert r["init_check"]["is_2b_pick"] and r["stats_sha"]["equal_across_evals"] and "amended" in r["design"]
    final_files_p(R, "ap", k_task=799)
    assert not run_final_p(R)["gate_pass"]
    final_files_p(R, "ap", wrong=50)
    assert not run_final_p(R)["gate"]["wrong_dest_lt_5"]
    final_files_p(R, "ap", c3=899, c2=950)
    assert not run_final_p(R)["gate"]["pick_ge_stage2_minus_5_points"]


def test_final_amended_gate_edges_3b(R, filled):
    final_files_p(R, "bp", k_task=700, wrong=49)
    r = run_final_p(R, "bp")
    assert r["gate"] == {"stack_success_ge_70": True, "wrong_object_lt_5": True, "pick_ge_stage2_minus_5_points": True}
    assert r["gate_pass"] and "stack_stability" in r["stack"] and "by_top_bottom_size" in r["stack"]
    final_files_p(R, "bp", k_task=699)
    assert not run_final_p(R, "bp")["gate"]["stack_success_ge_70"]
    final_files_p(R, "bp", k_task=700, wrong=50)
    assert not run_final_p(R, "bp")["gate"]["wrong_object_lt_5"]


@pytest.mark.parametrize("which", ["ap", "bp"])
def test_final_amended_init_and_stats_mutants(R, filled, which):
    final_files_p(R, which, init="0" * 64)  # wrong init (e.g. the 2M pick instead of the 2b pick)
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s3(which)
    final_files_p(R, which, ref_sha="1" * 64)  # s2bp.json names another checkpoint than the recorded init
    with pytest.raises(AssertionError, match="init_sha256"):
        S.final_s3(which)
    final_files_p(R, which, test_sha="T")  # the test served other normalizer stats
    with pytest.raises(AssertionError, match="stats_sha"):
        S.final_s3(which)
    final_files_p(R, which)
    assert not (R / "s2b.json").exists() and run_final_p(R, which)["gate_pass"]  # the old 2b files are never read


def test_final_amended_refuses_the_placeholder(R):
    final_files_p(R, "ap")
    with pytest.raises(AssertionError, match="placeholder"):
        S.final_s3("ap")
