"""scripts/s4b_report.py: pick rule, shown seed, McNemar, and the val -> shown -> final chain on synthetic eval JSONs."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("s4b_report", ROOT / "scripts" / "s4b_report.py")
M = importlib.util.module_from_spec(sp)
sp.loader.exec_module(M)

N300 = {t: 300 for t in M.TASKS}


def S(**kw):  # s[task] = {c050, c075, c100}
    return {t: dict(zip(("c050", "c075", "c100"), kw[t])) for t in M.TASKS}


def test_pick_c100_when_no_collapse_even_if_c050_is_higher():
    s = S(pick=(0.90, 0.89, 0.88), place=(0.70, 0.75, 0.80), stack=(0.5, 0.55, 0.6))  # pick: line 0.79, c100 0.88: not best-of
    r = M.pick_rule(s, N300)
    assert r["pick"] == "c100" and r["collapsed_tasks"] == [] and r["needs_extension_tasks"] == []


def test_collapse_picks_later_of_50_75_within_5_points():
    s = S(pick=(0.90, 0.88, 0.95), place=(0.70, 0.75, 0.40), stack=(0.5, 0.55, 0.6))  # place collapsed (line 0.65 vs 0.40)
    r = M.pick_rule(s, N300)
    assert r["collapsed_tasks"] == ["place"] and r["pick"] == "c075"
    s["place"] = {"c050": 0.80, "c075": 0.70, "c100": 0.40}  # c075 is 10 points below the max, c050 is the max
    assert M.pick_rule(s, N300)["pick"] == "c050"
    s["place"] = {"c050": 0.80, "c075": 0.75, "c100": 0.40}  # both within 5 -> the later
    assert M.pick_rule(s, N300)["pick"] == "c075"


def test_collapse_falls_back_to_mean_success_tie_later():
    s = S(pick=(0.90, 0.70, 0.30), place=(0.60, 0.90, 0.30), stack=(0.80, 0.80, 0.80))
    r = M.pick_rule(s, N300)
    assert set(r["collapsed_tasks"]) == {"pick", "place"} and r["pick"] == "c075"  # means 0.7667 vs 0.80
    s = S(pick=(0.90, 0.70, 0.30), place=(0.70, 0.90, 0.30), stack=(0.80, 0.80, 0.80))  # c050 mean == c075 mean -> later
    assert M.pick_rule(s, N300)["pick"] == "c075"
    s = S(pick=(0.90, 0.70, 0.30), place=(0.60, 0.90, 0.30), stack=(0.90, 0.60, 0.50))  # c050 mean higher
    assert M.pick_rule(s, N300)["pick"] == "c050"


def test_smart_extension_within_5_points_of_the_line_blocks_the_pick():
    s = S(pick=(0.90, 0.90, 0.78), place=(0.70, 0.75, 0.80), stack=(0.5, 0.55, 0.6))  # pick line 0.80, c100 0.78: within 5
    r = M.pick_rule(s, N300)
    assert r["pick"] is None and r["needs_extension_tasks"] == ["pick"]
    r = M.pick_rule(s, {"pick": 1000, "place": 300, "stack": 300})  # extended: decided at 1000 (collapsed: 0.78 < 0.80)
    assert r["pick"] == "c075" and r["collapsed_tasks"] == ["pick"]
    s["pick"]["c100"] = 0.75  # exactly 5 points under the line is still "within 5"
    assert M.pick_rule(s, N300)["pick"] is None
    s["pick"]["c100"] = 0.50
    assert M.pick_rule(s, N300)["pick"] == "c075"


def test_median_seed_and_ties():
    assert M.median_seed({0: 0.5, 1: 0.7, 2: 0.6})["shown_seed"] == 2
    assert M.median_seed({0: 0.7, 1: 0.7, 2: 0.6})["shown_seed"] == 0  # median score 0.7, tie -> lower index
    assert M.median_seed({0: 0.6, 1: 0.7, 2: 0.7})["shown_seed"] == 1
    assert M.median_seed({0: 0.6, 1: 0.6, 2: 0.6})["shown_seed"] == 0


def test_mcnemar():
    assert M.mcnemar([True, True, True, False], [False, False, True, True]) == {"true_only": 2, "shuffled_only": 1, "z": 1 / 3 ** 0.5}
    assert M.mcnemar([True], [True])["z"] is None


# ----------------------------------------------------------------------------- chain on synthetic files
def ev(task, seed, split, visual, n, ckpt, k, *, swap="none", wrong=0, palette="s4r"):
    eps = [{"index": i, "success": i < k, "wrong_object": i >= n - wrong, "wrong_grasp": False, "slam": False, "jam": False,
            "spec": json.dumps({"i": i}), "instruction": f"x{i}"} | ({"instruction_sent": f"y{i}"} if swap != "none" else {})
           for i in range(n)]
    agg = {m: {"count": sum(e[m] for e in eps), "n": n, "rate": sum(e[m] for e in eps) / n, "ci95": [0, 1]}
           for m in ("success", "wrong_object", "wrong_grasp", "slam", "jam")}
    j = {"config": {"policy": "baseline", "stage": M.STAGE[task], "split": split, "visual": visual, "table_palette": palette,
                    "seed": seed, "envs": 128, "image_size": 224, "n": 10, "cube_yaw_max_deg": 15.0, "instruction_swap": swap},
         "episodes": eps, "ckpt": f"/runs/s4_baseline_seed0/{ckpt}", "ckpt_sha": "sha", "gpu_h": 0.1,
         "aggregate": {task: agg}, "latency": {"server_s_p50": 0.001}}
    if swap != "none":
        j["instruction_swap_report"] = {}
    return j


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "R", tmp_path)
    monkeypatch.setattr(M, "RAW", tmp_path / "s4b")
    steps = {"c050": 100, "c075": 150, "c100": 200}
    for k, base in enumerate((210, 240, 180)):  # c100 success out of 300: .70 / .80 / .60 on every task
        d = tmp_path / "s4b" / f"seed{k}"
        d.mkdir(parents=True)
        (d / "train_summary.json").write_text(json.dumps({
            "save_steps": {"100": 0.5, "150": 0.75, "200": 1.0, "50": 0.25}, "params": 1, "S": 1, "steps_per_epoch": 1,
            "total_steps": 200, "epochs": 1, "final_step": 200, "git_sha": "x",
            "early_abort": {"a_nan": False, "b": {"abort": False}, "c": {"abort": False}}}))
        for c, st in steps.items():
            for t in M.TASKS:
                (d / f"val_{c}_{t}_s{M.VSEED[t]}.json").write_text(json.dumps(
                    ev(t, M.VSEED[t], "train", "nominal", 300, f"ckpt_{st:06d}", base if c == "c100" else base - 6)))
        for t, sd in (("pick", 21720), ("place", 21730)):
            for mode in ("true", "shuffle"):
                (d / f"i3_{mode}_{t}_s{sd}.json").write_text(json.dumps(ev(
                    t, sd, "train", "nominal", 100, "ckpt_000200", 70 if mode == "true" else 20,
                    swap="none" if mode == "true" else "shuffle")))
        for t in M.TASKS:
            (tmp_path / f"s4b_seed{k}_a_{t}_s{M.SPLITS['a'][3][t]}.json").write_text(json.dumps(
                ev(t, M.SPLITS["a"][3][t], "train", "nominal", 1000, "ckpt_000200", 700 + 50 * k)))
    return tmp_path


def test_chain_val_shown_final(world):
    assert M.val([0, 1, 2]) == 0
    v = {k: json.loads((world / f"s4b_val_seed{k}.json").read_text()) for k in range(3)}
    assert all(x["pick"] == "c100" and x["pick_step"] == 200 for x in v.values())
    assert [round(x["mean_success_picked"], 3) for x in v.values()] == [0.7, 0.8, 0.6]
    assert not any(x["early_abort_d"]["abort"] for x in v.values())
    M.shown([0, 1, 2])
    assert json.loads((world / "s4_shown_seed_baseline.json").read_text())["shown_seed"] == 0  # median score .7
    M.final([0, 1, 2])
    f = json.loads((world / "s4b.json").read_text())
    assert f["shown_seed"] == 0 and list(f["splits"]) == ["a"]
    assert f["splits"]["a"]["tasks"]["pick"]["success"]["per_seed"] == [0.7, 0.75, 0.8]
    assert f["splits"]["a"]["tasks"]["pick"]["success"]["mean"] == pytest.approx(0.75)
    ref = f["split_a_vs_smolvla_criteria"]
    assert ref["reference_only"] and ref["tasks"]["pick"]["meets_reference"] is False  # mean .75 < .85
    i3 = f["per_seed"]["0"]["i3"]["pick"]
    assert i3["success"]["true"]["count"] == 70 and i3["success"]["shuffle"]["count"] == 20
    assert i3["success_mcnemar"]["true_only"] == 50 and i3["success_mcnemar"]["z"] == pytest.approx(50 / 50 ** 0.5)


def test_checks_catch_wrong_files(world):
    d = world / "s4b" / "seed0"
    f = d / f"val_c050_pick_s{M.VSEED['pick']}.json"
    j = json.loads(f.read_text())
    f.write_text(json.dumps(j | {"ckpt": "/runs/x/ckpt_000999"}))
    with pytest.raises(AssertionError):
        M.val_seed(0)
    f.write_text(json.dumps(ev("pick", 1, "train", "nominal", 300, "ckpt_000100", 200)))
    with pytest.raises(AssertionError, match="eval config"):
        M.val_seed(0)
    f.write_text(json.dumps(ev("pick", M.VSEED["pick"], "train", "nominal", 200, "ckpt_000100", 100)))
    with pytest.raises(AssertionError, match="episodes"):
        M.val_seed(0)


def test_early_abort_d_and_extension_exit_code(world):
    d = world / "s4b" / "seed1"
    f = d / f"val_c050_pick_s{M.VSEED['pick']}.json"
    f.write_text(json.dumps(ev("pick", M.VSEED["pick"], "train", "nominal", 300, "ckpt_000100", 15)))  # 5 %
    assert M.val_seed(1)["early_abort_d"]["abort"] is True
    # c075 pick 0.80 -> line 0.70; c100 pick 0.73 is within 5 points of the line -> extension, exit code 2
    (d / f"val_c075_pick_s{M.VSEED['pick']}.json").write_text(json.dumps(ev("pick", M.VSEED["pick"], "train", "nominal", 300, "ckpt_000150", 240)))
    (d / f"val_c100_pick_s{M.VSEED['pick']}.json").write_text(json.dumps(ev("pick", M.VSEED["pick"], "train", "nominal", 300, "ckpt_000200", 219)))
    r = M.val_seed(1)
    assert r["pick"] is None and len(r["needs_extension"]) == 3 and r["needs_extension_tasks"] == ["pick"]
    assert M.val([1]) == 2


def _vp(vals, succ):
    return [{"index": i, "success": s, "table_value": v} for i, (v, s) in enumerate(zip(vals, succ))]


def test_value_bins_b_per_task_and_pooled():
    eps = {"pick": _vp([0.35, 0.42, 0.5, 0.55, 0.79, 0.8], [1, 0, 1, 1, 0, 1]),
           "place": _vp([0.31, 0.5, 0.65], [0, 1, 1]), "stack": _vp([0.7], [1])}
    r = M.value_bins("b", eps)
    assert r["bins"] == ["[0.3,0.4)", "[0.4,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8]"] and r["out_of_bins"] == 0
    assert (r["tasks"]["pick"]["[0.5,0.6)"]["count"], r["tasks"]["pick"]["[0.5,0.6)"]["n"]) == (2, 2)  # 0.5 -> upper bin
    assert (r["tasks"]["pick"]["[0.7,0.8]"]["count"], r["tasks"]["pick"]["[0.7,0.8]"]["n"]) == (1, 2)  # 0.8 closed
    p = r["pooled"]
    assert sum(p[b]["n"] for b in r["bins"]) == 10
    assert (p["[0.5,0.6)"]["count"], p["[0.5,0.6)"]["n"]) == (3, 3)
    assert p["[0.3,0.4)"]["ci95"] == pytest.approx(M.wilson_ci(1, 2))


def test_value_bins_bprime_bands_and_out_of_range():
    eps = {t: _vp([0.1, 0.9, 0.15, 0.5], [1, 0, 0, 1]) for t in M.TASKS}
    r = M.value_bins("bp", eps)
    assert r["out_of_bins"] == 3 and r["bins"] == ["below_nominal(<0.5)", "above_nominal(>0.5)"]
    assert (r["pooled"]["below_nominal(<0.5)"]["count"], r["pooled"]["below_nominal(<0.5)"]["n"]) == (3, 6)
    assert r["pooled"]["above_nominal(>0.5)"]["n"] == 3
    assert M.value_bins("b", {"pick": _vp([0.2], [1])})["out_of_bins"] == 1


def test_value_bins_missing_field_is_na_not_failure():
    eps = {"pick": [{"index": 0, "success": True}], "place": _vp([0.5], [1]), "stack": _vp([0.5], [1])}
    assert M.value_bins("b", eps) == "n/a (not recorded)"
    assert M.value_bins("bp", eps) == "n/a (not recorded)"
    assert M.value_bins("c", {"pick": _vp([0.5], [1])}) == "n/a (not recorded)"  # no such item for other splits
