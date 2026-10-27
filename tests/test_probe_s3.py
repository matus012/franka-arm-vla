"""s1_report probe_s3 on synthetic s3a / s3b paired eval JSONs: perfect / cube-blind / destination-blind policies, chance, relation split."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vla_common.catalog import ALL_CUBES, COLORS
from vla_tabletop.rollout import make_paired_stream

SPEC = importlib.util.spec_from_file_location("s1_report", Path(__file__).resolve().parents[1] / "scripts" / "s1_report.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)

STAGES = ("s3a", "s3b")


def dst(e) -> str:  # the instruction's destination, as the report names it
    return e.spec.dest if e.spec.task == "place" else "_".join(e.spec.dest)


def cube(e) -> str:
    return "_".join(e.spec.target)


def records(stage: str, policy, n_groups: int = 80) -> list[dict]:
    """policy(members) -> per member (grasped cube 'colour_size' | None, released-on destination | None); eval-style records."""
    eps, _ = make_paired_stream(33000, 0, n_groups, stage, "train", 15.0)
    out = []
    for g in sorted({e.layout_id for e in eps}):
        ms = [e for e in eps if e.layout_id == g]
        for e, (gc, gd) in zip(ms, policy(ms)):
            wg = gc is not None and gc != cube(e)
            rd = -1 if gd is None else COLORS.index(gd) if stage == "s3a" else ALL_CUBES.index(tuple(gd.split("_")))
            out.append({"index": e.index, "task": e.spec.task, "layout_id": g, "member": e.member, "relation": e.relation,
                        "target": cube(e), "n_cubes": len(e.spec.cubes), "instruction": e.instruction,
                        "spec": e.spec.to_json(), "success": gc == cube(e) and gd == dst(e), "wrong_grasp": wg,
                        "grasp_confirmed": gc == cube(e),
                        "rest_dest": rd, "wrong_grasp_cube": ALL_CUBES.index(tuple(gc.split("_"))) if wg else -1})
    return out


def perfect(ms):
    return [(cube(e), dst(e)) for e in ms]


def cube_blind(ms):  # member 0's cube and destination whatever is asked
    return [(cube(ms[0]), dst(ms[0]))] * len(ms)


def dest_blind(ms):  # right cube, member 0's destination
    return [(cube(e), dst(ms[0])) for e in ms]


@pytest.mark.parametrize("stage", STAGES)
def test_perfect_reads_100_percent(stage):
    st = S.probe_s3_stats(records(stage, perfect))
    for b in (st["all"], *st["by_relation"].values()):
        for k in ("named_cube_of_all", "named_cube_of_grasped", "named_dest_of_released", "both_right_of_all", "success"):
            assert b[k]["rate"] == 1.0, (k, b[k])
    assert st["same_choice_regardless_groups"]["rate"] == 0.0 and st["same_choice_regardless_groups"]["n"] == st["groups"]
    assert st["no_grasp_episodes"] == st["nothing_released_episodes"] == 0


@pytest.mark.parametrize("stage", STAGES)
def test_always_same_choice_reads_at_chance(stage):
    st = S.probe_s3_stats(records(stage, cube_blind))
    assert st["same_choice_regardless_groups"]["rate"] == 1.0
    a, r = st["all"], st["by_relation"]
    assert a["both_right_of_all"]["count"] == st["groups"]  # only member 0 (the anchor) is right
    assert a["both_right_of_all"]["rate"] == st["groups"] / st["episodes"] < 0.5
    assert r["same_target"]["named_cube_of_all"]["rate"] == 1.0  # same cube asked: right by construction
    assert r["same_target"]["named_dest_of_released"]["rate"] == 0.0  # only the word differs: never followed
    assert r["same_dest"]["named_dest_of_released"]["rate"] == 1.0 and r["same_dest"]["named_cube_of_all"]["rate"] == 0.0
    assert r["both_differ"]["both_right_of_all"]["count"] == 0 and r["both_differ"]["named_dest_of_released"]["rate"] == 0.0


@pytest.mark.parametrize("stage", STAGES)
def test_relation_split_separates_destination_and_cube_effects(stage):
    st = S.probe_s3_stats(records(stage, dest_blind))
    r = st["by_relation"]
    assert r["same_target"]["named_cube_of_all"]["rate"] == 1.0 and r["same_target"]["named_dest_of_released"]["rate"] == 0.0
    assert r["same_dest"]["named_dest_of_released"]["rate"] == 1.0 and r["same_dest"]["both_right_of_all"]["rate"] == 1.0
    assert r["both_differ"]["named_dest_of_released"]["rate"] == 0.0
    assert set(r) == {"anchor", "same_target", "same_dest", "both_differ"}
    assert sum(b["episodes"] for b in r.values()) == st["episodes"]


@pytest.mark.parametrize("stage", STAGES)
def test_chance_values(stage):
    recs = records(stage, perfect)
    a = S.probe_s3_stats(recs)["all"]
    n_d = [4 if stage == "s3a" else e["n_cubes"] - 1 for e in recs]
    assert a["named_cube_of_all"]["chance"] == pytest.approx(sum(1 / e["n_cubes"] for e in recs) / len(recs))
    assert a["named_dest_of_released"]["chance"] == pytest.approx(sum(1 / n for n in n_d) / len(recs))
    assert a["both_right_of_all"]["chance"] == pytest.approx(sum(1 / e["n_cubes"] / n for e, n in zip(recs, n_d)) / len(recs))
    assert 0 < S.probe_s3_stats(recs)["same_choice_regardless_groups"]["chance"] < 1


def test_same_choice_chance_is_collision_probability_over_cube_times_destination():
    recs = records("s3a", perfect)
    st = S.probe_s3_stats(recs)
    gs = {}
    for e in recs:
        gs.setdefault(e["layout_id"], []).append(e)
    exp = [S.p_same_cube_chance(len(ms), ms[0]["n_cubes"] * 4) for ms in gs.values()]
    assert st["same_choice_regardless_groups"]["chance"] == pytest.approx(sum(exp) / len(exp))


def test_nothing_released_and_no_grasp_are_excluded_from_their_denominators():
    st = S.probe_s3_stats(records("s3a", lambda ms: [(None, None)] * len(ms)))
    a = st["all"]
    assert st["no_grasp_episodes"] == st["nothing_released_episodes"] == st["episodes"]
    assert a["named_cube_of_grasped"]["n"] == 0 and a["named_dest_of_released"]["n"] == 0 and a["both_right_of_all"]["count"] == 0
    assert st["same_choice_regardless_groups"]["n"] == 0
    st = S.probe_s3_stats(records("s3b", lambda ms: [(cube(e), None) for e in ms]))  # right cube, nothing released
    assert st["all"]["named_cube_of_grasped"]["rate"] == 1.0 and st["all"]["named_dest_of_released"]["n"] == 0


def test_wrong_cube_right_destination_counts_only_the_destination():
    st = S.probe_s3_stats(records("s3a", lambda ms: [(cube(ms[(e.member + 1) % len(ms)]), dst(e)) for e in ms]))
    assert st["all"]["named_dest_of_released"]["rate"] == 1.0 and st["all"]["named_cube_of_all"]["rate"] < 1.0
    assert st["all"]["both_right_of_all"]["rate"] < 1.0


def test_old_json_rejected():
    r = records("s3a", perfect)
    with pytest.raises(AssertionError, match="s3"):
        S.probe_s3_stats([{k: v for k, v in e.items() if k != "relation"} for e in r])
    with pytest.raises(AssertionError, match="rest_dest"):
        S.probe_s3_stats([{k: v for k, v in e.items() if k != "rest_dest"} for e in r])


def test_cli_prints_and_writes_and_rejects_non_s3(tmp_path, capsys):
    recs = records("s3b", cube_blind, 10)
    ev = {"config": {"paired_targets": True, "stage": "s3b", "seed": 33000}, "ckpt": "x", "ckpt_sha": "y", "episodes": recs}
    f = tmp_path / "e.json"
    f.write_text(json.dumps(ev))
    S.probe_s3(str(f), "--out", str(tmp_path / "o.json"))
    out = capsys.readouterr().out
    assert "[same_target]" in out and "named destination" in out and "regardless of instruction" in out
    assert json.loads((tmp_path / "o.json").read_text())["stage"] == "s3b"
    f.write_text(json.dumps(ev | {"config": ev["config"] | {"stage": "s2a"}}))
    with pytest.raises(AssertionError, match="s3a / s3b"):
        S.probe_s3(str(f))
