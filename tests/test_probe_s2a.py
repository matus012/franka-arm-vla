"""s1_report probe_s2a on synthetic paired eval JSONs: perfectly grounded / perfectly ungrounded policies, chance levels."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vla_common.catalog import ALL_CUBES
from vla_tabletop.rollout import make_paired_stream

SPEC = importlib.util.spec_from_file_location("s1_report", Path(__file__).resolve().parents[1] / "scripts" / "s1_report.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def records(policy, n_groups: int = 60) -> list[dict]:
    """policy(members: list[episode]) -> grasped cube ids (colour_size) or None per member; builds eval-style records."""
    eps, _ = make_paired_stream(21645, 0, n_groups, "s2a", "train", 15.0)
    out = []
    for g in sorted({e.layout_id for e in eps}):
        ms = [e for e in eps if e.layout_id == g]
        for e, gc in zip(ms, policy(ms)):
            tgt = "_".join(e.spec.target)
            wg = gc is not None and gc != tgt
            out.append({"index": e.index, "layout_id": g, "member": e.member, "target": tgt, "n_cubes": len(e.spec.cubes),
                        "instruction": e.instruction, "spec": e.spec.to_json(), "success": gc == tgt, "wrong_grasp": wg,
                        "wrong_grasp_cube": ALL_CUBES.index(tuple(gc.split("_"))) if wg else -1})
    return out


def grounded(ms):
    return ["_".join(e.spec.target) for e in ms]


def ungrounded(ms):  # one fixed cube whatever is asked
    return ["_".join(ms[0].spec.target)] * len(ms)


def test_perfectly_grounded():
    st = S.probe_stats(records(grounded))
    assert st["success"]["rate"] == 1.0 and st["no_grasp_episodes"] == 0
    assert st["went_for_named_cube_of_all_episodes"]["rate"] == 1.0
    assert st["went_for_named_cube_of_grasped_episodes"]["rate"] == 1.0
    assert st["same_cube_regardless_groups"]["rate"] == 0.0 and st["same_cube_regardless_groups"]["n"] == st["groups"]


def test_perfectly_ungrounded():
    st = S.probe_stats(records(ungrounded))
    assert st["same_cube_regardless_groups"]["rate"] == 1.0
    named = st["went_for_named_cube_of_all_episodes"]  # only the member whose target is the fixed cube is named
    assert named["count"] == st["groups"] and named["n"] == st["episodes"]
    assert st["success_by_member"]["0"]["rate"] == 1.0 and st["success_by_member"]["1"]["rate"] == 0.0
    assert st["success_and_wrong_grasp_episodes"] == 0


def test_chance_levels():
    eps = records(grounded)
    st = S.probe_stats(eps)
    assert st["went_for_named_cube_of_all_episodes"]["chance"] == pytest.approx(sum(1 / e["n_cubes"] for e in eps) / len(eps))
    assert S.p_same_cube_chance(2, 2) == pytest.approx(0.5)
    assert S.p_same_cube_chance(3, 4) == pytest.approx(1 - 4 * 3 * 2 / 64)
    assert S.p_same_cube_chance(5, 4) == 1.0
    assert 0 < st["same_cube_regardless_groups"]["chance"] < 1


def test_no_grasp_episodes_are_excluded_from_grasped_rate_and_groups():
    st = S.probe_stats(records(lambda ms: [None] * len(ms)))
    assert st["no_grasp_episodes"] == st["episodes"] and st["went_for_named_cube_of_grasped_episodes"]["n"] == 0
    assert st["same_cube_regardless_groups"]["n"] == 0 and st["went_for_named_cube_of_all_episodes"]["rate"] == 0.0


def test_mixed_policy_counts():
    st = S.probe_stats(records(lambda ms: grounded(ms) if ms[0].layout_id % 2 else ungrounded(ms)))
    assert st["same_cube_regardless_groups"]["count"] == sum(g["group"] % 2 == 0 for g in st["per_group"])


def test_old_json_without_pairing_rejected_and_inconsistent_flags_rejected():
    r = records(grounded)
    with pytest.raises(AssertionError, match="paired"):
        S.probe_stats([{k: v for k, v in e.items() if k != "layout_id"} for e in r])
    r[0] = r[0] | {"wrong_grasp": True}  # wrong_grasp without a recorded cube
    with pytest.raises(AssertionError, match="disagree"):
        S.probe_stats(r)


def test_cli_prints_and_writes(tmp_path, capsys):
    recs = records(ungrounded, 10)  # 10 groups drawn, one skipped (< 2 valid targets)
    ev = {"config": {"paired_targets": True, "seed": 21645}, "ckpt": "x", "ckpt_sha": "y", "episodes": recs}
    f = tmp_path / "e.json"
    f.write_text(json.dumps(ev))
    S.probe_s2a(str(f), "--out", str(tmp_path / "o.json"))
    out = capsys.readouterr().out
    assert "same cube regardless" in out and "went for the named cube" in out
    assert json.loads((tmp_path / "o.json").read_text())["groups"] == len({r["layout_id"] for r in recs})
