"""scripts/s4_report.py on a synthetic /runs/k4_all tree: clean tree passes, each mutated input flips its item."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s4_report as R  # noqa: E402

RATE = {"kept": 380, "attempted": 384, "rate_pct": 99.0, "ci95_pct": [97.4, 99.6]}


def check_json(key: str) -> dict:
    kind = "s4r" if key == "s4r" else "grey"
    tasks = ["pick", "place", "stack"] if key == "s4r" else [{"s2g": "pick", "s3ag": "place", "s3bg": "stack"}[key]]
    return {"kind": kind, "items": {"1_count_eq_K": True, "6_table_region_median_saturation_le_0.10": True}, "pass": True,
            "episodes": 2000, "rows": 1, "T_per_task": {t: 350 for t in tasks}, "source_successful_episodes": 2200,
            "kept_stream_indices": [0, 2100], "expert": {t: {"ceiling_pct": 100.0, "pooled": RATE, "batches": [RATE, RATE]} for t in tasks},
            "video_check": {"psnr_min": 40.0, "psnr_mean": 45.0, "identity_fail": 0, "episodes": list(range(10))},
            "yaw": {}, "palette": {}, "columns_noise": {"noise_std_per_arm_dim": [0.1] * 4, "claw_exec_ne_label_rows": 0,
                                                       "column_mismatch_episodes": []},
            "table": {"box_u0_v0_u1_v1": [0, 190, 64, 222], "reason": "r", "numbers": {"saturation_max": 0.03}},
            "sheet": {"png": "x.png"}, "wall_s": 1.0}


def ds_entry(identity=True, slope=True, claw_rows=0, changed=0) -> dict:
    return {"identity_ok": identity, "slope_ok_4c": slope, "pass_4b_4c": identity and slope, "claw_rows_ne_stored": claw_rows,
            "mutant_rows_changed_vs_true_relabel": changed, "relabel_fn": "f", "rows_ne_relabel": 0,
            "relabel": {"slope_diff_on_prev_noise": [0.0] * 4, "std_label_diff": [0.1] * 4}}


def write_tree(raw: Path) -> None:
    for key, name in R.SETS.items():
        d = raw / key
        d.mkdir(parents=True)
        (d / f"0s4_check_{name}.json").write_text(json.dumps(check_json(key)))
        (d / "dscheck.json").write_text(json.dumps({"datasets": {name: ds_entry()}, "pass": True, "mutants_original_labels": (
            {R.M_A: ds_entry(identity=False)} if key == "s4r" else {})}))
        for m, e in (("shift", ds_entry(False, False)), ("claw", ds_entry(True, True, 0, 0)),
                     ("claw_offset", ds_entry(False, True, 5, 5))):
            (d / f"mutant_{m}.json").write_text(json.dumps({"datasets": {name: e}, "pass": True}))
        (d / "audit.log").write_text('{"dataset": "x", "violations": 0, "first": []}')
        (d / "chk.rc").write_text("0")
    u = raw / "union"
    u.mkdir()
    (u / "audit.log").write_text('"violations": 0')
    (u / "union_check.json").write_text(json.dumps({"pass": True, "items": {"union_count": True}, "columns_noise": {"episodes_checked": 8000},
                                                    "S_option_B_train_windows_per_epoch": {"total": 1220800}}))
    (u / "arms_build.json").write_text(json.dumps({"pass": True, "check_i_U": True, "check_i_mutant_fve_only": False, "counts": {}}))
    t = raw / "table_mutant"
    t.mkdir()
    (t / f"0s4_table_{R.TWIN_MUTANT}.json").write_text(json.dumps({"pass": False, "items": {"6_x": False},
                                                                    "table": {"numbers": {"saturation_max": 0.6}}}))


def audit_log(skip=None) -> str:
    return "".join(f"tests/test_data_audit.py::{n} PASSED\n" for n in R.AUDIT_MUTANT_TESTS if n != skip)


def test_report_clean_and_mutants(tmp_path):
    raw = tmp_path / "raw"
    write_tree(raw)
    log = tmp_path / "am.txt"
    log.write_text(audit_log())
    res = R.build(raw, log)
    assert res["pass"], [(k, s["failed_items"]) for k, s in res["sets"].items()]
    assert res["union"]["S_option_B_train_windows_per_epoch"]["total"] == 1220800 and res["table_mutant"]["flagged"]
    assert res["sets"]["s4r"]["items"]["mutant_m_a_flagged"] and "mutant_m_a_flagged" not in res["sets"]["s2g"]["items"]

    def mutate(path: str, fn):
        r2 = tmp_path / "raw2"
        if r2.exists():
            import shutil
            shutil.rmtree(r2)
        shutil_copy(raw, r2)
        p = r2 / path
        d = json.loads(p.read_text()) if p.suffix == ".json" else p.read_text()
        out = fn(copy.deepcopy(d))
        p.write_text(json.dumps(out) if p.suffix == ".json" else out)
        return R.build(r2, log)

    assert not mutate("s3ag/audit.log", lambda t: '"violations": 2')["sets"]["s3ag"]["items"]["2_audit_0_violations"]
    r = mutate(f"s4r/dscheck.json", lambda d: d | {"mutants_original_labels": {R.M_A: ds_entry(identity=True)}})
    assert not r["sets"]["s4r"]["items"]["mutant_m_a_flagged"] and not r["pass"]
    r = mutate("s2g/mutant_shift.json", lambda d: {"datasets": {"x": ds_entry()}})
    assert not r["sets"]["s2g"]["items"]["mutant_m_b_flagged"]
    r = mutate("s2g/mutant_claw_offset.json", lambda d: {"datasets": {"x": ds_entry()}})
    assert not r["sets"]["s2g"]["items"]["mutant_m_c_flagged_or_equivalent_with_m_c_prime_flagged"]
    r = mutate(f"table_mutant/0s4_table_{R.TWIN_MUTANT}.json", lambda d: d | {"pass": True})
    assert not r["table_mutant"]["flagged"] and not r["pass"]
    r = mutate("union/union_check.json", lambda d: d | {"pass": False})
    assert not r["union"]["pass"] and not r["pass"]
    r = mutate("s3bg/0s4_check_%s.json" % R.SETS["s3bg"], lambda d: d | {"pass": False, "items": d["items"] | {"6_table_region_median_saturation_le_0.10": False}})
    assert r["sets"]["s3bg"]["failed_items"] == ["6_table_region_median_saturation_le_0.10"] and not r["pass"]
    log.write_text(audit_log(skip="test_mutant_heldout_target"))
    assert not R.build(raw, log)["pass"]
    assert R.build(raw, None)["audit_mutants_not_supplied"]


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil
    shutil.copytree(src, dst)
