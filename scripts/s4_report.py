"""GATES Stage 4 re-plan (e2cef02) 0C bookkeeping on the pulled /runs/k4_all outputs (no simulator; project venv).

  python scripts/s4_report.py [--raw results/0c_s4/raw] [--audit-mutants-log results/0c_s4_audit_mutants.txt]
                              [--out results/0c_s4.json]

Raw layout = hpc/steps/s4_checks.txt output (copy /runs/k4_all to --raw):
  <set>/0s4_check_<name>.json (scripts/s4_check.py)  <set>/dscheck.json  <set>/mutant_{shift,claw,claw_offset}.json
  <set>/audit.log (tests/test_data_audit.py --root)  <set>/*.rc   for set in s2g s3ag s3bg s4r;
  union/{audit.log,union_check.json,arms_build.json}  table_mutant/0s4_table_<twin>.json
--audit-mutants-log: `pytest tests/test_data_audit.py -v` output (the synthetic audit mutants, run once on the laptop / PERUN).
Writes --out: every item pass / fail and the numbers the data RESULT needs; prints one line per set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = {"s2g": "0c_s2_pick_rl224_y15g", "s3ag": "0c_s3a_place_rl224_y15g", "s3bg": "0c_s3b_stack_rl224_y15g",
        "s4r": "0c_s4r_mix_rl224_y15"}
M_A = "0c_s4r_mix_224a_y15"
UNION = "0c_s4_all_rl224_y15g"
TWIN_MUTANT = "0c_s2_pick_rl224_y15"
AUDIT_MUTANT_TESTS = ("test_mutant_heldout_place_pair", "test_mutant_heldout_target", "test_mutant_paraphrase_instruction",
                      "test_mutant_heldout_band_visuals", "test_mutant_split_rule_and_collision", "test_split_rule_in_export_select",
                      "test_s4r_visuals_audit", "test_s4r_train_palette_exact_and_mutants", "test_stage4_collision_flagged_at_m15")
SLOPE_KEYS = ("slope_diff_on_prev_noise", "std_label_diff")


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def violations(log: Path) -> list[int]:
    return [int(x) for x in re.findall(r'"violations": (\d+)', log.read_text())]


def rcs(d: Path) -> dict[str, int]:
    return {p.stem: int(p.read_text().strip()) for p in sorted(d.glob("*.rc"))}


def one(raw: Path, key: str) -> dict:
    d, name = raw / key, SETS[key]
    ck = load(d / f"0s4_check_{name}.json")
    ds = load(d / "dscheck.json")
    mu = {m: load(d / f"mutant_{m}.json") for m in ("shift", "claw", "claw_offset")}
    v = next(iter(ds["datasets"].values()))
    mv = {m: next(iter(x["datasets"].values())) for m, x in mu.items()}
    viol = violations(d / "audit.log")
    mc_equiv = mv["claw"]["mutant_rows_changed_vs_true_relabel"] == 0
    items = dict(ck["items"])
    items |= {
        "2_audit_0_violations": viol == [0],
        "4b_identity_every_row": v["identity_ok"], "4c_abs_slope_lt_0.1": v["slope_ok_4c"],
        "mutant_m_b_flagged": not mv["shift"]["pass_4b_4c"] and not mv["shift"]["slope_ok_4c"],
        "mutant_m_c_flagged_or_equivalent_with_m_c_prime_flagged": (not mv["claw"]["pass_4b_4c"]) or (
            mc_equiv and mv["claw_offset"]["claw_rows_ne_stored"] > 0 and not mv["claw_offset"]["identity_ok"]),
    }
    ma = None
    if key == "s4r":
        ma = next(iter(ds["mutants_original_labels"].values()))
        items["mutant_m_a_flagged"] = not ma["identity_ok"]
    cn = ck["columns_noise"]
    out = {
        "set": key, "dataset": name, "kind": ck["kind"], "items": items, "pass": all(items.values()) and ck["pass"],
        "failed_items": [k for k, x in items.items() if not x],
        "counts": {"episodes": ck["episodes"], "rows": ck["rows"], "T_per_task": ck["T_per_task"],
                   "source_successful_episodes": ck["source_successful_episodes"], "kept_stream_indices": ck["kept_stream_indices"]},
        "keep_rates": {t: {"pooled": x["pooled"], "batches": x["batches"], "ceiling_pct": x["ceiling_pct"]}
                       for t, x in ck["expert"].items()},
        "images": {"psnr_min": ck["video_check"]["psnr_min"], "psnr_mean": ck["video_check"]["psnr_mean"],
                   "identity_fail": ck["video_check"]["identity_fail"], "episodes_checked": len(ck["video_check"]["episodes"])},
        "yaw": ck["yaw"], "palette": ck["palette"], "mix": ck.get("mix"), "twin": ck.get("twin"),
        "noise": {"std_per_arm_dim": cn["noise_std_per_arm_dim"], "claw_exec_ne_label_rows": cn["claw_exec_ne_label_rows"],
                  "column_mismatch_episodes": cn["column_mismatch_episodes"]},
        "slope_4c": v["relabel"]["slope_diff_on_prev_noise"], "std_label_diff": v["relabel"]["std_label_diff"],
        "table": {"box": ck["table"]["box_u0_v0_u1_v1"], "reason": ck["table"]["reason"], **ck["table"]["numbers"]},
        "sheet": ck["sheet"], "audit_violations": viol,
        "mutants": {
            **({"m_a": {"dataset": M_A, "identity_ok": ma["identity_ok"], "rows_ne_relabel": ma["rows_ne_relabel"]}} if ma else {}),
            **{f"m_{k}": {x: mm[x] for x in ("relabel_fn", "identity_ok", "slope_ok_4c", "claw_rows_ne_stored",
                                              "mutant_rows_changed_vs_true_relabel")}
               | {"slope": mm["relabel"]["slope_diff_on_prev_noise"]}
               for k, mm in (("b", mv["shift"]), ("c", mv["claw"]), ("c_prime", mv["claw_offset"]))}},
        "rc": rcs(d), "wall_s": ck["wall_s"],
    }
    return out


def union(raw: Path) -> dict:
    d = raw / "union"
    uc, ar = load(d / "union_check.json"), load(d / "arms_build.json")
    viol = violations(d / "audit.log")
    items = {"union_check_pass": uc["pass"], "2_audit_0_violations": viol == [0], "instrument_i_arms_pass": ar["pass"],
             **{f"union_{k}": x for k, x in uc["items"].items()}}
    return {"dataset": UNION, "items": items, "pass": all(items.values()), "failed_items": [k for k, x in items.items() if not x],
            "S_option_B_train_windows_per_epoch": uc.get("S_option_B_train_windows_per_epoch"),
            "episodes": uc["columns_noise"]["episodes_checked"], "arms": {k: ar[k] for k in ("check_i_U", "check_i_mutant_fve_only", "counts")},
            "audit_violations": viol, "rc": rcs(d)}


def audit_mutants(log: Path | None) -> dict | None:
    if log is None or not log.exists():
        return None
    t = log.read_text()
    passed = {n: bool(re.search(rf"::{n} PASSED", t)) for n in AUDIT_MUTANT_TESTS}
    return {"log": str(log), "tests": passed, "all_flagged": all(passed.values()), "failed_anywhere": "FAILED" in t}


def build(raw: Path, audit_log: Path | None = None) -> dict:
    sets = {k: one(raw, k) for k in SETS}
    tm = load(raw / "table_mutant" / f"0s4_table_{TWIN_MUTANT}.json")
    un = union(raw)
    am = audit_mutants(audit_log)
    extra = {"table_colour_mutant_flagged": not tm["pass"], "audit_mutants_flagged": None if am is None else am["all_flagged"]}
    res = {"entry": "GATES Stage 4 re-plan, language-free part (e2cef02), 0C data re-gate", "sets": sets, "union": un,
           "table_mutant": {"dataset": TWIN_MUTANT, "flagged": not tm["pass"], "numbers": tm["table"]["numbers"],
                            "items": tm["items"]}, "audit_mutants": am, **{"extra_items": extra}}
    res["item_3b"] = "carried over, not re-run (GATES Stage 4 re-plan item 3)"
    res["pass"] = all(s["pass"] for s in sets.values()) and un["pass"] and extra["table_colour_mutant_flagged"] and \
        extra["audit_mutants_flagged"] is not False
    res["audit_mutants_not_supplied"] = am is None
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=ROOT / "results" / "0c_s4" / "raw")
    p.add_argument("--audit-mutants-log", type=Path, default=ROOT / "results" / "0c_s4_audit_mutants.txt")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "0c_s4.json")
    a = p.parse_args()
    res = build(a.raw, a.audit_mutants_log)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for k, s in res["sets"].items():
        print(k, "PASS" if s["pass"] else "FAIL", s["failed_items"])
    print("union", "PASS" if res["union"]["pass"] else "FAIL", res["union"]["failed_items"],
          "S", (res["union"]["S_option_B_train_windows_per_epoch"] or {}).get("total"))
    print("table mutant flagged:", res["table_mutant"]["flagged"], "| audit mutants:", res["extra_items"]["audit_mutants_flagged"])
    print("ALL_PASS" if res["pass"] else "NOT_ALL_PASS")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
