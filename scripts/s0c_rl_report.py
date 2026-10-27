"""GATES 0C attempt 1 bookkeeping on the pulled JSONs (no simulator; project venv).

  python scripts/s0c_rl_report.py s1 [s2 s3a s3b]

Raw inputs per set in results/0c_rl/raw/<set>/: check.json (scripts/s0c_rl_check.py), dscheck.json + mutant_{shift,claw,
claw_offset}.json (scripts/s1_relabel.py dscheck), audit.log (tests/test_data_audit.py --root), replay.json (scripts/
s1_relabel.py replay, may hold several stages). Writes results/0c_rl.json (merged with sets already reported),
results/0c_rl/dscheck.json, results/0c_rl/replay.json, results/0c_rl_audit.txt, results/0c_rl_mutants.txt.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R, RAW = ROOT / "results", ROOT / "results" / "0c_rl" / "raw"
NAMES = {"s1": "0c_s1_pick_rl", "s2": "0c_s2_pick_rl", "s3a": "0c_s3a_place_rl", "s3b": "0c_s3b_stack_rl"}
EXEC_BAR = {"s1": 0.95, "s2": 0.95, "s3a": 0.95, "s3b": 0.923}  # GATES 0C attempt 1 item 3(b)
AUDIT_MUTANTS = 6  # GATES 0C item 2: the 6 synthetic audit mutants


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def one(s: str) -> dict:
    d = RAW / s
    ck, ds = load(d / "check.json"), load(d / "dscheck.json")
    me = {m: load(d / f"mutant_{m}.json") for m in ("shift", "claw", "claw_offset")}
    rp = load(d / "replay.json")["stages"][s]
    v = ds["datasets"][f"/data/datasets/{NAMES[s]}"]
    ma = next(iter(ds["mutants_original_labels"].values()))
    mv = {m: next(iter(x["datasets"].values())) for m, x in me.items()}
    audit = (d / "audit.log").read_text()
    viol = [int(x) for x in re.findall(r'"violations": (\d+)', audit)]
    am = (R / "0c_rl_audit_mutants.txt").read_text()
    am_pass = len(re.findall(r"::test_(mutant_\w+|split_rule_in_export_select) PASSED", am))
    b_ok, b_n = rp["B_relabel_open_loop"]["success_on_demo_successes"]
    mc_equiv = mv["claw"]["mutant_rows_changed_vs_true_relabel"] == 0
    items = {**{k: v_ for k, v_ in ck["items"].items()},
             "2_audit_0_violations": viol == [0],
             "2_audit_mutants_all_flagged": am_pass == AUDIT_MUTANTS,
             "3b_relabel_open_loop_on_A_successes_ge_bar": b_n > 0 and b_ok / b_n >= EXEC_BAR[s],
             "4b_identity_every_row": v["identity_ok"],
             "4c_abs_slope_lt_0.1": v["slope_ok_4c"],
             "mutant_m_a_flagged": not ma["identity_ok"],
             "mutant_m_b_flagged": not mv["shift"]["pass_4b_4c"] and not mv["shift"]["slope_ok_4c"],
             "mutant_m_c_flagged_or_equivalent_with_m_c_prime_flagged": (not mv["claw"]["pass_4b_4c"]) or (
                 mc_equiv and mv["claw_offset"]["claw_rows_ne_stored"] > 0 and not mv["claw_offset"]["identity_ok"])}
    E = rp["label_stats"]["E_clean_labels"]
    return {"set": s, "dataset": NAMES[s], "items": items, "pass_items_1_to_4": all(items.values()),
            "sheet": f"results/frames/0c_rl/{NAMES[s]}.png",
            "check": {k: ck[k] for k in ("episodes", "rows", "T", "videos", "label")} | {
                "noise_std_per_arm_dim": ck["columns_noise"]["noise_std_per_arm_dim"],
                "claw_exec_ne_label_rows": ck["columns_noise"]["claw_exec_ne_label_rows"],
                "column_mismatch_episodes": ck["columns_noise"]["column_mismatch_episodes"]},
            "labels": {"original": {k: v["original"][k] for k in ("std_label", "std_label_diff",
                                                                     "slope_diff_on_prev_noise", "r2")},
                       "relabel": {k: v["relabel"][k] for k in ("std_label", "std_label_diff",
                                                                    "slope_diff_on_prev_noise", "r2")},
                       "matched_sigma0_expert_E": {k: E[k] for k in ("std_label", "std_label_diff")},
                       "clip": v.get("clip")},
            "replay": {k: ({kk: vv for kk, vv in x.items() if kk != "success_per_ep"} if isinstance(x, dict) else x)
                       for k, x in rp.items() if k != "label_stats"} | {"bar_B_on_A": EXEC_BAR[s],
                                                                       "B_on_A": b_ok / b_n if b_n else None},
            "mutants": {"m_a": {"dataset": next(iter(ds["mutants_original_labels"])), "identity_ok": ma["identity_ok"],
                                "rows_ne_relabel": ma["rows_ne_relabel"]},
                        **{f"m_{k}": {kk: x[kk] for kk in ("relabel_fn", "identity_ok", "slope_ok_4c", "claw_rows_ne_stored",
                                                           "mutant_rows_changed_vs_true_relabel")}
                           | {"slope": x["relabel"]["slope_diff_on_prev_noise"]}
                           for k, x in (("b", mv["shift"]), ("c", mv["claw"]), ("c_prime", mv["claw_offset"]))}},
            "audit": {"violations": viol, "synthetic_mutants_flagged": am_pass}}


def main() -> None:
    out = R / "0c_rl.json"
    res = load(out) if out.exists() else {"entry": "GATES Stage 0C attempt 1 (c6dc514)", "sets": {}}
    for s in sys.argv[1:]:
        res["sets"][s] = one(s)
        print(s, json.dumps(res["sets"][s]["items"]))
    out.write_text(json.dumps(res, indent=1))
    sets = sorted(res["sets"])
    (R / "0c_rl" / "dscheck.json").write_text(json.dumps({s: load(RAW / s / "dscheck.json") for s in sets}, indent=1))
    (R / "0c_rl" / "replay.json").write_text(json.dumps({s: load(RAW / s / "replay.json")["stages"][s] for s in sets},
                                                        indent=1))
    (R / "0c_rl_audit.txt").write_text("".join(f"# {s}\n" + (RAW / s / "audit.log").read_text() for s in sets))
    lines = []
    for s in sets:
        m = res["sets"][s]["mutants"]
        lines += [f"{s} m-a (attempt-0 set, original labels): identity_ok={m['m_a']['identity_ok']} "
                  f"rows_ne_relabel={m['m_a']['rows_ne_relabel']} -> flagged={not m['m_a']['identity_ok']}",
                  f"{s} m-b (shift, shared function): identity_ok={m['m_b']['identity_ok']} slope="
                  f"{[round(x, 3) for x in m['m_b']['slope']]} -> flagged by (c)={not m['m_b']['slope_ok_4c']}",
                  f"{s} m-c (claw relabelled, as written): rows changed vs true relabel="
                  f"{m['m_c']['mutant_rows_changed_vs_true_relabel']} claw_rows_ne_stored={m['m_c']['claw_rows_ne_stored']}"
                  f" -> flagged={not m['m_c']['identity_ok']}",
                  f"{s} m-c' (arm correction one dim right, dyaw onto the claw): claw_rows_ne_stored="
                  f"{m['m_c_prime']['claw_rows_ne_stored']} -> flagged={not m['m_c_prime']['identity_ok']}"]
    (R / "0c_rl_mutants.txt").write_text("# GATES 0C attempt 1 item 4 mutants (scripts/s1_relabel.py dscheck)\n"
                                         + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
