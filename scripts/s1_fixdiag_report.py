"""Stage 1 fix diagnosis (validation only) -> results/s1_fixdiag.json, applying the orchestrator's pre-registered D1 / D2
criteria mechanically to the pulled JSONs in results/s1_fixdiag/.

  python scripts/s1_fixdiag_report.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "s1_fixdiag"
SIGMA0_RANGE = (0.014, 0.045)  # criterion text: the sigma-0 demos' step-to-step label std range
PLATEAU = 0.46


def rate(a: dict, m: str) -> dict:
    x = a[m]
    return {"count": x["count"], "n": x["n"], "rate": x["rate"], "ci95": x["ci95"]}


def main() -> None:
    d1 = {}
    for v in ("default", "zero", "mean8"):
        j = json.loads((D / f"s1_fixdiag_{v}_s21600.json").read_text())
        a = j["aggregate"]["pick"]
        d1[v] = {"json": f"results/s1_fixdiag/s1_fixdiag_{v}_s21600.json", "sampling": j["server_info"]["sampling"],
                 "k": j["server_info"]["k"], "ckpt": j["ckpt"], "ckpt_sha": j["ckpt_sha"], "gpu_h": j["gpu_h"],
                 "server_s_p50": j["latency"]["server_s_p50"],
                 **{m: rate(a, m) for m in ("success", "slam", "jam", "drop", "early_close", "clean", "invalid")}}
        q = d1[v]
        q["qualifies"] = q["success"]["rate"] >= 0.80 and q["slam"]["rate"] < 0.05 and q["jam"]["rate"] < 0.02
    succ = {v: [e["success"] for e in json.loads((D / f"s1_fixdiag_{v}_s21600.json").read_text())["episodes"]]
            for v in d1}
    paired = {f"{v}_vs_default": {"only_variant": sum(x and not y for x, y in zip(succ[v], succ["default"])),
                                  "only_default": sum(y and not x for x, y in zip(succ[v], succ["default"]))}
              for v in ("zero", "mean8")}
    qual = [v for v in ("zero", "mean8") if d1[v]["qualifies"]]
    d1_res = {"variants": d1, "paired": paired, "rule": "variant success >= 80 % on 300 validation episodes (seed "
              "21600, 0-299), slam < 5 %, jam < 2 %; both -> higher, tie -> zero",
              "qualifying": qual, "fix_a_qualifies": bool(qual),
              "chosen": (max(qual, key=lambda v: (d1[v]["success"]["rate"], v == "zero")) if qual else None),
              "best_variant": max(("zero", "mean8"), key=lambda v: d1[v]["success"]["rate"])}

    st = json.loads((D / "stats.json").read_text())
    rp = json.loads((D / "replay_s21600.json").read_text())
    s1 = {k: v for k, v in st.items() if "/0c/s1/" in k}
    rel = [v["relabel"] for v in s1.values()]
    slope_ok = all(abs(s) < 0.1 for r in rel for s in r["slope_diff_on_prev_noise"])
    std_ok = all(SIGMA0_RANGE[0] <= s <= SIGMA0_RANGE[1] for r in rel for s in r["std_label_diff"])
    ex = rp["B_relabel_open_loop"]
    exe_ok = ex["n"] >= 100 and ex["success_rate"] >= 0.95
    fit = {}
    for k in ("orig", "relabel"):
        f = D / f"vloss_{k}_003000.json"
        fit[k] = json.loads(f.read_text()) if f.exists() else None
    fit_ok = None
    if fit["orig"] and fit["relabel"]:
        fit_ok = fit["relabel"]["val_loss"] < PLATEAU and fit["relabel"]["val_loss"] < fit["orig"]["val_loss"]
    d2 = {"definition": "r_t = clip(a_t + (x_{t-1} - a_{t-1}), -1, 1) on the 4 arm dims, r_0 = a_0, claw unchanged "
                        "(scripts/s1_relabel.py)",
          "offline": {"stats_json": "results/s1_fixdiag/stats.json", "per_source": st,
                      "slope_abs_lt_0p1": slope_ok, "std_diff_in_sigma0_range": std_ok,
                      "sigma0_range_as_written": SIGMA0_RANGE,
                      "matched_sigma0_clean_expert_val": rp["label_stats"]["E_clean_labels"]["std_label_diff"],
                      "relabel_on_val_demos": rp["label_stats"]["A_relabel"]["std_label_diff"]},
          "executability": {"replay_json": "results/s1_fixdiag/replay_s21600.json",
                            **{k: {kk: vv for kk, vv in v.items() if kk != "success_per_ep"}
                               for k, v in rp.items() if k[:2] in ("A_", "B_", "C_", "D_", "E_")},
                            "pass": exe_ok},
          "short_fit": {"steps": 3000, "batch": 64, "arm": "U", "val_loss": {k: (v["val_loss"] if v else None)
                                                                            for k, v in fit.items()},
                        "plateau": PLATEAU, "pass": fit_ok}}
    d2["fix_b_qualifies"] = bool(slope_ok and std_ok and exe_ok and fit_ok)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"D1": d1_res, "D2": d2, "git_sha": sha}
    (ROOT / "results" / "s1_fixdiag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"D1": {v: (d1[v]["success"]["rate"], d1[v]["slam"]["rate"], d1[v]["jam"]["rate"]) for v in d1},
                      "fix_a": d1_res["fix_a_qualifies"], "D2": {"slope": slope_ok, "std": std_ok, "exec": exe_ok,
                                                                 "fit": d2["short_fit"]["val_loss"],
                                                                 "fix_b": d2["fix_b_qualifies"]}}))


if __name__ == "__main__":
    main()
