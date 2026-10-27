"""Task 004 (GATES "Inference-time instruction-guidance probe (004)"): readings from the four guided validation runs.
Reported only; NOT a gate. Pure python (either venv).

  python scripts/g4_report.py --evals results/s2a1_guidance/s2a1_guid_w{1,2,4,8}_s21640.json \
      --divs results/s2a1_guidance/div_w{1,2,4,8}_div.json --ref results/s2a1/s2a1_VT_val_c100_s21640.json \
      --out results/s2a1_guidance_v21640.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REF_RATE, SANITY_PTS = 0.280, 0.03  # s_100 validation 28.0 % (84 / 300), +-3 points
GAIN, PICK_OK, WRONG_OK, MOTOR, CUBE4_X = 0.10, 0.60, 0.15, 0.05, 2.0  # task 004 readings, verbatim
EPS = 1e-9  # rates are k / 300: a boundary reads as written, not by float rounding


def summary(ev: dict) -> dict:
    a = ev["aggregate"]["pick"]
    by = a["success_by_n_cubes"]
    return {"n": a["n"], "pick": a["success"]["rate"],
            "ceiling_pick_plus_wrong": (a["success"]["count"] + a["wrong_grasp"]["count"]) / a["n"], "pick_count": a["success"]["count"],
            "pick_ci95": a["success"]["ci95"], "wrong_cube": a["wrong_grasp"]["rate"],
            "wrong_cube_count": a["wrong_grasp"]["count"], "slam": a["slam"]["rate"], "jam": a["jam"]["rate"],
            "drop": a["drop"]["rate"], "by_n_cubes": {k: by[k]["rate"] for k in sorted(by)},
            "by_n_cubes_n": {k: by[k]["n"] for k in sorted(by)}, "gpu_h": ev["gpu_h"],
            "server_s_p50": ev["latency"]["server_s_p50"]}


def paired(ev_w: dict, ev_1: dict) -> dict:
    """Per-episode success discordance vs w = 1 (same episodes, same seeds)."""
    s1 = {r["index"]: r["success"] for r in ev_1["episodes"]}
    sw = {r["index"]: r["success"] for r in ev_w["episodes"]}
    assert s1.keys() == sw.keys()
    return {"gain": sum(sw[i] and not s1[i] for i in s1), "loss": sum(s1[i] and not sw[i] for i in s1)}


def readings(S: dict[float, dict]) -> dict:
    """S: w -> summary. The GATES 004 readings, mechanics as written there."""
    w1 = S[1.0]
    sanity = abs(w1["pick"] - REF_RATE) <= SANITY_PTS + EPS
    out: dict = {"sanity_w1": {"pick": w1["pick"], "ref": REF_RATE, "pass": sanity}}
    if not sanity:
        return out | {"best_w": None, "reading": "instrument finding (w = 1 outside 28.0 +- 3 points): no reading"}
    ws = sorted(w for w in S if w != 1.0)
    best = max(ws, key=lambda w: (S[w]["pick"], -w))  # tie -> the smaller w
    gains = {w: S[w]["pick"] - w1["pick"] for w in ws}
    helping = [w for w in ws if gains[w] >= GAIN - EPS]
    b = S[best]
    c4_1, c4_b = w1["by_n_cubes"].get("4"), b["by_n_cubes"].get("4")
    clauses = {"pick_ge_60": b["pick"] >= PICK_OK - EPS, "wrong_le_15": b["wrong_cube"] <= WRONG_OK + EPS,
               "cube4_ge_2x_w1": c4_b is not None and c4_1 is not None and c4_b >= CUBE4_X * c4_1 - EPS}
    motor = lambda w: S[w]["slam"] > MOTOR + EPS or S[w]["jam"] > MOTOR + EPS  # noqa: E731
    if not helping:
        reading = "no effect"
    elif len(helping) == 1 and motor(helping[0]):
        reading = "no effect"
    elif all(clauses.values()):
        reading = "guidance recovers grounding"
    else:
        reading = "partial"
    return out | {"best_w": best, "gain_vs_w1": gains, "helping_w": helping, "best_clauses": clauses,
                  "best_motor_over_5pct": motor(best), "reading": reading}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evals", type=Path, nargs="+", required=True)
    p.add_argument("--divs", type=Path, nargs="*", default=[])
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    E = {}
    for f in a.evals:
        ev = json.loads(f.read_text())
        c = ev["config"]
        assert (c["seed"], c["stage"], c["split"], ev["n_episodes"]) == (21640, "s2a", "train", 300), (f, c)
        E[float(c["guidance_w"])] = (f, ev)
    assert sorted(E) == [1.0, 2.0, 4.0, 8.0], sorted(E)
    ref = json.loads(a.ref.read_text())
    for w, (f, ev) in E.items():  # lead F4: the 2a validation protocol, only guidance_w differs
        rc = {k: v for k, v in ref["config"].items() if k != "ckpt"}
        ec = {k: v for k, v in ev["config"].items() if k not in ("ckpt", "guidance_w")}
        assert ec == rc, (f, {k: (ec.get(k), rc.get(k)) for k in set(ec) | set(rc) if ec.get(k) != rc.get(k)})
    shas = {ev["ckpt_sha"] for _, ev in E.values()} | {ref["ckpt_sha"]}
    assert len(shas) == 1, shas
    S = {w: summary(ev) | {"file": str(f)} for w, (f, ev) in E.items()}
    for w in S:
        S[w]["paired_vs_w1"] = paired(E[w][1], E[1.0][1])
    S_ref = summary(ref)
    D = {}
    for f in a.divs:
        d = json.loads(f.read_text())
        D[float(d["guidance_w"])] = {"file": str(f), **d["divergence"]}
    res = {"task": "004 inference-time instruction-guidance probe (diagnosis on validation, not an attempt)",
           "ckpt_sha": shas.pop(), "stream": "21640 eps 0-299 (s2a / train, m 15, H 224, N 10, B 128)",
           "ref_s100": S_ref | {"paired_vs_w1": paired(ref, E[1.0][1])},
           "by_w": {str(w): S[w] for w in sorted(S)}, "step2_by_w": {str(w): D[w] for w in sorted(D)},
           "gpu_h_evals": sum(S[w]["gpu_h"] for w in S),
           "ceiling_note": "ceiling_pick_plus_wrong = (success + wrong-cube) / n: the pick rate if every wrong grasp "
                           "became a success (reported only, lead F1)",
           "selection_disclosure":
           "best w chosen on the same 300 validation episodes it is read on (optimistic by selection); not a gate"}
    res |= readings(S)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("G4_REPORT=" + json.dumps({k: res[k] for k in ("sanity_w1", "best_w", "reading")} |
                                    {"pick": {w: S[w]["pick"] for w in sorted(S)},
                                     "wrong": {w: S[w]["wrong_cube"] for w in sorted(S)},
                                     "ratio": {w: D[w]["full_chunk"]["ratio"] for w in sorted(D)}}), flush=True)


if __name__ == "__main__":
    main()
