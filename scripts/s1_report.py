"""GATES Stage 1 bookkeeping on the pulled JSONs (no simulator, no model; project venv).

  python scripts/s1_report.py val    -> results/s1_{frozen,unfrozen}_val.json (s_50 / s_75 / s_100, checkpoint pick)
  python scripts/s1_report.py choice -> results/s1_choice.json (encoder choice on validation seed 21610)
  python scripts/s1_report.py final  -> results/s1.json (test numbers per arm, per colour / size, failure modes)
  python scripts/s1_report.py val_s3 <a|b>   -> results/s3{a,b}_val.json (two-skill checkpoint pick, raw in results/s3{a,b}/)
  python scripts/s1_report.py final_s3 <a|b> [--k K] -> results/s3{a,b}.json (gate; pick paired with Stage 2 (a), 11521)
  python scripts/s1_report.py val_s3 <ap|bp> | final_s3 <ap|bp> [--k K] -> results/s3{a,b}p_val.json / s3{a,b}p.json (task 007:
      3A / 3B amended, paired data + contrastive loss, arm VT, run /runs/s3{a,b}p_VT, init = the 2b amended gated checkpoint,
      pick reference results/s2bp.json; + guards (b1) / (b2) at c050, init sha, one stats_sha over every eval)
  python scripts/s1_report.py val_s3 <a1p|b1p> | final_s3 <a1p|b1p> [--k K] -> results/s3{a,b}1p_val.json / s3{a,b}1p.json (task 009b:
      one cube size per scene, samplers s3a1 / s3b1, init = the Stage 2M pick, pick skill s2a (validation 21640, test 11544 vs
      results/s2m*.json), guards c050 wrong target > 10 %, c075 success < bar - 15 / slam > 10 %)
  python scripts/s1_report.py expert_s3 <a1|b1> -> results/s3{a,b}1p_pre.json (009b precondition: expert / chunked, seed 11840 / 11850)
  python scripts/s1_report.py val_s2base | final_s2base [--ckpt-dir D] -> results/s2base_val.json / s2base.json (Stage 2 redesign (a))
  python scripts/s1_report.py lang_s2base <c050|c100> -> results/s2base_lang_{ck}.json (language check, reported only)
  python scripts/s1_report.py val_s2b | final_s2b [--ckpt-dir D] [--init-sha S] | lang_s2b <c050|c100> -> results/s2b_val.json / s2b.json /
      s2b_lang_{ck}.json (Stage 2 2b: arm VT, stage s2, init = 2a's gated checkpoint; --init-sha default results/s2a.json ckpt_sha)
  python scripts/s1_report.py init_disc <steps.jsonl> [--ref-threshold 0.5] [--expect gt|le] -> instrument I2 (exit 0 init
      confirmed, 1 finding; gt = mean loss steps 1-50 > threshold = base init, le = <= threshold = Stage-1-checkpoint init)
  python scripts/s1_report.py val_s2a | final_s2a [--ckpt-dir D] | lang_s2a <c050|c100> -> results/s2a_val.json / s2a.json /
      s2a_lang_{ck}.json (Stage 2 redesign #2, 2a: colours-only pick, arm VT from the Stage 1 V15 checkpoint, raw in results/s2a/)
  python scripts/s1_report.py probe_s3 <eval json> [--out results/<name>.json] -> the same probe for stage s3a / s3b (place /
      stack): named cube, named destination (rest_dest), both, split by member relation, same (cube, destination) regardless
  python scripts/s1_report.py probe_s2a <eval json> [--out results/<name>.json] -> counterfactual paired-scene probe (eval_policy
      --paired-targets): per member success + grasped cube, "went for the named cube", "same cube regardless", chance levels,
      per-group table (prints the table; writes the JSON only with --out)
Raw inputs live in results/s1/ (eval JSONs, vloss / offline JSONs, train_summary_<arm>.json, arms_*.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lift_rl.metrics import wilson_ci  # noqa: E402

R, RAW = ROOT / "results", ROOT / "results" / "s1"
ARMS = ("frozen", "unfrozen")
CKPTS = {"c050": "010332", "c075": "015498", "c100": "020664"}
FAILS = ("slam", "slam_10N", "slam_50N", "jam", "wrong_grasp", "wrong_object", "early_close", "drop",
         "distractor_disturbed", "invalid", "clean")


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def succ(ev: dict, n: int | None = None) -> float:
    eps = ev["episodes"][:n] if n else ev["episodes"]
    return sum(e["success"] for e in eps) / len(eps)


def rate(xs: list[bool]) -> dict:
    k, n = sum(xs), len(xs)
    return {"count": k, "n": n, "rate": k / n if n else None, "ci95": list(wilson_ci(k, n))}


def pick_rule(s: dict[str, float], n: int) -> dict:
    """GATES Stage 1 checkpoint pick: the 100 % checkpoint unless s_100 < max(s_50, s_75) - 10 points; if s_100 is
    within 5 points of that threshold at n = 300 -> extend all three to 1,000; on collapse the later of 50 / 75 % with
    success >= max - 5 points."""
    m = max(s["c050"], s["c075"])
    thr = m - 0.10
    if n < 1000 and abs(s["c100"] - thr) <= 0.05:
        return {"threshold": thr, "decision": "extend to 1000", "pick": None}
    if s["c100"] >= thr:
        return {"threshold": thr, "collapsed": False, "pick": "c100", "decision": f"no collapse at n={n}"}
    late = [c for c in ("c075", "c050") if s[c] >= m - 0.05]
    return {"threshold": thr, "collapsed": True, "pick": late[0], "decision": f"collapse at n={n}"}


def val() -> None:
    for arm in ARMS:
        ev = {c: load(RAW / f"s1_{arm}_val_{c}_s21600.json") for c in CKPTS}
        n = min(len(e["episodes"]) for e in ev.values())
        s300 = {c: succ(e, 300) for c, e in ev.items()}
        res = {"arm": arm, "seed": 21600, "s_300": s300, "pick_300": pick_rule(s300, 300)}
        if n >= 1000:
            s1000 = {c: succ(e, 1000) for c, e in ev.items()}
            res |= {"s_1000": s1000, "pick_1000": pick_rule(s1000, 1000)}
        final = res.get("pick_1000", res["pick_300"])
        res["pick"] = final["pick"]
        res["pick_step"] = CKPTS[final["pick"]] if final["pick"] else None
        res["per_ckpt"] = {c: {"step": CKPTS[c], "ckpt_sha": e["ckpt_sha"], "n": len(e["episodes"]),
                               **{m: e["aggregate"]["pick"][m] for m in ("success", "slam", "jam", "drop", "early_close")},
                               "server_s_p50": e["latency"]["server_s_p50"], "gpu_h": e["gpu_h"]}
                           for c, e in ev.items()}
        res["early_abort_d"] = {"s_50": s300["c050"], "abort": s300["c050"] <= 0.05}
        vl = {p.stem: load(p)["val_loss"] for p in sorted(RAW.glob(f"{arm}_vloss_*.json"))}
        res["offline_val_loss"] = vl
        a, b = vl.get(f"{arm}_vloss_005166"), vl.get(f"{arm}_vloss_010332")
        res["early_abort_c"] = {"loss_25": a, "loss_50": b, "abort": (b > 2 * a) if a and b else None}
        off = RAW / f"{arm}_offline_val_020664.json"
        if off.exists():
            o = load(off)
            res["offline_val_100"] = {"mae_arm_norm": o["mae_arm_norm"], "claw_acc": o["claw_acc"],
                                      "episodes": len(o["per_episode"])}
        (R / f"s1_{arm}_val.json").write_text(json.dumps(res, indent=1))
        print(arm, json.dumps({k: res[k] for k in ("s_300", "pick_300", "pick", "early_abort_c", "early_abort_d")}))


def choice() -> None:
    s = {arm: load(RAW / f"s1_{arm}_choice_s21610.json") for arm in ARMS}
    sf, su = succ(s["frozen"], 1000), succ(s["unfrozen"], 1000)
    assert all(len(e["episodes"]) == 1000 for e in s.values())
    res = {"seed": 21610, "episodes": [0, 999], "s_frozen": sf, "s_unfrozen": su, "diff_points": 100 * (su - sf),
           "rule": "U iff s_U > s_F + 3 points, else F", "choice": "unfrozen" if su > sf + 0.03 else "frozen",
           "ckpts": {a: e["ckpt"] for a, e in s.items()}, "ckpt_sha": {a: e["ckpt_sha"] for a, e in s.items()},
           "aggregate": {a: {m: e["aggregate"]["pick"][m] for m in ("success", "slam", "jam")} for a, e in s.items()}}
    (R / "s1_choice.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def breakdown(eps: list[dict]) -> dict:
    out = {"colour": {}, "size": {}, "cube": {}}
    for e in eps:
        col, size = e["target"].split("_")
        for k, v in (("colour", col), ("size", size), ("cube", e["target"])):
            out[k].setdefault(v, []).append(e["success"])
    return {k: {v: rate(x) for v, x in sorted(d.items())} for k, d in out.items()}


def final() -> None:
    ch = load(R / "s1_choice.json")
    res = {"choice": ch["choice"], "arms": {}}
    for arm in ARMS:
        t = load(R / f"s1_{arm}_test_s11500_0.json")
        eps = t["episodes"]
        fail = [e for e in eps if not e["success"]]
        res["arms"][arm] = {
            "gated": arm == ch["choice"], "ckpt": t["ckpt"], "ckpt_sha": t["ckpt_sha"], "n": len(eps),
            "success": rate([e["success"] for e in eps]), **{m: rate([e[m] for e in eps]) for m in FAILS},
            "smart_length": t["smart_length"], "bars": t["bars"], "by": breakdown(eps),
            "failures_with_no_flag": sum(not any(e[m] for m in ("slam", "jam", "wrong_object", "early_close", "drop",
                                                                 "invalid")) for e in fail),
            "failures_never_grasped": sum(not e["grasp_confirmed"] for e in fail),
            "latency": t["latency"], "eval_gpu_h": t["gpu_h"]}
    (R / "s1.json").write_text(json.dumps(res, indent=1))
    for arm, d in res["arms"].items():
        print(arm, d["gated"], d["success"]["rate"], d["slam"]["rate"], d["jam"]["rate"], d["smart_length"],
              {k: v["rate"] for k, v in d["by"]["colour"].items()}, {k: v["rate"] for k, v in d["by"]["size"].items()})


RAW_A1 = R / "s1a1"


def default_sampling(ev: dict) -> bool:
    """GATES Stage 1 attempt 1: the server ran --sampling default (asserted on every evaluation JSON)."""
    return ev["config"].get("sampling", "default") == "default" and ev["server_info"].get("sampling") == "default"


def val_a1() -> None:
    """Attempt 1 (U, relabelled labels): checkpoint pick on seed 21600 + paired comparison with attempt-0 U."""
    ev = {c: load(RAW_A1 / f"s1a1_unfrozen_val_{c}_s21600.json") for c in CKPTS}
    a0 = {c: load(RAW / f"s1_unfrozen_val_{c}_s21600.json") for c in CKPTS}
    assert all(default_sampling(e) for e in ev.values()), "a validation run did not use default sampling"
    n = min(len(e["episodes"]) for e in ev.values())
    s300 = {c: succ(e, 300) for c, e in ev.items()}
    res = {"arm": "unfrozen", "attempt": 1, "seed": 21600, "sampling": "default (asserted)", "s_300": s300,
           "pick_300": pick_rule(s300, 300)}
    if n >= 1000:
        s1000 = {c: succ(e, 1000) for c, e in ev.items()}
        res |= {"s_1000": s1000, "pick_1000": pick_rule(s1000, 1000)}
    final_ = res.get("pick_1000", res["pick_300"])
    res["pick"] = final_["pick"]
    res["pick_step"] = CKPTS[final_["pick"]] if final_["pick"] else None
    res["per_ckpt"] = {c: {"step": CKPTS[c], "ckpt": e["ckpt"], "ckpt_sha": e["ckpt_sha"], "n": len(e["episodes"]),
                           **{m: e["aggregate"]["pick"][m] for m in ("success", "slam", "jam", "drop", "early_close")},
                           "server_s_p50": e["latency"]["server_s_p50"], "gpu_h": e["gpu_h"]}
                       for c, e in ev.items()}
    pair = {}
    for c in CKPTS:
        x = [e["success"] for e in ev[c]["episodes"][:300]]
        y = [e["success"] for e in a0[c]["episodes"][:300]]
        assert [e["spec"] for e in ev[c]["episodes"][:300]] == [e["spec"] for e in a0[c]["episodes"][:300]]
        pair[c] = {"a1": sum(x) / 300, "a0": sum(y) / 300, "diff_points": 100 * (sum(x) - sum(y)) / 300,
                   "a1_only": sum(a and not b for a, b in zip(x, y)), "a0_only": sum(b and not a for a, b in zip(x, y))}
    res["paired_vs_attempt0_U_300"] = pair
    res["early_abort_d"] = {"s_50": s300["c050"], "abort": s300["c050"] <= 0.05}
    vl = {p.stem: load(p)["val_loss"] for p in sorted(RAW_A1.glob("vloss_*.json"))}
    res["offline_val_loss_vs_r"] = vl
    a, b = vl.get("vloss_005166"), vl.get("vloss_010332")
    res["early_abort_c"] = {"loss_25": a, "loss_50": b, "abort": (b > 2 * a) if a and b else None}
    off = RAW_A1 / "offline_val_020664.json"
    if off.exists():
        o = load(off)
        res["offline_val_100"] = {"mae_arm_norm": o["mae_arm_norm"], "claw_acc": o["claw_acc"],
                                  "episodes": len(o["per_episode"])}
    (R / "s1a1_unfrozen_val.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("s_300", "pick_300", "pick", "paired_vs_attempt0_U_300", "early_abort_c",
                                          "early_abort_d")}))


def final_a1() -> None:
    t = load(R / "s1_unfrozen_test_s11501_1.json")
    assert default_sampling(t) and t["config"]["seed"] == 11501 and len(t["episodes"]) == 1000
    v = load(R / "s1a1_unfrozen_val.json")
    assert t["ckpt"].endswith(f"/{v['pick_step']}/pretrained_model"), (t["ckpt"], v["pick_step"])
    eps = t["episodes"]
    fail = [e for e in eps if not e["success"]]
    res = {"attempt": 1, "arm": "unfrozen", "seed": 11501, "ckpt": t["ckpt"], "ckpt_sha": t["ckpt_sha"],
           "sampling": "default (asserted)", "n": len(eps), "success": rate([e["success"] for e in eps]),
           **{m: rate([e[m] for e in eps]) for m in FAILS}, "smart_length": t["smart_length"], "bars": t["bars"],
           "by": breakdown(eps),
           "failures_with_no_flag": sum(not any(e[m] for m in ("slam", "jam", "wrong_object", "early_close", "drop",
                                                                "invalid")) for e in fail),
           "failures_never_grasped": sum(not e["grasp_confirmed"] for e in fail),
           "first_failures": [e["index"] for e in fail[:20]], "latency": t["latency"], "eval_gpu_h": t["gpu_h"],
           "attempt0_U_test_11500": {m: load(R / "s1_unfrozen_test_s11500_0.json")["aggregate"]["pick"][m]
                                     for m in ("success", "slam", "jam")}}
    (R / "s1a1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("success", "slam", "jam", "smart_length", "failures_never_grasped")}))
    print({k: (v_["count"], v_["n"], round(v_["rate"], 3)) for k, v_ in res["by"]["cube"].items()})
    print({k: round(v_["rate"], 3) for d in ("colour", "size") for k, v_ in res["by"][d].items()})
    print({m: res[m]["rate"] for m in FAILS})


RAW_R = R / "s1r"
CKPTS_R = {"c050": "020664", "c075": "030996", "c100": "041328"}  # Stage 1 redesign: 48 epochs = 41,328 steps


def paired(x: list[bool], y: list[bool]) -> dict:
    n = len(x)
    b, c = sum(a and not b_ for a, b_ in zip(x, y)), sum(b_ and not a for a, b_ in zip(x, y))
    return {"n": n, "redesign": sum(x) / n, "attempt1": sum(y) / n, "diff_points": 100 * (sum(x) - sum(y)) / n,
            "redesign_only": b, "attempt1_only": c, "mcnemar_z": (b - c) / (b + c) ** 0.5 if b + c else None}


def val_r() -> None:
    """Stage 1 redesign (H 224, N 10, 48 epochs): checkpoint pick on seed 21600 (rule unchanged) + paired descriptive
    comparison with attempt 1 (H 128, N 20, 24 epochs) on the same 300 episodes."""
    ev = {c: load(RAW_R / f"s1r_U_val_{c}_s21600.json") for c in CKPTS_R}
    a1 = {c: load(RAW_A1 / f"s1a1_unfrozen_val_{c}_s21600.json") for c in CKPTS}
    assert all(default_sampling(e) for e in ev.values()), "a validation run did not use default sampling"
    assert all(e["config"]["n"] == 10 and e["config"]["image_size"] == 224 and e["config"]["seed"] == 21600
               for e in ev.values())
    assert all(e["ckpt"].endswith(f"/{CKPTS_R[c]}/pretrained_model") for c, e in ev.items())
    n = min(len(e["episodes"]) for e in ev.values())
    s300 = {c: succ(e, 300) for c, e in ev.items()}
    res = {"arm": "unfrozen", "design": "Stage 1 redesign (H 224, N 10, 48 epochs)", "seed": 21600,
           "sampling": "default (asserted)", "n_action_steps": 10, "image_size": 224, "s_300": s300,
           "pick_300": pick_rule(s300, 300)}
    if n >= 1000:
        s1000 = {c: succ(e, 1000) for c, e in ev.items()}
        res |= {"s_1000": s1000, "pick_1000": pick_rule(s1000, 1000)}
    final_ = res.get("pick_1000", res["pick_300"])
    res["pick"] = final_["pick"]
    res["pick_step"] = CKPTS_R[final_["pick"]] if final_["pick"] else None
    res["per_ckpt"] = {c: {"step": CKPTS_R[c], "ckpt": e["ckpt"], "ckpt_sha": e["ckpt_sha"], "n": len(e["episodes"]),
                           **{m: e["aggregate"]["pick"][m] for m in ("success", "slam", "jam", "drop", "early_close")},
                           "server_s_p50": e["latency"]["server_s_p50"], "gpu_h": e["gpu_h"]}
                       for c, e in ev.items()}
    pair = {}
    for c in CKPTS:
        assert [e["spec"] for e in ev[c]["episodes"][:300]] == [e["spec"] for e in a1[c]["episodes"][:300]]
        pair[c] = paired([e["success"] for e in ev[c]["episodes"][:300]],
                         [e["success"] for e in a1[c]["episodes"][:300]])
    res["paired_vs_attempt1_300"] = pair
    res["early_abort_d"] = {"s_50": s300["c050"], "abort": s300["c050"] <= 0.05}
    vl = {p_.stem: load(p_)["val_loss"] for p_ in sorted(RAW_R.glob("vloss_*.json"))}
    res["offline_val_loss_vs_r"] = vl
    a, b = vl.get("vloss_010332"), vl.get("vloss_020664")
    res["early_abort_c"] = {"loss_25": a, "loss_50": b, "abort": (b > 2 * a) if a and b else None}
    off = RAW_R / "offline_val_041328.json"
    if off.exists():
        o = load(off)
        res["offline_val_100"] = {"mae_arm_norm": o["mae_arm_norm"], "claw_acc": o["claw_acc"],
                                  "episodes": len(o["per_episode"])}
    (R / "s1r_U_val.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("s_300", "pick_300", "pick", "paired_vs_attempt1_300", "early_abort_c",
                                          "early_abort_d")}))


def final_r() -> None:
    t = load(R / "s1r_U_test_s11502.json")
    assert default_sampling(t) and t["config"]["seed"] == 11502 and len(t["episodes"]) == 1000
    assert t["config"]["n"] == 10 and t["config"]["image_size"] == 224
    v = load(R / "s1r_U_val.json")
    assert t["ckpt"].endswith(f"/{v['pick_step']}/pretrained_model"), (t["ckpt"], v["pick_step"])
    eps = t["episodes"]
    fail = [e for e in eps if not e["success"]]
    res = {"design": "Stage 1 redesign (H 224, N 10, 48 epochs)", "arm": "unfrozen", "seed": 11502, "ckpt": t["ckpt"],
           "ckpt_sha": t["ckpt_sha"], "sampling": "default (asserted)", "n": len(eps),
           "success": rate([e["success"] for e in eps]), **{m: rate([e[m] for e in eps]) for m in FAILS},
           "smart_length": t["smart_length"], "bars": t["bars"], "by": breakdown(eps),
           "failures_with_no_flag": sum(not any(e[m] for m in ("slam", "jam", "wrong_object", "early_close", "drop",
                                                                "invalid")) for e in fail),
           "failures_never_grasped": sum(not e["grasp_confirmed"] for e in fail),
           "first_failures": [e["index"] for e in fail[:20]], "latency": t["latency"], "eval_gpu_h": t["gpu_h"],
           "attempt1_U_test_11501": {m: load(R / "s1_unfrozen_test_s11501_1.json")["aggregate"]["pick"][m]
                                     for m in ("success", "slam", "jam")}}
    (R / "s1r.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("success", "slam", "jam", "smart_length", "failures_never_grasped")}))
    print({k: (v_["count"], v_["n"], round(v_["rate"], 3)) for k, v_ in res["by"]["cube"].items()})
    print({k: round(v_["rate"], 3) for d in ("colour", "size") for k, v_ in res["by"][d].items()})
    print({m: res[m]["rate"] for m in FAILS})





RAW_R2 = R / "s1r2"  # Stage 1 redesign attempt 1 (2,000 demos, same 41,328 steps = 12 epochs): same checkpoint steps


def val_r2() -> None:
    """Redesign attempt 1: checkpoint pick on 21600 (rule unchanged) + paired descriptive comparison with the redesign."""
    ev = {c: load(RAW_R2 / f"s1r2_U_val_{c}_s21600.json") for c in CKPTS_R}
    ref = {c: load(RAW_R / f"s1r_U_val_{c}_s21600.json") for c in CKPTS_R}
    assert all(default_sampling(e) for e in ev.values()), "a validation run did not use default sampling"
    assert all(e["config"]["n"] == 10 and e["config"]["image_size"] == 224 and e["config"]["seed"] == 21600
               and e["ckpt"].endswith(f"/s1r2_U/checkpoints/{CKPTS_R[c]}/pretrained_model") for c, e in ev.items())
    n = min(len(e["episodes"]) for e in ev.values())
    s300 = {c: succ(e, 300) for c, e in ev.items()}
    res = {"arm": "unfrozen", "design": "Stage 1 redesign attempt 1 (2,000 demos, H 224, N 10, 41,328 steps)",
           "seed": 21600, "sampling": "default (asserted)", "s_300": s300, "pick_300": pick_rule(s300, 300)}
    if n >= 1000:
        s1000 = {c: succ(e, 1000) for c, e in ev.items()}
        res |= {"s_1000": s1000, "pick_1000": pick_rule(s1000, 1000)}
    final_ = res.get("pick_1000", res["pick_300"])
    res["pick"] = final_["pick"]
    res["pick_step"] = CKPTS_R[final_["pick"]] if final_["pick"] else None
    res["per_ckpt"] = {c: {"step": CKPTS_R[c], "ckpt": e["ckpt"], "ckpt_sha": e["ckpt_sha"], "n": len(e["episodes"]),
                           **{m: e["aggregate"]["pick"][m] for m in ("success", "slam", "jam", "drop", "early_close")},
                           "gpu_h": e["gpu_h"]} for c, e in ev.items()}
    pair = {}
    for c in CKPTS_R:
        assert [e["spec"] for e in ev[c]["episodes"][:300]] == [e["spec"] for e in ref[c]["episodes"][:300]]
        p_ = paired([e["success"] for e in ev[c]["episodes"][:300]], [e["success"] for e in ref[c]["episodes"][:300]])
        pair[c] = {"attempt1_2k": p_["redesign"], "redesign_500": p_["attempt1"], "diff_points": p_["diff_points"],
                   "2k_only": p_["redesign_only"], "500_only": p_["attempt1_only"], "mcnemar_z": p_["mcnemar_z"]}
    res["paired_vs_redesign_300"] = pair
    res["early_abort_d"] = {"s_50": s300["c050"], "abort": s300["c050"] <= 0.05}
    vl = {p_.stem: load(p_)["val_loss"] for p_ in sorted(RAW_R2.glob("vloss_*.json"))}
    res["offline_val_loss_vs_r"] = vl
    a, b = vl.get("vloss_010332"), vl.get("vloss_020664")
    res["early_abort_c"] = {"loss_25": a, "loss_50": b, "abort": (b > 2 * a) if a and b else None}
    off = RAW_R2 / "offline_val_041328.json"
    if off.exists():
        o = load(off)
        res["offline_val_100"] = {"mae_arm_norm": o["mae_arm_norm"], "claw_acc": o["claw_acc"],
                                  "episodes": len(o["per_episode"])}
    (R / "s1r2_U_val.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("s_300", "pick_300", "pick", "paired_vs_redesign_300", "early_abort_c",
                                          "early_abort_d")}))


def final_r2() -> None:
    t = load(R / "s1r2_U_test_s11503.json")
    assert default_sampling(t) and t["config"]["seed"] == 11503 and len(t["episodes"]) == 1000
    assert t["config"]["n"] == 10 and t["config"]["image_size"] == 224
    v = load(R / "s1r2_U_val.json")
    assert t["ckpt"].endswith(f"/s1r2_U/checkpoints/{v['pick_step']}/pretrained_model"), (t["ckpt"], v["pick_step"])
    eps = t["episodes"]
    fail = [e for e in eps if not e["success"]]
    res = {"design": "Stage 1 redesign attempt 1 (2,000 demos)", "arm": "unfrozen", "seed": 11503, "ckpt": t["ckpt"],
           "ckpt_sha": t["ckpt_sha"], "sampling": "default (asserted)", "n": len(eps),
           "success": rate([e["success"] for e in eps]), **{m: rate([e[m] for e in eps]) for m in FAILS},
           "smart_length": t["smart_length"], "bars": t["bars"], "by": breakdown(eps),
           "failures_with_no_flag": sum(not any(e[m] for m in ("slam", "jam", "wrong_object", "early_close", "drop",
                                                                "invalid")) for e in fail),
           "failures_never_grasped": sum(not e["grasp_confirmed"] for e in fail),
           "first_failures": [e["index"] for e in fail[:20]], "latency": t["latency"], "eval_gpu_h": t["gpu_h"]}
    (R / "s1r2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("success", "slam", "jam", "smart_length", "failures_never_grasped")}))
    for d in ("cube", "colour", "size"):
        print(d, {k: (v_["count"], v_["n"], round(100 * v_["rate"], 1), [round(100 * x, 1) for x in v_["ci95"]])
                  for k, v_ in res["by"][d].items()})
    print({m: res[m]["rate"] for m in FAILS})


YAW_RULE = ("test V15 if v15 >= 0.90; else V0 if v0 >= 0.90; else the higher of v15 / v0 (tie -> V15); "
            "a missing variant cannot be selected; a single present variant is tested")


def check_yaw(ev: dict, m: float) -> None:
    assert ev["config"]["cube_yaw_max_deg"] == m, (ev["config"]["cube_yaw_max_deg"], m)
    ys = [e["spawn_yaw_fold_deg"] for e in ev["episodes"]]
    assert all(-m - 0.5 <= y <= m + 0.5 for y in ys), (min(ys), max(ys), m)


def val_y(V: str) -> None:
    """Redesign #2 variant V (yaw +-V deg): checkpoint pick on 21600 at the variant's m (rule unchanged)."""
    V = float(V)
    tag = f"s1y{int(V)}"
    raw = R / tag
    ev = {c: load(raw / f"{tag}_U_val_{c}_s21600.json") for c in CKPTS_R}
    assert all(default_sampling(e) for e in ev.values()), "a validation run did not use default sampling"
    for c, e in ev.items():
        assert e["config"]["n"] == 10 and e["config"]["image_size"] == 224 and e["config"]["seed"] == 21600
        assert e["ckpt"].endswith(f"/{tag}_U/checkpoints/{CKPTS_R[c]}/pretrained_model"), e["ckpt"]
        check_yaw(e, V)
    n = min(len(e["episodes"]) for e in ev.values())
    s300 = {c: succ(e, 300) for c, e in ev.items()}
    res = {"arm": "unfrozen", "design": f"Stage 1 redesign #2 variant V{int(V)} (cube yaw +-{V:g} deg, H 224, N 10, "
           "41,328 steps)", "cube_yaw_max_deg": V, "seed": 21600, "sampling": "default (asserted)",
           "s_300": s300, "pick_300": pick_rule(s300, 300)}
    if n >= 1000:
        s1000 = {c: succ(e, 1000) for c, e in ev.items()}
        res |= {"s_1000": s1000, "pick_1000": pick_rule(s1000, 1000)}
    final_ = res.get("pick_1000", res["pick_300"])
    res["pick"] = final_["pick"]
    res["pick_step"] = CKPTS_R[final_["pick"]] if final_["pick"] else None
    res["per_ckpt"] = {c: {"step": CKPTS_R[c], "ckpt": e["ckpt"], "ckpt_sha": e["ckpt_sha"], "n": len(e["episodes"]),
                           **{m: e["aggregate"]["pick"][m] for m in ("success", "slam", "jam", "drop", "early_close")},
                           "gpu_h": e["gpu_h"]} for c, e in ev.items()}
    res["early_abort_d"] = {"s_50": s300["c050"], "abort": s300["c050"] <= 0.05}
    vl = {p_.stem: load(p_)["val_loss"] for p_ in sorted(raw.glob("vloss_*.json"))}
    res["offline_val_loss_vs_r"] = vl
    a, b = vl.get("vloss_010332"), vl.get("vloss_020664")
    res["early_abort_c"] = {"loss_25": a, "loss_50": b, "abort": (b > 2 * a) if a and b else None}
    off = raw / "offline_val_041328.json"
    if off.exists():
        o = load(off)
        res["offline_val_100"] = {"mae_arm_norm": o["mae_arm_norm"], "claw_acc": o["claw_acc"],
                                  "episodes": len(o["per_episode"])}
    (R / f"{tag}_U_val.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("cube_yaw_max_deg", "s_300", "pick_300", "pick", "early_abort_c",
                                          "early_abort_d")}))


def select_variant(v15: float | None, v0: float | None) -> tuple[str, str]:
    """GATES redesign #2 selection rule (None = variant out)."""
    if v15 is not None and v15 >= 0.90:
        return "V15", "v15>=90"
    if v0 is not None and v0 >= 0.90:
        return "V0", "v0>=90"
    if v15 is not None and v0 is not None:
        return ("V0" if v0 > v15 else "V15"), "higher (none >= 90)"
    return ("V15" if v15 is not None else "V0"), "only variant with a v"


def select_y() -> None:
    res: dict = {"seed": 21790, "variants": {}}
    for V in (15, 0):
        p_ = R / f"s1y{V}_U_sel_s21790.json"
        if not p_.exists():
            continue
        t, v = load(p_), load(R / f"s1y{V}_U_val.json")
        assert default_sampling(t) and t["config"]["seed"] == 21790
        eps = t["episodes"]
        assert [e["index"] for e in eps] == list(range(1000))
        assert t["config"]["n"] == 10 and t["config"]["image_size"] == 224
        check_yaw(t, float(V))
        assert t["ckpt"].endswith(f"/s1y{V}_U/checkpoints/{v['pick_step']}/pretrained_model"), (t["ckpt"], v["pick_step"])
        res["variants"][f"V{V}"] = {
            "cube_yaw_max_deg": float(V), "pick_step": v["pick_step"], "ckpt": t["ckpt"], "ckpt_sha": t["ckpt_sha"],
            "success": rate([e["success"] for e in eps]),
            **{m: rate([e[m] for e in eps]) for m in ("slam", "jam", "drop", "early_close")}}
    g = res["variants"]
    v15, v0 = (g[k]["success"]["rate"] if k in g else None for k in ("V15", "V0"))
    res["tested"], res["branch"] = select_variant(v15, v0)
    res["tested_m"] = g[res["tested"]]["cube_yaw_max_deg"]
    res["tested_ckpt"] = g[res["tested"]]["ckpt"]
    res["rule"] = YAW_RULE
    (R / "s1y_select.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def final_y() -> None:
    t, sel = load(R / "s1y_U_test_s11504.json"), load(R / "s1y_select.json")
    m = sel["tested_m"]
    assert default_sampling(t) and t["config"]["seed"] == 11504 and len(t["episodes"]) == 1000
    assert t["config"]["n"] == 10 and t["config"]["image_size"] == 224
    assert t["ckpt"].endswith(sel["tested_ckpt"]), (t["ckpt"], sel["tested_ckpt"])
    check_yaw(t, m)
    eps = t["episodes"]
    fail = [e for e in eps if not e["success"]]
    ys = [e["spawn_yaw_fold_deg"] for e in eps]
    mean = sum(ys) / len(ys)
    res = {"design": f"Stage 1 redesign #2 variant {sel['tested']}", "tested": sel["tested"], "cube_yaw_max_deg": m,
           "arm": "unfrozen", "seed": 11504, "ckpt": t["ckpt"], "ckpt_sha": t["ckpt_sha"],
           "sampling": "default (asserted)", "n": len(eps), "success": rate([e["success"] for e in eps]),
           **{k: rate([e[k] for e in eps]) for k in FAILS},
           "smart_length": t["smart_length"], "bars": t["bars"], "by": breakdown(eps),
           "failures_with_no_flag": sum(not any(e[k] for k in ("slam", "jam", "wrong_object", "early_close", "drop",
                                                                "invalid")) for e in fail),
           "failures_never_grasped": sum(not e["grasp_confirmed"] for e in fail),
           "first_failures": [e["index"] for e in fail[:20]], "latency": t["latency"], "eval_gpu_h": t["gpu_h"],
           "spawn_yaw_fold": {"min": min(ys), "max": max(ys), "mean": mean,
                              "std": (sum((y - mean) ** 2 for y in ys) / len(ys)) ** 0.5}}
    if m > 0:
        bins = {"[0,5)": (0, 5), "[5,10)": (5, 10), f"[10,{m + 0.5:g}]": (10, m + 0.5 + 1e-9)}
        res["success_by_abs_yaw"] = {k: rate([e["success"] for e in eps if lo <= abs(e["spawn_yaw_fold_deg"]) < hi])
                                     for k, (lo, hi) in bins.items()}
    res["gate_pass"] = bool(res["success"]["rate"] >= 0.90 and res["slam"]["rate"] < 0.05 and res["jam"]["rate"] < 0.02)
    (R / "s1y.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("tested", "success", "slam", "jam", "smart_length", "gate_pass",
                                          "failures_never_grasped", "spawn_yaw_fold")}))
    for d in ("cube", "colour", "size"):
        print(d, {k: (v_["count"], v_["n"], round(100 * v_["rate"], 1)) for k, v_ in res["by"][d].items()})
    print(res.get("success_by_abs_yaw"))


# --------------------------------------------------------------------------- Stage 2 (GATES "Stage 2 re-plan after Stage 1 redesign #2")
RAW_S2 = R / "s2"
CKPTS_S2 = {"c050": "017220", "c075": "025830", "c100": "034440"}  # 10 epochs = 34,440 steps; 8,610 = 25 %
STEP25_S2 = "008610"
S2_RUNS = {  # attempt 0 (Stage 1 V15 init) and attempt 1 = redesign (a): same recipe / checkpoints; init + test seeds differ
    "s2": {"tag": "s2", "tseed": 11520, "hseed": 11530, "extra": {}, "design": "Stage 2 (s2 / train, cube yaw +-15 deg"},
    "s2base": {"tag": "s2base", "tseed": 11521, "hseed": 11531, "extra": {"init": "lerobot/smolvla_base", "attempt": 1},
               "design": "Stage 2 redesign (a), init smolvla_base (s2 / train, cube yaw +-15 deg"},
    # Stage 2 redesign #2 (2a): colours-only pick (stage s2a), arm VT (vision + text + expert) from the Stage 1 V15 041328
    # checkpoint; no held-out stream; every eval JSON must carry the same stats_sha.
    "s2a": {"tag": "s2a", "arm": "VT", "arm_name": "VT", "stage": "s2a", "vseed": 21640, "tseed": 11540, "hseed": None,
            "extra": {"init_sha256": "c3dcef76f53e00e5954469970c25c38980a396723f7bbbf2b85919e6cc6688f5"},
            "design": "Stage 2 redesign #2 (2a), arm VT from the Stage 1 V15 checkpoint (s2a / train, cube yaw +-15 deg"},
    # 2a attempt 1 (GATES d33e371): the same recipe on the counterfactual paired-scene set 0c_s2a_pair_rl224_y15; run
    # /runs/s2a1_VT, test 11541; raw in results/s2a1/.
    "s2a1": {"tag": "s2a1", "arm": "VT", "arm_name": "VT", "stage": "s2a", "vseed": 21640, "tseed": 11541, "hseed": None,
             "extra": {"init_sha256": "c3dcef76f53e00e5954469970c25c38980a396723f7bbbf2b85919e6cc6688f5", "attempt": 1,
                       "dataset": "0c_s2a_pair_rl224_y15"},
             "design": "Stage 2 redesign #2 (2a) attempt 1, counterfactual paired-scene data, arm VT from the Stage 1 V15 "
                       "checkpoint (s2a / train, cube yaw +-15 deg"},
    # Stage 2R (GATES "Stage 2R", task 005): 2a attempt 1's recipe + the instruction-contrastive hinge, init = 2a
    # attempt 1's 034440 (sha 07e87cc6...); run /runs/s2r_VT, test 11542; bars incl. slam < 5 % and jam < 2 % (task 005).
    "s2r": {"tag": "s2r", "arm": "VT", "arm_name": "VT", "stage": "s2a", "vseed": 21640, "tseed": 11542, "hseed": None,
            "extra": {"init_sha256": "07e87cc6f5b2533ab4b931e356e7cf944e6b53d49bbf2cc0109ab416290cb3ff", "attempt": "2R",
                      "dataset": "0c_s2a_pair_rl224_y15"},
            "design": "Stage 2R, instruction-contrastive flow loss on the paired-scene data, arm VT from 2a attempt 1's "
                      "034440 (s2a / train, cube yaw +-15 deg"},
    # Stage 2R one diagnosed fix (consultant 2026-10-25): more steps, +34,440 from Stage 2R's 034440 (sha 4dbe9303...),
    # same loss; run /runs/s2r1_VT, test 11543; bars as s2r.
    "s2r1": {"tag": "s2r1", "arm": "VT", "arm_name": "VT", "stage": "s2a", "vseed": 21640, "tseed": 11543, "hseed": None,
             "extra": {"init_sha256": "4dbe9303848db666d341937580425106e48d6dedc5d81a53741a00bc8e3dfcc1",
                       "attempt": "2R fix 1 (more steps)", "dataset": "0c_s2a_pair_rl224_y15"},
             "design": "Stage 2R one fix (more steps): +34,440 steps of the instruction-contrastive flow loss from Stage 2R's "
                       "034440 (s2a / train, cube yaw +-15 deg"},
    # Stage 2M (task 006, owner-approved new stage): +4,000 paired demos (union 6,000), 68,880 steps at lr 5e-5 from the
    # Stage 2R fix checkpoint (s2r1 034440, sha bc880e23...), same loss; run /runs/s2m_VT, test 11544; saves every 17,220.
    "s2m": {"tag": "s2m", "arm": "VT", "arm_name": "VT", "stage": "s2a", "vseed": 21640, "tseed": 11544, "hseed": None,
            "ckpts": {"c050": "034440", "c075": "051660", "c100": "068880"}, "step25": "017220",
            "extra": {"init_sha256": "bc880e236677284c360078794a8c11ca22e43f753c276a8f0c60fa1c40e2f777", "attempt": "2M", "dataset": "0c_s2a_pair6k_rl224_y15"},
            "design": "Stage 2M, grasp precision: 6,000 paired demos, 68,880 steps at lr 5e-5 of the instruction-contrastive "
                      "flow loss from the Stage 2R fix checkpoint (s2a / train, cube yaw +-15 deg"},
    # 2b AMENDED (GATES 3862001): paired s2 (colours + sizes) + contrastive loss, 34,440 steps lr 5e-5 from the 2M pick;
    # run /runs/s2bp_VT, tests 11550 / held-out 11560 (c88a467 bars: pick >= 85 %, wrong-cube < 5 %).
    "s2bp": {"tag": "s2bp", "arm": "VT", "arm_name": "VT", "stage": "s2", "vseed": 21650, "tseed": 11550, "hseed": 11560,
             "extra": {"init_sha256": "e4741cf66cbbb57686df5a2ae7948e46a3a35ac65bc8c79bb48a598a32ca0f64", "attempt": "2b amended", "dataset": "0c_s2_pair_rl224_y15"},
             "design": "Stage 2b amended: paired s2 scenes + instruction-contrastive flow loss, arm VT from the Stage 2M "
                       "picked checkpoint (s2 / train, cube yaw +-15 deg"},
    # 2b sizes round 2 (task 008): +8,000 paired s2 (union 12,000), 68,880 steps lr 5e-5 from the 2b fix pick; run
    # /runs/s2bp2_VT, tests 11552 / held-out 11562 (bars as Stage 2 + slam / jam, task 008).
    "s2bp2": {"tag": "s2bp2", "arm": "VT", "arm_name": "VT", "stage": "s2", "vseed": 21650, "tseed": 11552, "hseed": 11562,
              "ckpts": {"c050": "034440", "c075": "051660", "c100": "068880"}, "step25": "017220",
              "extra": {"init_sha256": "e87fd97bdd74dbf2625dd402c90fac734e2afc56414716d2270faf97abd1e503",
                        "attempt": "2b sizes round 2", "dataset": "0c_s2_pair12k_rl224_y15"},
              "design": "Stage 2b sizes round 2: 12,000 paired s2 scenes + instruction-contrastive flow loss, 68,880 steps at "
                        "lr 5e-5 from the 2b one-fix checkpoint (s2 / train, cube yaw +-15 deg"},
    # 2b amended one fix (consultant ruling; tests 11551 / 11561), run /runs/s2bp1_VT, init = the 2M pick.
    "s2bp1": {"tag": "s2bp1", "arm": "VT", "arm_name": "VT", "stage": "s2", "vseed": 21650, "tseed": 11551, "hseed": 11561,
             "extra": {"init_sha256": "e4741cf66cbbb57686df5a2ae7948e46a3a35ac65bc8c79bb48a598a32ca0f64", "attempt": "2b amended, one fix", "dataset": "0c_s2_pair_rl224_y15"},
             "design": "Stage 2b amended one fix: paired s2 scenes + instruction-contrastive flow loss, arm VT from the Stage 2M "
                       "picked checkpoint (s2 / train, cube yaw +-15 deg"},
    # Stage 2 redesign #2 (2b): the full s2 task (stage s2, split train), arm VT initialised from 2a's gated checkpoint.
    "s2b": {"tag": "s2b", "arm": "VT", "arm_name": "VT", "stage": "s2", "vseed": 21650, "tseed": 11550, "hseed": 11560,
            "extra": {}, "design": "Stage 2 redesign #2 (2b), arm VT from 2a's gated checkpoint (s2 / train, cube yaw +-15 deg"}}


def check_s2(ev: dict, seed: int, split: str, n: int | None = None, stage: str = "s2") -> None:
    c = ev["config"]
    assert default_sampling(ev), "not default sampling"
    assert c["n"] == 10 and c["image_size"] == 224 and c["seed"] == seed, (c["n"], c["image_size"], c["seed"], seed)
    assert c["stage"] == stage and c["split"] == split, (c["stage"], c["split"])
    check_yaw(ev, 15.0)
    if n is not None:
        assert len(ev["episodes"]) == n, (len(ev["episodes"]), n)


def val_s2(run: str = "s2") -> None:
    """Stage 2 checkpoint pick on validation 21620 (s2 / train, m 15): rule unchanged, early abort (c) / (d)."""
    cfg = S2_RUNS[run]
    CK, S25 = cfg.get("ckpts", CKPTS_S2), cfg.get("step25", STEP25_S2)  # per-run saves (Stage 2M: 68,880 steps)
    tag, raw, arm, vseed, stage = cfg["tag"], R / cfg["tag"], cfg.get("arm", "U"), cfg.get("vseed", 21620), cfg.get("stage", "s2")
    ev = {c: load(raw / f"{tag}_{arm}_val_{c}_s{vseed}.json") for c in CK}
    for c, e in ev.items():
        check_s2(e, vseed, "train", stage=stage)
        assert e["ckpt"].endswith(f"/{tag}_{arm}/checkpoints/{CK[c]}/pretrained_model"), e["ckpt"]
    n = min(len(e["episodes"]) for e in ev.values())
    s300 = {c: succ(e, 300) for c, e in ev.items()}
    res = {"arm": cfg.get("arm_name", "unfrozen"), "design": cfg["design"] + ", H 224, N 10, 34,440 steps)", **cfg["extra"],
           "cube_yaw_max_deg": 15.0, "seed": vseed, "sampling": "default (asserted)", "s_300": s300,
           "pick_300": pick_rule(s300, 300)}
    if stage == "s2a":  # one normalisation for all three checkpoints (they come from one run)
        shas = {c: e["stats_sha"] for c, e in ev.items()}
        assert len(set(shas.values())) == 1, f"stats_sha differs between the validation runs: {shas}"
        res["stats_sha"] = shas["c050"]
    if n >= 1000:
        s1000 = {c: succ(e, 1000) for c, e in ev.items()}
        res |= {"s_1000": s1000, "pick_1000": pick_rule(s1000, 1000)}
    final_ = res.get("pick_1000", res["pick_300"])
    res["pick"] = final_["pick"]
    res["pick_step"] = CK[final_["pick"]] if final_["pick"] else None
    res["per_ckpt"] = {c: {"step": CK[c], "ckpt": e["ckpt"], "ckpt_sha": e["ckpt_sha"], "n": len(e["episodes"]),
                           "success": rate([x["success"] for x in e["episodes"]]),
                           "wrong_grasp": rate([x["wrong_grasp"] for x in e["episodes"]]),
                           **{m: e["aggregate"]["pick"][m] for m in ("slam", "jam", "drop", "early_close")},
                           "gpu_h": e["gpu_h"]} for c, e in ev.items()}
    res["early_abort_d"] = {"s_50": s300["c050"], "abort": s300["c050"] <= 0.05}
    vl = {p_.stem: load(p_)["val_loss"] for p_ in sorted(raw.glob("vloss_*.json"))}
    res["offline_val_loss"] = vl
    a, b = vl.get(f"vloss_{S25}"), vl.get(f"vloss_{CK['c050']}")
    res["early_abort_c"] = {"loss_25": a, "loss_50": b, "abort": (b > 2 * a) if a and b else None}
    off = raw / f"offline_val_{CK['c100']}.json"
    if off.exists():
        o = load(off)
        res["offline_val_100"] = {"mae_arm_norm": o["mae_arm_norm"], "claw_acc": o["claw_acc"],
                                  "episodes": len(o["per_episode"])}
    (R / f"{tag}_val.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("s_300", "pick_300", "pick", "early_abort_c", "early_abort_d")}))


def val_s2base() -> None:
    val_s2("s2base")


def val_s2a() -> None:
    val_s2("s2a")


def val_s2b() -> None:
    val_s2("s2b")


def confusion_type(target: list[str], wrong_idx: int | None, spec_cubes: list) -> str:
    """wrong_grasp_cube (ALL_CUBES index = detector cube row, vla_common.catalog order, NOT spec.cubes order) ->
    same_colour_other_size / other_colour_same_size / other; 'unknown' when the eval JSON predates the field."""
    from vla_common.catalog import ALL_CUBES
    if wrong_idx is None:
        return "unknown"
    w = ALL_CUBES[wrong_idx]
    assert list(w) in [list(c) for c in spec_cubes] and list(w) != list(target), (w, target, spec_cubes)
    if w[0] == target[0]:
        return "same_colour_other_size"
    return "other_colour_same_size" if w[1] == target[1] else "other"


def confusion(eps: list[dict]) -> dict:
    types: dict[str, int] = {}
    for e in eps:
        if e["wrong_grasp"]:
            w = e.get("wrong_grasp_cube")
            assert w is None or w >= 0, "wrong_grasp without a recorded cube"
            t = confusion_type(e["target"].split("_"), w, json.loads(e["spec"])["cubes"])
            types[t] = types.get(t, 0) + 1
        else:
            assert e.get("wrong_grasp_cube", -1) == -1, "wrong_grasp_cube set without wrong_grasp"
    return {"wrong_grasp_total": sum(types.values()), **{t: types.get(t, 0) for t in (
        "same_colour_other_size", "other_colour_same_size", "other", "unknown")}}


def grasped_cube(e: dict) -> str | None:
    """The cube a probe episode went for, 'colour_size': the first non-target cube that triggered wrong_grasp
    (wrong_grasp_cube, ALL_CUBES index), else the target when the episode succeeded, else None (no grasp recorded)."""
    from vla_common.catalog import ALL_CUBES
    w = e.get("wrong_grasp_cube", -1)
    assert w is not None and (w < 0) == (not e["wrong_grasp"]), f"episode {e['index']}: wrong_grasp / wrong_grasp_cube disagree"
    if w >= 0:
        return "_".join(ALL_CUBES[w])
    return e["target"] if e["success"] else None


def p_same_cube_chance(k: int, n: int) -> float:
    """P(at least two of k members pick the same cube) when each picks uniformly among n cubes."""
    if k > n:
        return 1.0
    p = 1.0
    for j in range(k):
        p *= (n - j) / n
    return 1 - p


def probe_stats(eps: list[dict]) -> dict:
    """Counterfactual probe numbers from eval records carrying layout_id / member (see probe_s2a)."""
    assert eps and all("layout_id" in e and "member" in e for e in eps), "not a --paired-targets eval JSON"
    gs: dict[int, list[dict]] = {}
    for e in sorted(eps, key=lambda e: (e["layout_id"], e["member"])):
        gs.setdefault(e["layout_id"], []).append(e)
    rows, named_all, named_g, chance_all, chance_g, same, same_chance = [], [], [], [], [], [], []
    for g, ms in gs.items():
        assert len({e["target"] for e in ms}) == len(ms), f"group {g}: duplicate targets"
        grasp = [grasped_cube(e) for e in ms]
        n = ms[0]["n_cubes"]
        for e, gc in zip(ms, grasp):
            named_all.append(gc == e["target"])
            chance_all.append(1 / n)
            if gc is not None:
                named_g.append(gc == e["target"])
                chance_g.append(1 / n)
        known = [gc for gc in grasp if gc is not None]
        if len(known) >= 2:
            same.append(len(set(known)) < len(known))
            same_chance.append(p_same_cube_chance(len(known), n))
        rows.append({"group": g, "n_cubes": n, "members": [
            {"member": e["member"], "target": e["target"], "instruction": e["instruction"], "success": e["success"],
             "wrong_grasp": e["wrong_grasp"], "grasped": gc, "named": gc == e["target"] if gc else None}
            for e, gc in zip(ms, grasp)]})
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"groups": len(gs), "episodes": len(eps),
            "success": rate([e["success"] for e in eps]),
            "success_by_member": {str(m): rate([e["success"] for e in eps if e["member"] == m])
                                  for m in sorted({e["member"] for e in eps})},
            "success_and_wrong_grasp_episodes": sum(e["success"] and e["wrong_grasp"] for e in eps),
            "no_grasp_episodes": sum(r["grasped"] is None for g in rows for r in g["members"]),
            "went_for_named_cube_of_all_episodes": rate(named_all) | {"chance": mean(chance_all)},
            "went_for_named_cube_of_grasped_episodes": rate(named_g) | {"chance": mean(chance_g)},
            "same_cube_regardless_groups": rate(same) | {"chance": mean(same_chance), "note":
                "groups with >= 2 known grasps; a cube grasped by >= 2 members under different instructions"},
            "per_group": rows}


def probe_s2a(path: str, *args: str) -> None:
    ev = load(Path(path))
    assert ev["config"].get("paired_targets"), "eval JSON is not a --paired-targets run"
    st = probe_stats(ev["episodes"]) | {"eval": str(path), "seed": ev["config"]["seed"], "ckpt": ev.get("ckpt"),
                                       "ckpt_sha": ev.get("ckpt_sha")}
    for g in st["per_group"]:
        print(f"group {g['group']:4d} n_cubes {g['n_cubes']}  " + "  ".join(
            f"[{m['target']} -> {m['grasped']}{' OK' if m['success'] else ''}]" for m in g["members"]))
    f = lambda r: f"{r['count']}/{r['n']} = {r['rate']:.3f}" + (f" (chance {r['chance']:.3f})" if r.get("chance") is not None else "")  # noqa: E731
    print(f"groups {st['groups']} episodes {st['episodes']}  success {f(st['success'])}")
    print(f"went for the named cube: {f(st['went_for_named_cube_of_all_episodes'])} of all episodes, "
          f"{f(st['went_for_named_cube_of_grasped_episodes'])} of grasped")
    print(f"same cube regardless of instruction: {f(st['same_cube_regardless_groups'])} of groups")
    if "--out" in args:
        o = Path(args[args.index("--out") + 1])
        o.write_text(json.dumps(st, indent=1))
        print(f"wrote {o}")


def grasped_cube_s3(e: dict) -> str | None:
    """grasped_cube for place / stack: a right-cube grasp that failed to reach the destination is not a success, so also
    count the target as grasped when the detector latched grasp_confirmed (absent = not grasped)."""
    if e.get("wrong_grasp") or e["success"]:
        return grasped_cube(e)
    return e["target"] if e.get("grasp_confirmed") else None


def settled_dest(e: dict) -> str | None:
    """Destination a probe episode released a cube on (rest_dest, detectors): place pad colour / stack bottom 'colour_size';
    None = nothing settled released on a pad / cube. Needs a record written after rest_dest existed."""
    from vla_common.catalog import ALL_CUBES, COLORS
    assert "rest_dest" in e, f"episode {e['index']}: no rest_dest (eval JSON predates the s3 probe)"
    d = e["rest_dest"]
    if d < 0:
        return None
    return COLORS[d] if e["task"] == "place" else "_".join(ALL_CUBES[d])


def named_dest(e: dict) -> str:
    d = json.loads(e["spec"])["dest"]
    return d if isinstance(d, str) else "_".join(d)


def n_destinations(e: dict) -> int:
    """Destinations a uniform chooser has: the 4 pads (place) / every cube but the one moved (stack)."""
    return len(json.loads(e["spec"])["pads"]) if e["task"] == "place" else e["n_cubes"] - 1


def probe_s3_stats(eps: list[dict]) -> dict:
    """s3a / s3b counterfactual probe: per episode the cube it went for (grasped_cube) and the destination it released on
    (settled_dest), vs the instruction's (target, dest). Chance = uniform over n_cubes / n_destinations."""
    assert eps and all("layout_id" in e and "member" in e and "relation" in e for e in eps), "not an s3 --paired-targets eval JSON"
    rows = []
    for e in sorted(eps, key=lambda e: (e["layout_id"], e["member"])):
        gc, gd, nd = grasped_cube_s3(e), settled_dest(e), n_destinations(e)
        rows.append({"group": e["layout_id"], "member": e["member"], "relation": e["relation"], "target": e["target"],
                     "dest": named_dest(e), "instruction": e["instruction"], "success": e["success"], "cube": gc,
                     "released_on": gd, "cube_ok": gc == e["target"], "dest_ok": gd == named_dest(e),
                     "ch_cube": 1 / e["n_cubes"], "ch_dest": 1 / nd, "n_options": e["n_cubes"] * nd})
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    rc = lambda rs, key, ch: rate([r[key] for r in rs]) | {"chance": mean([r[ch] for r in rs])}  # noqa: E731

    def block(rs: list[dict]) -> dict:
        g = [r for r in rs if r["cube"] is not None]
        d = [r for r in rs if r["released_on"] is not None]
        return {"episodes": len(rs), "success": rate([r["success"] for r in rs]),
                "named_cube_of_all": rc(rs, "cube_ok", "ch_cube"), "named_cube_of_grasped": rc(g, "cube_ok", "ch_cube"),
                "named_dest_of_released": rc(d, "dest_ok", "ch_dest"),
                "both_right_of_all": rate([r["cube_ok"] and r["dest_ok"] for r in rs])
                | {"chance": mean([r["ch_cube"] * r["ch_dest"] for r in rs])}}

    same, same_chance = [], []
    for g in sorted({r["group"] for r in rows}):
        ch = [(r["cube"], r["released_on"]) for r in rows if r["group"] == g and r["cube"] and r["released_on"]]
        if len(ch) >= 2:
            same.append(len(set(ch)) < len(ch))
            same_chance.append(p_same_cube_chance(len(ch), next(r["n_options"] for r in rows if r["group"] == g)))
    return {"groups": len({r["group"] for r in rows}), "episodes": len(rows), "all": block(rows),
            "by_relation": {rel: block([r for r in rows if r["relation"] == rel])
                            for rel in ("anchor", "same_target", "same_dest", "both_differ")
                            if any(r["relation"] == rel for r in rows)},
            "relation_note": "relation = member vs member 0. same_target: same cube, other destination (destination effect); "
                             "same_dest: other cube, same destination (cube effect); both_differ; anchor = member 0.",
            "same_choice_regardless_groups": rate(same) | {"chance": mean(same_chance), "note":
                "groups with >= 2 members that grasped a cube and released on a destination; the same (cube, destination) "
                "produced by >= 2 members under different instructions; chance = P(collision) over n_cubes * n_destinations"},
            "no_grasp_episodes": sum(r["cube"] is None for r in rows),
            "nothing_released_episodes": sum(r["released_on"] is None for r in rows),
            "per_episode": rows}


def probe_s3(path: str, *args: str) -> None:
    ev = load(Path(path))
    assert ev["config"].get("paired_targets") and ev["config"]["stage"] in ("s3a", "s3b", "s3a1", "s3b1"),         "eval JSON is not a --paired-targets s3a / s3b / s3a1 / s3b1 run"
    st = probe_s3_stats(ev["episodes"]) | {"eval": str(path), "stage": ev["config"]["stage"], "seed": ev["config"]["seed"],
                                           "ckpt": ev.get("ckpt"), "ckpt_sha": ev.get("ckpt_sha")}
    f = lambda r: f"{r['count']}/{r['n']} = {r['rate']:.3f}" + (f" (chance {r['chance']:.3f})" if r.get("chance") is not None else "") if r["n"] else "0/0"  # noqa: E731
    print(f"stage {st['stage']}  groups {st['groups']} episodes {st['episodes']}  no grasp {st['no_grasp_episodes']}  "
          f"nothing released {st['nothing_released_episodes']}")
    for name, b in [("all", st["all"])] + list(st["by_relation"].items()):
        print(f"[{name}] n {b['episodes']}  success {f(b['success'])}")
        print(f"  named cube: {f(b['named_cube_of_all'])} of all, {f(b['named_cube_of_grasped'])} of grasped")
        print(f"  named destination: {f(b['named_dest_of_released'])} of released")
        print(f"  both right: {f(b['both_right_of_all'])} of all")
    print(f"same (cube, destination) regardless of instruction: {f(st['same_choice_regardless_groups'])} of groups")
    if "--out" in args:
        o = Path(args[args.index("--out") + 1])
        o.write_text(json.dumps(st, indent=1))
        print(f"wrote {o}")


def by_n_cubes(eps: list[dict]) -> dict:
    return {str(n): {"success": rate([e["success"] for e in eps if e["n_cubes"] == n]),
                     "wrong_grasp": rate([e["wrong_grasp"] for e in eps if e["n_cubes"] == n])}
            for n in sorted({e["n_cubes"] for e in eps})}


def stats_check(ckpt_dir: str | None, norm_stats: dict) -> str | dict:
    """The checkpoint's pre / postprocessor normalizer stats (safetensors) equal the trainer's stats exactly."""
    if not ckpt_dir:
        return "not run (checkpoint not synced)"
    import glob
    from safetensors.numpy import load_file
    d = Path(ckpt_dir)
    pre = load_file(glob.glob(str(d / "policy_preprocessor_step_*_normalizer_processor.safetensors"))[0])
    post = load_file(glob.glob(str(d / "policy_postprocessor_step_*_unnormalizer_processor.safetensors"))[0])
    out = {}
    for name, f, key in (("pre_state", pre, "observation.state"), ("pre_action", pre, "action"),
                         ("post_action", post, "action")):
        out[name] = all(f[f"{key}.{s}"].astype("float32").tolist() == norm_stats[key][s] for s in ("mean", "std"))
    out["equal"] = all(out.values())
    assert out["equal"], f"checkpoint normalizer stats differ from train_summary_U.json norm_stats: {out}"
    return out


def final_s2(*args: str, run: str = "s2") -> None:
    cfg = S2_RUNS[run]
    tag, tseed, hseed = cfg["tag"], cfg["tseed"], cfg["hseed"]
    arm, stage = cfg.get("arm", "U"), cfg.get("stage", "s2")
    ckpt_dir = args[args.index("--ckpt-dir") + 1] if "--ckpt-dir" in args else None
    v = load(R / f"{tag}_val.json")
    t = load(R / f"{tag}_test_s{tseed}.json")
    check_s2(t, tseed, "train", 1000, stage)
    assert t["ckpt"].endswith(f"/{tag}_{arm}/checkpoints/{v['pick_step']}/pretrained_model"), (t["ckpt"], v["pick_step"])
    eps = t["episodes"]
    fail = [e for e in eps if not e["success"]]
    n = len(eps)
    train_summary = load(R / tag / f"train_summary_{arm}.json")
    res = {"design": cfg["design"] + ")", **cfg["extra"], "arm": cfg.get("arm_name", "unfrozen"), "seed": tseed, "ckpt": t["ckpt"],
           "ckpt_sha": t["ckpt_sha"], "pick_step": v["pick_step"], "sampling": "default (asserted)", "n": n,
           "success": rate([e["success"] for e in eps]), **{m: rate([e[m] for e in eps]) for m in FAILS},
           "smart_length": t["smart_length"], "bars": t["bars"], "by": breakdown(eps),
           "by_n_cubes": by_n_cubes(eps), "wrong_cube_confusion": confusion(eps),
           "failures_with_no_flag": sum(not any(e[m] for m in ("slam", "jam", "wrong_object", "early_close", "drop",
                                                                "invalid")) for e in fail),
           "failures_never_grasped": sum(not e["grasp_confirmed"] for e in fail),
           "first_failures": [e["index"] for e in fail[:20]], "latency": t["latency"], "eval_gpu_h": t["gpu_h"],
           "stats_check": stats_check(ckpt_dir, train_summary["norm_stats"])}
    ks, kw = res["success"]["count"], res["wrong_grasp"]["count"]
    res["gate_pass"] = bool(100 * ks >= 85 * n and 100 * kw < 5 * n)  # integer arithmetic: pick >= 85 %, wrong-cube < 5 %
    if run in ("s2r", "s2r1", "s2m", "s2bp2"):  # task 005 / 006 / 008 bars: + slam < 5 %, jam < 2 %
        res["gate_pass"] = bool(res["gate_pass"] and 100 * res["slam"]["count"] < 5 * n and 100 * res["jam"]["count"] < 2 * n)
    i3 = run in ("s2base", "s2b")
    if i3:  # GATES redesign (a) / 2b instruments I1 / I3 (I2 = init_disc on the training job's steps.jsonl)
        a0_norm = load(R / "s2" / "train_summary_U.json")["norm_stats"]
        a0_sha = load(R / "s2_test_s11520.json")["stats_sha"]
    if run == "s2b":  # I1: the init is 2a's gated checkpoint (--init-sha, default results/s2a.json ckpt_sha)
        want = args[args.index("--init-sha") + 1] if "--init-sha" in args else load(R / "s2a.json")["ckpt_sha"]
        res["i1_init"] = {"init": train_summary["init"], "init_sha256": train_summary["init_sha256"], "expected_sha256": want,
                          "is_2a_gated_checkpoint": train_summary["init_sha256"] == want}
        assert res["i1_init"]["is_2a_gated_checkpoint"], f"I1: init_sha256 {train_summary['init_sha256']} != {want}"
        val_sha = {c: load(R / tag / f"{tag}_{arm}_val_{c}_s{cfg['vseed']}.json")["stats_sha"] for c in CKPTS_S2}
        assert all(x == a0_sha for x in val_sha.values()), f"I3: validation stats_sha {val_sha} != attempt 0 {a0_sha}"
    if run == "s2base":
        init = train_summary["argv"]["init"]
        res["i1_init"] = {"init": init, "is_smolvla_base": init == cfg["extra"]["init"]}
    if i3:
        res["i3_norm"] = {"norm_stats_equal_attempt0": train_summary["norm_stats"] == a0_norm, "attempt0_stats_sha": a0_sha,
                          "test_stats_sha": t["stats_sha"], "test_stats_sha_equal_attempt0": t["stats_sha"] == a0_sha}
    if run == "s2base":
        assert res["i1_init"]["is_smolvla_base"], f"I1: init is {init}"
    if i3:
        assert res["i3_norm"]["norm_stats_equal_attempt0"], "I3: norm_stats differ from attempt 0 (results/s2/train_summary_U.json)"
        assert res["i3_norm"]["test_stats_sha_equal_attempt0"], f"I3: stats_sha {t['stats_sha']} != attempt 0 {a0_sha}"
    if run in ("s2a", "s2a1", "s2r", "s2r1", "s2m", "s2bp", "s2bp1", "s2bp2"):  # init identity (train_summary init path + sha256) and one stats_sha over all evals and the training run
        want = cfg["extra"]["init_sha256"]
        key = {"s2r": "is_2a1_034440", "s2r1": "is_2r_034440", "s2m": "is_2r1_034440", "s2bp": "is_2m_pick", "s2bp1": "is_2m_pick", "s2bp2": "is_2b_fix_pick"}.get(run, "is_stage1_v15_041328")
        res["init_check"] = {"init": train_summary["init"], "init_sha256": train_summary["init_sha256"],
                             key: train_summary["init_sha256"] == want}
        assert res["init_check"][key], f"init_sha256 {train_summary['init_sha256']} != {want}"
        res["stats_sha"] = stats_sha_check(tag, arm, cfg["vseed"], t["stats_sha"], ckpt_dir, cfg.get("ckpts"))
    ho = R / f"{tag}_heldout_s{hseed}.json"
    if hseed and ho.exists():
        h = load(ho)
        check_s2(h, hseed, "heldout_comp", 300)
        if i3:
            assert h["stats_sha"] == a0_sha, f"I3: held-out stats_sha {h['stats_sha']} != attempt 0 {a0_sha}"
            res["i3_norm"]["heldout_stats_sha_equal_attempt0"] = True
        assert h["ckpt"] == t["ckpt"], (h["ckpt"], t["ckpt"])
        he = h["episodes"]
        res["heldout"] = {"n": len(he), "success": rate([e["success"] for e in he]),
                          "wrong_grasp": rate([e["wrong_grasp"] for e in he]), "by": breakdown(he),
                          "by_n_cubes": by_n_cubes(he), "wrong_cube_confusion": confusion(he),
                          "latency": h["latency"], "eval_gpu_h": h["gpu_h"]}
    res["gpu_h"] = t["gpu_h"] + (res["heldout"]["eval_gpu_h"] if "heldout" in res else 0.0)
    (R / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("success", "wrong_grasp", "slam", "jam", "gate_pass", "wrong_cube_confusion",
                                          "stats_check", "failures_never_grasped")}))
    print("by_n_cubes", {k: (v_["success"]["rate"], v_["wrong_grasp"]["rate"]) for k, v_ in res["by_n_cubes"].items()})


def final_s2base(*args: str) -> None:
    final_s2(*args, run="s2base")


def stats_sha_check(tag: str, arm: str, vseed: int, test_sha: str, ckpt_dir: str | None, ck: dict | None = None) -> dict:
    """Every eval JSON's stats_sha (the three validation runs and the test) is one value; with --ckpt-dir also equal to the
    sha of the training run's own normalizer files (scripts/eval_policy.stats_sha of the checkpoint)."""
    val = {c: load(R / tag / f"{tag}_{arm}_val_{c}_s{vseed}.json")["stats_sha"] for c in (ck or CKPTS_S2)}
    res = {"validation": val, "test": test_sha, "equal_across_evals": len(set(val.values()) | {test_sha}) == 1}
    assert res["equal_across_evals"], f"stats_sha differs between the validation runs and the test: {res}"
    if ckpt_dir:
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_policy import stats_sha
        res["train_run"] = stats_sha(ckpt_dir, "")
        res["equal_train_run"] = res["train_run"] == test_sha
        assert res["equal_train_run"], f"evals' stats_sha {test_sha} != the training run's {res['train_run']}"
    else:
        res["train_run"] = "not run (checkpoint not synced)"
    return res


def final_s2a(*args: str) -> None:
    final_s2(*args, run="s2a")


def val_s2a1() -> None:
    val_s2("s2a1")


def final_s2a1(*args: str) -> None:
    final_s2(*args, run="s2a1")


def lang_s2a1(ck: str) -> None:
    lang_single("s2a1", ck)


def val_s2r() -> None:
    val_s2("s2r")


def final_s2r(*args: str) -> None:
    final_s2(*args, run="s2r")


def lang_s2r(ck: str) -> None:
    lang_single("s2r", ck)


def val_s2r1() -> None:
    val_s2("s2r1")


def final_s2r1(*args: str) -> None:
    final_s2(*args, run="s2r1")


def val_s2m() -> None:
    val_s2("s2m")


def final_s2m(*args: str) -> None:
    final_s2(*args, run="s2m")


def val_s2bp() -> None:
    val_s2("s2bp")


def final_s2bp(*args: str) -> None:
    final_s2(*args, run="s2bp")


def val_s2bp1() -> None:
    val_s2("s2bp1")


def final_s2bp1(*args: str) -> None:
    final_s2(*args, run="s2bp1")


def val_s2bp2() -> None:
    val_s2("s2bp2")


def final_s2bp2(*args: str) -> None:
    final_s2(*args, run="s2bp2")


def final_s2b(*args: str) -> None:
    final_s2(*args, run="s2b")


def lang_arm(run: str, ck: str, mode: str) -> dict:
    """One language-check eval JSON (validation 21620 / 2a: 21640, eps 0-99, s2 / train): config and checkpoint asserted."""
    cfg = S2_RUNS[run]
    tag, arm, vseed = cfg["tag"], cfg.get("arm", "U"), cfg.get("vseed", 21620)
    e = load(R / f"{tag}_lang" / f"{tag}_{arm}_{ck}_{'true' if mode == 'none' else 'shuffle'}_s{vseed}.json")
    check_s2(e, vseed, "train", 100, cfg.get("stage", "s2"))
    assert e["config"].get("instruction_swap", "none") == mode, (e["config"].get("instruction_swap"), mode)
    assert ("instruction_swap_report" in e) == (mode != "none"), "instruction_swap_report presence != swap mode"
    assert e["ckpt"].endswith(f"/{tag}_{arm}/checkpoints/{CKPTS_S2[ck]}/pretrained_model"), e["ckpt"]
    return e


def lang_pair(run: str, ck: str) -> dict:
    """Paired true vs shuffled counts (McNemar z = (b - c) / sqrt(b + c), None if b + c = 0) on the same 100 episodes."""
    t, s = lang_arm(run, ck, "none"), lang_arm(run, ck, "shuffle")
    key = lambda e: [(x["index"], x["spec"], x["instruction"]) for x in e["episodes"]]  # noqa: E731
    assert key(t) == key(s), "true / shuffled episodes differ in index, spec or instruction (or order)"
    assert all(x["instruction_sent"] != x["instruction"] for x in s["episodes"]), "a shuffled episode sent its own instruction"
    xt, xs = ([x["success"] for x in e["episodes"]] for e in (t, s))
    b, c = sum(p and not q for p, q in zip(xt, xs)), sum(q and not p for p, q in zip(xt, xs))
    return {"t": t, "s": s, "true_success": sum(xt), "shuffled_success": sum(xs), "b_true_only": b, "c_shuffled_only": c,
            "mcnemar_z": (b - c) / (b + c) ** 0.5 if b + c else None,
            "wrong_grasp_true": sum(x["wrong_grasp"] for x in t["episodes"]),
            "wrong_grasp_shuffled": sum(x["wrong_grasp"] for x in s["episodes"])}


def lang_s2base(ck: str) -> None:
    """Language check (reported only, no pass / fail): attempt 1 paired true vs shuffled instruction on validation 21620
    eps 0-99, next to attempt 0's; the shuffled arms must have sent the same instructions (same donor permutation)."""
    new, old = lang_pair("s2base", ck), lang_pair("s2", ck)
    sent = lambda p: [(x["index"], x["spec"], x["instruction_sent"]) for x in p["s"]["episodes"]]  # noqa: E731
    assert sent(new) == sent(old), "shuffled episodes / sent instructions differ from attempt 0's"
    assert new["s"]["instruction_swap_report"]["donor_of"] == old["s"]["instruction_swap_report"]["donor_of"]
    pub = lambda p: {k: v for k, v in p.items() if k not in ("t", "s")} | {"ckpt": p["t"]["ckpt"], "ckpt_sha": p["t"]["ckpt_sha"]}  # noqa: E731
    res = {"ck": ck, "step": CKPTS_S2[ck], "seed": 21620, "episodes": [0, 99], "swap": "shuffle",
           "attempt1": pub(new), "attempt0": pub(old), "same_donor_permutation_as_attempt0": True,
           "note": "reported only: no pass / fail, triggers nothing"}
    (R / f"s2base_lang_{ck}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("ck", "attempt1", "attempt0")}))


def lang_single(run: str, ck: str) -> None:
    """2a / 2b language check (reported only, no pass / fail, no attempt-0 comparison): paired true vs shuffled instruction
    on the run's validation stream, eps 0-99."""
    assert ck in ("c050", "c100"), ck
    cfg = S2_RUNS[run]
    p = lang_pair(run, ck)
    pub = {k: v for k, v in p.items() if k not in ("t", "s")} | {"ckpt": p["t"]["ckpt"], "ckpt_sha": p["t"]["ckpt_sha"]}
    res = {"ck": ck, "step": CKPTS_S2[ck], "seed": cfg["vseed"], "episodes": [0, 99], "swap": "shuffle", "pair": pub,
           "note": "reported only: no pass / fail, triggers nothing"}
    (R / f"{cfg['tag']}_lang_{ck}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("ck", "pair")}))


def lang_s2a(ck: str) -> None:
    lang_single("s2a", ck)


def lang_s2b(ck: str) -> None:
    lang_single("s2b", ck)


def init_disc(path: str, *args: str) -> int:
    """GATES I2 init discriminator: mean training loss over steps 1-50 of a steps.jsonl. --expect gt (default): mean >
    threshold -> base init confirmed; --expect le: mean <= threshold -> Stage-1-checkpoint init confirmed. Else a finding."""
    thr = float(args[args.index("--ref-threshold") + 1]) if "--ref-threshold" in args else 0.5
    expect = args[args.index("--expect") + 1] if "--expect" in args else "gt"
    assert expect in ("gt", "le"), expect
    loss = [r["loss"] for r in map(json.loads, Path(path).read_text().splitlines()) if 1 <= r["step"] <= 50]
    assert len(loss) == 50, f"steps 1..50: found {len(loss)} rows"
    mean = sum(loss) / 50
    ok = mean > thr if expect == "gt" else mean <= thr
    what = "base init" if expect == "gt" else "Stage-1-checkpoint init"
    print(f"mean loss steps 1-50 = {mean:.4f} (threshold {thr}): " + (f"{what} confirmed" if ok else "INSTRUMENT FINDING"))
    return int(not ok)


# ------------------------------------------------------------------------------------------ Stage 3A / 3B
CKPTS_S3 = {"c050": "017754", "c075": "026631", "c100": "035508"}  # 4 epochs = 35,508 steps; 25 % = 8,877
STEP25_S3 = "008877"
S3 = {"a": {"stage": "s3a", "task": "place", "vseed": 21660, "tseed": 11570, "bar": 80, "wrong": "wrong_dest",
            "dest": "pad"},
      "b": {"stage": "s3b", "task": "stack", "vseed": 21690, "tseed": 11590, "bar": 70, "wrong": "wrong_object",
            "dest": "size"}}
# Task 007 (3A / 3B amended: paired data + contrastive loss, init = the 2b amended gated checkpoint): same streams, bars,
# checkpoints (35,508 steps, saves every 8,877) and two-skill pick rule; run /runs/<tag>_VT, Stage 2 reference = s2bp.
# init_sha256: PLACEHOLDER (None) until the 2b PASS -- filled with results/s2bp.json ckpt_sha; final_s3 refuses None.
S3 |= {w + "p": S3[w] | {"tag": S3[w]["stage"] + "p", "arm": "VT", "s2ref": "s2bp", "init_sha256": None} for w in ("a", "b")}
# Task 009b (3A / 3B one cube size per scene; used only if task 008 fails): samplers s3a1 / s3b1, paired, init = the Stage 2M
# picked checkpoint (colours-only init), pick skill = s2a (validation 21640, test 11544 eps 0-999, reference results/s2m*.json),
# bars as 3A / 3B; guards: c050 wrong target > 10 % and c075 task success < bar - 15 / slam > 10 % -> stop. Runs /runs/s3a1p_VT.
S3 |= {"a1p": S3["a"] | {"stage": "s3a1", "tag": "s3a1p", "vseed": 21665, "tseed": 11580},
       "b1p": S3["b"] | {"stage": "s3b1", "tag": "s3b1p", "vseed": 21695, "tseed": 11600}}
S3 |= {w: S3[w] | {"arm": "VT", "s2ref": "s2m", "init_sha256": "e4741cf66cbbb57686df5a2ae7948e46a3a35ac65bc8c79bb48a598a32ca0f64",
                   "pick_stage": "s2a", "pick_vseed": 21640, "pick_tseed": 11544, "guards": "one", "init_key": "is_2m_pick"}
       for w in ("a1p", "b1p")}
# Task 009 3A one diagnosed fix (consultant 113, 2026-10-27): init = the 3A pick (s3a1p_VT 035508), +17,754 steps, saves
# 008877 / 017754 -> two-save pick rule (c050 / c100), guard place < 70 % at c050, test place 11581 (k 1) + pick 11544.
S3 |= {"a1pf": S3["a1p"] | {"tag": "s3a1pf", "ckpts": {"c050": "008877", "c100": "017754"}, "guards": "fix",
                            "init_sha256": "f6adef8890138871f025994b9aeaa5714e67440c9601a272e001a998bc562f84",
                            "init_key": "is_3a_pick", "init_is_s2ref": False}}
PICK_VSEED, PICK_TSEED = 21650, 11550  # pick = Stage 2b's gated streams (GATES 3A / 3B after Stage 2 redesign #2 (2b))


def check_stage(ev: dict, seed: int, stage: str, split: str, task: str, n: int | None = None) -> None:
    c = ev["config"]
    assert default_sampling(ev), "not default sampling"
    assert c["n"] == 10 and c["image_size"] == 224 and c["seed"] == seed, (c["n"], c["image_size"], c["seed"], seed)
    assert c["stage"] == stage and c["split"] == split, (c["stage"], c["split"], stage, split)
    check_yaw(ev, 15.0)
    assert all(e["task"] == task for e in ev["episodes"]), f"an episode is not a {task} episode"
    if n is not None:
        assert len(ev["episodes"]) == n, (len(ev["episodes"]), n)


def skill_rule(k: dict[str, int], n: int) -> dict:
    """One skill of the two-skill pick rule, integer arithmetic on success counts k (c050 / c075 / c100) out of n:
    collapse iff s_100 < max(s_50, s_75) - 10 points; within 5 points (inclusive) of that line at n < 1000 -> extend."""
    early = [c for c in ("c050", "c075") if c in k]  # the 3A fix has two saves: c050 / c100 only
    km = max(k[c] for c in early)
    near = 20 * abs(k["c100"] - km + n / 10) <= n
    return {"n": n, "max_50_75": km, "line_points": 100 * km / n - 10, "collapsed": 10 * (km - k["c100"]) > n,
            "extend": n < 1000 and near, "within5_of_max": {c: 20 * (km - k[c]) <= n for c in early}}


def pick_rule2(skills: dict[str, dict[str, int]], n: dict[str, int], main: str) -> dict:
    """GATES 3A / 3B checkpoint pick (the Stage 1 rule extended to two skills). skills: {skill: {ckpt: success count}},
    n: episodes per skill, main: the skill this stage adds (place / stack; breaks ties). Pick = c100 unless it collapsed on
    either skill; a skill within 5 points of its collapse line at n < 1000 -> extend (that skill's three checkpoints to
    episodes 0-999; pick None). On collapse: the later of c075 / c050 within 5 points of the max on every collapsed
    skill; else the one with the higher `main` success (tie -> later)."""
    per = {sk: skill_rule(skills[sk], n[sk]) for sk in skills}
    ext = sorted(sk for sk, r in per.items() if r["extend"])
    out = {"skills": per, "extend_skills": ext}
    if ext:
        return out | {"decision": "extend " + ", ".join(ext) + " to 1000", "collapsed_skills": None, "pick": None}
    col = sorted(sk for sk, r in per.items() if r["collapsed"])
    out["collapsed_skills"] = col
    if not col:
        return out | {"decision": "no collapse", "pick": "c100"}
    late = [c for c in ("c075", "c050") if all(per[sk]["within5_of_max"].get(c, False) for sk in col)]
    if late:
        return out | {"decision": f"collapse on {col}: later checkpoint within 5 points of the max", "pick": late[0]}
    pick = "c075" if skills[main].get("c075", -1) >= skills[main]["c050"] else "c050"
    return out | {"decision": f"collapse on {col}: neither within 5 points; higher {main} success (tie -> later)", "pick": pick}


def arm_of(args: tuple, default: str = "U") -> str:
    return args[args.index("--arm") + 1] if "--arm" in args else default


def val_s3(which: str, *args: str) -> None:
    sp = S3[which]
    arm = arm_of(args, sp.get("arm", "U"))
    tag, raw = sp.get("tag", sp["stage"]), R / sp.get("tag", sp["stage"])
    pv, pst = sp.get("pick_vseed", PICK_VSEED), sp.get("pick_stage", "s2")
    ck = sp.get("ckpts", CKPTS_S3)
    ev = {sk: {c: load(raw / f"{tag}_{arm}_val_{sk}_{c}_s{seed}.json") for c in ck}
          for sk, seed in ((sp["task"], sp["vseed"]), ("pick", pv))}
    for sk, seed, stage in ((sp["task"], sp["vseed"], sp["stage"]), ("pick", pv, pst)):
        for c, e in ev[sk].items():
            check_stage(e, seed, stage, "train", sk)
            assert e["ckpt"].endswith(f"/{tag}_{arm}/checkpoints/{ck[c]}/pretrained_model"), e["ckpt"]
    for c in ck:
        assert ev[sp["task"]][c]["ckpt_sha"] == ev["pick"][c]["ckpt_sha"], f"{c}: the two skills ran different checkpoints"
    n = {sk: min(len(e["episodes"]) for e in ev[sk].values()) for sk in ev}
    assert all(v >= 300 for v in n.values()), n
    cnt = lambda sk, m: {c: sum(x["success"] for x in e["episodes"][:m]) for c, e in ev[sk].items()}  # noqa: E731
    res = {"arm_tag": arm, "design": f"Stage 3{which[0].upper()} ({sp['task']} s{sp['vseed']} + {pst} pick s{pv}, "
           "cube yaw +-15 deg, H 224, N 10, 35,508 steps" + (", one cube size per scene)" if "guards" in sp else ")"),
           "cube_yaw_max_deg": 15.0, "seeds": {sp["task"]: sp["vseed"], "pick": pv},
           "sampling": "default (asserted)",
           "s_300": {sk: {c: v / 300 for c, v in cnt(sk, 300).items()} for sk in ev},
           "pick_300": pick_rule2({sk: cnt(sk, 300) for sk in ev}, {sk: 300 for sk in ev}, sp["task"])}
    final_ = res["pick_300"]
    if any(v >= 1000 for v in n.values()):  # extension is per skill: that skill at 1,000, the other at 300
        m = {sk: 1000 if n[sk] >= 1000 else 300 for sk in ev}
        res["s_extended"] = {sk: {c: v / m[sk] for c, v in cnt(sk, m[sk]).items()} for sk in ev}
        res["pick_extended"] = pick_rule2({sk: cnt(sk, m[sk]) for sk in ev}, m, sp["task"]) | {"n_per_skill": m}
        final_ = res["pick_extended"]
    res["pick"] = final_["pick"]
    res["pick_step"] = ck[final_["pick"]] if final_["pick"] else None
    res["per_ckpt"] = {sk: {c: {"step": ck[c], "ckpt": e["ckpt"], "ckpt_sha": e["ckpt_sha"], "n": len(e["episodes"]),
                                "success": rate([x["success"] for x in e["episodes"]]),
                                "wrong_grasp": rate([x["wrong_grasp"] for x in e["episodes"]]),
                                **{m_: e["aggregate"][sk][m_] for m_ in ("slam", "jam", "drop", "early_close")},
                                "gpu_h": e["gpu_h"]} for c, e in ev[sk].items()} for sk in ev}
    k50 = cnt(sp["task"], 300)["c050"]
    res["early_abort_d"] = {"skill": sp["task"], "s_50": k50 / 300, "abort": 20 * k50 <= 300}  # s_50 <= 5 % on 300 episodes
    vl = {p_.stem: load(p_)["val_loss"] for p_ in sorted(raw.glob("vloss_*.json"))}
    res["offline_val_loss"] = vl
    a, b = vl.get(f"vloss_{STEP25_S3}"), vl.get(f"vloss_{CKPTS_S3['c050']}")
    res["early_abort_c"] = {"loss_25": a, "loss_50": b, "abort": (b > 2 * a) if a and b else None}
    if "tag" in sp:  # task 007: guards at c050 on the stage skill eps 0-299 ((b) not applied: converged init)
        e50 = ev[sp["task"]]["c050"]["episodes"][:300]
        kw = sum(x[sp["wrong"]] for x in e50)
        b2 = {"rule": f"{sp['wrong']} > 10 % -> stop", "count": kw, "n": 300, "fires": 10 * kw > 300}
        if sp.get("guards") == "fix":  # task 009 3A fix: place < 70 % at the first save (eps 0-299) -> stop
            res["guards_c050"] = {"place_lt_70": {"rule": f"{sp['task']} success < 70 % -> stop", "success_count": k50,
                                                  "n": 300, "fires": 10 * k50 < 7 * 300}}
        elif sp.get("guards") == "one":  # task 009b: c050 wrong target; c075 success < bar - 15 and slam > 10 % (stage skill, eps 0-299)
            e75 = ev[sp["task"]]["c075"]["episodes"][:300]
            k75, ks = sum(x["success"] for x in e75), sum(x["slam"] for x in e75)
            res["guards_c050"] = {"b2": b2}
            res["guards_c075"] = {
                "success": {"rule": f"{sp['task']} success < bar - 15 = {sp['bar'] - 15} % -> stop", "success_count": k75,
                            "n": 300, "fires": 100 * k75 < (sp["bar"] - 15) * 300},
                "slam": {"rule": f"{sp['task']} slam > 10 % -> stop", "count": ks, "n": 300, "fires": 10 * ks > 300}}
        else:
            res["guards_c050"] = {
                "b1": {"rule": f"{sp['task']} success < bar - 20 = {sp['bar'] - 20} % -> stop", "success_count": k50, "n": 300,
                       "fires": 100 * k50 < (sp["bar"] - 20) * 300},
                "b2": b2}
        shas = {f"{sk}_{c}": e["stats_sha"] for sk in ev for c, e in ev[sk].items()}
        assert len(set(shas.values())) == 1, f"stats_sha differs between the validation runs: {shas}"
        res["stats_sha"] = shas[f"{sp['task']}_c050"]
    (R / f"{tag}_val.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("s_300", "pick_300", "pick", "early_abort_c", "early_abort_d")}))


def paired_counts(x: list[bool], y: list[bool], resamples: int = 10_000) -> dict:
    """Pick, 3A / 3B (x) vs Stage 2 (y) on the same episodes: discordant counts and the paired bootstrap 95 % CI of the
    difference in points (numpy default_rng(0); `resamples` resamples of the episode pairs, drawn 1,000 at a time)."""
    import numpy as np
    xa, ya = np.asarray(x, bool), np.asarray(y, bool)
    d = xa.astype(np.int64) - ya
    n = len(d)
    rng = np.random.default_rng(0)
    means = np.concatenate([d[rng.integers(0, n, (1000, n))].mean(1) for _ in range(resamples // 1000)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": n, "stage3_only": int((d == 1).sum()), "stage2_only": int((d == -1).sum()),
            "both": int((xa & ya).sum()), "neither": int((~xa & ~ya).sum()), "diff_points": 100 * float(d.mean()),
            "ci95_points": [100 * float(lo), 100 * float(hi)], "resamples": len(means), "rng": "numpy default_rng(0)"}


def same_stream(a: dict, b: dict) -> None:
    """Two evals ran the same stream: identical config keys of the stream and identical (index, spec, instruction)."""
    for k in ("seed", "stage", "split", "cube_yaw_max_deg", "mix", "visual"):
        assert a["config"].get(k) == b["config"].get(k), (k, a["config"].get(k), b["config"].get(k))
    ka, kb = ([(e["index"], e["spec"], e["instruction"]) for e in x["episodes"]] for x in (a, b))
    assert ka == kb, "the two pick files do not hold identical episode specs"


def by_dest(eps: list[dict], mode: str) -> dict:
    """3A: success by destination pad colour; 3B: by (top size, bottom size)."""
    out: dict[str, list] = {}
    for e in eps:
        sp = json.loads(e["spec"])
        key = sp["dest"] if mode == "pad" else f"top_{sp['target'][1]}__bottom_{sp['dest'][1]}"
        out.setdefault(key, []).append(e["success"])
    return {k: rate(v) for k, v in sorted(out.items())}


def final_s3(which: str, *args: str) -> None:
    sp = S3[which]
    tag, task, ref = sp.get("tag", sp["stage"]), sp["task"], sp.get("s2ref", "s2b")
    pv, pt, pst = sp.get("pick_vseed", PICK_VSEED), sp.get("pick_tseed", PICK_TSEED), sp.get("pick_stage", "s2")
    k = int(args[args.index("--k") + 1]) if "--k" in args else 0  # attempt number: the place / stack stream advances
    tseed = sp["tseed"] + k
    v = load(R / f"{tag}_val.json")
    arm = arm_of(args, v.get("arm_tag", "U"))  # --arm, else the arm val_s3 recorded
    t = load(R / f"{tag}_test_{task}_s{tseed}.json")
    p = load(R / f"{tag}_test_pick_s{pt}.json")
    s2, s2t = load(R / f"{ref}.json"), load(R / f"{ref}_test_s{pt}.json")
    check_stage(t, tseed, sp["stage"], "train", task, 1000)
    check_stage(p, pt, pst, "train", "pick", 1000)
    check_stage(s2t, pt, pst, "train", "pick", 1000)
    same_stream(p, s2t)
    assert s2["seed"] == pt and s2["n"] == 1000, f"results/{ref}.json is not the Stage 2b test result"
    for x in (t, p):
        assert x["ckpt"].endswith(f"/{tag}_{arm}/checkpoints/{v['pick_step']}/pretrained_model"), (x["ckpt"], v["pick_step"])
    assert t["ckpt_sha"] == p["ckpt_sha"]
    eps, pe, se = t["episodes"], p["episodes"], s2t["episodes"]
    n = len(eps)
    assert all(e["wrong_object"] == (e["wrong_grasp"] or e["wrong_dest"]) for e in eps), "wrong_object != wrong_grasp or wrong_dest"
    kt, kw = sum(e["success"] for e in eps), sum(e[sp["wrong"]] for e in eps)
    c3, c2 = sum(e["success"] for e in pe), sum(e["success"] for e in se)
    assert c2 == s2["success"]["count"], f"Stage 2b per-episode successes {c2} != results/{ref}.json {s2['success']['count']}"
    fail = [e for e in eps if not e["success"]]
    gate = {f"{task}_success_ge_{sp['bar']}": 100 * kt >= sp["bar"] * n,
            f"{sp['wrong']}_lt_5": 100 * kw < 5 * n,
            "pick_ge_stage2_minus_5_points": 20 * (c3 - c2) >= -n}  # c3A >= cS2 - 50 of 1,000: exactly 5.0 below passes
    rel = ("release_failure", "release_fail_not_released", "release_fail_off_destination", "release_fail_not_clear",
           "release_fail_unstable")
    res = {"design": f"Stage 3{which[0].upper()} ({sp['stage']} / train, cube yaw +-15 deg, init = the Stage {'2M' if ref == 's2m' else '2b'} checkpoint"
                     + (", amended: paired data + contrastive loss, arm VT" if "tag" in sp else "")
                     + (", one cube size per scene" if "guards" in sp else "") + (")" if "tag" in sp else ")"),
           "arm_tag": arm, "seed": tseed, "attempt_k": k, "ckpt": t["ckpt"], "ckpt_sha": t["ckpt_sha"], "pick_step": v["pick_step"],
           "sampling": "default (asserted)", "n": n, task: {
               **{m: rate([e[m] for e in eps]) for m in ("success", "wrong_grasp", "wrong_dest", "wrong_object") + rel
                  + ("slam", "slam_10N", "slam_50N", "jam", "drop", "early_close", "distractor_disturbed", "invalid")},
               "success_by_target": breakdown(eps)["cube"],
               "by_destination" if which[0] == "a" else "by_top_bottom_size": by_dest(eps, sp["dest"]),
               "by_n_cubes": by_n_cubes(eps), "failures_never_grasped": sum(not e["grasp_confirmed"] for e in fail),
               "first_failures": [e["index"] for e in fail[:20]], "latency": t["latency"], "eval_gpu_h": t["gpu_h"]},
           "pick": {"success": rate([e["success"] for e in pe]), "wrong_grasp": rate([e["wrong_grasp"] for e in pe]),
                    "wrong_cube_confusion": confusion(pe), "slam": rate([e["slam"] for e in pe]),
                    "jam": rate([e["jam"] for e in pe]), "by": breakdown(pe), "by_n_cubes": by_n_cubes(pe),
                    "stage2_success_count": c2, "stage2_ref": f"results/{ref}.json (Stage {'2M' if ref == 's2m' else '2b'} test pick, seed {pt})",
                    "paired_vs_stage2": paired_counts([e["success"] for e in pe], [e["success"] for e in se]),
                    "latency": p["latency"], "eval_gpu_h": p["gpu_h"]},
           "gate": gate, "gate_pass": all(gate.values()), "gpu_h": t["gpu_h"] + p["gpu_h"]}
    if "tag" in sp:  # task 007: init = the 2b amended gated checkpoint; one stats_sha over every eval
        want = sp["init_sha256"]
        assert want is not None, f"S3[{which!r}]['init_sha256'] is the placeholder: fill it with results/{ref}.json ckpt_sha"
        ts = load(R / tag / f"train_summary_{arm}.json")
        res["init_check"] = {"init": ts["init"], "init_sha256": ts["init_sha256"], "expected_sha256": want,
                             f"{ref}_ckpt_sha": s2["ckpt_sha"],
                             sp.get("init_key", "is_2b_pick"): ts["init_sha256"] == want == s2["ckpt_sha"]}
        if sp.get("init_is_s2ref", True) is False:  # the 3A fix: init = the 3A pick, not the Stage 2 reference
            res["init_check"][sp["init_key"]] = ts["init_sha256"] == want
        assert res["init_check"][sp.get("init_key", "is_2b_pick")], f"init_sha256 {ts['init_sha256']} != {want} / {ref} ckpt_sha {s2['ckpt_sha']}"
        val = {f"{sk}_{c}": load(R / tag / f"{tag}_{arm}_val_{sk}_{c}_s{seed}.json")["stats_sha"]
               for sk, seed in ((task, sp["vseed"]), ("pick", pv)) for c in sp.get("ckpts", CKPTS_S3)}
        res["stats_sha"] = {"validation": val, f"test_{task}": t["stats_sha"], "test_pick": p["stats_sha"],
                            "equal_across_evals": len(set(val.values()) | {t["stats_sha"], p["stats_sha"]}) == 1}
        assert res["stats_sha"]["equal_across_evals"], f"stats_sha differs between the evals: {res['stats_sha']}"
    if which[0] == "b":
        st = [e for e in eps if e["stack_release_rest_seen"]]
        res[task]["stack_stability"] = {"released_resting_on_bottom": len(st),
                                        "stable_50_steps": sum(e["stack_stable_after_release"] for e in st),
                                        "rate": rate([e["stack_stable_after_release"] for e in st])}
    (R / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"gate": gate, "gate_pass": res["gate_pass"],
                      task: {m: res[task][m]["rate"] for m in ("success", "wrong_grasp", "wrong_dest")},
                      "pick": res["pick"]["success"]["rate"], "paired": res["pick"]["paired_vs_stage2"]}))


EXPERT_S3 = {"a1": {"stage": "s3a1", "task": "place", "seed": 11840, "fresh": 90, "chunked": 80},
             "b1": {"stage": "s3b1", "task": "stack", "seed": 11850, "fresh": 85, "chunked": 70}}


def expert_s3(which: str) -> None:
    """Task 009b precondition (as 007's s3a / s3b): fresh unchunked expert >= `fresh` %, chunked (N 10) reachability >= `chunked` %
    and within 3 points of the fresh expert (inclusive), 1,000 episodes each, stream s3a1 11840 / s3b1 11850 at m 15.
    Reads results/s3{a,b}1p_pre/s3{a,b}1_expert_{fresh,N10}_s<seed>.json, writes results/s3{a,b}1p_pre.json."""
    sp = EXPERT_S3[which]
    st, task, seed = sp["stage"], sp["task"], sp["seed"]
    raw = R / f"{st}p_pre"
    ev = {k: load(raw / f"{st}_expert_{k}_s{seed}.json") for k in ("fresh", "N10")}
    for k, e in ev.items():
        c = e["config"]
        assert c["policy"] == "expert" and c["stage"] == st and c["split"] == "train" and c["seed"] == seed, c
        assert c["expert_fresh"] == (k == "fresh") and c["n"] == 10 and c["image_size"] == 224, c
        check_yaw(e, 15.0)
        assert len(e["episodes"]) == 1000 and all(x["task"] == task for x in e["episodes"]), (k, len(e["episodes"]))
    kf, kc = (sum(x["success"] for x in ev[k]["episodes"]) for k in ("fresh", "N10"))
    n = 1000
    gate = {f"expert_fresh_ge_{sp['fresh']}": 100 * kf >= sp["fresh"] * n,
            f"chunked_ge_{sp['chunked']}": 100 * kc >= sp["chunked"] * n,
            "chunked_within_3_points": 100 * (kf - kc) <= 3 * n}
    res = {"design": f"009b precondition, {st} / train, m 15, seed {seed}, B 128, paired off, 1,000 episodes", "seed": seed, "n": n,
           "fresh": rate([x["success"] for x in ev["fresh"]["episodes"]]), "chunked_N10": rate([x["success"] for x in ev["N10"]["episodes"]]),
           "diff_points": 100 * (kf - kc) / n, "gate": gate, "gate_pass": all(gate.values())}
    (R / f"{st}p_pre.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"gate": gate, "gate_pass": res["gate_pass"], "fresh": kf / n, "chunked": kc / n}))


if __name__ == "__main__":
    sys.exit({"val": val, "choice": choice, "final": final, "val_a1": val_a1, "final_a1": final_a1, "val_r": val_r,
     "final_r": final_r, "val_r2": val_r2, "final_r2": final_r2, "val_y": val_y, "select_y": select_y,
     "final_y": final_y, "val_s2": val_s2, "final_s2": final_s2, "val_s3": val_s3, "final_s3": final_s3,
     "val_s2base": val_s2base, "final_s2base": final_s2base, "lang_s2base": lang_s2base,
     "val_s2a": val_s2a, "final_s2a": final_s2a, "lang_s2a": lang_s2a,
     "val_s2a1": val_s2a1, "final_s2a1": final_s2a1, "lang_s2a1": lang_s2a1,
     "val_s2r": val_s2r, "final_s2r": final_s2r, "lang_s2r": lang_s2r, "val_s2r1": val_s2r1, "final_s2r1": final_s2r1, "val_s2m": val_s2m, "final_s2m": final_s2m, "val_s2bp": val_s2bp, "final_s2bp": final_s2bp, "val_s2bp1": val_s2bp1, "final_s2bp1": final_s2bp1, "val_s2bp2": val_s2bp2, "final_s2bp2": final_s2bp2,
     "val_s2b": val_s2b, "final_s2b": final_s2b, "lang_s2b": lang_s2b,
     "init_disc": init_disc, "probe_s2a": probe_s2a, "probe_s3": probe_s3, "expert_s3": expert_s3}[sys.argv[1]](*sys.argv[2:]) or 0)
