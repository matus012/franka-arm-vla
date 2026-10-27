"""GATES Stage 4 re-plan (e2cef02) baseline bookkeeping on pulled JSONs (no simulator, no model; project venv).

  python scripts/s4b_report.py val [--seeds 0 1 2]  -> results/s4b_val_seed{k}.json  (checkpoint pick + early abort (d) + (b) / (c))
  python scripts/s4b_report.py shown                -> results/s4_shown_seed_baseline.json  (median seed by validation)
  python scripts/s4b_report.py final [--seeds ...]  -> results/s4b.json  (per seed / split / task, 3-seed mean + spread,
                                                       split (a) next to the SmolVLA criteria for reference only, B-I3)

Raw inputs (pulled from /runs) live in results/s4b/seed{k}/:
  train_summary.json                                  vla_policy.train_baseline (save_steps, early_abort (b) / (c))
  val_{c050|c075|c100}_{pick|place|stack}_s{21720|21730|21740}.json   validation, 300 eps (extended in place to 1,000 by
                                                       eval_policy --extend-to when the pick rule asks for it)
  i3_{true|shuffle}_{pick|place}_s{21720|21730}.json  B-I3, picked checkpoint, eps 0-99
Test JSONs: results/s4b_seed{k}_{a|b|c|bp}_{pick|place|stack}_s{seed}.json (a: 1,000 per task; b, c, bp: 334 / 333 / 333).

Checkpoint pick (GATES Stage 4, per seed over the 3 tasks): pick = c100 unless it collapsed on any task (s_100 < max(s_50,
s_75) - 10 points on that task). Smart extension: a task with |s_100 - line| <= 5 points at n = 300 must have all three of
its runs extended to 1,000 (the pick waits: `needs_extension` lists the runs, exit code 2). On collapse: the later of
c075 / c050 whose success is within 5 points of max(s_50, s_75) on EVERY collapsed task; else the higher mean success over
the three tasks of c050 / c075 (tie -> later). Never best-of. Early abort (d): s_50 <= 5 % on any task.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lift_rl.metrics import wilson_ci  # noqa: E402

R = ROOT / "results"
RAW = R / "s4b"
TASKS = ("pick", "place", "stack")
STAGE = {"pick": "s2", "place": "s3a", "stack": "s3b"}
VSEED = {"pick": 21720, "place": 21730, "stack": 21740}
FRAC = {"c050": 0.5, "c075": 0.75, "c100": 1.0}
EPS = 1e-9
# split -> (visual, table palette, split flag, seeds per task, episodes per task); GATES Stage 4 re-plan "Tests"
SPLITS = {"a": ("nominal", "s4r", "train", {"pick": 11610, "place": 11620, "stack": 11630}, {"pick": 1000, "place": 1000, "stack": 1000}),
          "b": ("train_rand", "s4r", "train", {"pick": 11640, "place": 11650, "stack": 11660}, {"pick": 334, "place": 333, "stack": 333}),
          "c": ("nominal", "s4r", "heldout_comp", {"pick": 11670, "place": 11680, "stack": 11690}, {"pick": 334, "place": 333, "stack": 333}),
          "bp": ("heldout", "s4r", "train", {"pick": 11730, "place": 11740, "stack": 11750}, {"pick": 334, "place": 333, "stack": 333})}
# SmolVLA final criteria, split (a), per task (GATES Stage 4): REFERENCE ONLY here -- they gate SmolVLA, never the baseline
SMOLVLA_REF = {"success_min": {"pick": 0.85, "place": 0.75, "stack": 0.60},
               "wrong_object_max": 0.05, "slam_max": 0.05, "jam_max": 0.02}
EXPERT_CEILING = {"pick": 1.0, "place": 1.0, "stack": 0.973}  # 0A.2 attempt-1 ceilings (GATES 0C data items)


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def rate(k: int, n: int) -> dict:
    return {"count": k, "n": n, "rate": k / n if n else None, "ci95": list(wilson_ci(k, n))}


def check_eval(ev: dict, *, seed: int, stage: str, split: str, visual: str, palette: str, n: int | None = None,
               at_least: int | None = None, ckpt_name: str | None = None) -> None:
    c = {"table_palette": "112", "instruction_swap": "none"} | ev["config"]
    want = {"policy": "baseline", "stage": stage, "split": split, "visual": visual, "table_palette": palette, "seed": seed,
            "envs": 128, "image_size": 224, "n": 10, "cube_yaw_max_deg": 15.0}
    bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
    assert not bad, f"eval config (got, want): {bad}"
    m = len(ev["episodes"])
    assert (n is None or m == n) and (at_least is None or m >= at_least), f"{m} episodes (want {n or ('>=' + str(at_least))})"
    assert [e["index"] for e in ev["episodes"]] == list(range(m)), "episodes are not the contiguous stream prefix"
    if ckpt_name:
        assert Path(ev["ckpt"]).name == ckpt_name, (ev["ckpt"], ckpt_name)


def ckpt_steps(summ: dict) -> dict[str, int]:
    """c050 / c075 / c100 -> the step of the checkpoint saved at that fraction (train_summary.save_steps)."""
    by_f = {float(f): int(s) for s, f in summ["save_steps"].items()}
    return {c: by_f[f] for c, f in FRAC.items()}


def succ(ev: dict, n: int) -> tuple[int, int]:
    eps = ev["episodes"][:n]
    return sum(e["success"] for e in eps), len(eps)


# --------------------------------------------------------------------------- checkpoint pick
def pick_rule(s: dict[str, dict[str, float]], n: dict[str, int]) -> dict:
    """s[task][c] = success rate at n[task] eps. Pure. -> decision dict (pick None + needs_extension when it must wait)."""
    line = {t: max(s[t]["c050"], s[t]["c075"]) - 0.10 for t in TASKS}
    need = [t for t in TASKS if n[t] < 1000 and abs(s[t]["c100"] - line[t]) <= 0.05 + EPS]
    collapsed = [t for t in TASKS if s[t]["c100"] < line[t] - EPS]
    out = {"line": line, "collapsed_tasks": collapsed, "needs_extension_tasks": need, "n": n}
    if need:  # the smart extension is decided per task before any pick (GATES: within 5 points of the line)
        return out | {"pick": None, "decision": "extend " + ",".join(need) + " to 1000"}
    if not collapsed:
        return out | {"pick": "c100", "decision": "no collapse"}
    mx = {t: max(s[t]["c050"], s[t]["c075"]) for t in collapsed}
    late = [c for c in ("c075", "c050") if all(s[t][c] >= mx[t] - 0.05 - EPS for t in collapsed)]
    if late:
        return out | {"pick": late[0], "decision": f"collapse on {collapsed}: later of 50/75 % within 5 points of the max"}
    mean = {c: sum(s[t][c] for t in TASKS) / len(TASKS) for c in ("c050", "c075")}
    p = "c075" if mean["c075"] >= mean["c050"] - EPS else "c050"  # higher mean over tasks, tie -> later
    return out | {"pick": p, "mean_success_over_tasks": mean, "decision": f"collapse on {collapsed}: higher mean over tasks"}


def val_seed(k: int) -> dict:
    d = RAW / f"seed{k}"
    summ = load(d / "train_summary.json")
    steps = ckpt_steps(summ)
    ev = {t: {c: load(d / f"val_{c}_{t}_s{VSEED[t]}.json") for c in FRAC} for t in TASKS}
    for t in TASKS:
        for c in FRAC:
            check_eval(ev[t][c], seed=VSEED[t], stage=STAGE[t], split="train", visual="nominal", palette="s4r",
                       at_least=300, ckpt_name=f"ckpt_{steps[c]:06d}")
    n = {t: 1000 if all(len(ev[t][c]["episodes"]) >= 1000 for c in FRAC) else 300 for t in TASKS}
    s = {t: {c: succ(ev[t][c], n[t])[0] / n[t] for c in FRAC} for t in TASKS}
    res = {"seed": k, "val_seeds": VSEED, "n": n, "success": s, "steps": steps, **pick_rule(s, n)}
    res["per_ckpt"] = {c: {t: {"n": n[t], **{m: rate(sum(e[m] for e in ev[t][c]["episodes"][:n[t]]), n[t])
                                           for m in ("success", "wrong_grasp", "wrong_object", "slam", "jam")},
                              "ckpt_sha": ev[t][c]["ckpt_sha"], "gpu_h": ev[t][c]["gpu_h"]} for t in TASKS} for c in FRAC}
    res["pick_step"] = steps[res["pick"]] if res["pick"] else None
    res["mean_success_picked"] = sum(s[t][res["pick"]] for t in TASKS) / len(TASKS) if res["pick"] else None
    res["early_abort_d"] = {"s_50": {t: s[t]["c050"] for t in TASKS}, "abort": any(s[t]["c050"] <= 0.05 + EPS for t in TASKS)}
    res["early_abort_b"] = summ["early_abort"]["b"]
    res["early_abort_c"] = summ["early_abort"]["c"]
    res["early_abort_a_nan"] = summ["early_abort"]["a_nan"]
    res["needs_extension"] = [f"{c} {t} (val seed {VSEED[t]}, stage {STAGE[t]})" for t in res["needs_extension_tasks"] for c in FRAC]
    res["train"] = {k_: summ[k_] for k_ in ("params", "S", "steps_per_epoch", "total_steps", "epochs", "final_step", "git_sha")}
    return res


def val(seeds: list[int]) -> int:
    rc = 0
    for k in seeds:
        r = val_seed(k)
        (R / f"s4b_val_seed{k}.json").write_text(json.dumps(r, indent=1))
        print(f"seed {k}", json.dumps({x: r[x] for x in ("n", "decision", "pick", "pick_step", "needs_extension")}),
              "abort_d", r["early_abort_d"]["abort"])
        rc |= 2 if r["pick"] is None else 0
    return rc


# --------------------------------------------------------------------------- shown seed
def median_seed(score: dict[int, float]) -> dict:
    """Median seed by validation (mean over tasks of the picked checkpoint's success); ties on the median score -> lower index."""
    vals = sorted(score.values())
    med = vals[len(vals) // 2]
    cands = sorted(k for k, v in score.items() if abs(v - med) <= EPS)
    return {"shown_seed": cands[0], "median_score": med, "scores": {str(k): v for k, v in sorted(score.items())},
            "tied_candidates": cands}


def shown(seeds: list[int]) -> None:
    v = {k: load(R / f"s4b_val_seed{k}.json") for k in seeds}
    assert all(x["pick"] for x in v.values()), "a seed has no pick yet"
    res = median_seed({k: x["mean_success_picked"] for k, x in v.items()}) | {
        "picked": {str(k): {"ckpt": x["pick"], "step": x["pick_step"]} for k, x in v.items()},
        "rule": "median seed by validation (mean over tasks of the picked checkpoint's validation success); tie -> lower index"}
    (R / "s4_shown_seed_baseline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


# --------------------------------------------------------------------------- final
def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """a = true labels, b = shuffled labels, same episodes: discordant counts and z = (n_a_only - n_b_only) / sqrt(sum)."""
    x, y = sum(p and not q for p, q in zip(a, b)), sum(q and not p for p, q in zip(a, b))
    return {"true_only": x, "shuffled_only": y, "z": (x - y) / math.sqrt(x + y) if x + y else None}


def i3(k: int, pick: str, step: int) -> dict:
    d, out = RAW / f"seed{k}", {}
    for t in ("pick", "place"):
        arms = {}
        for mode in ("true", "shuffle"):
            e = load(d / f"i3_{mode}_{t}_s{VSEED[t]}.json")
            check_eval(e, seed=VSEED[t], stage=STAGE[t], split="train", visual="nominal", palette="s4r", n=100,
                       ckpt_name=f"ckpt_{step:06d}")
            assert e["config"].get("instruction_swap", "none") == ("none" if mode == "true" else "shuffle")
            arms[mode] = e
        key = lambda e: [(x["index"], x["spec"], x["instruction"]) for x in e["episodes"]]  # noqa: E731
        assert key(arms["true"]) == key(arms["shuffle"]), "B-I3 arms differ in index, spec or instruction"
        assert "instruction_swap_report" in arms["shuffle"] and all(
            x["instruction_sent"] != x["instruction"] for x in arms["shuffle"]["episodes"])
        row = {m: {mode: rate(sum(x[m] for x in arms[mode]["episodes"]), 100) for mode in arms}
               for m in ("success", "wrong_object", "wrong_grasp")}
        row |= {m + "_mcnemar": mcnemar([x[m] for x in arms["true"]["episodes"]], [x[m] for x in arms["shuffle"]["episodes"]])
                for m in ("success", "wrong_object")}
        out[t] = row
    return out | {"picked": pick, "note": "reported, not gated: if success does not fall under shuffled labels the baseline "
                                          "does not use its labels and its numbers do not measure 'perfect parsing'"}


def test_file(k: int, split: str, t: str) -> dict:
    vis, pal, spl, seeds, ns = SPLITS[split]
    ev = load(R / f"s4b_seed{k}_{split}_{t}_s{seeds[t]}.json")
    check_eval(ev, seed=seeds[t], stage=STAGE[t], split=spl, visual=vis, palette=pal, n=ns[t])
    return ev


def spread(xs: list[float]) -> dict:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return {"mean": m, "min": min(xs), "max": max(xs), "sd": sd, "per_seed": xs}


def three_seed(per_seed: list[dict]) -> dict:
    """per_seed: aggregate dicts (metric -> {count, n, rate, ci95} | other); -> metric -> spread of rates."""
    return {m: spread([a[m]["rate"] for a in per_seed]) for m in per_seed[0]
            if isinstance(per_seed[0][m], dict) and "rate" in per_seed[0][m] and all(a.get(m, {}).get("rate") is not None for a in per_seed)}


def pooled(aggs: dict[str, dict]) -> dict:
    """Mixed-split pool over the three task streams: counts and n summed per metric present in every task."""
    ms = [m for m in aggs["pick"] if all(isinstance(aggs[t].get(m), dict) and "count" in aggs[t][m] for t in aggs)]
    return {m: rate(sum(aggs[t][m]["count"] for t in aggs), sum(aggs[t][m]["n"] for t in aggs)) for m in ms}


NOMINAL_TABLE_VALUE = 0.5  # S4R_DEFAULT_TABLE grey level (vla_tabletop.visual_rand)
NA = "n/a (not recorded)"


def value_bins(split: str, eps: dict[str, list[dict]]) -> dict | str:
    """Success by recorded table value (eval JSON per-episode `table_value`, scripts/eval_policy.visual_record), per task
    and pooled. (b): fixed bins of width 0.1 over [0.3, 0.8] (last bin closed); (b'): below / above nominal 0.5.
    Reported, never gated. Any episode without the field -> NA (never a failure). Values outside every bin are counted."""
    if split not in ("b", "bp") or any("table_value" not in e for es in eps.values() for e in es):
        return NA
    edges = [round(0.3 + 0.1 * i, 1) for i in range(6)]
    if split == "b":
        names = [f"[{a:.1f},{b:.1f}{']' if b == edges[-1] else ')'}" for a, b in zip(edges, edges[1:])]
        which = lambda v: next((i for i in range(5) if edges[i] <= v < edges[i + 1] or (i == 4 and v == edges[5])), None)  # noqa: E731
    else:
        names = ["below_nominal(<0.5)", "above_nominal(>0.5)"]
        which = lambda v: 0 if v < NOMINAL_TABLE_VALUE else 1 if v > NOMINAL_TABLE_VALUE else None  # noqa: E731
    cnt = {t: [[0, 0] for _ in names] for t in eps}
    out_of_bins = 0
    for t, es in eps.items():
        for e in es:
            i = which(e["table_value"])
            if i is None:
                out_of_bins += 1
            else:
                cnt[t][i][0] += int(e["success"])
                cnt[t][i][1] += 1
    row = lambda c: {nm: rate(k, n) for nm, (k, n) in zip(names, c)}  # noqa: E731
    pool = [[sum(cnt[t][i][j] for t in eps) for j in (0, 1)] for i in range(len(names))]
    return {"bins": names, "tasks": {t: row(c) for t, c in cnt.items()}, "pooled": row(pool), "out_of_bins": out_of_bins}


def final(seeds: list[int]) -> None:
    v = {k: load(R / f"s4b_val_seed{k}.json") for k in seeds}
    assert all(x["pick"] for x in v.values()), "a seed has no pick yet"
    shown_f = load(R / "s4_shown_seed_baseline.json")
    res: dict = {"seeds": seeds, "shown_seed": shown_f["shown_seed"], "picked": {str(k): {"ckpt": x["pick"], "step": x["pick_step"]} for k, x in v.items()},
                 "expert_ceiling_reference": EXPERT_CEILING, "per_seed": {}, "splits": {}}
    gpu_h = {str(k): 0.0 for k in seeds}
    for k in seeds:
        gpu_h[str(k)] += sum(c[t]["gpu_h"] for c in v[k]["per_ckpt"].values() for t in TASKS)
        res["per_seed"][str(k)] = {}
        for sp in SPLITS:
            if not all((R / f"s4b_seed{k}_{sp}_{t}_s{SPLITS[sp][3][t]}.json").exists() for t in TASKS):
                continue
            files = {t: test_file(k, sp, t) for t in TASKS}
            for f in files.values():
                assert Path(f["ckpt"]).name == f"ckpt_{v[k]['pick_step']:06d}", "test ckpt != the picked checkpoint"
            aggs = {t: files[t]["aggregate"][t] for t in TASKS}
            gpu_h[str(k)] += sum(f["gpu_h"] for f in files.values())
            res["per_seed"][str(k)][sp] = {"table_value_bins": value_bins(sp, {t: files[t]["episodes"] for t in TASKS}),
                                           "tasks": aggs, "latency": {t: files[t]["latency"] for t in TASKS},
                                           "ckpt_sha": {t: files[t]["ckpt_sha"] for t in TASKS}}
            if sp != "a":
                res["per_seed"][str(k)][sp]["mixed"] = pooled(aggs)
        res["per_seed"][str(k)]["i3"] = i3(k, v[k]["pick"], v[k]["pick_step"])
    res["gpu_h_evaluations_per_seed_excl_i3_frames"] = gpu_h
    for sp in SPLITS:
        have = [k for k in seeds if sp in res["per_seed"][str(k)]]
        if len(have) != len(seeds):
            continue
        ps = lambda k: res["per_seed"][str(k)][sp]  # noqa: E731
        res["splits"][sp] = {"tasks": {t: three_seed([ps(k)["tasks"][t] for k in seeds]) for t in TASKS}}
        if sp != "a":
            res["splits"][sp]["mixed"] = three_seed([ps(k)["mixed"] for k in seeds])
    if "a" in res["splits"]:  # split (a) vs the SmolVLA criteria: REFERENCE ONLY
        ref = {}
        for t in TASKS:
            m = res["splits"]["a"]["tasks"][t]
            ref[t] = {"success_mean": m["success"]["mean"], "success_min_ref": SMOLVLA_REF["success_min"][t],
                      "wrong_object_mean": m["wrong_object"]["mean"], "slam_mean": m["slam"]["mean"], "jam_mean": m["jam"]["mean"],
                      "meets_reference": bool(m["success"]["mean"] >= SMOLVLA_REF["success_min"][t]
                                              and m["wrong_object"]["mean"] < SMOLVLA_REF["wrong_object_max"]
                                              and m["slam"]["mean"] < SMOLVLA_REF["slam_max"] and m["jam"]["mean"] < SMOLVLA_REF["jam_max"])}
        res["split_a_vs_smolvla_criteria"] = {"reference_only": True, "criteria": SMOLVLA_REF, "tasks": ref,
                                              "note": "the criteria gate SmolVLA, never the baseline"}
    (R / "s4b.json").write_text(json.dumps(res, indent=1))
    print("s4b.json written; splits:", list(res["splits"]))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1:]] if "--seeds" in sys.argv else [0, 1, 2]
    rc = {"val": lambda: val(seeds), "shown": lambda: shown(seeds) or 0, "final": lambda: final(seeds) or 0}[mode]()
    sys.exit(rc)


if __name__ == "__main__":
    main()
