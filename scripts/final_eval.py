"""Task 010 final evaluation table (reported only; no gate) from the pulled eval JSONs (no simulator, no model).

  python scripts/final_eval.py [--dir results/final] [--out results/final_eval.json]
      -> per (model, split) cell: success / wrong object / wrong pad / slam / jam / early close / drop with Wilson 95 % CIs,
         server latency p50 / p95, success by cube count; 2M vs 3A pick paired on the shared s2a streams; prints a markdown
         table (the README / REPORT table) and writes the JSON.
Cells and seeds are the GATES 010 registration (4a7f7b6); a missing cell file is an error, an n/a cell carries its reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lift_rl.metrics import wilson_ci  # noqa: E402

SPLITS = {"a": "seen", "b": "rand", "c": "heldout", "d": "para"}
SPLIT_NAMES = {"a": "(a) seen, nominal", "b": "(b) randomized visuals (OOD)", "c": "(c) held-out combinations",
               "d": "(d) held-out paraphrases"}
# model key -> (label, skill, verdict, {split: seed | n/a reason})
CELLS: dict[str, tuple[str, str, str, dict[str, int | str]]] = {
    "s1": ("S1 (single cube)", "pick", "PASS", {"a": 11905, "b": 11906, "c": "not in the 010 table (single cube)",
                                                 "d": 11908}),
    "2m": ("2M (colours, 2-4 cubes)", "pick", "PASS", {"a": 11900, "b": 11901, "c": 11902, "d": 11903}),
    "3aplace": ("3A attempt (place)", "place", "FAIL", {"a": 11910, "b": 11911,
                                                        "c": "heldout_comp undefined for s3a1 (tasks.py:197)", "d": 11913}),
    "3apick": ("3A attempt (pick retention)", "pick", "FAIL", {"a": 11900, "b": 11901, "c": 11902, "d": 11903}),
}
FLAGS = ("success", "wrong_object", "wrong_dest", "slam", "jam", "early_close", "drop")


def cell_path(d: Path, model: str, split: str, seed: int) -> Path:
    return d / f"e10_{model}" / f"f10_{model}_{split}_{SPLITS[split]}_s{seed}.json"


def summarize(ev: dict) -> dict:
    eps = ev["episodes"]
    n = len(eps)
    out: dict = {"n": n, "ckpt_sha": ev.get("ckpt_sha"), "stats_sha": ev.get("stats_sha"),
                 "latency": {k: ev["latency"][k] for k in ("server_s_p50", "server_s_p95")}}
    for f in FLAGS:
        k = sum(bool(e[f]) for e in eps)
        out[f] = {"count": k, "rate": k / n, "ci95": list(wilson_ci(k, n))}
    by: dict[int, list[int]] = {}
    for e in eps:
        by.setdefault(e["n_cubes"], [0, 0])
        by[e["n_cubes"]][0] += bool(e["success"])
        by[e["n_cubes"]][1] += 1
    out["success_by_n_cubes"] = {str(c): {"count": k, "n": m, "rate": k / m} for c, (k, m) in sorted(by.items())}
    return out


def paired(a: dict, b: dict) -> dict:
    sa = {e["index"]: bool(e["success"]) for e in a["episodes"]}
    sb = {e["index"]: bool(e["success"]) for e in b["episodes"]}
    assert sa.keys() == sb.keys(), "paired streams differ in episode indices"
    both = sum(sa[i] and sb[i] for i in sa)
    a_only = sum(sa[i] and not sb[i] for i in sa)
    b_only = sum(sb[i] and not sa[i] for i in sa)
    return {"n": len(sa), "both": both, "a_only": a_only, "b_only": b_only, "neither": len(sa) - both - a_only - b_only,
            "diff_points": 100 * (a_only - b_only) / len(sa)}


def build(d: Path) -> dict:
    res: dict = {"design": "GATES 010 (4a7f7b6): reported only, no gate; (b) is out of distribution for every model",
                 "cells": {}, "paired_2m_vs_3apick": {}}
    raw: dict = {}
    for m, (label, skill, verdict, splits) in CELLS.items():
        res["cells"][m] = {"label": label, "skill": skill, "verdict": verdict, "splits": {}}
        for s, seed in splits.items():
            if isinstance(seed, str):
                res["cells"][m]["splits"][s] = {"na": seed}
                continue
            p = cell_path(d, m, s, seed)
            ev = json.loads(p.read_text())
            assert ev["config"]["seed"] == seed and len(ev["episodes"]) == 1000, p
            raw[(m, s)] = ev
            res["cells"][m]["splits"][s] = {"seed": seed, "file": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT)
                                            else str(p), **summarize(ev)}
    for s in "abcd":
        res["paired_2m_vs_3apick"][s] = paired(raw[("2m", s)], raw[("3apick", s)])
    return res


def table(res: dict) -> str:
    rows = ["| model | skill | " + " | ".join(SPLIT_NAMES[s] for s in "abcd") + " |",
            "|---|---|" + "---|" * 4]
    for c in res["cells"].values():
        cells = []
        for s in "abcd":
            x = c["splits"][s]
            if "na" in x:
                cells.append(f"n/a ({x['na']})")
            else:
                lo, hi = x["success"]["ci95"]
                cells.append(f"{100 * x['success']['rate']:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}], slam "
                             f"{100 * x['slam']['rate']:.1f} %")
        rows.append(f"| {c['label']} ({c['verdict']}) | {c['skill']} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results" / "final")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "final_eval.json")
    a = ap.parse_args()
    res = build(a.dir)
    a.out.write_text(json.dumps(res, indent=1))
    print(table(res))
    for s, p in res["paired_2m_vs_3apick"].items():
        print(f"paired 2M vs 3A pick ({s}): both {p['both']}, 2M-only {p['a_only']}, 3A-only {p['b_only']}, "
              f"neither {p['neither']} -> {p['diff_points']:+.1f} pts")


if __name__ == "__main__":
    main()
