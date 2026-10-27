"""Task 011: the demo video's card text, generated (never hand-typed) from the evaluator's constants and the audited
result JSONs -> media/cards.json. `check()` re-derives every number and asserts it against the sources and the audited
write-ups (G2 / G4); tests/test_demo_facts.py runs it.

  python scripts/demo_facts.py            (write media/cards.json, then check it)
  python scripts/demo_facts.py --check    (check only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "media" / "cards.json"
AUDITED = ("REPORT.md", "reports/2026-10-27-final.md", "README.md")  # the write-ups the numbers must appear in
CONTROL_HZ = 50

# Skill chapters: model, the stage gate it was judged on (result JSON + metric), the 010 re-check cell (or None).
SKILLS = [
    {"key": "s1", "title": "One cube: pick it up", "task": "pick", "templates": "s1",
     "model": "Stage 1 model", "data": "2,000 scripted-expert demos",
     "scene": "one cube (4 colours x 2 sizes), its yaw within +-15 deg of straight (the owner made the task easier "
              "after two misses at any yaw)",
     "gate": "results/s1y_U_test_s11504.json", "split_label": "1,000 unseen test scenes", "final": ("s1", "a")},
    {"key": "2m", "title": "Colours: pick the cube named by its colour", "task": "pick", "templates": "pick",
     "model": "colour model", "data": "6,000 scripted-expert demos (each scene shown once per cube that can be named)",
     "scene": "2-4 cubes of different colours, one size per scene, 4 coloured pads",
     "gate": "results/s2m_test_s11544.json", "split_label": "1,000 unseen test scenes", "final": ("2m", "a")},
    {"key": "sizes", "title": "Colours and sizes: \"the small red cube\"", "task": "pick", "templates": "pick",
     "model": "sizes model", "data": "12,000 scripted-expert demos (paired scenes)",
     "scene": "2-5 cubes, two sizes, so the size word is sometimes needed",
     "gate": "results/s2bp2_test_s11552.json", "split_label": "1,000 unseen test scenes", "final": None},
    {"key": "3aplace", "title": "Place the named cube on the named pad", "task": "place", "templates": "place",
     "model": "place model", "data": "~4,000 place demos + the colour-pick demos (paired scenes)",
     "scene": "2-4 cubes, 4 coloured pads",
     "gate": "results/s3a1p_test_place_s11580.json", "split_label": "1,000 unseen test scenes",
     "final": ("3aplace", "a")},
    {"key": "3bstack", "title": "Stack the named cube on another cube", "task": "stack", "templates": "stack",
     "model": "stack model (training stopped at 75 %)", "data": "4,000 stack demos + the colour-pick demos (paired scenes)",
     "scene": "2-4 cubes, 4 coloured pads",
     "gate": "results/s3b1p/c075/s3b1p_VT_val_stack_c075_s21695.json", "bar": 0.70, "guard": 0.55,
     "split_label": "300 validation scenes; stopped by its guard before any test", "final": None},
]
BREAKS = [  # (cell, split, title, what it shows)
    ("2m", "b", "Randomized table colour and lighting",
     "The colour model on table colours / lighting it never saw in training (no 113 model was trained on these)."),
    ("2m", "c", "Colour-size combinations never used as targets",
     "\"the large green cube\", \"the small yellow cube\": each word seen in training, never this combination as a target."),
    ("3apick", "d", "Reworded instructions",
     "The place model asked to pick, in words it was not trained on (\"grab ...\", \"get hold of ...\")."),
]
WRONG = {"pick": "wrong_object", "place": "wrong_dest", "stack": "wrong_object"}


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def rules() -> dict[str, str]:
    """Success rule per task in plain words, from the evaluator's own constants (vla_tabletop.detectors)."""
    from lift_rl.metrics import Thresholds
    from vla_tabletop.detectors import Consts113

    k, th = Consts113(), Thresholds()
    T = {t: json.loads((ROOT / "results" / f"0a2_{t}_timing.json").read_text())["T_steps"] for t in ("pick", "place", "stack")}
    s = {t: T[t] / CONTROL_HZ for t in T}
    common = (f"Also scored: wrong cube; slam (hand or finger pushes on the table or a pad with > {th.slam_n:.0f} N); "
              f"drop (the grasped cube is not held for > {k.drop_steps} steps away from the destination).")
    return {
        "pick": (f"Success = at the end of the {s['pick']:.0f} s episode the named cube is held by both fingers and "
                 f"lifted at least {100 * k.pick_dz:.0f} cm above where it rested. " + common),
        "place": (f"Success = at the end of the {s['place']:.0f} s episode the named cube rests flat (< {k.flat_deg:.0f} deg "
                  f"tilt) and still on the named pad (centre inside the pad), released for the last {k.k_place} control "
                  f"steps ({k.k_place / CONTROL_HZ:.1f} s), with the gripper at least {100 * k.clear_dz:.0f} cm above it. "
                  + common),
        "stack": (f"Success = at the end of the {s['stack']:.0f} s episode the named cube rests flat and still on top of "
                  f"the named cube (centre over its top face), both released, for the last {k.k_stack} control steps "
                  f"({k.k_stack / CONTROL_HZ:.1f} s), with the gripper at least {100 * k.clear_dz:.0f} cm above. " + common),
        "_T_s": s,
    }


def gate_numbers(ch: dict) -> dict:
    d = json.loads((ROOT / ch["gate"]).read_text())
    a = d["aggregate"][ch["task"]]
    bars = {(t, m): b for t, m, op, b in d.get("bars") or []}
    bar = bars.get((ch["task"], "success"), ch.get("bar"))
    succ = a["success"]
    passed = succ["rate"] >= bar and not ch.get("guard")
    out = {"rate": succ["rate"], "count": succ["count"], "n": d["n_episodes"], "ci95": succ["ci95"], "bar": bar,
           "verdict": "PASS" if passed else "FAIL", "wrong": a[WRONG[ch["task"]]]["rate"]}
    if ch.get("guard"):
        out["guard"] = ch["guard"]
    return out


def final_cell(model: str, split: str) -> dict:
    d = json.loads((ROOT / "results" / "final_eval.json").read_text())["cells"][model]["splits"][split]
    return {"rate": d["success"]["rate"], "count": d["success"]["count"], "n": d["n"], "ci95": d["success"]["ci95"],
            "wrong": d["wrong_object"]["rate"], "drop": d["drop"]["rate"]}


def build() -> dict:
    from vla_common.tasks import TEMPLATES

    R = rules()
    chapters = []
    for ch in SKILLS:
        g = gate_numbers(ch)
        lines_result = [f"Judged on {ch['split_label']}: {pct(g['rate'])} success (95 % CI {pct(g['ci95'][0])} to "
                        f"{pct(g['ci95'][1])}), bar >= {100 * g['bar']:.0f} % -> {g['verdict']}"]
        if ch.get("guard"):
            lines_result.append(f"(training was stopped: it needed >= {100 * ch['guard']:.0f} % at 75 % of training "
                                f"to continue)")
        f = None
        if ch["final"]:
            f = final_cell(*ch["final"])
            lines_result.append(f"Re-checked later on 1,000 fresh scenes: {pct(f['rate'])}")
        tpl = TEMPLATES["train"][ch["templates"]]
        chapters.append({"key": ch["key"], "title": ch["title"], "task": ch["task"], "model": ch["model"],
                         "scene": ch["scene"], "data": ch["data"],
                         "templates": [t.replace("{t}", "<cube>").replace("{d}", "<pad>" if ch["task"] == "place" else "<cube>")
                                       for t in tpl],
                         "rule": R[ch["task"]], "time_limit_s": R["_T_s"][ch["task"]], "result_lines": lines_result,
                         "gate": g, "final": f, "verdict": g["verdict"]})
    breaks = []
    for model, split, title, what in BREAKS:
        f = final_cell(model, split)
        seen = final_cell(model, "a")
        extra = (f"; wrong cube {pct(f['wrong'])}" if split == "c" else f"; dropped {pct(f['drop'])}" if split == "d" else "")
        breaks.append({"key": f"{model}_{split}", "title": title, "what": what, "final": f, "seen": seen,
                       "line": f"Measured on 1,000 fresh scenes: {pct(f['rate'])} success{extra} (same model on "
                               f"ordinary scenes: {pct(seen['rate'])})"})
    plain = json.loads((ROOT / "results" / "s2a1_test_s11541.json").read_text())["aggregate"]["pick"]
    return {"source": "scripts/demo_facts.py (generated; do not edit by hand)", "chapters": chapters, "breaks": breaks,
            "grounding": {"plain_rate": plain["success"]["rate"], "plain_wrong": plain["wrong_object"]["rate"],
                          "final_rate": gate_numbers(SKILLS[1])["rate"]},
            "budget_h200_h": 178.95}


def check(cards: dict) -> list[str]:
    """G2 / G4: every number on a card equals a fresh derivation from the sources, and appears in an audited write-up."""
    fresh = build()
    errs = []
    if json.dumps(fresh, sort_keys=True) != json.dumps(cards, sort_keys=True):
        errs.append("cards.json differs from a fresh build from the sources (G2 / G4)")
    text = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in AUDITED)
    nums = [c["gate"]["rate"] for c in cards["chapters"]] + [b["final"]["rate"] for b in cards["breaks"]]
    nums += [c["final"]["rate"] for c in cards["chapters"] if c["final"]]
    nums += [cards["grounding"]["plain_rate"]]
    for x in nums:
        if f"{100 * x:.1f}" not in text:
            errs.append(f"{100 * x:.1f} % on a card is not in any audited write-up {AUDITED}")
    if f"{cards['budget_h200_h']:.2f}" not in text:
        errs.append("budget not in the audited write-ups")
    return errs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if not a.check:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(build(), indent=1))
    errs = check(json.loads(OUT.read_text()))
    print("\n".join(errs) or f"G2/G4 OK: {OUT.relative_to(ROOT)}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
