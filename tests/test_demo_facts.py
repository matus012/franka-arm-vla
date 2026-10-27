"""Task 011 G2 / G4: the demo cards are generated from the evaluator's constants and the audited result JSONs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import demo_facts as F  # noqa: E402


def test_cards_match_a_fresh_build_and_the_audited_writeups():
    assert F.check(json.loads(F.OUT.read_text())) == []


def test_rules_come_from_the_detector_constants():
    from vla_tabletop.detectors import Consts113

    r = F.rules()
    k = Consts113()
    assert f"{100 * k.pick_dz:.0f} cm" in r["pick"] and f"last {k.k_place} control steps" in r["place"]
    assert f"last {k.k_stack} control steps" in r["stack"] and r["_T_s"] == {"pick": 7.0, "place": 11.0, "stack": 11.0}


def test_check_catches_a_hand_edited_number():
    cards = json.loads(F.OUT.read_text())
    cards["chapters"][0]["gate"]["rate"] = 0.99
    assert F.check(cards)


def test_verdicts_follow_the_bars():
    c = {x["key"]: x for x in F.build()["chapters"]}
    assert [c[k]["verdict"] for k in ("s1", "2m", "sizes", "3aplace", "3bstack")] == ["PASS", "PASS", "FAIL", "FAIL", "FAIL"]
