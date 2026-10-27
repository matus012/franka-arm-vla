"""Task 004 readings (scripts/g4_report.py): the GATES 004 rule on synthetic summaries, boundaries included."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import g4_report as R  # noqa: E402


def s(pick: float, wrong: float = 0.3, c4: float = 0.15, slam: float = 0.0, jam: float = 0.0) -> dict:
    return {"pick": pick, "wrong_cube": wrong, "by_n_cubes": {"2": 0.5, "3": 0.3, "4": c4}, "slam": slam, "jam": jam}


def test_sanity_fails_outside_3_points():
    r = R.readings({1.0: s(0.31 + 0.004), 2.0: s(0.9), 4.0: s(0.9), 8.0: s(0.9)})
    assert not r["sanity_w1"]["pass"] and r["best_w"] is None and r["reading"].startswith("instrument finding")
    assert R.readings({1.0: s(93 / 300), 2.0: s(0.3), 4.0: s(0.3), 8.0: s(0.3)})["sanity_w1"]["pass"]  # 31.0 %
    assert R.readings({1.0: s(75 / 300), 2.0: s(0.3), 4.0: s(0.3), 8.0: s(0.3)})["sanity_w1"]["pass"]  # 25.0 %


def test_no_effect_below_10_points():
    r = R.readings({1.0: s(84 / 300), 2.0: s(113 / 300), 4.0: s(100 / 300), 8.0: s(90 / 300)})  # +9.7 points
    assert r["reading"] == "no effect" and r["best_w"] == 2.0 and r["helping_w"] == []


def test_partial_at_exactly_10_points():
    r = R.readings({1.0: s(84 / 300), 2.0: s(114 / 300), 4.0: s(0.2), 8.0: s(0.2)})  # +10.0 points
    assert r["reading"] == "partial" and r["helping_w"] == [2.0]


def test_recovers_needs_all_three_clauses():
    base = {1.0: s(84 / 300, c4=0.15), 2.0: s(0.4), 4.0: s(0.4)}
    assert R.readings(base | {8.0: s(0.60, wrong=0.15, c4=0.30)})["reading"] == "guidance recovers grounding"
    assert R.readings(base | {8.0: s(0.60, wrong=0.16, c4=0.30)})["reading"] == "partial"
    assert R.readings(base | {8.0: s(0.59, wrong=0.10, c4=0.30)})["reading"] == "partial"
    assert R.readings(base | {8.0: s(0.70, wrong=0.10, c4=0.29)})["reading"] == "partial"


def test_motor_failure_at_the_only_helping_w():
    r = R.readings({1.0: s(84 / 300), 2.0: s(0.25), 4.0: s(0.25), 8.0: s(0.7, wrong=0.1, c4=0.4, jam=0.06)})
    assert r["helping_w"] == [8.0] and r["reading"] == "no effect"
    r = R.readings({1.0: s(84 / 300), 2.0: s(0.25), 4.0: s(0.25), 8.0: s(0.7, wrong=0.1, c4=0.4, slam=0.05)})
    assert r["reading"] == "guidance recovers grounding"  # 5 % is not > 5 %


def test_tie_takes_smaller_w():
    r = R.readings({1.0: s(84 / 300), 2.0: s(0.3), 4.0: s(0.5), 8.0: s(0.5)})
    assert r["best_w"] == 4.0
