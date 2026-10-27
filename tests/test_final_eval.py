"""scripts/final_eval.py (task 010 table): synthetic eval JSONs, no simulator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("final_eval", ROOT / "scripts" / "final_eval.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)


def _ev(seed: int, k: int, n: int = 1000, slam: int = 0) -> dict:
    eps = [{"index": i, "n_cubes": 2 + i % 3, "success": i < k, "wrong_object": False, "wrong_dest": False,
            "slam": k <= i < k + slam, "jam": False, "early_close": i >= k, "drop": False} for i in range(n)]
    return {"config": {"seed": seed}, "episodes": eps, "ckpt_sha": "x", "stats_sha": "y",
            "latency": {"server_s_p50": 2.0, "server_s_p95": 2.1}}


def _write_all(d: Path, k: dict[tuple[str, str], int]) -> None:
    for m, (_, _, _, splits) in F.CELLS.items():
        for s, seed in splits.items():
            if isinstance(seed, str):
                continue
            p = F.cell_path(d, m, s, seed)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(_ev(seed, k.get((m, s), 900), slam=10)))


def test_cells_match_registration():
    assert {m: {s: v for s, v in c[3].items() if not isinstance(v, str)} for m, c in F.CELLS.items()} == {
        "s1": {"a": 11905, "b": 11906, "d": 11908},
        "2m": {"a": 11900, "b": 11901, "c": 11902, "d": 11903},
        "3aplace": {"a": 11910, "b": 11911, "d": 11913},
        "3apick": {"a": 11900, "b": 11901, "c": 11902, "d": 11903}}
    assert F.CELLS["3aplace"][3]["c"].startswith("heldout_comp undefined")


def test_build_rates_ci_bycubes_and_paired(tmp_path):
    _write_all(tmp_path, {("2m", "a"): 967, ("3apick", "a"): 902})
    res = F.build(tmp_path)
    x = res["cells"]["2m"]["splits"]["a"]
    assert x["success"]["count"] == 967 and x["success"]["rate"] == pytest.approx(0.967)
    lo, hi = x["success"]["ci95"]
    assert lo < 0.967 < hi
    assert x["slam"]["count"] == 10 and sum(v["n"] for v in x["success_by_n_cubes"].values()) == 1000
    p = res["paired_2m_vs_3apick"]["a"]
    assert (p["both"], p["a_only"], p["b_only"], p["neither"]) == (902, 65, 0, 33)
    assert p["diff_points"] == pytest.approx(6.5)
    assert res["cells"]["3aplace"]["splits"]["c"].keys() == {"na"}
    t = F.table(res)
    assert "96.7 %" in t and "n/a (heldout_comp undefined" in t and t.count("\n") == 5


def test_wrong_seed_or_missing_cell_fails(tmp_path):
    _write_all(tmp_path, {})
    p = F.cell_path(tmp_path, "s1", "a", 11905)
    p.write_text(json.dumps(_ev(11999, 900)))
    with pytest.raises(AssertionError):
        F.build(tmp_path)
    p.unlink()
    with pytest.raises(FileNotFoundError):
        F.build(tmp_path)
