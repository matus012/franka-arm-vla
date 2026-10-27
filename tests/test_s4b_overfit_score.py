"""scripts/s4b_overfit_score.py on a synthetic eval JSON + train_summary."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sp = importlib.util.spec_from_file_location("s4b_overfit_score", Path(__file__).resolve().parents[1] / "scripts" / "s4b_overfit_score.py")
S = importlib.util.module_from_spec(sp)
sp.loader.exec_module(S)


def rec(i, **kw):
    return {"index": i, "success": False, "early_close": False, "grasp_confirmed": False, "wrong_object": False} | kw


def fixture():
    # 10 eval episodes; trained on stream indices 0, 2, 3, 5 (seed 7) + one episode of another seed + one beyond the eval
    eps = [rec(0, success=True, grasp_confirmed=True), rec(1, success=True), rec(2, early_close=True), rec(3, success=True),
           rec(4, wrong_object=True), rec(5, grasp_confirmed=True, early_close=True)] + [rec(i) for i in range(6, 10)]
    ev = {"config": {"seed": 7}, "episodes": eps}
    kept = [{"lerobot_episode_index": k, "seed": 7, "episode_index": i} for k, i in enumerate((0, 2, 3, 5))]
    kept += [{"lerobot_episode_index": 4, "seed": 8, "episode_index": 1}, {"lerobot_episode_index": 5, "seed": 7, "episode_index": 99}]
    return ev, {"kept_train_episodes": kept}


def test_counts_split_and_wilson():
    r = S.score(*fixture())
    t, o = r["trained_on"], r["not_trained_on"]
    assert t["n"] == 4 and t["success"]["count"] == 2 and t["early_close"]["count"] == 2 and t["grasp_confirmed"]["count"] == 2
    assert t["wrong_object"]["count"] == 0 and t["success"]["rate"] == 0.5
    assert o["n"] == 6 and o["success"]["count"] == 1 and o["wrong_object"]["count"] == 1 and o["early_close"]["count"] == 0
    assert r["kept_other_seed"][0]["episode_index"] == 1 and r["kept_without_eval_record"] == [99]
    lo, hi = t["success"]["ci95"]
    assert lo == pytest.approx(0.15, abs=0.01) and hi == pytest.approx(0.85, abs=0.01)  # Wilson 2/4
    assert t["wrong_object"]["ci95"][0] == 0.0
    assert r["n_eval_records"] == 10


def test_summary_without_ids_is_rejected():
    ev, _ = fixture()
    with pytest.raises(AssertionError):
        S.score(ev, {"kept_train_episodes": None})
    with pytest.raises(AssertionError):
        S.score(ev, {"kept_train_episodes": [{"lerobot_episode_index": 0}]})


def test_cli(tmp_path, monkeypatch):
    ev, sm = fixture()
    (tmp_path / "e.json").write_text(json.dumps(ev))
    (tmp_path / "s.json").write_text(json.dumps(sm))
    monkeypatch.setattr(sys, "argv", ["x", "--eval", str(tmp_path / "e.json"), "--summary", str(tmp_path / "s.json"),
                                      "--out", str(tmp_path / "o" / "r.json")])
    S.main()
    assert json.loads((tmp_path / "o" / "r.json").read_text())["trained_on"]["n"] == 4
