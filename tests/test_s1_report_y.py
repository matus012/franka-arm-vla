"""s1_report redesign #2 modes (val_y / select_y / final_y): selection rule branches + bookkeeping assertions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("s1_report", Path(__file__).resolve().parents[1] / "scripts" / "s1_report.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


@pytest.mark.parametrize("v15,v0,want", [
    (0.92, None, ("V15", "v15>=90")), (0.92, 0.99, ("V15", "v15>=90")), (0.90, 0.5, ("V15", "v15>=90")),
    (0.85, 0.93, ("V0", "v0>=90")), (0.85, 0.90, ("V0", "v0>=90")), (None, 0.95, ("V0", "v0>=90")),
    (0.85, 0.80, ("V15", "higher (none >= 90)")), (0.80, 0.85, ("V0", "higher (none >= 90)")),
    (0.8, 0.8, ("V15", "higher (none >= 90)")),
    (None, 0.5, ("V0", "only variant with a v")), (0.4, None, ("V15", "only variant with a v"))])
def test_select_variant(v15, v0, want):
    assert S.select_variant(v15, v0) == want


def ev(V, seed, n, ok, step="041328", yaw=None, extra=None):
    eps = [{"index": i, "success": i < ok, "slam": False, "jam": False, "drop": False, "early_close": False,
            "spawn_yaw_fold_deg": (yaw if yaw is not None else ((i % 7) - 3) * V / 3), "target": "red_small",
            "grasp_confirmed": True, **{k: False for k in S.FAILS if k not in ("slam", "jam", "drop", "early_close")}}
           for i in range(n)]
    d = {"config": {"n": 10, "image_size": 224, "seed": seed, "cube_yaw_max_deg": float(V)},
         "server_info": {"sampling": "default"}, "ckpt": f"/runs/s1y{V}_U/checkpoints/{step}/pretrained_model",
         "ckpt_sha": "abc", "episodes": eps, "smart_length": 1000, "bars": {}, "latency": {}, "gpu_h": 0.1,
         "aggregate": {"pick": {k: 0.5 for k in ("success", "slam", "jam", "drop", "early_close")}}}
    return d | (extra or {})


def put(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "R", tmp_path)
    steps = S.CKPTS_R
    for V in (15, 0):
        for c, k in zip(steps, (0.5, 0.6, 0.7)):
            put(tmp_path / f"s1y{V}" / f"s1y{V}_U_val_{c}_s21600.json",
                ev(V, 21600, 300, int(300 * k), step=steps[c]))
        put(tmp_path / f"s1y{V}" / "vloss_010332.json", {"val_loss": 1.0})
        put(tmp_path / f"s1y{V}" / "vloss_020664.json", {"val_loss": 1.5})
    # wrong seed / wrong cube_yaw_max_deg fire
    bad = ev(15, 21601, 300, 100, step=steps["c050"])
    put(tmp_path / "s1y15" / "s1y15_U_val_c050_s21600.json", bad)
    with pytest.raises(AssertionError):
        S.val_y("15")
    bad = ev(15, 21600, 300, 100, step=steps["c050"])
    bad["config"]["cube_yaw_max_deg"] = 0.0
    put(tmp_path / "s1y15" / "s1y15_U_val_c050_s21600.json", bad)
    with pytest.raises(AssertionError):
        S.val_y("15")
    put(tmp_path / "s1y15" / "s1y15_U_val_c050_s21600.json", ev(15, 21600, 300, 150, step=steps["c050"]))
    S.val_y("15")
    S.val_y("0")
    v = json.loads((tmp_path / "s1y15_U_val.json").read_text())
    assert v["pick"] == "c100" and v["pick_step"] == "041328" and v["cube_yaw_max_deg"] == 15.0
    assert v["early_abort_c"]["abort"] is False and v["early_abort_d"]["abort"] is False

    # yaw outside the range fires
    put(tmp_path / "s1y0" / "s1y0_U_val_c050_s21600.json", ev(0, 21600, 300, 150, step=steps["c050"], yaw=1.0))
    with pytest.raises(AssertionError):
        S.val_y("0")
    put(tmp_path / "s1y0" / "s1y0_U_val_c050_s21600.json", ev(0, 21600, 300, 150, step=steps["c050"]))

    # selection: V15 missing -> V0 only; both present with wrong seed fires
    put(tmp_path / "s1y0_U_sel_s21790.json", ev(0, 21790, 1000, 800))
    S.select_y()
    s = json.loads((tmp_path / "s1y_select.json").read_text())
    assert s["tested"] == "V0" and s["branch"] == "only variant with a v" and s["tested_m"] == 0.0
    put(tmp_path / "s1y15_U_sel_s21790.json", ev(15, 21790, 1000, 950))
    S.select_y()
    s = json.loads((tmp_path / "s1y_select.json").read_text())
    assert s["tested"] == "V15" and s["branch"] == "v15>=90" and s["tested_m"] == 15.0
    assert s["tested_ckpt"].endswith("/s1y15_U/checkpoints/041328/pretrained_model")
    put(tmp_path / "s1y15_U_sel_s21790.json", ev(15, 21791, 1000, 950))
    with pytest.raises(AssertionError):
        S.select_y()
    put(tmp_path / "s1y15_U_sel_s21790.json", ev(15, 21790, 999, 950))
    with pytest.raises(AssertionError):
        S.select_y()
    put(tmp_path / "s1y15_U_sel_s21790.json", ev(15, 21790, 1000, 950))
    S.select_y()

    # final
    put(tmp_path / "s1y_U_test_s11504.json", ev(15, 11505, 1000, 950))
    with pytest.raises(AssertionError):
        S.final_y()
    put(tmp_path / "s1y_U_test_s11504.json", ev(0, 11504, 1000, 950))
    with pytest.raises(AssertionError):
        S.final_y()
    put(tmp_path / "s1y_U_test_s11504.json", ev(15, 11504, 1000, 950))
    S.final_y()
    f = json.loads((tmp_path / "s1y.json").read_text())
    assert f["gate_pass"] is True and f["tested"] == "V15" and f["cube_yaw_max_deg"] == 15.0
    assert set(f["success_by_abs_yaw"]) == {"[0,5)", "[5,10)", "[10,15.5]"}
    assert sum(b["n"] for b in f["success_by_abs_yaw"].values()) == 1000
    put(tmp_path / "s1y_U_test_s11504.json", ev(15, 11504, 1000, 800))
    S.final_y()
    assert json.loads((tmp_path / "s1y.json").read_text())["gate_pass"] is False
