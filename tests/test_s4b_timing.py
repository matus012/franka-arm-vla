"""scripts/s4b_timing.py on synthetic steps.jsonl: overlap window, charged rate, E_B, gpu util join."""
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("s4b_timing", Path(__file__).resolve().parents[1] / "scripts" / "s4b_timing.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

BS, T0 = 128, 1_800_000_000.0


def write_proc(d: Path, start: float, dt: float, c_frac: float) -> None:
    d.mkdir(parents=True)
    rows = [{"step": k, "epoch": 0, "loss": 1.0, "index": list(range(BS)), "t": start + k * dt, "w": dt, "c": dt * c_frac}
            for k in range(1, 351)]
    (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "train_summary.json").write_text(json.dumps({"timing": {"gpu_s_per_sample": dt / BS}}))


@pytest.fixture
def runs(tmp_path):
    write_proc(tmp_path / "A", T0, 0.20, 0.9)  # one process: 0.2 s / 128 samples
    for k, off in enumerate((0.0, 15.0, 30.0)):  # three processes, staggered starts, 0.45 s per step each -> 3 x 128 / 0.45 s
        write_proc(tmp_path / f"B{k}", T0 + 1000 + off, 0.45, 0.8)
    ep = T0 + 1000
    loc = datetime.fromtimestamp(ep).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    (tmp_path / "clock_epoch.txt").write_text(f"{ep}\n")
    (tmp_path / "clock_local.txt").write_text(loc + "\n")
    lines = []
    for s in range(0, 400):  # utilisation 50 % before the B window, 90 % inside B
        e = ep + s
        u = 90 if e > ep + 40 else 50
        lines.append(datetime.fromtimestamp(e).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3] + f", {u}, 1000")
    (tmp_path / "gpu_util.csv").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_rates_overlap_and_e_b(runs):
    a = M.config(runs, ["A"])
    assert a["charged_rate_gpu_s_per_sample"] == pytest.approx(0.20 / BS, rel=1e-6)
    assert a["data_wait_share"] == pytest.approx(0.1) and a["compute_only_rate_gpu_s_per_sample"] == pytest.approx(0.18 / BS)
    b = M.config(runs, ["B0", "B1", "B2"])
    lo, hi = b["overlap_window_epoch_s"]
    assert lo == pytest.approx(T0 + 1000 + 30.0 + 50 * 0.45) and hi == pytest.approx(T0 + 1000 + 350 * 0.45)  # B2's 50 .. B0's 350
    assert b["charged_rate_gpu_s_per_sample"] == pytest.approx(0.45 / (3 * BS), rel=0.02)
    assert b["data_wait_share"] == pytest.approx(0.2)
    # mutant: counting every process's full own window against the overlap wall overstates the samples
    naive = b["overlap_wall_s"] / (3 * 300 * BS)
    assert naive < 0.95 * b["charged_rate_gpu_s_per_sample"]
    assert M.e_b(0.2 / BS, 1_220_800) == min(6, int(6.0 * 3600 / (3 * 1_220_800 * 0.2 / BS)))
    assert M.e_b(1e-6, 1_220_800) == 6


def test_cli_picks_lower_rate_and_gpu_util(runs, tmp_path, monkeypatch):
    import sys
    out = tmp_path / "o" / "t.json"
    monkeypatch.setattr(sys, "argv", ["s4b_timing", "--runs", str(runs), "--out", str(out)])
    M.main()
    j = json.loads(out.read_text())
    assert j["chosen_config"] == "B" and j["S"] == 1_220_800 and j["r_B"] == j["configs"]["B"]["charged_rate_gpu_s_per_sample"]
    assert j["E_B"] == j["configs"]["B"]["E_B"]
    gu = j["configs"]["B"]["gpu_util"]
    assert gu["mean_util_pct"] == pytest.approx(90.0) and gu["samples"] > 100  # clock offset applied, window inside the 90 % phase


def test_non_overlapping_windows_are_refused(tmp_path):
    write_proc(tmp_path / "B0", T0, 0.1, 1.0)
    write_proc(tmp_path / "B1", T0 + 500, 0.1, 1.0)
    with pytest.raises(AssertionError, match="overlap"):
        M.config(tmp_path, ["B0", "B1"])
