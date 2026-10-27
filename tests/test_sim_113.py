"""113 simulator tests (GPU; run through ops/run_job.ps1). Each check runs a script in a subprocess (one sim per process).
Camera-side checks (parked objects 0 px in both cameras, distinct segmentation ids, wrist mount, foreign pixels) are in
scripts/s0a1_scene.py --part cams (results/0a1_cams.json); layout rules are tested without a simulator in test_layout.py.
Run: .venv/Scripts/python -m pytest tests/test_sim_113.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str], timeout: int = 900) -> str:
    r = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def test_scene_loads_obs_whitelist_and_parking():
    out = run(["scripts/check_scene_113.py"])
    r = json.loads([ln for ln in out.splitlines() if ln.startswith("CHECK_SCENE=")][-1].split("=", 1)[1])
    assert r["obs_groups"] == ["policy"]
    assert r["policy_terms"] == ["joint_pos", "joint_vel", "gripper_width"]
    assert r["policy_dims"] == [[9], [9], [1]] and r["policy_shape"][1] == 19
    assert r["parked_max_move_m"] < 1e-3  # contract v2 j
    assert not r["parked_in_reach_box"]
    assert r["on_table_max_move_m"] < 5e-3  # dropped 1 mm, then at rest
    assert r["teleport_moved_min_m"] > 0.05  # every cube and pad really moved (GATES 0B.4 item 1)
    assert r["teleport_state_max_abs_change"] == 0.0  # ... and the policy state did not change at all
