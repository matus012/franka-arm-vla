"""112 simulator tests (GPU). Each check runs a real script in a subprocess (one sim per process) on a few envs and
asserts on its JSON. Run: .venv/Scripts/python -m pytest tests/test_sim_112.py"""

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


def test_wrist_camera_follows_hand_and_each_env_sees_only_itself(tmp_path):
    out = tmp_path / "s02.json"
    run(["scripts/s02_cameras.py", "--num_envs", "4", "--sheet_envs", "2", "--seed", "3", "--out", str(out), "--sheet_dir", str(tmp_path / "sheets")])
    r = json.loads(out.read_text())
    m = r["mounting_checks"]
    assert m["steps"] == 400
    assert m["wrist_pos_err_max_m"] < 1e-3 and m["wrist_rot_err_max_deg"] < 0.5
    assert m["foreign_pixels"] == 0


def test_student_env_banned_inputs_and_real_alignment():
    out = run(["scripts/check_student_env.py", "--num_envs", "4"])
    r = json.loads([ln for ln in out.splitlines() if ln.startswith("CHECK_STUDENT_ENV=")][-1].split("=", 1)[1])
    assert r["student_terms"] == ["joint_pos", "joint_vel", "gripper_width", "target", "prev_action"]
    assert r["student_dim"] == [27]
    assert r["teacher_tcp_to_cube_change"] > 0.05  # the teleport really moved the cube
    assert r["student_proprio_max_change_after_cube_teleport"] == 0.0
    assert r["frame_index_rule_ok"] and r["frame_vs_step_proprio_max_diff"] == 0.0
