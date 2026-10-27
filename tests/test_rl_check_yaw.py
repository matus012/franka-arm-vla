"""s0c_rl_check.yaw_items (redesign #2 yaw faithfulness) on synthetic sidecar rows. Needs the policy venv (lerobot)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("lerobot")
from s0c_rl_check import fold, yaw_items  # noqa: E402


def rows(raw, m, attrs=True):
    return [{"cube_yaw_max_deg": m} | ({"spawn_yaw_deg": float(y), "spawn_yaw_fold_deg": float(fold(y))} if attrs else {})
            for y in raw]


def test_fold():
    assert np.allclose(fold([0, 15, -15, 90, 100, 135, -170]), [0, 15, -15, 0, 10, -45, 10])


@pytest.mark.parametrize("m", [15.0, 0.0])
def test_good_set_passes(m):
    raw = np.random.default_rng(0).uniform(-m, m, 300) if m else np.zeros(300)
    raw[:2] = [m, -m]
    items, nums = yaw_items(rows(raw, m), m, list(raw + 90.0))  # a 90 deg offset is the same cube
    assert all(items.values()), items
    assert nums["spawn_fold"]["max"] <= m + 0.5


def test_mutant_180_set_expect_15_fails():
    raw = np.random.default_rng(0).uniform(-180, 180, 300)
    items, _ = yaw_items(rows(raw, 180.0, attrs=False), 15.0, list(raw))  # no spawn attrs, 180 layout
    assert not items["yaw_spawn_attrs_present"] and not items["yaw_regenerated_layout_fold_in_range"]
    assert not items["1_cube_yaw_max_deg_eq_expected"]
    items2, _ = yaw_items(rows(raw, 180.0), 15.0, list(raw))
    assert not items2["yaw_spawn_fold_in_range"] and not items2["yaw_regenerated_layout_fold_in_range"]


def test_disagreement_and_missing_coverage_fail():
    raw = np.random.default_rng(1).uniform(-15, 15, 300)
    items, _ = yaw_items(rows(raw, 15.0), 15.0, list(raw + 3.0))
    assert not items["yaw_spawn_agrees_with_regenerated_layout"]
    items, _ = yaw_items(rows(raw / 3, 15.0), 15.0, list(raw / 3))
    assert not items["yaw_coverage"]
