"""Per-episode object layout (pure numpy; no simulator): where the spec's cubes and pads go on the table, and where the
unused objects are parked. Deterministic given the rng.

Placement rules (brief section 3: random, all in reach, minimum spacing, no overlaps):
- region (env frame) REGION_X x REGION_Y holds every cube centre and every whole pad footprint; it lies inside the 111
  IK workspace box (x 0.25..0.80, y +-0.45);
- pads axis-aligned, >= PAD_PAD_GAP apart (squares never overlap);
- cube yaw U(-pi, pi) by default; cube_yaw_max_deg = m < 180 narrows it to U(-m, m) deg (Stage 1 redesign #2: V15 =
  15, V0 = 0 axis-aligned). The rng call sequence is the same for every m: the U(-pi, pi) draw is always consumed and
  scaled by m / 180 (exactly 1.0 at the default, so 180 reproduces the old layouts bit for bit; 0 gives yaw 0), hence
  every position (cubes and pads) and every later draw from the rng is identical across m for the same rng state.
  Pads stay axis-aligned (yaw 0) for every m;
- cube-cube centre distance >= FINGER_REACH + the larger half-diagonal + CUBE_GAP (the open claw around either cube
  cannot touch the other); cube-pad: distance from the cube centre to the pad square >=
  cube half-diagonal + PAD_GAP (no cube starts on or overlapping a pad).
Parking (contract v2 j): on the ground (FLOOR_Z), at rest, spaced 0.2 m apart, PARK_X = 15 m in front of the robot:
outside the reach box, behind the side camera, and beyond the wrist camera's widest possible view (optical axis <= 30 deg
from straight down + 51 deg half-diagonal FOV = 81 deg from nadir; 15 m at <= 1.7 m height is 84 deg).
"""

from __future__ import annotations

import math

import numpy as np

from vla_common.catalog import CUBE_NAMES, EDGE_M, PAD_EDGE_M, PAD_NAMES, PAD_THICKNESS_M, CubeId, cube_name, pad_name

# Table-top height in the env frame, MEASURED (results/s0a1_measure111.json): the 111 cube (edge 0.048 m from its
# collision box) rests with its centre at z = 0.02099 -> top = 0.02099 - 0.024 = -0.0030 m. Re-measured per size in 0A.1.
TABLE_Z = -0.0030
OBJ_NAMES: tuple[str, ...] = CUBE_NAMES + PAD_NAMES  # 12 objects, row order of every per-object tensor
CUBE_HALF = {s: EDGE_M[s] / 2 for s in EDGE_M}
PAD_HALF = PAD_EDGE_M / 2

REGION_X = (0.33, 0.67)
REGION_Y = (-0.30, 0.30)
FINGER_REACH = 0.05  # open claw: 4 cm half opening + ~1 cm finger, from the grasped cube's centre
CUBE_GAP = 0.01
PAD_GAP = 0.015
PAD_PAD_GAP = 0.01
REACH_BOX_LO = (0.25, -0.45, 0.005)  # lift_rl.task_space BOX_LO / BOX_HI (fingertip workspace)
REACH_BOX_HI = (0.80, 0.45, 0.60)
FLOOR_Z = -1.05  # lift_rl.env_cfg ground plane
PARK_X = 15.0
PARK_Y0, PARK_DY = -1.1, 0.2


def obj_half_height(name: str) -> float:
    return PAD_THICKNESS_M / 2 if name.startswith("pad_") else CUBE_HALF[name.rsplit("_", 1)[1]]


def rest_z(name: str) -> float:
    """Centre height of an object resting on the table top (env frame)."""
    return TABLE_Z + obj_half_height(name)


def park_pose(name: str) -> tuple[float, float, float]:
    """(x, y, z) of a parked object (env frame), resting on the ground."""
    j = OBJ_NAMES.index(name)
    return PARK_X, PARK_Y0 + PARK_DY * j, FLOOR_Z + obj_half_height(name)


def cube_half_diag(c: CubeId) -> float:
    return CUBE_HALF[c[1]] * math.sqrt(2.0)


def dist_to_square(p: np.ndarray, centre: np.ndarray, half: float) -> float:
    d = np.maximum(np.abs(p - centre) - half, 0.0)
    return float(np.linalg.norm(d))


def sample_layout(spec, rng: np.random.Generator, max_restarts: int = 500, tries_per_object: int = 200,
                  cube_yaw_max_deg: float = 180.0) -> tuple[dict[str, tuple[float, float, float]], dict[str, int]]:
    """({object name: (x, y, yaw)} for spec.cubes + spec.pads, {"restarts", "draws", "objects"}): pads first, then
    cubes; draws / objects - 1 = per-object rejection rate. Cube yaw in [-cube_yaw_max_deg, cube_yaw_max_deg] deg
    (module docstring: same rng sequence for every value). Raises if no layout is found."""
    assert 0.0 <= cube_yaw_max_deg <= 180.0, cube_yaw_max_deg
    yaw_scale = cube_yaw_max_deg / 180.0  # 180 / 180 == 1.0 exactly
    draws = 0
    for restart in range(max_restarts):
        out: dict[str, tuple[float, float, float]] = {}
        pads: list[np.ndarray] = []
        ok = True
        for color in spec.pads:
            for _ in range(tries_per_object):
                draws += 1
                p = np.array([rng.uniform(REGION_X[0] + PAD_HALF, REGION_X[1] - PAD_HALF),
                              rng.uniform(REGION_Y[0] + PAD_HALF, REGION_Y[1] - PAD_HALF)])
                if all(np.max(np.abs(p - q)) >= PAD_EDGE_M + PAD_PAD_GAP for q in pads):
                    pads.append(p)
                    out[pad_name(color)] = (float(p[0]), float(p[1]), 0.0)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        cubes: list[tuple[np.ndarray, float]] = []
        for c in spec.cubes:
            hd = cube_half_diag(c)
            for _ in range(tries_per_object):
                draws += 1
                p = np.array([rng.uniform(*REGION_X), rng.uniform(*REGION_Y)])
                if all(np.linalg.norm(p - q) >= FINGER_REACH + max(hd, hq) + CUBE_GAP for q, hq in cubes) and \
                        all(dist_to_square(p, q, PAD_HALF) >= hd + PAD_GAP for q in pads):
                    cubes.append((p, hd))
                    yaw = float(rng.uniform(-math.pi, math.pi)) * yaw_scale + 0.0  # + 0.0: no -0.0 at scale 0
                    out[cube_name(c)] = (float(p[0]), float(p[1]), yaw)
                    break
            else:
                ok = False
                break
        if ok:
            return out, {"restarts": restart, "draws": draws, "objects": len(out)}
    raise RuntimeError(f"no layout found for {spec}")


def layout_array(layouts: list[dict[str, tuple[float, float, float]]], drop: float = 0.001) -> np.ndarray:
    """(N, 12, 5) float32 per object: x, y, z, yaw, on_table (1) / parked (0); env frame; row order OBJ_NAMES. Objects
    on the table start `drop` above their resting height."""
    out = np.zeros((len(layouts), len(OBJ_NAMES), 5), np.float32)
    for i, lay in enumerate(layouts):
        for j, name in enumerate(OBJ_NAMES):
            if name in lay:
                x, y, yaw = lay[name]
                out[i, j] = (x, y, rest_z(name) + (drop if name.startswith("cube_") else 0.0), yaw, 1.0)
            else:
                out[i, j, :3] = park_pose(name)
    return out
