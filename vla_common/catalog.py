"""World catalog: the 8 cubes (4 colours x 2 sizes) and 4 pads. Pure stdlib; shared by scene, randomizer, tasks."""

from __future__ import annotations

COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow")
SIZES: tuple[str, ...] = ("small", "large")
CubeId = tuple[str, str]  # (color, size)

ALL_CUBES: tuple[CubeId, ...] = tuple((c, s) for c in COLORS for s in SIZES)

# Linear 0-1 RGB, saturated and clearly distinct under the 112 lighting. The 0A builder may edit NOMINAL_RGB and EDGE_M only.
NOMINAL_RGB: dict[str, tuple[float, float, float]] = {
    "red": (0.8, 0.08, 0.08),
    "green": (0.1, 0.65, 0.15),
    "blue": (0.1, 0.2, 0.85),
    "yellow": (0.9, 0.8, 0.1),
}

# MEASURED 0A.1 (results/s0a1_measure111.json, results/0a1_measure.json): small = the 111 cube's collision box, 2 x 0.024 m;
# large = 6.0 cm, the brief's maximum (6.0 x (cos 10 + sin 10) = 6.95 cm <= 7.0 cm).
EDGE_M: dict[str, float] = {"small": 0.048, "large": 0.060}
DENSITY: float = 1950.0  # kg/m^3, same for both sizes (large ~0.42 kg at 6 cm edge)
PAD_EDGE_M: float = 0.10  # large cube footprint at 10 deg yaw error is 6.9 cm; margin on every side
PAD_THICKNESS_M: float = 0.002


def cube_name(cube: CubeId) -> str:
    return f"cube_{cube[0]}_{cube[1]}"


def pad_name(color: str) -> str:
    return f"pad_{color}"


CUBE_NAMES: tuple[str, ...] = tuple(cube_name(c) for c in ALL_CUBES)
PAD_NAMES: tuple[str, ...] = tuple(pad_name(c) for c in COLORS)
