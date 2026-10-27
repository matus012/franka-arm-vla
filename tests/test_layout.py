"""Layout sampler (pure numpy): deterministic per rng, all objects in the region / reach box, minimum spacing, no
overlaps, parked objects outside the reach box and beyond both cameras' view (geometric bound), on the ground."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vla_common.catalog import PAD_EDGE_M
from vla_common.tasks import sample_spec
from vla_tabletop import layout as L


def _pairs(lay):
    names = list(lay)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            yield names[a], names[b]


@pytest.mark.parametrize("stage", ["s1", "s2", "s3a", "s3b", "s4"])
def test_layout_rules(stage):
    rng = np.random.default_rng(3)
    for _ in range(1500):
        spec = sample_spec(stage, "train", rng)
        lay, info = L.sample_layout(spec, rng)
        assert set(lay) == {L.cube_name(c) for c in spec.cubes} | {L.pad_name(p) for p in spec.pads}
        for n, (x, y, _) in lay.items():
            h = L.PAD_HALF if n.startswith("pad_") else 0.0
            assert L.REGION_X[0] + h - 1e-9 <= x <= L.REGION_X[1] - h + 1e-9
            assert L.REGION_Y[0] + h - 1e-9 <= y <= L.REGION_Y[1] - h + 1e-9
            assert L.REACH_BOX_LO[0] <= x <= L.REACH_BOX_HI[0] and L.REACH_BOX_LO[1] <= y <= L.REACH_BOX_HI[1]
        for a, b in _pairs(lay):
            pa, pb = np.array(lay[a][:2]), np.array(lay[b][:2])
            if a.startswith("pad_") and b.startswith("pad_"):
                assert np.max(np.abs(pa - pb)) >= PAD_EDGE_M + L.PAD_PAD_GAP - 1e-9
            elif a.startswith("cube_") and b.startswith("cube_"):
                ha = L.CUBE_HALF[a.rsplit("_", 1)[1]] * math.sqrt(2)
                hb = L.CUBE_HALF[b.rsplit("_", 1)[1]] * math.sqrt(2)
                assert np.linalg.norm(pa - pb) >= L.FINGER_REACH + max(ha, hb) + L.CUBE_GAP - 1e-9
                assert np.linalg.norm(pa - pb) > ha + hb  # no overlap
            else:
                c, p = (pa, pb) if a.startswith("cube_") else (pb, pa)
                cn = a if a.startswith("cube_") else b
                hd = L.CUBE_HALF[cn.rsplit("_", 1)[1]] * math.sqrt(2)
                assert L.dist_to_square(c, p, L.PAD_HALF) >= hd + L.PAD_GAP - 1e-9


def test_layout_deterministic():
    a = [L.sample_layout(sample_spec("s3a", "train", r), r)[0] for r in [np.random.default_rng([5, i]) for i in range(20)]]
    b = [L.sample_layout(sample_spec("s3a", "train", r), r)[0] for r in [np.random.default_rng([5, i]) for i in range(20)]]
    assert a == b


# sha256 over repr((stage, i, sorted(layout.items()), info)) for stages s1..s4, i < 300, rng([7, i]): computed with the
# sampler at commit dacf0c3 (before cube_yaw_max_deg existed; old and new also compared element-wise at that time).
OLD_DIGEST = "348b96592a17987dccc5009b2b735ea6c6e09455f3098e724e17c51f7b00b5b3"


def _draw(stage: str, i: int, m: float | None):
    r = np.random.default_rng([7, i])
    spec = sample_spec(stage, "train", r)
    lay, info = L.sample_layout(spec, r) if m is None else L.sample_layout(spec, r, cube_yaw_max_deg=m)
    return lay, info, r.random()  # next draw: the rng sequence after the layout


def test_cube_yaw_default_reproduces_old_layouts():
    import hashlib

    for m in (None, 180.0):
        h = hashlib.sha256()
        for stage in ("s1", "s2", "s3a", "s3b", "s4"):
            for i in range(300):
                lay, info, _ = _draw(stage, i, m)
                h.update(repr((stage, i, sorted(lay.items()), info)).encode())
        assert h.hexdigest() == OLD_DIGEST


@pytest.mark.parametrize("stage", ["s1", "s3b", "s4"])
def test_cube_yaw_range_changes_only_cube_yaw(stage):
    for i in range(300):
        ref, ref_info, ref_next = _draw(stage, i, 180.0)
        for m in (15.0, 0.0):
            lay, info, nxt = _draw(stage, i, m)
            assert info == ref_info and nxt == ref_next and set(lay) == set(ref)
            for k, (x, y, yaw) in lay.items():
                assert (x, y) == ref[k][:2]
                if k.startswith("cube_"):
                    assert yaw == ref[k][2] * (m / 180.0)  # the same draw, scaled
                else:
                    assert yaw == ref[k][2] == 0.0  # pads: axis-aligned for every m


def test_cube_yaw_15_and_0():
    yaws = {15.0: [], 0.0: []}
    for m in yaws:
        for i in range(2000):
            lay, _, _ = _draw("s1", i, m)
            yaws[m] += [math.degrees(v[2]) for k, v in lay.items() if k.startswith("cube_")]
    y15 = np.array(yaws[15.0])
    assert len(y15) >= 2000 and np.abs(y15).max() <= 15.0
    assert y15.min() < -14.9 and y15.max() > 14.9  # both extremes approached
    assert abs(y15.mean()) < 0.5 and 0.2 < (np.abs(y15) < 7.5).mean() < 0.8  # spread over the range, not clumped
    y0 = np.array(yaws[0.0])
    assert (y0 == 0.0).all() and not np.signbit(y0).any()


def test_make_stream_threads_cube_yaw():
    from vla_tabletop.rollout import make_stream

    def mk(m):
        return make_stream(31700, 0, 50, lambda r: sample_spec("s1", "train", r), "train", cube_yaw_max_deg=m)

    old, v15, v0 = make_stream(31700, 0, 50, lambda r: sample_spec("s1", "train", r), "train"), mk(15.0), mk(0.0)
    assert [e.layout for e in old] == [e.layout for e in mk(180.0)]
    for a, b, c in zip(old, v15, v0):
        assert a.spec == b.spec == c.spec and a.instruction == b.instruction == c.instruction
        for k, v in a.layout.items():
            assert v[:2] == b.layout[k][:2] == c.layout[k][:2]
            if k.startswith("cube_"):
                assert abs(b.layout[k][2]) <= math.radians(15.0) and c.layout[k][2] == 0.0


def test_cube_yaw_out_of_range_rejected():
    spec = sample_spec("s1", "train", np.random.default_rng(0))
    for m in (-1.0, 181.0):
        with pytest.raises(AssertionError):
            L.sample_layout(spec, np.random.default_rng(0), cube_yaw_max_deg=m)


def test_parked_poses_outside_reach_and_views():
    arr = L.layout_array([{}])[0]
    assert (arr[:, 4] == 0).all()
    pos = arr[:, :3]
    assert (pos[:, 0] > L.REACH_BOX_HI[0]).all()  # outside the reach box
    assert np.allclose(pos[:, 2], [L.FLOOR_Z + L.obj_half_height(n) for n in L.OBJ_NAMES])  # on the ground
    d = np.linalg.norm(pos[:, None, :2] - pos[None, :, :2], axis=-1) + np.eye(len(pos)) * 9
    assert d.min() >= 0.15  # spaced apart
    # wrist camera: optical axis <= 30 deg from straight down (hand held vertical), half-diagonal FOV 51 deg (12 mm on
    # 20.955 mm) -> sees at most 81 deg from nadir; the hand stays inside the reach box (z <= 0.6 + 0.1 m)
    cam_hi = np.array([L.REACH_BOX_HI[0], 0.0, L.REACH_BOX_HI[2] + 0.1])
    for p in pos:
        ang = math.degrees(math.atan2(np.linalg.norm(p[:2] - cam_hi[:2]) - 0.5, cam_hi[2] - p[2]))
        assert ang > 30 + math.degrees(math.atan(math.sqrt(2) * 10.4775 / 12.0)) + 1
    # side camera (eye (1.15, 0.45, 0.55) looking at (0.5, 0, 0.18)): every parked object is behind it
    eye, look = np.array([1.15, 0.45, 0.55]), np.array([0.50, 0.00, 0.18])
    fwd = (look - eye) / np.linalg.norm(look - eye)
    assert ((pos - eye) @ fwd < 0).all()


def test_fold_yaw_deg():
    """Spawn-yaw fold mod 90 into [-45, 45); +-45 and 135 fold to -45 (documented tie rule)."""
    import torch

    from vla_tabletop.rollout import fold_yaw_deg

    cases = {0: 0, 15: 15, -15: -15, 90: 0, 100: 10, 135: -45, -170: 10, -45: -45, 45: -45, 180: 0, -180: 0}
    got = fold_yaw_deg(torch.tensor([float(k) for k in cases], dtype=torch.float64))
    assert torch.allclose(got, torch.tensor([float(v) for v in cases.values()], dtype=torch.float64), atol=1e-9)


def test_quat_yaw_matches_raw_degrees():
    import torch

    from vla_tabletop.rollout import quat_yaw

    a = math.radians(-170.0)
    q = torch.tensor([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)], dtype=torch.float64)
    assert abs(math.degrees(quat_yaw(q).item()) + 170.0) < 1e-9
