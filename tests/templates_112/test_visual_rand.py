"""Stage 4 fix: the train visual distribution contains the nominal scene; held-out is disjoint from train and nominal."""

import math

import torch

from vision_lift.visual_rand import DEFAULT_LIGHT, MODES, NOMINAL_FRACTION, _hsv_to_rgb, light_elevation_deg, rgb_to_hsv

DEFAULT_CUBE = torch.tensor([0.402, 0.639, 0.906])  # measured default shape colours (renderer, 30 Sep 2026)
DEFAULT_TABLE = torch.tensor([0.267, 0.467, 0.667])


def _in_bands(x, bands):
    return any(lo <= x < hi for lo, hi in bands)


def _in_mode(hsv, mode):
    m = MODES[mode]
    h, s, v = [float(t) for t in hsv]
    return _in_bands(h, m["hues"]) and m["sat"][0] <= s <= m["sat"][1] and m["value"][0] <= v <= m["value"][1]


def test_nominal_scene_is_inside_the_train_distribution():
    for c in (DEFAULT_CUBE, DEFAULT_TABLE):
        assert _in_mode(rgb_to_hsv(c), "train"), rgb_to_hsv(c)
        assert not _in_mode(rgb_to_hsv(c), "heldout")
    el = light_elevation_deg(DEFAULT_LIGHT)
    assert MODES["train"]["light_elev"][0] <= el <= MODES["train"]["light_elev"][1]
    assert not MODES["heldout"]["light_elev"][0] <= el <= MODES["heldout"]["light_elev"][1]
    assert 0.0 < NOMINAL_FRACTION < 1.0  # exact nominal scenes are also sampled in train mode


def test_heldout_disjoint_from_train_with_margins():
    tr, ho = MODES["train"], MODES["heldout"]
    for lo, hi in ho["hues"]:
        for tlo, thi in tr["hues"]:
            assert hi + 0.03 <= tlo or thi + 0.03 <= lo, ((lo, hi), (tlo, thi))
    assert ho["value"][1] + 0.05 <= tr["value"][0]
    assert ho["light_elev"][1] + 5.0 <= tr["light_elev"][0]


def test_sampled_heldout_tints_never_fall_in_train():
    g = torch.Generator().manual_seed(0)
    n = 20000
    ho = MODES["heldout"]
    h = ho["hues"][0][0] + torch.rand(n, generator=g) * (ho["hues"][0][1] - ho["hues"][0][0])
    s = ho["sat"][0] + torch.rand(n, generator=g) * (ho["sat"][1] - ho["sat"][0])
    v = ho["value"][0] + torch.rand(n, generator=g) * (ho["value"][1] - ho["value"][0])
    back = rgb_to_hsv(_hsv_to_rgb(h, s, v))
    assert torch.allclose(back[:, 0], h, atol=1e-4) and torch.allclose(back[:, 2], v, atol=1e-5)
    assert not any(_in_mode(back[i], "train") for i in range(0, n, 97))
    assert math.isclose(light_elevation_deg(DEFAULT_LIGHT), 35.26, abs_tol=0.01)
