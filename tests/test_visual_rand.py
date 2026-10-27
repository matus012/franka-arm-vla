"""113 visual randomizer: train/heldout disjoint, train contains nominal, object colours stay nearest their named colour."""

import numpy as np
import torch

from vla_common.catalog import ALL_CUBES, COLORS, NOMINAL_RGB
from vla_tabletop.visual_rand import (DEFAULT_LIGHT, DEFAULT_TABLE, MODES, NOMINAL_FRACTION, light_elevation_deg, rgb_to_hsv,
                                      sample_params)


def _in_bands(x, bands):
    return any(lo <= x < hi for lo, hi in bands)


def _in_mode(hsv, mode):
    m = MODES[mode]
    h, s, v = [float(t) for t in hsv]
    return _in_bands(h, m["hues"]) and m["sat"][0] <= s <= m["sat"][1] and m["value"][0] <= v <= m["value"][1]


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (..., 3) in 0-1 -> CIELAB (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _de00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference between CIELAB arrays (broadcast over leading dims; Sharma, Wu & Dalal 2005)."""
    L1, a1, b1 = np.moveaxis(lab1, -1, 0)
    L2, a2, b2 = np.moveaxis(lab2, -1, 0)
    cm = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cm ** 7 / (cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1, c2 = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1, h2 = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    zero = c1 * c2 == 0
    dh = h2 - h1
    dh = np.where(zero, 0.0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dH = 2 * np.sqrt(c1 * c2) * np.sin(np.radians(dh / 2))
    lm, cpm, hs = (L1 + L2) / 2, (c1 + c2) / 2, h1 + h2
    hm = np.where(zero, hs, np.where(np.abs(h1 - h2) > 180, np.where(hs < 360, hs + 360, hs - 360), hs) / 2)
    t = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm)) + 0.32 * np.cos(np.radians(3 * hm + 6))
         - 0.20 * np.cos(np.radians(4 * hm - 63)))
    rc = 2 * np.sqrt(cpm ** 7 / (cpm ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (lm - 50) ** 2 / np.sqrt(20 + (lm - 50) ** 2)
    sc, sh = 1 + 0.045 * cpm, 1 + 0.015 * cpm * t
    rt = -np.sin(np.radians(60 * np.exp(-(((hm - 275) / 25) ** 2)))) * rc
    x, y, z = (L2 - L1) / sl, (c2 - c1) / sc, dH / sh
    return np.sqrt(x ** 2 + y ** 2 + z ** 2 + rt * y * z)


def test_nominal_scene_is_inside_the_train_distribution():
    tab = rgb_to_hsv(torch.tensor(DEFAULT_TABLE))
    assert _in_mode(tab, "train") and not _in_mode(tab, "heldout")
    el = light_elevation_deg(DEFAULT_LIGHT)
    assert MODES["train"]["light_elev"][0] <= el <= MODES["train"]["light_elev"][1]
    assert not MODES["heldout"]["light_elev"][0] <= el <= MODES["heldout"]["light_elev"][1]
    g = torch.Generator().manual_seed(0)
    p = sample_params("train", 10000, g)
    nom = p["nominal"]
    assert abs(float(nom.float().mean()) - NOMINAL_FRACTION) < 0.02
    k = int(nom.sum())
    assert torch.equal(p["table_rgb"][nom], torch.tensor(DEFAULT_TABLE).expand(k, 3))
    for j, (c, _) in enumerate(ALL_CUBES):
        assert torch.allclose(p["cube_rgb"][nom][:, j], torch.tensor(NOMINAL_RGB[c]).expand(k, 3))
    assert float(p["side_eye_off"][nom].abs().max()) == 0.0 and float(p["wrist_off"][nom].abs().max()) == 0.0
    assert float(p["side_eye_off"][~nom].abs().max()) > 0.005  # non-nominal envs are jittered
    lights = [sample_params("train", 1, torch.Generator().manual_seed(s))["light"] for s in range(200)]
    assert any(torch.allclose(l, torch.tensor(DEFAULT_LIGHT)) for l in lights)
    pn = sample_params("nominal", 4, g)
    assert pn["nominal"].all() and torch.allclose(pn["light"], torch.tensor(DEFAULT_LIGHT))


def test_heldout_disjoint_from_train_with_margins():
    tr, ho = MODES["train"], MODES["heldout"]
    for lo, hi in ho["hues"]:
        for tlo, thi in tr["hues"]:
            assert hi + 0.03 <= tlo or thi + 0.03 <= lo, ((lo, hi), (tlo, thi))
    assert ho["value"][1] + 0.05 <= tr["value"][0]
    assert ho["light_elev"][1] + 5.0 <= tr["light_elev"][0]


def test_sampled_table_and_light_respect_the_splits():
    g = torch.Generator().manual_seed(1)
    ph = sample_params("heldout", 10000, g)
    assert not ph["nominal"].any()
    hsv = rgb_to_hsv(ph["table_rgb"])
    assert all(_in_mode(hsv[i], "heldout") for i in range(0, 10000, 7))
    assert not any(_in_mode(hsv[i], "train") for i in range(0, 10000, 7))
    lo, hi = MODES["heldout"]["light_elev"]
    els = [light_elevation_deg(sample_params("heldout", 1, torch.Generator().manual_seed(s))["light"]) for s in range(200)]
    assert all(lo - 1e-3 <= e <= hi + 1e-3 for e in els)
    pt = sample_params("train", 10000, g)
    hsv = rgb_to_hsv(pt["table_rgb"][~pt["nominal"]])
    assert all(_in_mode(hsv[i], "train") for i in range(0, len(hsv), 7))
    assert abs(float(pt["side_eye_off"].abs().max()) - 0.01) < 1e-3
    els = [light_elevation_deg(sample_params("train", 1, torch.Generator().manual_seed(s))["light"]) for s in range(200)]
    tlo, thi = MODES["train"]["light_elev"]
    assert all(tlo - 1e-3 <= e <= thi + 1e-3 for e in els)


def test_object_colours_stay_nearest_their_named_colour():
    ref = _lab(np.array([NOMINAL_RGB[c] for c in COLORS], dtype=np.float64))
    named = torch.tensor([NOMINAL_RGB[c] for c in COLORS])
    gen = torch.Generator().manual_seed(2)
    for mode in ("train", "heldout"):
        p = sample_params(mode, 10000, gen)
        for key, truth in (("cube_rgb", [COLORS.index(c) for c, _ in ALL_CUBES]), ("pad_rgb", [0, 1, 2, 3])):
            rgb = p[key].double().numpy()  # (n, K, 3)
            assert rgb.min() >= 0.0 and rgb.max() <= 1.0 + 1e-6
            d = np.linalg.norm(_lab(rgb)[:, :, None, :] - ref[None, None], axis=-1)  # (n, K, 4)
            assert (d.argmin(-1) == np.array(truth)[None]).all(), (mode, key)
            h0 = rgb_to_hsv(named)[torch.tensor(truth), 0]  # hue is fixed at the named colour
            assert torch.allclose(rgb_to_hsv(p[key])[..., 0], h0[None].expand(10000, -1), atol=1e-3)
        assert float((p["cube_rgb"] != p["cube_rgb"][0:1]).float().mean()) > 0.5  # but jitter does happen


# ----------------------------------------------------------------------------- s4r table palette (visual_rand doc)
# Light model (rendering is multiplicative: rendered RGB = colour x shading, per-channel ratios equal within 0.03 in
# results/visual_rand_check.json). Table shading = min(1, 0.21 + sin(light elevation)): fit to the measured table
# means 0.416 / 0.613 / 0.784 / 0.97 at 12 / 24.3 / 35.3 / 50.1 deg (results/frames/s4r_palette/check.json
# "light_12deg", results/visual_rand_check.json otherwise); it predicts the later-measured 0.58 at 22 deg (0.585).
# Object shading = r x table shading, r in R_OBJ = [0.33, 1.55]: the bulk of the measured object / table mask-mean
# ratios (nominal 0.48-1.02, held-out 0.39-1.19, low light 22 deg 0.25-1.22, 12 deg 0.29-1.51). NOT covered: the
# smallest mask means, r 0.14-0.29 (mostly wrist camera). The one inspected (held-out dark, env 4, wrist,
# cube_green_large, measured CIEDE2000 9.2 vs the table) is a sliver of a dark side face at the frame edge; the others
# were not inspected. These are reported, not gated.
# Contrast metric: CIEDE2000 between the rendered object colour and the rendered table colour. Bars:
#   ANY_LIGHT_DE00 = 10 at every light of the mode (CIEDE2000 > 10 = different colours, not shades of one; JND ~ 2.3);
#   NOMINAL_LIGHT_DE00 = 20 at the nominal light for every s4r table: must reject the finding behind the palette, blue
#   cube on the 112 nominal table = 12.2 over the jitter box (16.8 for the exact nominal blue; tested below).
ANY_LIGHT_DE00, NOMINAL_LIGHT_DE00 = 10.0, 20.0
R_OBJ = (0.33, 1.55)
S4R = ("s4r_nominal", "s4r_train", "s4r_heldout")


def _k_table(elev_deg):
    return np.minimum(1.0, 0.21 + np.sin(np.radians(elev_deg)))


def _objects() -> np.ndarray:
    """(4, 9, 3): every named colour at the corners and midpoints of the jitter box (saturation x value scale)."""
    from vla_tabletop.visual_rand import OBJ_SAT_SCALE, OBJ_VAL_SCALE, hsv_to_rgb
    hsv = rgb_to_hsv(torch.tensor([NOMINAL_RGB[c] for c in COLORS]))
    s, v = [torch.tensor(x.ravel()) for x in np.meshgrid(np.linspace(*OBJ_SAT_SCALE, 3), np.linspace(*OBJ_VAL_SCALE, 3))]
    return np.stack([hsv_to_rgb(hsv[i, 0].expand(9), hsv[i, 1] * s, hsv[i, 2] * v).numpy() for i in range(4)]).astype(float)


def _tables(mode: str, sat_cap: float | None = None) -> np.ndarray:
    """(N, 3) dense grid over every table colour the mode can draw: 24 hues x 3 saturations x 9 values per band."""
    from vla_tabletop.visual_rand import S4R_DEFAULT_TABLE, S4R_MODES, hsv_to_rgb
    if mode == "s4r_nominal":
        return np.array([S4R_DEFAULT_TABLE], float)
    cfg = S4R_MODES[mode.removeprefix("s4r_")]
    cap = cfg["sat"][1] if sat_cap is None else sat_cap
    v = np.concatenate([np.linspace(lo, hi, 9) for lo, hi in cfg["values"]])
    h, s, v = np.meshgrid(np.linspace(0, 1, 24, endpoint=False), np.linspace(0, cap, 3), v, indexing="ij")
    return hsv_to_rgb(*(torch.tensor(x.ravel()) for x in (h, s, v))).numpy().astype(float)


def _lit(mode: str):
    """Yield (object shading factors (10,), table shading factor) over the mode's light-elevation range."""
    from vla_tabletop.visual_rand import S4R_MODES
    els = ([light_elevation_deg(DEFAULT_LIGHT)] if mode == "s4r_nominal" else
           np.linspace(*S4R_MODES[mode.removeprefix("s4r_")]["light_elev"], 7))
    for kt in _k_table(np.array(els)):
        yield np.minimum(1.0, kt * np.linspace(*R_OBJ, 10)), float(kt)


def _min_contrast(mode: str, sat_cap: float | None = None) -> np.ndarray:
    """(4,) min CIEDE2000 per named colour between any rendered object and any rendered table of the mode."""
    obj, tab = _objects(), _tables(mode, sat_cap)
    out = np.full(4, np.inf)
    for ko, kt in _lit(mode):
        d = _de00(_lab(obj[:, :, None, None, :] * ko[:, None, None]), _lab(tab * kt)[None])  # (4, 9, 10, N)
        out = np.minimum(out, d.reshape(4, -1).min(1))
    return out


def _nominal_light_contrast(tables: np.ndarray) -> np.ndarray:
    k = float(_k_table(light_elevation_deg(DEFAULT_LIGHT)))
    return _de00(_lab(_objects()[:, :, None, :] * k), _lab(tables * k)[None, None]).reshape(4, -1).min(1)


def test_de00_matches_sharma_reference_pairs():
    pairs = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425), ((50, 0, 0), (50, -1, 2), 2.3669),
             ((50, 2.5, 0), (73, 25, -18), 27.1492), ((50, 2.49, -0.001), (50, -2.49, 0.0009), 7.1792),
             ((22.7233, 20.0904, -46.694), (23.0331, 14.973, -42.5619), 2.0373),
             ((2.0776, 0.0795, -1.135), (0.9033, -0.0636, -0.5514), 0.9082)]
    for a, b, want in pairs:
        assert abs(float(_de00(np.array(a, float), np.array(b, float))) - want) < 1e-4


def test_existing_modes_bit_identical():
    """(d) nominal / train / heldout output pinned: digest computed with the randomizer at 52ac1e4 (before s4r)."""
    import hashlib
    h = hashlib.sha256()
    for mode in ("nominal", "train", "heldout"):
        for seed, n in ((0, 1), (1, 1), (7, 64), (21, 1000)):
            p = sample_params(mode, n, torch.Generator().manual_seed(seed))
            for k in sorted(p):
                h.update(k.encode())
                h.update(p[k].numpy().tobytes() if torch.is_tensor(p[k]) else str(p[k]).encode())
    assert h.hexdigest() == "5ddf43104daf9ceb0af29495b227156a733da0dce064c16603ef05decaf939ab"


def test_s4r_differs_from_its_112_twin_only_in_table_and_light():
    from vla_tabletop.visual_rand import S4R_DEFAULT_TABLE, run_mode
    for mode in ("nominal", "train", "heldout"):
        a = sample_params(mode, 500, torch.Generator().manual_seed(3))
        b = sample_params(f"s4r_{mode}", 500, torch.Generator().manual_seed(3))
        assert b["mode"] == f"s4r_{mode}"
        for k in a:
            if k not in ("mode", "table_rgb", "light"):
                assert torch.equal(a[k], b[k]), (mode, k)
        k = int(b["nominal"].sum())
        assert torch.equal(b["table_rgb"][b["nominal"]], torch.tensor(S4R_DEFAULT_TABLE).expand(k, 3))
    assert [run_mode(v, "112") for v in ("nominal", "train_rand", "heldout")] == [None, "train", "heldout"]
    assert [run_mode(v, "s4r") for v in ("nominal", "train_rand", "heldout")] == list(S4R)


def test_s4r_train_and_heldout_disjoint():
    """(c) value bands disjoint with 0.05 margin, light elevations with 5 deg; nominal inside train; samples obey."""
    from vla_tabletop.visual_rand import S4R_DEFAULT_TABLE, S4R_MODES, S4R_SAT_CAP
    tr, ho = S4R_MODES["train"], S4R_MODES["heldout"]
    for lo, hi in ho["values"]:
        for tlo, thi in tr["values"]:
            assert hi + 0.05 <= tlo or thi + 0.05 <= lo, ((lo, hi), (tlo, thi))
    assert ho["light_elev"][1] + 5.0 <= tr["light_elev"][0]
    nom = rgb_to_hsv(torch.tensor(S4R_DEFAULT_TABLE))
    assert float(nom[1]) == 0.0 and any(lo <= float(nom[2]) <= hi for lo, hi in tr["values"])
    assert tr["light_elev"][0] <= light_elevation_deg(DEFAULT_LIGHT) <= tr["light_elev"][1]
    g = torch.Generator().manual_seed(4)
    for mode, cfg, other in (("s4r_train", tr, ho), ("s4r_heldout", ho, tr)):
        p = sample_params(mode, 10000, g)
        hsv = rgb_to_hsv(p["table_rgb"][~p["nominal"]])
        assert float(hsv[:, 1].max()) <= S4R_SAT_CAP + 1e-5
        val = hsv[:, 2]
        inb = lambda bands: torch.stack([(val >= lo - 1e-5) & (val <= hi + 1e-5) for lo, hi in bands]).any(0)  # noqa: E731
        assert inb(cfg["values"]).all() and not inb(other["values"]).any(), mode
        for lo, hi in cfg["values"]:  # every band is drawn
            assert ((val >= lo) & (val <= hi)).sum() > 1000, (mode, lo, hi)
        lo, hi = cfg["light_elev"]
        els = [light_elevation_deg(sample_params(mode, 1, torch.Generator().manual_seed(s))["light"]) for s in range(200)]
        assert all(lo - 1e-3 <= e <= hi + 1e-3 for e in els), mode


def test_s4r_objects_stay_nearest_their_own_name():
    """(a) Every jittered cube / pad colour, rendered at every light of the mode, is nearest (CIEDE2000) its own named
    colour rendered under the same light, among the four names and the rendered table, for every table colour."""
    obj = _objects()
    named = np.array([NOMINAL_RGB[c] for c in COLORS])
    for mode in S4R:
        tab = _tables(mode)
        for ko, kt in _lit(mode):
            o = _lab(obj[:, :, None, :] * ko[:, None])  # (4, 9, 10, 3)
            ref = _lab(named[None, :, :] * ko[:, None, None])  # (10, 4, 3)
            d_names = _de00(o[:, :, :, None, :], ref[None, None])  # (4, 9, 10, 4)
            assert (d_names.argmin(-1) == np.arange(4)[:, None, None]).all(), (mode, kt)
            own = np.take_along_axis(d_names, np.arange(4)[:, None, None, None], -1)[..., 0]  # (4, 9, 10)
            d_tab = _de00(o[:, :, :, None, :], _lab(tab * kt)[None, None, None]).min(-1)
            assert (own < d_tab).all(), (mode, kt, float((own - d_tab).max()))


def test_s4r_min_contrast_against_every_table():
    """(b) CIEDE2000 >= ANY_LIGHT_DE00 at every light of every s4r mode; >= NOMINAL_LIGHT_DE00 at the nominal light."""
    for mode in S4R:
        m = _min_contrast(mode)
        assert (m >= ANY_LIGHT_DE00).all(), (mode, dict(zip(COLORS, m.round(1))))
        n = _nominal_light_contrast(_tables(mode))
        assert (n >= NOMINAL_LIGHT_DE00).all(), (mode, dict(zip(COLORS, n.round(1))))


def test_contrast_bar_rejects_the_112_nominal_table_and_a_tinted_s4r_table():
    """The bars measure the design: blue on the 112 nominal table fails the nominal-light bar, and widening the s4r
    saturation cap so a table can take a cube's hue fails the any-light bar."""
    n = _nominal_light_contrast(np.array([DEFAULT_TABLE], float))
    assert n[COLORS.index("blue")] < NOMINAL_LIGHT_DE00
    assert (_min_contrast("s4r_train", sat_cap=0.9) < ANY_LIGHT_DE00).all()
