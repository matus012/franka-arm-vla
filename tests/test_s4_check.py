"""scripts/s4_check.py pure item functions (GATES Stage 4 re-plan 0C items 1, 5, 6) on synthetic data; each check has a
mutant it must flag. Policy venv (imports lerobot via s0c_check)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s4_check as S  # noqa: E402
from vla_tabletop import visual_rand as VR  # noqa: E402


def vrow(params: dict, j: int) -> dict:  # scripts/collect.py visual_row
    return {"mode": params["mode"], "light": params["light"].tolist(),
            **{k: params[k][j].tolist() for k in ("nominal", "table_rgb", "cube_rgb", "pad_rgb")}}


def s4r_side(n: int = 90, seed: int = 0) -> list[dict]:
    """Sidecar-like rows of an s4r_train set; light per 'batch' of 30 (global per batch, as the collector)."""
    side = []
    for b in range(0, n, 30):
        p = VR.sample_params("s4r_train", 30, torch.Generator().manual_seed(seed + b))
        for j in range(30):
            k = b + j
            side.append({"episode_index": k, "task": ("pick", "place", "stack")[k % 3], "visual_mode": "train_rand",
                         "visual_json": json.dumps(vrow(p, j)), "table_palette": "s4r", "seed": 32100})
    return side


def grey_side(n: int = 20) -> list[dict]:
    p = VR.sample_params("s4r_nominal", n, torch.Generator().manual_seed(0))
    return [{"episode_index": k, "task": "pick", "visual_mode": "nominal", "visual_json": json.dumps(vrow(p, k)),
             "table_palette": "s4r", "spec_json": f"s{k}", "instruction": f"i{k}"} for k in range(n)]


# ----------------------------------------------------------------------------- expert floors per task
def es(rates: dict[str, list[tuple[int, int]]]) -> dict:
    nb = len(next(iter(rates.values())))
    return {"batches": [{"attempted": sum(rates[t][b][1] for t in rates), "per_task": {
        t: {"kept": rates[t][b][0], "attempted": rates[t][b][1]} for t in rates}} for b in range(nb)]}


def test_expert_items_floors_and_mutants():
    ok = es({"pick": [(384, 384)] * 2, "place": [(380, 384)] * 2, "stack": [(375, 384)] * 2})  # 100 / 99 / 97.7
    items, nums = S.expert_items(ok, ["pick", "place", "stack"])
    assert all(items.values()) and nums["stack"]["ceiling_pct"] == 97.3
    bad_batch = es({"pick": [(384, 384), (340, 384)], "place": [(384, 384)] * 2, "stack": [(375, 384)] * 2})  # 88.5 < 90
    assert not S.expert_items(bad_batch, ["pick", "place", "stack"])[0]["1_expert_every_batch_pick_ge_ceiling_minus_10"]
    bad_pool = es({"pick": [(384, 384)] * 2, "place": [(384, 384)] * 2, "stack": [(345, 384)] * 2})  # 89.8: < 92.3, >= 87.3
    it = S.expert_items(bad_pool, ["pick", "place", "stack"])[0]
    assert not it["1_expert_pooled_stack_ge_ceiling_minus_5"] and it["1_expert_every_batch_stack_ge_ceiling_minus_10"]
    small = es({"pick": [(100, 100)] * 2})
    assert not S.expert_items(small, ["pick"])[0]["1_expert_every_batch_n_ge_200"]


# ----------------------------------------------------------------------------- mix share
def test_mix_items():
    side = [{"task": ("pick", "place", "stack")[k % 3]} for k in range(2000)]
    assert S.mix_items(side)[0]["1_mix_share_each_task_in_band"]
    skew = [{"task": "pick" if k < 800 else ("place", "stack")[k % 2]} for k in range(2000)]  # pick 0.40
    assert not S.mix_items(skew)[0]["1_mix_share_each_task_in_band"]
    none = [{"task": ("pick", "place")[k % 2]} for k in range(2000)]  # stack missing
    assert not S.mix_items(none)[0]["1_mix_share_each_task_in_band"]


# ----------------------------------------------------------------------------- palette attributes
def test_palette_grey_and_mutants():
    side = grey_side()
    assert all(S.palette_items(side, "grey", "s4r")[0].values())
    for name, fn in {"palette_table_palette_eq_expected_all": lambda s: s.update(table_palette="112"),
                     "palette_visual_json_s4r_nominal_all": lambda s: s.update(visual_json=json.dumps({"mode": "nominal"})),
                     "palette_visual_mode_nominal_all": lambda s: s.update(visual_mode="train_rand")}.items():
        m = copy.deepcopy(side)
        fn(m[3])
        assert not S.palette_items(m, "grey", "s4r")[0][name], name


def test_palette_s4r_and_mutants():
    side = s4r_side()
    items, nums = S.palette_items(side, "s4r", "s4r")
    assert all(items.values()), items
    assert nums["nonnominal_table"] > 20 and nums["nondefault_light"] > 20
    held = copy.deepcopy(side)  # one episode with held-out band parameters (dark grey, low light)
    vj = json.loads(held[5]["visual_json"]) | {"table_rgb": [0.12, 0.12, 0.12], "light": [-0.92, 0.0, -0.39]}
    held[5]["visual_json"] = json.dumps(vj)
    assert not S.palette_items(held, "s4r", "s4r")[0]["palette_params_inside_train_outside_heldout_all"]
    old = copy.deepcopy(side)  # one episode with the 112 table (saturation 0.6) under the s4r mode
    old[7]["visual_json"] = json.dumps(json.loads(old[7]["visual_json"]) | {"table_rgb": list(VR.DEFAULT_TABLE)})
    assert not S.palette_items(old, "s4r", "s4r")[0]["palette_params_inside_train_outside_heldout_all"]
    old112 = copy.deepcopy(side)  # a 112-mode record in an s4r set
    old112[7]["visual_json"] = json.dumps(json.loads(old112[7]["visual_json"]) | {"mode": "train"})
    assert not S.palette_items(old112, "s4r", "s4r")[0]["palette_visual_json_s4r_train_all"]


# ----------------------------------------------------------------------------- twin consistency
def test_twin_items():
    a = grey_side(20)
    twin = [dict(s) for s in a[3:]] + [{"episode_index": 99, "spec_json": "x", "instruction": "y"}]
    items, nums = S.twin_items(a, twin)
    assert all(items.values()) and nums["kept_shared"] == 17 and nums["only_here"] == 3 and nums["only_twin"] == 1
    bad = copy.deepcopy(twin)
    bad[0]["instruction"] = "other"
    assert not S.twin_items(a, bad)[0]["1_twin_shared_specs_instructions_equal"]
    assert not S.twin_items(a, [{"episode_index": 500, "spec_json": "x", "instruction": "y"}])[0]["1_twin_overlap_nonempty"]


# ----------------------------------------------------------------------------- sheet selection
def test_pick_sheet_episodes():
    side = s4r_side()
    ch = S.pick_sheet_episodes(side, "s4r")
    assert S.pick_sheet_episodes(side, "s4r") == ch and len(set(ch)) == 6
    assert all(S.sheet_items(side, ch, "s4r").values())
    assert [side[i]["task"] for i in ch] == ["pick", "pick", "place", "place", "stack", "stack"]
    assert S.pick_sheet_episodes(grey_side(2000), "grey") == [0, 1000, 1999]
    assert "table rgb" in S.note(side[ch[0]]) and "light elev" in S.note(side[ch[0]])
    nom = [dict(s, visual_json=json.dumps({"mode": "s4r_train", "table_rgb": [0.5] * 3, "light": list(S.DEFAULT_LIGHT)}))
           for s in side]
    it = S.sheet_items(nom, S.pick_sheet_episodes(nom, "s4r"), "s4r")
    assert not it["5_sheet_has_nonnominal_table"] and not it["5_sheet_has_nondefault_light"]


# ----------------------------------------------------------------------------- item 6
def flat(rgb, h=224) -> np.ndarray:
    return np.full((h, h, 3), np.round(np.asarray(rgb) * 255), np.uint8)


def test_region_stats_and_saturation_mutant():
    s, v = S.region_stats(flat([0.5, 0.5, 0.5]))
    assert s == 0.0 and abs(v - 0.5) < 0.01
    s, v = S.region_stats(flat([0.5, 0.5, 0.52]))  # the rendered nominal table is ~0.03 saturated
    assert s < 0.05
    s112, _ = S.region_stats(flat(VR.DEFAULT_TABLE))
    assert s112 > 0.5
    grey = [S.region_stats(flat([x] * 3))[0] for x in np.linspace(0.35, 0.75, 50)]
    assert S.table_items(grey, [0.5] * 50, [0.5] * 50, "grey")[0]["6_table_region_median_saturation_le_0.10"]
    tint = [s112] * 50  # the 112-table twin checked as a grey set
    assert not S.table_items(tint, [0.6] * 50, [0.5] * 50, "grey")[0]["6_table_region_median_saturation_le_0.10"]
    one_bad = grey[:-1] + [s112]
    assert not S.table_items(one_bad, [0.5] * 50, [0.5] * 50, "grey")[0]["6_table_region_median_saturation_le_0.10"]


def test_spearman_and_value_mutants():
    rec = list(np.linspace(0.35, 0.75, 50))
    rng = np.random.default_rng(0)
    val = [r * 0.9 + rng.normal(0, 0.02) for r in rec]  # shading + noise
    items, nums = S.table_items([0.0] * 50, val, rec, "s4r")
    assert all(items.values()) and nums["spearman_value_vs_recorded_table_value"] > 0.9
    assert not S.table_items([0.0] * 50, [0.5] * 50, rec, "s4r")[0]["6_table_value_spearman_ge_0.5"]  # table not reaching pixels
    assert not S.table_items([0.0] * 50, list(rng.permutation(val)), rec, "s4r")[0]["6_table_value_spearman_ge_0.5"]
    assert abs(S.spearman([1, 2, 3, 4], [1, 2, 2, 4]) - 0.9486832980505138) < 1e-9
    assert "6_table_value_spearman_ge_0.5" not in S.table_items([0.0], [0.5], [0.5], "grey")[0]


def project(p, H=224):
    """Side camera pinhole (vision_lift/cameras.py: eye / look, focal 20 mm, aperture 20.955 mm)."""
    eye, look = np.array([1.15, 0.45, 0.55]), np.array([0.5, 0.0, 0.18])
    fwd = (look - eye) / np.linalg.norm(look - eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = np.asarray(p) - eye
    z = d @ fwd
    return H / 2 + (d @ right / z) * 20.0 / 20.955 * H, H / 2 - (d @ up / z) * 20.0 / 20.955 * H


def object_mask(margin: int = 12) -> np.ndarray:
    from scipy import ndimage

    from vla_tabletop import layout as L

    m = np.zeros((224, 224), bool)
    for x in np.linspace(L.REGION_X[0] - 0.05, L.REGION_X[1] + 0.05, 45):
        for y in np.linspace(L.REGION_Y[0] - 0.05, L.REGION_Y[1] + 0.05, 71):
            for z in (L.TABLE_Z, 0.03, 0.07):
                u, v = project((x, y, z))
                if 0 <= u < 224 and 0 <= v < 224:
                    m[int(v), int(u)] = True
    return ndimage.binary_dilation(ndimage.binary_closing(ndimage.binary_dilation(m, iterations=3), iterations=6),
                                   iterations=margin)


def test_table_box_free_of_objects_in_any_layout():
    """TABLE_BOX stays clear of every cube / pad point (layout.py region +- half diagonal, up to 0.07 m high) projected
    through the side camera, with a 12 px margin for the s4r camera jitter. Mutant: a box in the object band is hit."""
    m = object_mask()
    u0, v0, u1, v1 = S.TABLE_BOX
    assert not m[v0:v1, u0:u1].any()
    assert m[130:170, 90:140].any()


class FakeDS:
    """Stand-in for LeRobotDataset: episode e starts at row 10 e; row 10 e is its t = 0 side frame (CHW uint8)."""

    def __init__(self, imgs: list[np.ndarray]):
        self.imgs = imgs
        self.meta = type("M", (), {"episodes": [{"dataset_from_index": 10 * e} for e in range(len(imgs))]})()

    def __getitem__(self, i: int) -> dict:
        return {"observation.images.side": torch.from_numpy(self.imgs[i // 10]).permute(2, 0, 1)}


def test_table_check_end_to_end_with_fake_dataset():
    """table_check on rendered-like frames (object band bright coloured blocks outside TABLE_BOX, table grey with value v):
    s4r passes with the table reaching the pixels; the 112 tint and a frame whose table ignores the recorded value fail."""
    rng = np.random.default_rng(0)
    side = s4r_side(120)
    rec = [S.table_value(s) for s in side]

    def frame(v: float, tint=(0.0, 0.0, 0.0)) -> np.ndarray:
        g = np.clip(np.array([v, v, v]) * 0.8 + np.array(tint), 0, 1)
        im = np.tile(np.round(g * 255).astype(np.uint8), (224, 224, 1))
        im[100:160, 80:160] = (200, 30, 30)  # a cube band; never inside TABLE_BOX
        return im

    good = FakeDS([frame(r) for r in rec])
    tc = S.table_check(good, side, "s4r")
    assert all(tc["items"].values()) and len(tc["episodes"]) == 50 and tc["numbers"]["spearman_value_vs_recorded_table_value"] > 0.95
    grey = S.table_check(FakeDS([frame(0.5)] * 120), grey_side(120), "grey")
    assert all(grey["items"].values())
    twin = S.table_check(FakeDS([np.tile(np.round(np.array(VR.DEFAULT_TABLE) * 255).astype(np.uint8), (224, 224, 1))] * 120),
                         grey_side(120), "grey")
    assert not twin["items"]["6_table_region_median_saturation_le_0.10"] and twin["numbers"]["saturation_max"] > 0.5
    flat_ = S.table_check(FakeDS([frame(0.5)] * 120), side, "s4r")  # palette not reaching the pixels
    assert not flat_["items"]["6_table_value_spearman_ge_0.5"]
    assert tc["box_u0_v0_u1_v1"] == list(S.TABLE_BOX) and "projected" in tc["reason"] or "side camera" in tc["reason"]
    _ = rng


def build_fake_set(tmp: Path, K: int = 60, palette="s4r") -> tuple[Path, list[dict]]:
    """A dataset dir with the meta files check() reads: sidecar rows regenerated from (seed, index, s2 sampler), an HDF5 source
    with the kept indices; the heavy readers (video, parquet columns, sheet, expert success) are monkeypatched by the caller."""
    import argparse  # noqa: F401

    import h5py

    from vla_common.catalog import cube_name
    from vla_common.tasks import labels, sample_spec
    from vla_tabletop.layout import sample_layout
    import math

    root, src = tmp / "0c_fake_pick_rl224_y15g", tmp / "src"
    (root / "meta").mkdir(parents=True)
    src.mkdir()
    side = []
    with h5py.File(src / "p.h5", "w") as f:
        for k in range(K):
            i = 3 * k
            rng = np.random.default_rng([32500, i])
            spec = sample_spec("s2", "train", rng)
            lay = sample_layout(spec, rng, cube_yaw_max_deg=15.0)[0]
            yaw = math.degrees(lay[cube_name(tuple(spec.target))][2])
            f.create_group(f"ep_{k}").attrs["episode_index"] = i
            side.append({"episode_index": i, "lerobot_episode_index": k, "seed": 32500, "task": "pick", "T": 350,
                         "split": "val" if k % 20 == 19 else "train", "spec_json": spec.to_json(), "instruction": "x", "labels": json.dumps(labels(spec).tolist()),
                         "image_size": 224, "cube_yaw_max_deg": 15.0, "spawn_yaw_deg": yaw, "spawn_yaw_fold_deg": float(S.fold(yaw)) if hasattr(S, "fold") else yaw,
                         "visual_mode": "nominal", "visual_json": json.dumps({"mode": "s4r_nominal", "table_rgb": [0.5] * 3, "light": list(S.DEFAULT_LIGHT)}),
                         "table_palette": palette, "file_attrs": {"stage": "s2", "seed": 32500, "mix": "1,1,1"}, "source": str(src / "p.h5"),
                         "group": f"ep_{k}"})
    (root / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))
    (root / "meta" / "export_113.json").write_text(json.dumps({"episodes": K, "rows": 350 * K, "sources": [str(src)]}))
    (root / "meta" / "split_113.json").write_text(json.dumps({"kept_per_stream": K, "val": [k for k in range(K) if k % 20 == 19],
                                                                 "train": [k for k in range(K) if k % 20 != 19]}))
    return root, side


def test_check_driver_with_patched_readers(tmp_path, monkeypatch):
    import argparse

    import s0c_rl_check as RL

    root, side = build_fake_set(tmp_path)
    for s in side:  # yaw fold as the collector stores it
        s["spawn_yaw_fold_deg"] = float(RL.fold(s["spawn_yaw_deg"]))
    (root / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))
    rate = {"kept": 380, "attempted": 384, "rate_pct": 99.0, "ci95_pct": [97, 99.5]}
    es = {"batches": [{"attempted": 384, "per_task": {"pick": {"kept": 384, "attempted": 384}}}] * 2, "pooled": rate}
    ds = FakeDS([np.tile(np.array([128, 128, 130], np.uint8), (224, 224, 1))] * len(side))
    ds.meta.features = {f"observation.images.{c}": {"shape": [224, 224, 3]} for c in ("side", "wrist")}
    cn = {"rows_per_episode_mismatch": [], "column_mismatch_episodes": [], "noise_std_per_arm_dim": [0.1] * 4,
          "claw_exec_ne_label_rows": 0}
    monkeypatch.setattr(S.C, "open_ds", lambda r: ds)
    monkeypatch.setattr(S.C, "columns_and_noise", lambda *a, **k: cn)
    monkeypatch.setattr(S.C, "video_check", lambda *a: {"psnr_min": 40.0, "psnr_mean": 44.0, "identity_fail": 0, "episodes": [0]})
    monkeypatch.setattr(S.C, "expert_success", lambda srcs: es)
    monkeypatch.setattr(S.C, "sheet", lambda *a, **k: [])

    def run(**kw):
        a = argparse.Namespace(dataset=root, kind="grey", out=tmp_path / "o", twin=None, rerender=224, expect_yaw=15.0, kept=60,
                               expect_palette="s4r", only=None)
        for k, v in kw.items():
            setattr(a, k, v)
        return S.check(a)

    monkeypatch.setattr(S, "timing_T", lambda: {"pick": 350, "place": 550, "stack": 550})
    res = run()
    assert res["pass"], [k for k, v in res["items"].items() if not v]
    assert (tmp_path / "o" / f"0s4_check_{root.name}.json").exists()
    # mutants: 112-palette attribute, a wrong T, an off-stream kept list, a wrong yaw range expectation, 112 image size
    assert not run(expect_palette="112")["items"]["palette_table_palette_eq_expected_all"]
    assert not run(expect_yaw=0.0)["pass"]
    assert not run(kept=61)["items"]["1_count_eq_K"]
    assert not run(rerender=128)["items"]["1_image_size"]
    monkeypatch.setattr(S, "timing_T", lambda: {"pick": 349, "place": 550, "stack": 550})
    assert not run()["items"]["1_T_per_task_eq_timing"]
    monkeypatch.setattr(S, "timing_T", lambda: {"pick": 350, "place": 550, "stack": 550})
    only = run(only="table")
    assert only["pass"] and (tmp_path / "o" / f"0s4_table_{root.name}.json").exists()
    ds.imgs = [np.tile(np.round(np.array(VR.DEFAULT_TABLE) * 255).astype(np.uint8), (224, 224, 1))] * len(side)
    assert not run(only="table")["pass"] and not run()["items"]["6_table_region_median_saturation_le_0.10"]
    # sidecar row order is shard order (task-grouped for s4r), not episode-index order: the order must not matter
    path = root / "meta" / "episodes_113.jsonl"
    orig = list(side)

    def write(rows):
        path.write_text("".join(json.dumps(s) + "\n" for s in rows))

    write(orig[1::2] + orig[0::2])  # shuffled, same kept set and split labels
    items = run()["items"]
    assert items["1_kept_are_first_K_successful_in_order"] and items["1_split_rule"]
    wrong = copy.deepcopy(orig)  # last kept index replaced by a later success
    wrong[-1]["episode_index"] = 3 * 60 + 3
    write(wrong[1::2] + wrong[0::2])
    assert not run()["items"]["1_kept_are_first_K_successful_in_order"]
    moved = copy.deepcopy(orig)  # val label moved from rank 19 to rank 18
    moved[19]["split"], moved[18]["split"] = "train", "val"
    write(moved[1::2] + moved[0::2])
    assert not run()["items"]["1_split_rule"]
    write(orig)
