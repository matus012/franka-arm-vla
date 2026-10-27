"""scripts/paired_check.py on synthetic sidecars + frames (CPU): a clean set passes; mutants m3 (group split across
train / val), m4 (re-sampled layout = different t=0 frame), m5 (duplicate targets) are flagged. Also the selection rule."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paired_check as PC  # noqa: E402
from paired_check import _labels  # noqa: E402
from vla_common.tasks import labels  # noqa: E402
from vla_tabletop.rollout import make_paired_stream  # noqa: E402

SEED, M = 32800, 15.0


def synth(n_groups: int = 60, slots: str = "random"):
    """Sidecar rows of the real paired stream + identical robot t=0 states + env slots (independent of the target)."""
    eps, _ = make_paired_stream(SEED, 0, n_groups, "s2a", "train", M)
    ranks = {g: k for k, g in enumerate(sorted({e.layout_id for e in eps}))}
    rng = np.random.default_rng(3)
    side, state0, frames = [], {}, {}
    for k, e in enumerate(eps):
        lab = labels(e.spec)
        slot = int(rng.integers(0, 128)) if slots == "random" else int(lab[1]) * 32 + int(rng.integers(0, 32))
        side.append({"layout_id": e.layout_id, "member": e.member, "lerobot_episode_index": k, "paired": True,
                     "labels": json.dumps(lab.tolist()), "instruction": e.instruction, "slot": slot, "env_id": slot,
                     "split": "val" if ranks[e.layout_id] % 20 == 19 else "train", "spec_json": e.spec.to_json(),
                     "seed": SEED, "episode_index": e.index, "cube_yaw_max_deg": M, "file_attrs": {"stage": "s2a"}})
        state0[k] = np.arange(19, dtype=np.float32) * 0.01  # bit-identical robot start
        img = np.random.default_rng([7, e.layout_id]).integers(0, 255, (8, 8, 3), dtype=np.uint8)
        frames[k] = {"side": img.copy(), "wrist": (255 - img).copy()}
    return side, state0, frames


def run(side, state0, frames=None, layouts=None):
    items, nums = PC.p_items(side, state0, layouts=layouts, t0=frames)
    items["P_regenerates"] = not PC.regenerates(side)
    return items, nums


def test_clean_set_passes_and_reports_numbers():
    side, state0, frames = synth()
    items, nums = run(side, state0, frames)
    assert all(items.values()), items
    assert nums["P2a"]["zero_variance_dims"] == list(range(19)) and nums["P2a"]["min_p"] is None
    assert nums["P2b"]["accuracy"] == pytest.approx(nums["P2b"]["majority_rate"])  # no usable feature: majority
    assert nums["P2c"]["chi2_p"] >= 0.01 and set(nums["P2c"]["member_x_slot_octile"]) == {"0", "1", "2"}
    assert nums["reported_t0_frames"]["max_abs_diff"] == {"side": 0, "wrist": 0}
    assert nums["groups"] == len({s["layout_id"] for s in side}) and nums["episodes"] == len(side)


def test_nonzero_target_independent_state_noise_passes():
    side, state0, _ = synth()
    rng = np.random.default_rng(5)
    state0 = {k: v + rng.normal(0, 1e-3, 19).astype(np.float32) for k, v in state0.items()}  # random, not target related
    items, nums = run(side, state0)
    assert items["P2a_state_delta_vs_target"] and items["P2b_state_classifier"], nums["P2a"]["min_p"]
    assert nums["P2a"]["nonzero_variance_dims"] == list(range(19))


def test_frame_differences_are_reported_not_gated():
    side, state0, frames = synth()
    for k in range(0, len(side), 3):  # slot noise: a few pixels differ in one member
        frames[k]["wrist"] = frames[k]["wrist"].copy()
        frames[k]["wrist"][0, :2] ^= 17
    items, nums = run(side, state0, frames)
    rf = nums["reported_t0_frames"]
    assert all(items.values()) and rf["groups_all_cams_sha_equal"] < rf["groups"] and rf["max_abs_diff"]["wrist"] > 0
    assert rf["differing_px_per_group"]["wrist"]["max"] == 2 and rf["differing_px_per_group"]["side"]["max"] == 0


def test_m3_group_split_across_train_val():
    side, state0, _ = synth()
    side[1]["split"] = "val" if side[1]["split"] == "train" else "train"
    items, nums = run(side, state0)
    assert not items["P_iv_group_in_one_split"] and nums["bad_groups"]["split"] == [side[1]["layout_id"]]
    assert items["P_i_distinct_targets"] and items["P1_layout_identical"]


def test_m4_layout_resampled_per_member_fails_P1():
    from vla_common.tasks import TaskSpec
    from vla_tabletop.layout import sample_layout

    side, state0, _ = synth()
    lay = PC.regenerated_layouts(side)
    assert PC.p1_bad_groups(PC.groups_of(side), lay) == {"layout": [], "spec": []}
    g = side[0]["layout_id"]
    spec = TaskSpec.from_json(side[0]["spec_json"])
    bad = dict(lay)
    bad[(g, 1)] = sample_layout(spec, np.random.default_rng([SEED, g, 1, 99]), cube_yaw_max_deg=M)[0]  # re-sampled for member 1
    items, nums = run(side, state0, layouts=bad)
    assert not items["P1_layout_identical"] and nums["bad_groups"]["p1_layout"] == [g]


def test_layout_perturbed_by_1e4_fails_P1_and_1e7_passes():
    side, state0, _ = synth()
    lay = PC.regenerated_layouts(side)
    g = side[0]["layout_id"]
    name = next(iter(lay[(g, 1)]))
    x, y, yaw = lay[(g, 1)][name]
    for eps_, ok in ((1e-4, False), (1e-7, True)):
        bad = dict(lay)
        bad[(g, 1)] = bad[(g, 1)] | {name: (x + eps_, y, yaw)}
        assert run(side, state0, layouts=bad)[0]["P1_layout_identical"] == ok
    bad = dict(lay)
    bad[(g, 1)] = bad[(g, 1)] | {name: (x, y, yaw + 1e-4)}  # yaw in rad
    assert not run(side, state0, layouts=bad)[0]["P1_layout_identical"]


def test_spec_differing_beyond_target_fails_P1():
    side, state0, _ = synth()
    g = next(s["layout_id"] for s in side if s["member"] == 2)
    k = next(i for i, s in enumerate(side) if s["layout_id"] == g and s["member"] == 2)
    sp = json.loads(side[k]["spec_json"])
    sp["cubes"] = sp["cubes"][:-1]
    side[k]["spec_json"] = json.dumps(sp)
    items, _ = run(side, state0)
    assert not items["P1_layout_identical"] and not items["P_same_scene_spec"]


def test_m6_target_dependent_state_offset_fails_P2a_and_P2b():
    side, state0, _ = synth()
    for k, s in enumerate(side):
        state0[k] = state0[k] + np.float32(0.02) * np.float32(_labels(s)[1])  # robot start shifts with the target colour
    items, nums = run(side, state0)
    assert not items["P2a_state_delta_vs_target"] and not items["P2b_state_classifier"], nums["P2b"]
    assert nums["P2b"]["accuracy"] > nums["P2b"]["majority_rate"] + 3 * nums["P2b"]["se"]
    assert items["P1_layout_identical"] and items["P2c_slot_independent"]


def test_m7_slot_assignment_sorted_by_target_fails_P2c():
    side, state0, _ = synth(slots="sorted")
    items, nums = run(side, state0)
    assert not items["P2c_slot_independent"] and nums["P2c"]["chi2_p"] < 0.01
    assert items["P2a_state_delta_vs_target"] and items["P2b_state_classifier"]


def test_m5_duplicate_targets():
    side, state0, _ = synth()
    a = next(i for i, s in enumerate(side) if s["member"] == 0)
    for key in ("labels", "instruction", "spec_json"):  # member 1 is a copy of member 0 (same target)
        side[a + 1][key] = side[a][key]
    items, nums = run(side, state0)
    assert not items["P_i_distinct_targets"] and not items["P_iii_labels_instr_differ"]
    assert nums["bad_groups"]["distinct"] == [side[a]["layout_id"]]
    assert not items["P_regenerates"]


def test_single_member_group_flagged():
    side, state0, _ = synth()
    two = [g for g, ms in PC.groups_of(side).items() if len(ms) == 2][0]
    side2 = [s for s in side if not (s["layout_id"] == two and s["member"] == 1)]
    assert not PC.p_items(side2, state0)[0]["P_i_distinct_targets"]


# ---- selection rule = the export's own function (remainder-1 skip included)
def _sized(sizes, n=0):
    exp, ok = {}, set()
    for g, sz in enumerate(sizes):
        exp[g] = list(range(n, n + sz))
        ok |= set(exp[g])
        n += sz
    return exp, ok


def test_expected_selection_is_export_select_paired_including_remainder_one():
    exp, ok = _sized([3, 2, 2, 3])
    assert PC.expected_selection(exp, ok, 6, SEED) == [(0, "train"), (3, "train")]  # the 2s would leave remainder 1
    exp, ok = _sized([3, 4, 2, 2])
    assert [g for g, _ in PC.expected_selection(exp, ok, 5, SEED)] == [0, 2]
    exp, ok = _sized([2, 2, 2, 2])  # a failed member drops its group
    ok.discard(exp[1][0])
    assert [g for g, _ in PC.expected_selection(exp, ok, 6, SEED)] == [0, 2, 3]


def test_expected_selection_on_a_real_stream_matches_export():
    import export_dataset as X

    eps, _ = make_paired_stream(SEED, 0, 90, "s2a", "train", M)
    exp = {}
    for e in eps:
        exp.setdefault(e.layout_id, []).append(e.index)
    fail = {exp[3][0], exp[10][1]}
    rows = [{"file": "f", "group": f"ep_{e.index}", "stage": "s2a", "seed": SEED, "episode_index": e.index, "paired": True,
             "layout_id": e.layout_id} for e in eps if e.index not in fail]
    ok = {r["episode_index"] for r in rows}
    for kept in (60, 101, 150, 151):
        ch, _ = X.select_paired(rows, {(SEED, g): v for g, v in exp.items()}, kept)
        want = PC.expected_selection(exp, ok, kept, SEED)
        got = {}
        for r in rows:
            if ("f", r["group"]) in ch:
                got.setdefault(r["layout_id"], set()).add(ch[("f", r["group"])])
        assert [g for g, _ in want] == sorted(got) and all(got[g] == {sp} for g, sp in want) and 3 not in got
        assert sum(len(exp[g]) for g, _ in want) == kept


def test_selection_items_flag_wrong_split_and_wrong_groups():
    side, _, _ = synth(60)
    exp = {}
    for s in side:
        exp.setdefault(s["layout_id"], []).append(s["episode_index"])
    ok = {s["episode_index"] for s in side}
    kept = len(side)
    items, _ = PC.selection_items(side, exp, ok, kept)
    assert all(items.values()), items
    bad = copy.deepcopy(side)
    for s in bad:
        if s["layout_id"] == side[-1]["layout_id"]:
            s["split"] = "val"  # rank of the last group is not 19 mod 20
    assert not PC.selection_items(bad, exp, ok, kept)[0]["P_split_rule"]
    dropped = [s for s in side if s["layout_id"] != side[0]["layout_id"]]  # first group missing from the export
    assert not PC.selection_items(dropped, exp, ok, len(dropped))[0]["P_kept_groups_rule"]


# ---- s0c_rl_check.check_fresh on a fake paired export: the paired rules replace "first K in order" / "k % 20"
def _fake_export(tmp_path, kept: int, tamper=None):
    import math
    from types import SimpleNamespace

    import h5py

    from vla_common.catalog import cube_name

    eps, _ = make_paired_stream(SEED, 0, 80, "s2a", "train", M)
    src = tmp_path / "src"
    src.mkdir()
    exp = {}
    for e in eps:
        exp.setdefault(e.layout_id, []).append(e)
    (src / "paired.json").write_text(json.dumps({"seed": SEED, "groups_map": {
        str(g): {"episodes": [e.index for e in v]} for g, v in exp.items()}}))
    with h5py.File(src / "a.h5", "w") as h:
        for e in eps:
            h.create_group(f"ep_{e.index}").attrs["episode_index"] = e.index
    keep, total, rank = [], 0, 0
    for g, v in exp.items():
        if total + len(v) <= kept:
            keep.append((g, "val" if rank % 20 == 19 else "train", v))
            total, rank = total + len(v), rank + 1
        if total == kept:
            break
    assert total == kept
    side = []
    for g, sp, v in keep:
        for e in v:
            yaw = math.degrees(e.layout[cube_name(tuple(e.spec.target))][2])
            side.append({"episode_index": e.index, "seed": SEED, "file_attrs": {"stage": "s2a"}, "image_size": 128, "T": 5,
                         "label": "action", "split": sp, "lerobot_episode_index": len(side), "layout_id": g,
                         "member": e.member, "paired": True, "cube_yaw_max_deg": M, "spawn_yaw_deg": yaw,
                         "spawn_yaw_fold_deg": float(PCK.fold(yaw))})
    if tamper:
        tamper(side)
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))
    (root / "meta" / "export_113.json").write_text(json.dumps({"sources": [str(src)], "episodes": len(side), "rows": 5 * len(side)}))
    (root / "meta" / "split_113.json").write_text(json.dumps({
        "kept_per_stream": kept, "paired": {}, "val": [s["lerobot_episode_index"] for s in side if s["split"] == "val"]}))
    ds = SimpleNamespace(meta=SimpleNamespace(features={f"observation.images.{c}": {"shape": (128, 128, 3)} for c in ("side", "wrist")}))
    return root, ds


PCK = pytest.importorskip("s0c_rl_check")


def _run_fresh(tmp_path, monkeypatch, kept, tamper=None):
    root, ds = _fake_export(tmp_path, kept, tamper)
    C = PCK.C
    monkeypatch.setattr(C, "columns_and_noise", lambda *a, **k: {"rows_per_episode_mismatch": [], "column_mismatch_episodes": [],
                                                                 "noise_std_per_arm_dim": [0.1] * 4, "claw_exec_ne_label_rows": 0})
    monkeypatch.setattr(C, "open_ds", lambda r: ds)
    monkeypatch.setattr(C, "video_check", lambda *a: {"psnr_min": 40.0, "psnr_mean": 41.0, "identity_fail": 0})
    monkeypatch.setattr(C, "sheet", lambda *a, **k: [])
    out = tmp_path / "out"
    out.mkdir()
    return PCK.check_fresh(root, out, 128, M, kept)["items"]


def test_check_fresh_paired_rules(tmp_path, monkeypatch):
    items = _run_fresh(tmp_path, monkeypatch, 100)
    assert items["1_kept_are_first_K_successful_in_order"] and items["1_split_rule"] and items["1_count_eq_K"], items
    assert items["yaw_spawn_agrees_with_regenerated_layout"]


def test_check_fresh_paired_flags_split_group_and_wrong_groups(tmp_path, monkeypatch):
    def flip(side):  # one member of the first group moves to val
        side[0]["split"] = "val"
    (tmp_path / "a").mkdir()
    assert not _run_fresh(tmp_path / "a", monkeypatch, 100, flip)["1_split_rule"]

    def drop_first_group(side):
        g = side[0]["layout_id"]
        side[:] = [s for s in side if s["layout_id"] != g]
        for k, s in enumerate(side):
            s["lerobot_episode_index"] = k
    (tmp_path / "b").mkdir()
    items = _run_fresh(tmp_path / "b", monkeypatch, 100, drop_first_group)
    assert not items["1_kept_are_first_K_successful_in_order"]


def test_stream_kept_applies_the_exports_kept_stream_override():
    # 009: the s3a1 union was exported with --kept 4000 --kept-stream s3a1:33200=3999; the check must apply the same
    spj = {"kept_per_stream": 4000, "kept_stream_overrides": ["s3a1:33200=3999"]}
    assert PC.stream_kept(spj, ("s3a1", 33200)) == 3999
    assert PC.stream_kept(spj, ("s2a", 32800)) == 4000
    assert PC.stream_kept({"kept_per_stream": 4000, "kept_stream_overrides": None}, ("s3a1", 33200)) == 4000
