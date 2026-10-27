"""GATES 0C attempt 1 (relabelled labels) items 1, 4(a), 5 on one exported `_rl` set (PERUN, policy venv, cpu_short;
item 2 = tests/test_data_audit.py --root, items 4(b) / 4(c) + mutants = scripts/s1_relabel.py dscheck, 3(b) = replay).

  python scripts/s0c_rl_check.py --dataset /data/datasets/0c_s1_pick_rl --ref /data/datasets/0c_s1_pick --out /runs/k0crl_s1
  python scripts/s0c_rl_check.py --dataset <set> --fresh --rerender 224 --expect-yaw 15 --kept 2000 --out <dir>
      (redesign #2 fresh-seed set without a reference export: check_fresh)

 1. episode list (LeRobot episode index -> HDF5 source / group, spec_json, split, and every other sidecar field except
    `label`; the shard dir name compared with the dataset name swapped) identical to the attempt-0 export `--ref`; export counts equal; rows per episode = T; every non-image column
    exact against HDF5 with `action` compared to r = s1_relabel.relabel(action, action_exec); images: sha256 of every
    video file identical to `--ref` (else the attempt-0 PSNR >= 35 dB + frame identity check on 10 episodes).
 4(a). std(action_exec - ORIGINAL clean label) per arm dim over |label| <= 0.7 in [0.095, 0.105]; claw columns identical.
 5. sheet <out>/<name>.png: the attempt-0 episodes (first, middle, last), original label, r and executed action printed.
Writes <out>/0c_rl_check_<name>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import s0c_check as C  # noqa: E402
from s1_relabel import relabel  # noqa: E402


def sha_videos(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "videos").rglob("*.mp4"))}


def check(root: Path, ref: Path, out: Path) -> dict:
    t0 = time.time()
    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    sref = [json.loads(x) for x in (ref / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    # every field except `label` (new) and the exporter's temporary shard name, which embeds the dataset name
    strip = lambda s: {k: (v.replace(root.name, ref.name) if k == "shard" else v)  # noqa: E731
                       for k, v in s.items() if k != "label"}
    ep_diff = [i for i, (a, b) in enumerate(zip(side, sref, strict=True)) if strip(a) != strip(b)]
    exp, eref = (json.loads((d / "meta" / "export_113.json").read_text()) for d in (root, ref))
    cn = C.columns_and_noise(root, side, expect=lambda act, exe: relabel(act.astype("float32"), exe.astype("float32")))
    vs, vr = sha_videos(root), sha_videos(ref)
    vid_same = vs == vr and len(vs) > 0
    ds = C.open_ds(root)
    vc = None if vid_same else C.video_check(ds, side)
    sh = C.sheet(ds, side, out / f"{root.name}.png", relabel=relabel)
    T = {int(s["T"]) for s in side}
    items = {
        "1_episode_list_identical_to_attempt0": not ep_diff and len(side) == len(sref),
        "1_counts_eq_attempt0": exp["episodes"] == eref["episodes"] == len(side) and exp["rows"] == eref["rows"],
        "1_rows_per_episode_eq_T": not cn["rows_per_episode_mismatch"] and exp["rows"] == len(side) * max(T)
        and len(T) == 1,
        "1_columns_exact_action_eq_r": not cn["column_mismatch_episodes"],
        "1_images": vid_same or (vc["psnr_min"] >= C.PSNR_BAR and vc["identity_fail"] == 0),
        "4a_noise_std_in_band": all(C.NOISE_BAND[0] <= x <= C.NOISE_BAND[1] for x in cn["noise_std_per_arm_dim"]),
        "4a_claw_identical": cn["claw_exec_ne_label_rows"] == 0,
    }
    res = {"dataset": str(root), "ref": str(ref), "name": root.name, "label": side[0].get("label"),
           "episodes": len(side), "rows": exp["rows"], "T": sorted(T), "episode_list_diff": ep_diff[:20],
           "videos": {"files": len(vs), "sha256_identical_to_ref": vid_same,
                      "differing": sorted(k for k in set(vs) | set(vr) if vs.get(k) != vr.get(k))[:20]},
           "video_check": vc, "columns_noise": cn, "sheet": {"png": f"{root.name}.png", "episodes": sh},
           "items": items, "pass_items_1_4a": all(items.values()), "git_sha": (ROOT / "GIT_HEAD").read_text().strip()
           if (ROOT / "GIT_HEAD").exists() else None, "wall_s": round(time.time() - t0, 1)}
    (out / f"0c_rl_check_{root.name}.json").write_text(json.dumps(res, indent=1))
    print("RLCHECK=" + json.dumps({"name": root.name, "items": items, "noise_std": cn["noise_std_per_arm_dim"],
                                   "videos_identical": vid_same}), flush=True)
    return res


def check_rerender(root: Path, ref: Path, out: Path, image_size: int) -> dict:
    """GATES Stage 1 redesign: the same items on a RE-RENDERED set (s1 at H = 224, sim not bit-reproducible). Item 1's
    "episode list identical to attempt 0" becomes: overlap of the kept stream episodes with `--ref` reported (and the
    spec of every shared stream index equal), expert success within the 0C item-3 tolerance (s0c_check.expert_success
    on the collect batches), count = the 0C plan count; images re-checked (PSNR >= 35 dB + frame identity on 10
    episodes, torchcodec) since no video can be byte-identical; every episode's image_size == --image-size."""
    t0 = time.time()
    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    sref = [json.loads(x) for x in (ref / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    exp = json.loads((root / "meta" / "export_113.json").read_text())
    ki, kr = {s["episode_index"]: s for s in side}, {s["episode_index"]: s for s in sref}
    shared = sorted(set(ki) & set(kr))
    # spec + instruction only: the split follows the kept rank, which shifts when another episode fails (redesign RESULT)
    spec_diff = [i for i in shared if (ki[i]["spec_json"], ki[i]["instruction"]) != (kr[i]["spec_json"], kr[i]["instruction"])]
    # a larger set continuing the same stream (redesign attempt 1, 2,000 demos): its first len(ref) episodes must be the
    # ref's (every sidecar field except the exporter's temporary shard dir, which depends on the worker split)
    strip = lambda s_: {k: v for k, v in s_.items() if k != "shard"}  # noqa: E731
    prefix_diff = ([i for i, (a_, b_) in enumerate(zip(side, sref)) if strip(a_) != strip(b_)]
                   if len(side) > len(sref) else None)
    es = C.expert_success([Path(x) for x in exp["sources"]])
    ceil = C.CEILING["pick"]
    cn = C.columns_and_noise(root, side, expect=lambda act, exe: relabel(act.astype("float32"), exe.astype("float32")))
    ds = C.open_ds(root)
    vc = C.video_check(ds, side)
    sh = C.sheet(ds, side, out / f"{root.name}.png", relabel=relabel)
    T = {int(s["T"]) for s in side}
    plan = int(json.loads((root / "meta" / "split_113.json").read_text())["kept_per_stream"])
    items = {
        "1_count_eq_plan": exp["episodes"] == plan == len(side) and len(side) >= len(sref),
        "1_first_ref_episodes_identical": prefix_diff is None or prefix_diff == [],
        "1_rows_per_episode_eq_T": not cn["rows_per_episode_mismatch"] and exp["rows"] == len(side) * max(T)
        and len(T) == 1,
        "1_columns_exact_action_eq_r": not cn["column_mismatch_episodes"],
        "1_image_size": {int(s["image_size"]) for s in side} == {image_size}
        and list(ds.meta.features["observation.images.side"]["shape"][:2]) == [image_size, image_size],
        "1_psnr_min_ge_35": vc["psnr_min"] >= C.PSNR_BAR, "1_frame_identity": vc["identity_fail"] == 0,
        "1_shared_stream_specs_equal": not spec_diff,
        "1_expert_pooled_ge_ceiling_minus_5": es["pooled"]["rate_pct"] >= ceil - C.POOLED_TOL,
        "1_expert_every_batch_ge_ceiling_minus_10": all(b["rate_pct"] >= ceil - C.BATCH_TOL for b in es["batches"]),
        "1_expert_every_batch_n_ge_200": all(b["attempted"] >= C.BATCH_MIN_N for b in es["batches"]),
        "4a_noise_std_in_band": all(C.NOISE_BAND[0] <= x <= C.NOISE_BAND[1] for x in cn["noise_std_per_arm_dim"]),
        "4a_claw_identical": cn["claw_exec_ne_label_rows"] == 0,
    }
    res = {"dataset": str(root), "ref": str(ref), "name": root.name, "label": side[0].get("label"),
           "episodes": len(side), "rows": exp["rows"], "T": sorted(T), "image_size": image_size,
           "overlap_with_ref": {"kept_shared": len(shared), "only_here": len(set(ki) - set(kr)),
                                "only_ref": len(set(kr) - set(ki)), "shared_spec_or_instruction_diff": spec_diff[:20],
                                "prefix_episodes_differing_from_ref": None if prefix_diff is None else prefix_diff[:20],
                                "same_position_same_stream_index": sum(a["episode_index"] == b["episode_index"]
                                                                       for a, b in zip(side, sref))},
           "expert_success": es, "ceiling_pct": ceil, "video_check": vc, "columns_noise": cn,
           "sheet": {"png": f"{root.name}.png", "episodes": sh}, "items": items,
           "pass_items_1_4a": all(items.values()), "git_sha": (ROOT / "GIT_HEAD").read_text().strip()
           if (ROOT / "GIT_HEAD").exists() else None, "wall_s": round(time.time() - t0, 1)}
    (out / f"0c_rl_check_{root.name}.json").write_text(json.dumps(res, indent=1))
    print("RLCHECK=" + json.dumps({"name": root.name, "items": items, "noise_std": cn["noise_std_per_arm_dim"],
                                   "overlap": res["overlap_with_ref"], "pooled": es["pooled"],
                                   "psnr": [vc["psnr_min"], vc["psnr_mean"]]}), flush=True)
    return res


def fold(y):
    """Raw yaw (deg) folded mod 90 into [-45, 45) (vla_tabletop.rollout.fold_yaw_deg for numpy)."""
    return (np.asarray(y, float) + 45.0) % 90.0 - 45.0


def yaw_items(side: list[dict], m: float, regen_deg: list[float]) -> tuple[dict, dict]:
    """GATES redesign #2 yaw items. side: sidecar rows (spawn_yaw_deg / spawn_yaw_fold_deg / cube_yaw_max_deg attrs, any
    may be missing); regen_deg: the layout yaw (deg) of each episode regenerated from (seed, index, sampler, the
    episode's own sidecar m). Returns (items, numbers). Missing spawn attrs fail their items."""
    tol = 0.5
    has = all("spawn_yaw_deg" in s and "spawn_yaw_fold_deg" in s for s in side)
    rec_fold = np.array([s["spawn_yaw_fold_deg"] for s in side]) if has else np.array([])
    rec_raw = np.array([s["spawn_yaw_deg"] for s in side]) if has else np.array([])
    regen = np.asarray(regen_deg, float)
    rg = fold(regen)
    lo, hi = -m - tol, m + tol
    items = {
        "1_cube_yaw_max_deg_eq_expected": all(float(s.get("cube_yaw_max_deg", 180.0)) == float(m) for s in side),
        "yaw_spawn_attrs_present": has,
        "yaw_spawn_fold_in_range": has and bool(((rec_fold >= lo) & (rec_fold <= hi)).all()),
        "yaw_regenerated_layout_fold_in_range": bool(((rg >= lo) & (rg <= hi)).all()),
        # difference of the two raw yaws, mod 90, within 0.5 deg
        "yaw_spawn_agrees_with_regenerated_layout": has and bool((np.abs(fold(rec_raw - regen)) <= tol).all()),
        "yaw_coverage": has and (bool((np.abs(rec_fold) <= tol).all()) if m == 0 else
                                 bool(rec_fold.min() < -(m - 1) and rec_fold.max() > (m - 1))),
    }
    nums = {"expected_m": m, "n": len(side)}
    for name, a in (("spawn_fold", rec_fold), ("regenerated_fold", rg)):
        if len(a):
            nums[name] = {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}
    return items, nums


def check_fresh(root: Path, out: Path, image_size: int, m: float, kept: int) -> dict:
    """GATES 0C items for a fresh-seed re-render (no reference export): export counts == K, rows == K * T, one T, image
    size == H in every episode and in the dataset features, every non-image column exact vs HDF5 (action = relabel),
    kept episodes = the first K successful stream indices in order (from the source HDF5) with split k % 20 == 19, images
    PSNR >= 35 dB + frame identity on 10 episodes, cube_yaw_max_deg == m, the yaw-faithfulness items (yaw_items), 4(a)
    noise band, 5 sheet. Missing spawn-yaw attrs fail their items (a set collected before the instrument)."""
    import h5py
    from export_dataset import VAL_EVERY
    from vla_common.catalog import cube_name
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import make_stream

    t0 = time.time()
    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    exp = json.loads((root / "meta" / "export_113.json").read_text())
    split_json = json.loads((root / "meta" / "split_113.json").read_text())
    idx = [int(s["episode_index"]) for s in side]
    allidx = []
    for d in exp["sources"]:
        for f in sorted(Path(d).glob("*.h5")):
            with h5py.File(f, "r") as h:
                allidx += [int(h[n].attrs["episode_index"]) for n in h]
    allidx.sort()
    paired = "paired" in split_json  # export_dataset --paired: whole groups, split per group (scripts/paired_check.py)
    regen = []
    if paired:
        from vla_tabletop.rollout import make_paired_stream

        s0 = side[0]
        ref, _ = make_paired_stream(int(s0["seed"]), 0, max(int(s["layout_id"]) for s in side) + 1, s0["file_attrs"]["stage"],
                                    "train", float(s0.get("cube_yaw_max_deg", 180.0)))
        by = {(e.layout_id, e.member): e for e in ref}
        for s in side:
            ep = by[(int(s["layout_id"]), int(s["member"]))]
            regen.append(math.degrees(ep.layout[cube_name(tuple(ep.spec.target))][2]))
    else:
        for s in side:
            stage = s["file_attrs"]["stage"]
            ep = make_stream(int(s["seed"]), int(s["episode_index"]), 1, lambda rng, st=stage: sample_spec(st, "train", rng),
                             split="train", cube_yaw_max_deg=float(s.get("cube_yaw_max_deg", 180.0)))[0]
            regen.append(math.degrees(ep.layout[cube_name(tuple(ep.spec.target))][2]))
    yi, yn = yaw_items(side, m, regen)
    if paired:  # paired rules replace "first K successful in order" / "k % 20 == 19" (per group, paired_check)
        import paired_check as PC

        expected, ok = PC.source_success([Path(d) for d in exp["sources"]])
        pi, _ = PC.selection_items(side, expected, ok, kept)
        rule_items = {"1_kept_are_first_K_successful_in_order": pi["P_kept_groups_rule"],
                      "1_split_rule": pi["P_split_rule"]
                      and split_json["val"] == [s["lerobot_episode_index"] for s in side if s["split"] == "val"]}
    else:
        rule_items = {
            "1_kept_are_first_K_successful_in_order": len(allidx) >= kept and idx == allidx[:kept],
            "1_split_rule": [s["split"] for s in side] == ["val" if k % VAL_EVERY == VAL_EVERY - 1 else "train"
                                                           for k in range(len(side))]
            and split_json["val"] == [s["lerobot_episode_index"] for s in side if s["split"] == "val"]}
    cn = C.columns_and_noise(root, side, expect=lambda act, exe: relabel(act.astype("float32"), exe.astype("float32")))
    ds = C.open_ds(root)
    vc = C.video_check(ds, side)
    sh = C.sheet(ds, side, out / f"{root.name}.png", relabel=relabel)
    T = {int(s["T"]) for s in side}
    items = {
        "1_count_eq_K": exp["episodes"] == kept == len(side) == int(split_json["kept_per_stream"]),
        "1_rows_eq_K_T": len(T) == 1 and exp["rows"] == kept * max(T),
        "1_rows_per_episode_eq_T": not cn["rows_per_episode_mismatch"],
        "1_columns_exact_action_eq_r": not cn["column_mismatch_episodes"],
        "1_image_size": {int(s["image_size"]) for s in side} == {image_size}
        and all(list(ds.meta.features[f"observation.images.{c}"]["shape"][:2]) == [image_size, image_size]
                for c in C.CAMS),
        "1_psnr_min_ge_35": vc["psnr_min"] >= C.PSNR_BAR, "1_frame_identity": vc["identity_fail"] == 0,
        **rule_items,
        **yi,
        "4a_noise_std_in_band": all(C.NOISE_BAND[0] <= x <= C.NOISE_BAND[1] for x in cn["noise_std_per_arm_dim"]),
        "4a_claw_identical": cn["claw_exec_ne_label_rows"] == 0,
    }
    res = {"dataset": str(root), "name": root.name, "label": side[0].get("label"), "episodes": len(side),
           "rows": exp["rows"], "T": sorted(T), "image_size": image_size, "expect_yaw": m, "kept": kept,
           "yaw": yn, "source_successful_episodes": len(allidx), "kept_stream_indices": [idx[0], idx[-1]],
           "video_check": vc, "columns_noise": cn, "sheet": {"png": f"{root.name}.png", "episodes": sh},
           "items": items, "pass_items_1_4a": all(items.values()), "git_sha": (ROOT / "GIT_HEAD").read_text().strip()
           if (ROOT / "GIT_HEAD").exists() else None, "wall_s": round(time.time() - t0, 1)}
    (out / f"0c_rl_check_{root.name}.json").write_text(json.dumps(res, indent=1))
    print("RLCHECK=" + json.dumps({"name": root.name, "items": items, "yaw": yn,
                                   "noise_std": cn["noise_std_per_arm_dim"],
                                   "psnr": [vc["psnr_min"], vc["psnr_mean"]]}), flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--ref", type=Path, help="the attempt-0 export of the same set (not with --fresh)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rerender", type=int, default=0, help="image size of a re-rendered set (Stage 1 redesign)")
    p.add_argument("--fresh", action="store_true", help="fresh-seed set without a reference export (redesign #2)")
    p.add_argument("--expect-yaw", type=float, help="--fresh: the cube yaw range m the set must have")
    p.add_argument("--kept", type=int, help="--fresh: the planned kept episode count K")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.fresh:
        assert a.rerender and a.expect_yaw is not None and a.kept, "--fresh needs --rerender, --expect-yaw and --kept"
        res = check_fresh(a.dataset, a.out, a.rerender, a.expect_yaw, a.kept)
        sys.exit(0 if res["pass_items_1_4a"] else 1)
    assert a.ref, "--ref is required without --fresh"
    if a.rerender:
        check_rerender(a.dataset, a.ref, a.out, a.rerender)
    else:
        check(a.dataset, a.ref, a.out)


if __name__ == "__main__":
    main()
