"""GATES Stage 4 re-plan (e2cef02) 0C items 1, 4(a), 5, 6 on one exported set (PERUN, policy venv, cpu_short). Item 2 =
tests/test_data_audit.py --root, 4(b)/(c) + mutants = scripts/s1_relabel.py dscheck, union = scripts/s3_union_check.py.

  python scripts/s4_check.py --kind grey --dataset /data/datasets/0c_s2_pick_rl224_y15g --twin /data/datasets/0c_s2_pick_rl224_y15 \
      --rerender 224 --expect-yaw 15 --kept 2000 --expect-palette s4r --out /runs/k4_all/s2g
  python scripts/s4_check.py --kind s4r  --dataset /data/datasets/0c_s4r_mix_rl224_y15 ... --out /runs/k4_all/s4r
  python scripts/s4_check.py --kind grey --only table --dataset /data/datasets/0c_s2_pick_rl224_y15 --out <dir>   (item 6 mutant: exit 1)

kind grey = a nominal grey-table re-render of one task stream (s2 / s3a / s3b); kind s4r = the mixed randomized set (s4).
Items (names carry the GATES item number): 1 counts / integrity as s0c_rl_check.check_fresh with T per task
(results/0a2_<task>_timing.json), expert success per task against the 0A.2 ceilings (pooled >= c - 5, every batch >= c - 10),
s4r mix share, palette attributes, re-render consistency vs --twin; 4(a) noise; 5 sheets; 6 table-colour faithfulness.
Item 6: decoded side frame at t = 0 of 50 episodes spread over the set; median HSV saturation of the fixed region TABLE_BOX
(224 px image) <= 0.10; s4r: Spearman(region median value, recorded table value) >= 0.5.
TABLE_BOX = (u0, v0, u1, v1) = (0, 190, 64, 222): the lower-left corner of the side image. Reason: side camera eye
(1.15, 0.45, 0.55) -> look (0.5, 0, 0.18), focal 20 / aperture 20.955 (vision_lift/cameras.py); every cube / pad centre lies in
x 0.33..0.67, y +-0.30 (vla_tabletop/layout.py REGION_*), so objects (half diagonal / pad half 0.05, height <= 0.07 m) project
into a band that, dilated by 12 px for the s4r camera jitter (+-1 cm, +-2 deg ~ 9 px), leaves this near-camera corner free; the
robot at t = 0 is in the upper half. Projected and checked in the commit that introduced it; same view as
results/frames/s4r_palette/*.png (that corner is plain table).
"""

from __future__ import annotations

import argparse
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
from s0c_rl_check import yaw_items  # noqa: E402
from s1_relabel import relabel  # noqa: E402

STAGE_TASK = {"s2": "pick", "s3a": "place", "s3b": "stack"}
TABLE_BOX = (0, 190, 64, 222)  # u0, v0, u1, v1 on the 224 x 224 side image
TABLE_N = 50
SAT_MAX, RHO_MIN = 0.10, 0.5
MIX_BAND = (0.30, 0.37)
DEFAULT_LIGHT = (-0.57735026, 0.57735026, -0.57735026)


def timing_T() -> dict[str, int]:
    return {t: int(json.loads((ROOT / "results" / f"0a2_{t}_timing.json").read_text())["T_steps"])
            for t in ("pick", "place", "stack")}


def vjson(s: dict) -> dict:
    v = s.get("visual_json")
    return json.loads(v) if isinstance(v, str) else (v or {})


def table_value(s: dict) -> float:
    """HSV value of the episode's recorded table colour (max channel); nominal / missing -> None."""
    t = vjson(s).get("table_rgb")
    return None if t is None else float(max(t))


def nonnominal_table(s: dict) -> bool:
    t = vjson(s).get("table_rgb")
    return t is not None and not np.allclose(t, 0.5, atol=1e-4)


def nondefault_light(s: dict) -> bool:
    d = vjson(s).get("light")
    return d is not None and not np.allclose(d, DEFAULT_LIGHT, atol=1e-4)


def light_elev(s: dict) -> float | None:
    d = vjson(s).get("light")
    return None if d is None else float(np.degrees(np.arcsin(-d[2] / np.linalg.norm(d))))


# ----------------------------------------------------------------------------- pure item functions
def expert_items(es: dict, tasks: list[str]) -> tuple[dict, dict]:
    """Per task: pooled rate >= ceiling - 5 and every batch >= ceiling - 10 (per-task counts from collect_summary per_task);
    every batch n >= 200 (batch total). Returns (items, numbers)."""
    items, nums = {}, {}
    for t in tasks:
        ceil = C.CEILING[t]
        per = [b["per_task"][t] for b in es["batches"]]
        pooled = C.rate(sum(x["kept"] for x in per), sum(x["attempted"] for x in per))
        batches = [C.rate(x["kept"], x["attempted"]) for x in per]
        nums[t] = {"ceiling_pct": ceil, "pooled": pooled, "batches": batches}
        items[f"1_expert_pooled_{t}_ge_ceiling_minus_5"] = pooled["rate_pct"] >= ceil - C.POOLED_TOL
        items[f"1_expert_every_batch_{t}_ge_ceiling_minus_10"] = all(b["rate_pct"] >= ceil - C.BATCH_TOL for b in batches)
    items["1_expert_every_batch_n_ge_200"] = all(b["attempted"] >= C.BATCH_MIN_N for b in es["batches"])
    return items, nums


def mix_items(side: list[dict]) -> tuple[dict, dict]:
    n = len(side)
    share = {t: sum(s["task"] == t for s in side) / n for t in ("pick", "place", "stack")}
    return {"1_mix_share_each_task_in_band": all(MIX_BAND[0] <= v <= MIX_BAND[1] for v in share.values())}, \
        {"share": share, "counts": {t: round(v * n) for t, v in share.items()}, "band": list(MIX_BAND)}


def palette_items(side: list[dict], kind: str, expect: str) -> tuple[dict, dict]:
    from export_dataset import visual_ok

    items = {"palette_table_palette_eq_expected_all": all(s.get("table_palette") == expect for s in side)}
    mode = {s.get("visual_mode") for s in side}, {vjson(s).get("mode") for s in side}
    if kind == "grey":
        items["palette_visual_mode_nominal_all"] = mode[0] == {"nominal"}
        items["palette_visual_json_s4r_nominal_all"] = mode[1] == {"s4r_nominal"}
    else:
        items["palette_visual_mode_train_rand_all"] = mode[0] == {"train_rand"}
        items["palette_visual_json_s4r_train_all"] = mode[1] == {"s4r_train"}
        items["palette_params_inside_train_outside_heldout_all"] = all(visual_ok(s) is None for s in side)
    nums = {"visual_mode": sorted(map(str, mode[0])), "visual_json_mode": sorted(map(str, mode[1])),
            "table_palette": sorted({str(s.get("table_palette")) for s in side}),
            "nonnominal_table": sum(map(nonnominal_table, side)), "nondefault_light": sum(map(nondefault_light, side))}
    return items, nums


def twin_items(side: list[dict], twin: list[dict]) -> tuple[dict, dict]:
    """Re-render consistency: kept stream indices shared with the 112-table twin; spec + instruction equal on every shared
    index (the split follows the kept rank, which shifts, so it is not compared)."""
    a, b = {s["episode_index"]: s for s in side}, {s["episode_index"]: s for s in twin}
    shared = sorted(set(a) & set(b))
    diff = [i for i in shared if (a[i]["spec_json"], a[i]["instruction"]) != (b[i]["spec_json"], b[i]["instruction"])]
    return {"1_twin_overlap_nonempty": bool(shared), "1_twin_shared_specs_instructions_equal": not diff}, \
        {"kept_shared": len(shared), "only_here": len(set(a) - set(b)), "only_twin": len(set(b) - set(a)),
         "overlap_frac_of_set": len(shared) / len(a), "spec_or_instruction_diff": diff[:20]}


def pick_sheet_episodes(side: list[dict], kind: str) -> list[int]:
    """grey: sidecar indices 0, middle, last (= episodes 0 / 1000 / 1999). s4r: 2 per task (pick, place, stack order), the
    first of each pair = the first episode of the task with a non-nominal table AND a non-default light (else the first
    non-nominal table, else the first of the task), the second = the first episode of the task that is not the first."""
    if kind == "grey":
        return [0, len(side) // 2, len(side) - 1]
    out = []
    for t in ("pick", "place", "stack"):
        ix = [i for i, s in enumerate(side) if s["task"] == t]
        both = [i for i in ix if nonnominal_table(side[i]) and nondefault_light(side[i])]
        a = (both or [i for i in ix if nonnominal_table(side[i])] or ix)[0]
        out += [a, next(i for i in ix if i != a)]
    return out


def sheet_items(side: list[dict], chosen: list[int], kind: str) -> dict:
    if kind == "grey":
        return {"5_sheet_three_episodes_first_middle_last": chosen == [0, len(side) // 2, len(side) - 1]}
    return {"5_sheet_six_episodes_two_per_task": sorted(side[i]["task"] for i in chosen) == ["pick"] * 2 + ["place"] * 2 + ["stack"] * 2,
            "5_sheet_has_nonnominal_table": any(nonnominal_table(side[i]) for i in chosen),
            "5_sheet_has_nondefault_light": any(nondefault_light(side[i]) for i in chosen)}


def note(s: dict) -> str:
    t, e = vjson(s).get("table_rgb"), light_elev(s)
    return (f"table rgb {tuple(round(float(x), 3) for x in t)} light elev {e:.1f} deg" if t is not None and e is not None
            else "table rgb / light ?")


def region_stats(img: np.ndarray) -> tuple[float, float]:
    """(median HSV saturation, median HSV value) of TABLE_BOX in a uint8 (H, W, 3) side frame."""
    from matplotlib.colors import rgb_to_hsv

    u0, v0, u1, v1 = TABLE_BOX
    assert img.shape[0] >= v1 and img.shape[1] >= u1, img.shape
    hsv = rgb_to_hsv(img[v0:v1, u0:u1].reshape(-1, 3).astype(np.float64) / 255.0)
    return float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2]))


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def table_items(sat: list[float], val: list[float], recorded: list[float | None], kind: str) -> tuple[dict, dict]:
    items = {"6_table_region_median_saturation_le_0.10": bool(len(sat)) and max(sat) <= SAT_MAX}
    nums = {"n": len(sat), "saturation_max": max(sat), "saturation_median": float(np.median(sat)),
            "value_min": min(val), "value_max": max(val)}
    if kind == "s4r":
        rho = spearman(val, [r for r in recorded])
        nums["spearman_value_vs_recorded_table_value"] = rho
        items["6_table_value_spearman_ge_0.5"] = bool(rho >= RHO_MIN)  # nan -> False
    return items, nums


def table_check(ds, side: list[dict], kind: str) -> dict:
    eps = ds.meta.episodes
    pick = sorted({int(round(x)) for x in np.linspace(0, len(side) - 1, TABLE_N)})
    sat, val = [], []
    for e in pick:
        s_, v_ = region_stats(ds[int(eps[e]["dataset_from_index"])]["observation.images.side"].permute(1, 2, 0).numpy())
        sat.append(s_)
        val.append(v_)
    items, nums = table_items(sat, val, [table_value(side[e]) for e in pick], kind)
    return {"box_u0_v0_u1_v1": list(TABLE_BOX), "image_size": 224, "episodes": pick,
            "reason": "lower-left corner, outside the object footprint x 0.33..0.67 / y +-0.30 projected through the side "
                      "camera (eye 1.15, 0.45, 0.55 -> look 0.5, 0, 0.18) with object size and 12 px camera-jitter margin; "
                      "see module docstring", "per_episode_saturation": sat, "per_episode_value": val,
            "per_episode_recorded_table_value": [table_value(side[e]) for e in pick], "numbers": nums, "items": items}


# ----------------------------------------------------------------------------- driver
def regen_yaw(side: list[dict]) -> list[float]:
    from vla_common.catalog import cube_name
    from vla_common.tasks import sample_spec
    from vla_tabletop.layout import sample_layout

    out = []
    for s in side:
        fa = s["file_attrs"]
        rng = np.random.default_rng([int(s["seed"]), int(s["episode_index"])])
        spec = sample_spec(fa["stage"], "train", rng, tuple(float(x) for x in fa.get("mix", "1,1,1").split(",")))
        lay = sample_layout(spec, rng, cube_yaw_max_deg=float(s.get("cube_yaw_max_deg", 180.0)))[0]
        out.append(math.degrees(lay[cube_name(tuple(spec.target))][2]))
    return out


def check(a: argparse.Namespace) -> dict:
    import h5py
    from export_dataset import VAL_EVERY

    t0 = time.time()
    root, out = a.dataset, a.out
    out.mkdir(parents=True, exist_ok=True)
    side = [json.loads(x) for x in (root / "meta" / "episodes_113.jsonl").read_text().splitlines()]
    ds = C.open_ds(root)
    if a.only == "table":
        tc = table_check(ds, side, a.kind)
        res = {"dataset": str(root), "name": root.name, "kind": a.kind, "table": tc, "items": tc["items"],
               "pass": all(tc["items"].values())}
        (out / f"0s4_table_{root.name}.json").write_text(json.dumps(res, indent=1))
        print("S4TABLE=" + json.dumps({"name": root.name, "items": tc["items"], "numbers": tc["numbers"]}), flush=True)
        return res
    exp = json.loads((root / "meta" / "export_113.json").read_text())
    split_json = json.loads((root / "meta" / "split_113.json").read_text())
    Tt = timing_T()
    tasks = sorted({s["task"] for s in side})
    streams = {(s["file_attrs"]["stage"], int(s["seed"])) for s in side}
    assert len(streams) == 1, streams
    idx = [int(s["episode_index"]) for s in side]
    allidx = []
    for d in exp["sources"]:
        for f in sorted(Path(d).glob("*.h5")):
            with h5py.File(f, "r") as h:
                allidx += [int(h[n].attrs["episode_index"]) for n in h]
    allidx.sort()
    rank = {k: r for r, k in enumerate(sorted(idx))}  # rank in ascending episode-index order (one stream)
    yi, yn = yaw_items(side, a.expect_yaw, regen_yaw(side))
    cn = C.columns_and_noise(root, side, expect=lambda act, exe: relabel(act.astype("float32"), exe.astype("float32")))
    vc = C.video_check(ds, side)
    es = C.expert_success([Path(x) for x in exp["sources"]])
    ei, en = expert_items(es, [t for t in ("pick", "place", "stack") if t in tasks])
    pi, pn = palette_items(side, a.kind, a.expect_palette)
    chosen = pick_sheet_episodes(side, a.kind)
    sh = C.sheet(ds, side, out / f"{root.name}.png", relabel=relabel, chosen=chosen, note=note)
    tc = table_check(ds, side, a.kind)
    items = {
        "1_count_eq_K": exp["episodes"] == a.kept == len(side) == int(split_json["kept_per_stream"]),
        "1_tasks_as_expected": tasks == (["pick", "place", "stack"] if a.kind == "s4r" else [STAGE_TASK[next(iter(streams))[0]]]),
        "1_T_per_task_eq_timing": all(int(s["T"]) == Tt[s["task"]] for s in side),
        "1_rows_eq_sum_T": exp["rows"] == sum(Tt[s["task"]] for s in side),
        "1_rows_per_episode_eq_T": not cn["rows_per_episode_mismatch"],
        "1_columns_exact_action_eq_r": not cn["column_mismatch_episodes"],
        "1_image_size": {int(s["image_size"]) for s in side} == {a.rerender}
        and all(list(ds.meta.features[f"observation.images.{c}"]["shape"][:2]) == [a.rerender] * 2 for c in C.CAMS),
        "1_psnr_min_ge_35": vc["psnr_min"] >= C.PSNR_BAR, "1_frame_identity": vc["identity_fail"] == 0,
        "1_kept_are_first_K_successful_in_order": len(allidx) >= a.kept and len(set(idx)) == len(idx) == a.kept
        and sorted(idx) == allidx[:a.kept],  # sidecar order is shard order (grouped by task for s4r), so compare sorted
        "1_split_rule": [s["split"] for s in side] == [
            "val" if rank[int(s["episode_index"])] % VAL_EVERY == VAL_EVERY - 1 else "train" for s in side]
        and sorted(split_json["val"]) == sorted(s["lerobot_episode_index"] for s in side if s["split"] == "val"),
        **yi, **ei, **pi, **sheet_items(side, chosen, a.kind), **tc["items"],
        "4a_noise_std_in_band": all(C.NOISE_BAND[0] <= x <= C.NOISE_BAND[1] for x in cn["noise_std_per_arm_dim"]),
        "4a_claw_identical": cn["claw_exec_ne_label_rows"] == 0,
    }
    extra = {}
    if a.kind == "s4r":
        mi, extra["mix"] = mix_items(side)
        items |= mi
    if a.twin:
        twin = [json.loads(x) for x in (a.twin / "meta" / "episodes_113.jsonl").read_text().splitlines()]
        ti, extra["twin"] = twin_items(side, twin)
        items |= ti
        tes = C.expert_success([Path(x) for x in json.loads((a.twin / "meta" / "export_113.json").read_text())["sources"]])
        extra["twin"] |= {"name": a.twin.name, "expert_pooled": tes["pooled"], "set_expert_pooled": es["pooled"]}
    res = {"dataset": str(root), "name": root.name, "kind": a.kind, "episodes": len(side), "rows": exp["rows"],
           "T_per_task": {t: Tt[t] for t in tasks}, "image_size": a.rerender, "expect_yaw": a.expect_yaw, "kept": a.kept,
           "expect_palette": a.expect_palette, "yaw": yn, "source_successful_episodes": len(allidx),
           "kept_stream_indices": [idx[0], idx[-1]], "video_check": vc, "columns_noise": cn, "expert_success": es,
           "expert": en, "palette": pn, "table": tc, "sheet": {"png": f"{root.name}.png", "episodes": sh,
                                                                "chosen_sidecar_indices": chosen,
                                                                "notes": [note(side[i]) for i in chosen]},
           **extra, "items": items, "pass": all(items.values()), "wall_s": round(time.time() - t0, 1)}
    (out / f"0s4_check_{root.name}.json").write_text(json.dumps(res, indent=1))
    print("S4CHECK=" + json.dumps({"name": root.name, "items": items, "failed": [k for k, v in items.items() if not v],
                                   "noise_std": cn["noise_std_per_arm_dim"], "psnr": [vc["psnr_min"], vc["psnr_mean"]],
                                   "table": tc["numbers"]}), flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--kind", required=True, choices=("grey", "s4r"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--twin", type=Path, help="the 112-table export of the same stream (grey kind)")
    p.add_argument("--rerender", type=int, default=224)
    p.add_argument("--expect-yaw", type=float, default=15.0)
    p.add_argument("--kept", type=int, default=2000)
    p.add_argument("--expect-palette", default="s4r")
    p.add_argument("--only", choices=("table",), help="run item 6 only (the table-colour mutant)")
    a = p.parse_args()
    sys.exit(0 if check(a)["pass"] else 1)


if __name__ == "__main__":
    main()
