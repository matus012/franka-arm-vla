"""Instrument D1 (substitution, Stage 4 re-plan): closed-loop bridge inputs vs the training dataset's own rows (policy venv, CPU).

  python scripts/s4b_d1_compare.py --dump /runs/d4b_d1/d1_s32500_dump.npz --dataset /data/datasets/0c_s2_pick_rl224_y15g \
      --ckpt /runs/s4_baseline_seed2/ckpt_028638 --out /runs/d4b_d1/d1_compare.json [--video-backend torchcodec]

`dump` = scripts/eval_policy.py --dump-slots K on the SAME stream (seed, stage, flags) the dataset was collected from.
Slot j (stream seed S, episode i) -> dataset episode via meta/episodes_113.jsonl (`seed`, `episode_index` ->
`lerobot_episode_index`); slots whose demo was not kept (failed expert, --kept cut) are listed as unmatched.
Per slot: PSNR of the bridge t=0 frames vs the dataset row-0 decoded frames (uint8 vs round(255 x decoded)); max |state|
diff at t=0; labels equal; BaselinePolicy.predict on the dataset row-0 inputs vs the chunk the server returned; both
chunks vs the dataset action rows 0..N-1 (arm MAE, claw agreement); first executed claw close vs the dataset's; the
per-step |state - dataset state| curve. `pass` = the t=0 input items only: PSNR >= 35 dB (GATES 0B.1 bar) on both cameras,
state diff <= TOL (s4b_parity.TOL, 1e-5), labels equal. Server-vs-local predict on the bridge's own inputs is reported
(`server_vs_local_bridge`, the server runs on another device) but not part of `pass`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_policy import train_baseline as TB  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402
from vla_policy.baseline import BaselinePolicy  # noqa: E402

PSNR_BAR_DB = 35.0  # GATES 0B.1: decoded video vs source frames
TOL = 1e-5  # scripts/s4b_parity.TOL
CURVE_STEPS = (0, 10, 30, 100)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def first_close(claw: np.ndarray) -> int:
    i = np.flatnonzero(np.asarray(claw) > 0.5)
    return int(i[0]) if i.size else -1


def episode_map(root: Path) -> dict[tuple[int, int], int]:
    """(stream seed, stream episode index) -> LeRobot episode index, from the exporter's sidecar."""
    out = {}
    for line in (root / "meta" / "episodes_113.jsonl").read_text().splitlines():
        s = json.loads(line)
        out[(int(s["seed"]), int(s["episode_index"]))] = int(s["lerobot_episode_index"])
    return out


def u8(x: np.ndarray) -> np.ndarray:
    """(3,H,W) float [0,1] decoded frame -> (H,W,3) uint8 as the sim would send it."""
    return np.rint(x * 255.0).astype(np.uint8).transpose(1, 2, 0)


def compare(dump: dict, ds, ckpt: Path) -> dict:
    model = BaselinePolicy.load(ckpt, "cpu")
    seed = int(dump["seed"])
    emap = episode_map(Path(ds.root))
    ep_col = T.column(ds, "episode_index").reshape(-1)
    st_col = T.column(ds, "observation.state").astype(np.float32)
    ac_col = T.column(ds, "action").astype(np.float32)
    K, steps = len(dump["episode_index"]), dump["states"].shape[1]
    slots, unmatched = [], []
    for j in range(K):
        key = (seed, int(dump["episode_index"][j]))
        if key not in emap:
            unmatched.append(key[1])
            continue
        rows = np.flatnonzero(ep_col == emap[key])
        it = ds[int(rows[0])]
        side, wrist = u8(it["observation.images.side"].numpy()), u8(it["observation.images.wrist"].numpy())
        st0, lab = it["observation.state"].numpy().astype(np.float32), TB.episode_labels(it)
        ref = model.predict(it["observation.images.side"].numpy()[None], it["observation.images.wrist"].numpy()[None],
                            st0[None], lab[None])[0]  # the trainer's view of row 0
        srv = dump["chunk0"][j]
        N = srv.shape[0]
        loc = model.predict(dump["side0"][j:j + 1], dump["wrist0"][j:j + 1], dump["state0"][j:j + 1],
                            dump["labels0"][j:j + 1])[0]  # same inputs the server got, CPU
        act = it["action"].numpy()[:N]  # dataset action rows 0..N-1 (r)
        n = min(steps, len(rows))
        dst, ex = st_col[rows[:n]], dump["actions"][j]
        d = np.abs(dump["states"][j][:n] - dst)
        fc_d, fc_e = first_close(ac_col[rows][:, 4]), first_close(ex[:, 4])
        slots.append({
            "slot": j, "episode_index": key[1], "lerobot_episode": emap[key],
            "psnr_side": psnr(dump["side0"][j], side), "psnr_wrist": psnr(dump["wrist0"][j], wrist),
            "state0_max_abs_diff": float(np.abs(dump["state0"][j] - st0).max()),
            "labels_equal": bool(np.array_equal(dump["labels0"][j], lab)),
            "ds_pred_vs_server": {"arm_max_abs": float(np.abs(ref[:, :4] - srv[:, :4]).max()),
                                  "claw_equal": bool(np.array_equal(ref[:, 4], srv[:, 4]))},
            "server_vs_local_bridge": {"arm_max_abs": float(np.abs(loc[:, :4] - srv[:, :4]).max()),
                                       "claw_equal": bool(np.array_equal(loc[:, 4], srv[:, 4]))},
            "vs_dataset_action": {
                name: {"arm_mae": float(np.abs(c[:, :4] - act[:, :4]).mean()),
                       "claw_agreement": float((c[:, 4] == act[:, 4]).mean())}
                for name, c in (("server", srv), ("ds_pred", ref))},
            "first_close": {"executed": fc_e, "dataset": fc_d},
            "state_diff_mean_dims": d.mean(1).tolist(), "state_diff_max_dims": d.max(1).tolist()})
    agg: dict = {"n_slots": K, "n_matched": len(slots), "unmatched_episode_index": unmatched}
    if slots:
        col = lambda f: np.array([f(s) for s in slots])  # noqa: E731
        ps, pw, s0 = col(lambda s: s["psnr_side"]), col(lambda s: s["psnr_wrist"]), col(lambda s: s["state0_max_abs_diff"])
        both = [s for s in slots if s["first_close"]["executed"] >= 0 and s["first_close"]["dataset"] >= 0]
        agg |= {
            "psnr_side": {"min": float(ps.min()), "mean": float(ps.mean())},
            "psnr_wrist": {"min": float(pw.min()), "mean": float(pw.mean())},
            "state0_max_abs_diff": float(s0.max()), "labels_equal_all": bool(col(lambda s: s["labels_equal"]).all()),
            "ds_pred_vs_server": {"arm_max_abs": float(col(lambda s: s["ds_pred_vs_server"]["arm_max_abs"]).max()),
                                  "claw_equal_slots": int(col(lambda s: s["ds_pred_vs_server"]["claw_equal"]).sum())},
            "server_vs_local_bridge": {"arm_max_abs": float(col(lambda s: s["server_vs_local_bridge"]["arm_max_abs"]).max()),
                                       "claw_equal_slots": int(col(lambda s: s["server_vs_local_bridge"]["claw_equal"]).sum())},
            "vs_dataset_action": {name: {"arm_mae_mean": float(col(lambda s: s["vs_dataset_action"][name]["arm_mae"]).mean()),
                                         "claw_agreement_mean": float(col(lambda s: s["vs_dataset_action"][name]["claw_agreement"]).mean())}
                                  for name in ("server", "ds_pred")},
            "first_close": {"executed_never": sum(s["first_close"]["executed"] < 0 for s in slots),
                            "dataset_never": sum(s["first_close"]["dataset"] < 0 for s in slots),
                            "both": len(both),
                            "executed_minus_dataset_median": float(np.median([s["first_close"]["executed"] - s["first_close"]["dataset"] for s in both])) if both else None,
                            "executed_at_step0": sum(s["first_close"]["executed"] == 0 for s in slots)},
            "state_diff_at_step": {str(t): {"mean_dims_mean_slots": float(np.mean([s["state_diff_mean_dims"][t] for s in slots])),
                                            "max_dims_max_slots": float(np.max([s["state_diff_max_dims"][t] for s in slots]))}
                                   for t in CURVE_STEPS if all(len(s["state_diff_mean_dims"]) > t for s in slots)}}
        agg["pass"] = bool(ps.min() >= PSNR_BAR_DB and pw.min() >= PSNR_BAR_DB and s0.max() <= TOL
                           and agg["labels_equal_all"])
        agg["items"] = {"psnr_side": bool(ps.min() >= PSNR_BAR_DB), "psnr_wrist": bool(pw.min() >= PSNR_BAR_DB),
                        "state0": bool(s0.max() <= TOL), "labels": agg["labels_equal_all"]}
    else:
        agg["pass"] = False
    return {"seed": seed, "psnr_bar_db": PSNR_BAR_DB, "tol": TOL, "aggregate": agg, "slots": slots}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-backend", default="pyav")
    a = ap.parse_args()
    dump = dict(np.load(a.dump))
    n = dump["chunk0"].shape[1]
    meta = TB.make_ds(a.dataset, None, n, a.video_backend)
    ds = TB.make_ds(a.dataset, None, n, a.video_backend, meta.fps)
    r = compare(dump, ds, a.ckpt) | {"dump": str(a.dump), "dataset": str(a.dataset), "ckpt": str(a.ckpt),
                                     "git_sha": T.git_sha()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r["aggregate"].items() if k != "unmatched_episode_index"}, indent=1))


if __name__ == "__main__":
    main()
