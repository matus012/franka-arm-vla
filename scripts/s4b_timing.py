"""Baseline timing job -> results/s4b_timing.json (GATES Stage 4 re-plan, "Epochs E_B").

  python scripts/s4b_timing.py --runs <copy of /runs/b4_timing> [--S 1220800] [--out results/s4b_timing.json]

Reads per process <runs>/{A,B0,B1,B2}/steps.jsonl (per step: t = unix time at step end, w / c = wall / compute seconds
of the step, written by vla_policy.train_baseline) and train_summary.json; optionally <runs>/gpu_util.csv with
clock_epoch.txt + clock_local.txt (written by the first step of hpc/steps/s4b_timing.txt: the node's local-time <->
epoch offset for the nvidia-smi timestamps).

Charged rate r (GPU-s per training sample) = (wall of the timed window x 1 GPU) / samples trained in that window by all
processes of the config. The timed window of a process = steps 51..350 (50 warm-up), from the end time of step 50 to the
end time of step 350. A: that window. B: the three windows differ in position, so the OVERLAPPING window
[max_p t_p(50), min_p t_p(350)] is used and only steps that ended inside it count (granularity <= 1 step per process
of 300). Compute-only rate = per-process compute seconds / sample (c summed over the in-window steps), divided by the
number of processes (the GPU time one sample would cost if the GPU were the only limit and fully shared). Data-wait
share = sum(w - c) / sum(w) over the in-window steps of all processes. E_B = min(6, floor(L_B * 3600 / (3 * S * r))).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

WARMUP, LAST = 50, 350
L_B_HOURS = 6.0
S_PLANNING = 1_220_800


def load_steps(d: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["step"] <= LAST]
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1)) and len(rows) == LAST, f"{d}: need steps 1..{LAST}"
    return {k: np.array([r[k] for r in rows], float) for k in ("step", "t", "w", "c")} | {"n": np.array([len(r["index"]) for r in rows])}


def epoch_offset(runs: Path) -> float | None:
    try:
        ep = float((runs / "clock_epoch.txt").read_text().strip())
        loc = datetime.strptime((runs / "clock_local.txt").read_text().strip(), "%Y/%m/%d %H:%M:%S.%f")
    except (FileNotFoundError, ValueError):
        return None
    return loc.timestamp() - ep  # local-naive (read as this machine's local time) minus epoch; undone in gpu_util()


def gpu_util(runs: Path, lo: float, hi: float) -> dict | None:
    off = epoch_offset(runs)
    f = runs / "gpu_util.csv"
    if off is None or not f.exists():
        return None
    ts, ut = [], []
    for line in f.read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            ts.append(datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp() - off)
            ut.append(float(p[1]))
        except (ValueError, IndexError):
            continue
    ts, ut = np.array(ts), np.array(ut)
    m = (ts > lo) & (ts <= hi)
    return {"mean_util_pct": float(ut[m].mean()), "samples": int(m.sum())} if m.any() else None


def config(runs: Path, procs: list[str]) -> dict:
    s = {p: load_steps(runs / p) for p in procs}
    t50 = {p: s[p]["t"][WARMUP - 1] for p in procs}
    t350 = {p: s[p]["t"][LAST - 1] for p in procs}
    lo, hi = max(t50.values()), min(t350.values())
    assert hi > lo, f"timed windows do not overlap: {t50} {t350}"
    wall = hi - lo
    per, samples, wsum, csum = {}, 0, 0.0, 0.0
    for p in procs:
        m = (s[p]["t"] > lo) & (s[p]["t"] <= hi)
        n, w, c = int(s[p]["n"][m].sum()), float(s[p]["w"][m].sum()), float(s[p]["c"][m].sum())
        samples, wsum, csum = samples + n, wsum + w, csum + c
        per[p] = {"window_s": [t50[p], t350[p]], "own_window_wall_s": t350[p] - t50[p], "steps_in_overlap": int(m.sum()),
                  "samples_in_overlap": n, "compute_s_per_sample": c / n, "data_wait_share": (w - c) / w}
    summ = {p: json.loads((runs / p / "train_summary.json").read_text())["timing"] for p in procs}
    return {"processes": procs, "overlap_window_epoch_s": [lo, hi], "overlap_wall_s": wall, "samples_in_window": samples,
            "charged_rate_gpu_s_per_sample": wall * 1.0 / samples,  # 1 GPU
            "compute_only_rate_gpu_s_per_sample": float(np.mean([v["compute_s_per_sample"] for v in per.values()])) / len(procs),
            "data_wait_share": (wsum - csum) / wsum, "gpu_util": gpu_util(runs, lo, hi), "per_process": per,
            "trainer_summary_gpu_s_per_sample": {p: summ[p]["gpu_s_per_sample"] for p in procs},
            "how": "window = [max_p t_p(step 50), min_p t_p(step 350)] from steps.jsonl step-end times; samples = batch rows of "
                   "all processes' steps that ended inside it; rate = window wall x 1 GPU / samples"}


def e_b(r: float, S: int) -> int:
    return min(6, math.floor(L_B_HOURS * 3600 / (3 * S * r)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--S", type=int, default=S_PLANNING, help="Stage 4 union train windows per epoch (planning value 1,220,800)")
    ap.add_argument("--out", type=Path, default=Path("results/s4b_timing.json"))
    a = ap.parse_args()
    cfgs = {"A": config(a.runs, ["A"]), "B": config(a.runs, ["B0", "B1", "B2"])}
    for k, c in cfgs.items():
        c["E_B"] = e_b(c["charged_rate_gpu_s_per_sample"], a.S)
    best = min(cfgs, key=lambda k: cfgs[k]["charged_rate_gpu_s_per_sample"])
    out = {"S": a.S, "L_B_hours": L_B_HOURS, "formula": "E_B = min(6, floor(L_B*3600/(3*S*r_B)))", "configs": cfgs,
           "chosen_config": best, "r_B": cfgs[best]["charged_rate_gpu_s_per_sample"], "E_B": cfgs[best]["E_B"],
           "E_B_lt_3_needs_decode_efficient_loader": cfgs[best]["E_B"] < 3}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("S", "chosen_config", "r_B", "E_B")} | {
        "A": cfgs["A"]["charged_rate_gpu_s_per_sample"], "B": cfgs["B"]["charged_rate_gpu_s_per_sample"]}))


if __name__ == "__main__":
    main()
