"""Task 005 (GATES "Stage 2R — instruction-contrastive flow loss") tools, policy venv, CPU.

  margin  the hinge margin m by the GATES 005 rule, from the paired training set (before training):
          m = 0.5 x E || a_true - a_other ||^2, over the TRAIN layout groups, first chunk (rows 0 .. 49 of each
          episode = the first query's label chunk), the 5 action dims (4 arm + claw) normalized as the trainer's loss
          sees them (MEAN_STD from the train rows, LeRobot's (x - mean) / (std + 1e-8)), ||.||^2 = the MEAN over the
          50 x 5 entries (the per-sample flow loss is a mean over the same entries). E = mean over train episodes e of
          the mean over e's other group members o (the training distribution of l': uniform over the others).
          Groups = contrastive.layout_table's (stage, data seed, layout_id). --stage S (task 007, union sets): only the
          train episodes of stage S's groups enter E (the normalization stays the whole set's train rows, as the trainer
          normalizes the union); default: every group, unchanged.
  steps   the run's per-step contrastive log (steps.jsonl): L_true / gap / hinge / active fraction per window, the
          void-contrast flag (spec item (iii)), and instrument I2 (mean L_true over steps 1-50 <= --i2-max).
  union   task 006 data gate (consultant): the union set only ADDS data -- every episode of the old set is in the union
          (matched by source identity: data seed + source episode_index, with the same layout_id / member), with the same
          split, and the same label rows (sha of `action` + `observation.state` per episode) on a sample; union groups
          inside the old layout range that the old set lacks are listed (allowed only if the old export skipped them;
          only groups of the old set's data seeds count -- a union with another stream, task 007, adds other seeds).
  runcheck  lead F3: the trainer ran with exactly the recorded m (results/s2r/margin.json, repr-equal), lambda 1.0,
          share_vision true, the 034440 init sha and lr 1e-4 -- read from the job log's [train] start lines (at the
          step-100 guard) and, when given, from train_summary.json (after training).
  agree   fix candidate (cross-target, not registered): per paired layout group, how many leading frames k = 0, 1, 2, ...
          have the members' `observation.state` agreeing (max abs difference over members and dims <= --tol) -- the frames
          where the other member's chunk is still the right target for this observation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_policy import contrastive as C  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402

LEROBOT_EPS = 1e-8


def margin(dataset: Path, backend: str = "pyav", stage: str | None = None) -> dict:
    ds = T.open_dataset(Path(dataset), backend)
    is_val, train_eps, val_eps = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    act = T.column(ds, "action").astype(np.float64).reshape(len(ds), -1)
    ep = T.column(ds, "episode_index").reshape(-1)
    tix = T.column(ds, "task_index").reshape(-1)
    task = [ds.meta.tasks.iloc[int(i)].name for i in tix]  # LeRobot's own row -> task string map (dataset_reader)
    A = act.shape[1]
    mean = stats["action"]["mean"].astype(np.float64)
    std = stats["action"]["std"].astype(np.float64)
    an = (act - mean) / (std + LEROBOT_EPS)
    starts = {int(e): int(i) for i, e in zip(*np.unique(ep, return_index=True)[::-1])}
    table = C.read_table(Path(dataset))
    train = {e for e in train_eps if stage is None or table[e]["group"][0] == stage}
    assert train, f"no train episodes of stage {stage}"
    groups: dict[tuple, list[int]] = {}
    for e, r in table.items():
        groups.setdefault(r["group"], []).append(e)
    for g, mem in groups.items():  # a layout is wholly in one split (data gate item P(iv))
        assert len({e in train for e in mem}) == 1, f"layout {g} straddles train / val"
    first = {}
    for e in train:
        s = starts[e]
        assert np.all(ep[s:s + T.CHUNK] == e), f"episode {e} shorter than one chunk"
        first[e] = an[s:s + T.CHUNK]
    per_ep, per_dim, pairs = [], [], 0
    for e in sorted(train):
        mem = [o for o in groups[table[e]["group"]] if o != e]
        assert mem, f"episode {e}: no other member"
        d = [((first[e] - first[o]) ** 2).mean() for o in mem]
        per_dim.append(np.mean([((first[e] - first[o]) ** 2).mean(0) for o in mem], 0))
        per_ep.append(float(np.mean(d)))
        pairs += len(mem)
    per_ep = np.array(per_ep)
    gap = float(per_ep.mean())
    return {
        "rule": "m = 0.5 x mean over train episodes of mean over other members of mean_{50 x 5}((a_e - a_o)^2), "
                "first chunk, action normalized MEAN_STD (train rows), LeRobot eps 1e-8",
        "dataset": str(dataset), **({"stage": stage} if stage else {}), "train_episodes": len(train),
        "val_episodes": len(val_eps), "train_layouts": len({table[e]["group"] for e in train}), "ordered_pairs": pairs, "action_dims": A,
        "chunk": T.CHUNK, "expected_sq_diff": gap, "margin": 0.5 * gap,
        "per_episode_quantiles": {q: float(np.quantile(per_ep, q / 100)) for q in (10, 25, 50, 75, 90)},
        "frac_episodes_below_margin": float((per_ep < 0.5 * gap).mean()),
        "per_dim_expected_sq_diff": np.mean(per_dim, 0).tolist(),
        "action_mean": mean.tolist(), "action_std": std.tolist(),
        "task_column_checked": all(task[starts[e]].strip() == table[e]["instruction"].strip() for e in train),
    }


def window_means(rows: list[dict], keys: list[str], width: int) -> list[dict]:
    out = []
    for s in range(0, len(rows), width):
        w = rows[s:s + width]
        out.append({"steps": [w[0]["step"], w[-1]["step"]], **{k: float(np.mean([r[k] for r in w])) for k in keys}})
    return out


def early_b(rows: list[dict], at: int) -> dict:
    """GATES Stage 2R early abort (b) (consultant R1): mean over the last 100 steps up to `at` (the 25 % save) vs 50 % of
    the mean over steps 1-50, read on loss_total (fires iff ratio >= 0.5); L_true's ratio reported beside it."""
    by = {r["step"]: r for r in rows}
    first = [by[s] for s in range(1, 51)]
    last = [by[s] for s in range(at - 99, at + 1)]
    out = {"at_step": at}
    for k in ("loss_total", "loss"):
        a, b = float(np.mean([r[k] for r in first])), float(np.mean([r[k] for r in last]))
        out[k] = {"mean_steps_1_50": a, f"mean_steps_{at - 99}_{at}": b, "ratio": b / a}
    out["fires"] = out["loss_total"]["ratio"] >= 0.5
    return out


def steps_report(path: Path, margin_: float, i2_max: float, width: int = 1000, b_at: int | None = None) -> dict:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    rows = sorted({r["step"]: r for r in rows}.values(), key=lambda r: r["step"])  # a resume rewrites no step twice
    first50 = [r["loss"] for r in rows if 1 <= r["step"] <= 50]
    i2 = {"mean_loss_steps_1_50": float(np.mean(first50)) if len(first50) == 50 else None, "max": i2_max}
    i2["pass"] = i2["mean_loss_steps_1_50"] is not None and i2["mean_loss_steps_1_50"] <= i2_max
    out = {"steps_jsonl": str(path), "steps": len(rows), "last_step": rows[-1]["step"] if rows else None, "i2": i2,
           "loss_step1": rows[0]["loss"] if rows and rows[0]["step"] == 1 else None}  # lead F7: the pure init read
    crows = [r for r in rows if "gap" in r]
    out["contrastive_steps"] = len(crows)
    if crows:
        keys = ["loss", "l_other", "gap", "hinge", "hinge_active", "loss_total"]
        out["windows"] = window_means(crows, keys, width)
        out["first50"] = window_means(crows[:50], keys, 50)[0]
        out["last50"] = window_means(crows[-50:], keys, 50)[0]
        out["void"] = C.step_flags(crows, margin_)
        if b_at is not None:
            out["early_b"] = early_b(crows, b_at)
    return out


def agree_counts(states_by_member: list[np.ndarray], tol: float) -> int:
    """Leading frames k = 0, 1, ... at which all members' state rows agree: max abs difference across members over all
    dims <= tol (the shorter member bounds k). Stops at the first disagreement."""
    d = member_diffs(states_by_member)
    bad = np.flatnonzero(d.max(1) > tol)
    return int(bad[0]) if len(bad) else len(d)


def member_diffs(states_by_member: list[np.ndarray]) -> np.ndarray:
    """(min length, D): per frame and dim, the max abs difference across members (max - min)."""
    n = min(len(x) for x in states_by_member)
    s = np.stack([np.asarray(x, np.float64)[:n] for x in states_by_member])
    return s.max(0) - s.min(0)


def agree(dataset: Path, tol: float, backend: str = "pyav") -> dict:
    ds = T.open_dataset(Path(dataset), backend)
    st = T.column(ds, "observation.state").astype(np.float64).reshape(len(ds), -1)
    ep = T.column(ds, "episode_index").reshape(-1)
    table = C.read_table(Path(dataset))
    groups: dict[tuple, list[int]] = {}
    for e, r in table.items():
        groups.setdefault(r["group"], []).append(e)
    counts, d1, d2 = {}, [], []
    for g, mem in sorted(groups.items()):
        states = [st[ep == e] for e in mem]
        counts[g[2] if len({k[:2] for k in groups}) == 1 else "/".join(map(str, g))] = agree_counts(states, tol)
        d = member_diffs(states)
        if len(d) > 2:
            d1.append(d[1]), d2.append(d[2])
    c = np.array(list(counts.values()))
    return {"dataset": str(dataset), "tol": tol, "groups": len(c), "count_per_group": counts,
            "quantiles": {q: float(np.quantile(c, q / 100)) for q in (0, 10, 25, 50, 75, 90, 100)},
            "min": int(c.min()), "max": int(c.max()),
            "histogram_0_to_10plus": [int((c == k).sum()) for k in range(10)] + [int((c >= 10).sum())],
            "median_perdim_maxdiff_frame1": np.median(d1, 0).tolist() if d1 else None,
            "median_perdim_maxdiff_frame2": np.median(d2, 0).tolist() if d2 else None}


def _side(ds_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (Path(ds_dir) / "meta" / "episodes_113.jsonl").read_text().splitlines()]


def _label_cols(ds) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """episode_index + the label columns, each read ONCE (a full-column read per sampled episode does not scale)."""
    return T.column(ds, "episode_index").reshape(-1), {k: T.column(ds, k) for k in ("action", "observation.state")}


def _ep_sha(cols: tuple[np.ndarray, dict[str, np.ndarray]], e: int) -> str:
    import hashlib
    ep_col, data = cols
    rows = np.flatnonzero(ep_col == e)
    h = hashlib.sha256()
    for k in ("action", "observation.state"):
        h.update(np.ascontiguousarray(data[k][rows]).astype(np.float32).tobytes())
    return h.hexdigest()


def union_check(old: Path, new: Path, every: int = 40, backend: str = "pyav") -> dict:
    so, sn = _side(old), _side(new)
    key = lambda r: (int(r["seed"]), int(r["episode_index"]))  # noqa: E731
    on, nn = {key(r): r for r in so}, {key(r): r for r in sn}
    assert len(on) == len(so) and len(nn) == len(sn), "duplicate source identity in a sidecar"
    missing = sorted(set(on) - set(nn))
    split_diff = [k for k in on if k in nn and on[k]["split"] != nn[k]["split"]]
    pair_diff = [k for k in on if k in nn and (int(on[k]["layout_id"]), int(on[k]["member"]))
                 != (int(nn[k]["layout_id"]), int(nn[k]["member"]))]
    old_layouts = {(int(r["seed"]), int(r["layout_id"])) for r in so}
    max_old = max(l for _, l in old_layouts)
    old_seeds = {sd for sd, _ in old_layouts}
    extra_in_range = sorted({(int(r["seed"]), int(r["layout_id"])) for r in sn if int(r["seed"]) in old_seeds
                             and int(r["layout_id"]) <= max_old and (int(r["seed"]), int(r["layout_id"])) not in old_layouts})
    old_skipped = []
    sp = Path(old) / "meta" / "split_113.json"
    if sp.exists():
        for rep_ in (json.loads(sp.read_text()).get("paired") or {}).values():
            old_skipped += [int(g["group"]) for g in rep_.get("skipped_groups", [])]
    unexplained = [g for g in extra_in_range if g[1] not in old_skipped]
    co, cn = _label_cols(T.open_dataset(Path(old), backend)), _label_cols(T.open_dataset(Path(new), backend))
    sample = sorted(on)[::every]
    sha_bad = [k for k in sample if k in nn and _ep_sha(co, int(on[k]["lerobot_episode_index"]))
               != _ep_sha(cn, int(nn[k]["lerobot_episode_index"]))]
    out = {"old": str(old), "new": str(new), "old_episodes": len(so), "new_episodes": len(sn),
           "missing_in_union": missing[:20], "n_missing": len(missing), "split_differs": split_diff[:20],
           "n_split_differs": len(split_diff), "layout_member_differs": pair_diff[:20],
           "old_val_episodes": sum(r["split"] == "val" for r in so),
           "union_val_of_old": sum(nn[k]["split"] == "val" for k in on if k in nn),
           "extra_groups_in_old_layout_range": extra_in_range, "old_export_skipped_groups": old_skipped,
           "unexplained_extra_groups": unexplained, "label_sha_sample": len(sample), "label_sha_differs": sha_bad}
    out["ok"] = bool(not missing and not split_diff and not pair_diff and not unexplained and not sha_bad)
    return out


INIT_034440 = "07e87cc6f5b2533ab4b931e356e7cf944e6b53d49bbf2cc0109ab416290cb3ff"


def runcheck(log: Path, margin_json: Path, summary: Path | None = None, lam: float = 1.0,
             init_sha: str = INIT_034440, lr: float = 1e-4) -> dict:
    want_m = json.loads(Path(margin_json).read_text())["margin"]
    text = Path(log).read_text(errors="replace")
    c = re.findall(r"\[train\] contrastive lambda=(\S+) margin=(\S+) share_vision=(\S+)", text)
    sha = re.findall(r"\[train\] init_sha256=([0-9a-f]{64})", text)
    out = {"log": str(log), "margin_recorded": want_m, "log_contrastive": c, "log_init_sha256": sha}
    ok_log = (len(c) >= 1 and all(float(x[0]) == lam and float(x[1]) == want_m and x[2] == "True" for x in c)
              and len(sha) >= 1 and all(x == init_sha for x in sha))
    out["log_ok"] = bool(ok_log)
    ok = ok_log
    if summary is not None:
        t = json.loads(Path(summary).read_text())
        cs = t.get("contrastive") or {}
        lrs = {k: v for k, v in (("argv_lr", t["argv"].get("lr")), ("optimizer_lr", (t.get("optimizer") or {}).get("lr")),
                                 ("scheduler_peak_lr", (t.get("scheduler") or {}).get("peak_lr"))) if v is not None}
        out["summary"] = {"contrastive": cs, "init_sha256": t.get("init_sha256"), "lr": lrs}
        out["summary_ok"] = bool(cs.get("margin") == want_m and cs.get("lambda") == lam and cs.get("share_vision") is True
                                 and t.get("init_sha256") == init_sha and lrs and all(v == lr for v in lrs.values()))
        ok = ok and out["summary_ok"]
    out["ok"] = bool(ok)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("margin")
    m.add_argument("--dataset", type=Path, required=True)
    m.add_argument("--video-backend", default="pyav")
    m.add_argument("--stage", default=None, help="union sets: only this stage's groups (s2a / s2 / s3a / s3b); default all")
    m.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("steps")
    s.add_argument("--steps", type=Path, required=True, help="the run's steps.jsonl")
    s.add_argument("--margin", type=float, required=True)
    s.add_argument("--i2-max", type=float, required=True)
    s.add_argument("--b-at", type=int, default=None, help="early abort (b) at this step (the 25 %% save, 8610)")
    s.add_argument("--expect-i2", choices=["pass", "fail"], default="pass",
                   help="fail = a mutant read (another init's run) that the check must flag")
    s.add_argument("--out", type=Path, required=True)
    ag = sub.add_parser("agree")
    ag.add_argument("--dataset", type=Path, required=True)
    ag.add_argument("--tol", type=float, default=1e-4)
    ag.add_argument("--video-backend", default="pyav")
    ag.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("union")
    u.add_argument("--old", type=Path, required=True)
    u.add_argument("--new", type=Path, required=True)
    u.add_argument("--every", type=int, default=40, help="label-sha sample: every k-th old episode (source order)")
    u.add_argument("--out", type=Path, required=True)
    rc = sub.add_parser("runcheck")
    rc.add_argument("--log", type=Path, required=True, help="the training job's stdout (rank 0 [train] lines)")
    rc.add_argument("--margin-json", type=Path, required=True)
    rc.add_argument("--summary", type=Path, default=None)
    rc.add_argument("--init-sha", default=INIT_034440, help="expected init sha256 (default: 2a attempt 1's 034440)")
    rc.add_argument("--lr", type=float, default=1e-4, help="expected resolved peak lr (Stage 2M: 5e-5)")
    rc.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "margin":
        r = margin(a.dataset, a.video_backend, a.stage)
        ok = r["task_column_checked"] and r["margin"] > 0
    elif a.cmd == "agree":
        r = agree(a.dataset, a.tol, a.video_backend)
        ok = True
    elif a.cmd == "union":
        r = union_check(a.old, a.new, a.every)
        ok = r["ok"]
    elif a.cmd == "runcheck":
        r = runcheck(a.log, a.margin_json, a.summary, init_sha=a.init_sha, lr=a.lr)
        ok = r["ok"]
    else:
        r = steps_report(a.steps, a.margin, a.i2_max, b_at=a.b_at)
        ok = r["i2"]["pass"] == (a.expect_i2 == "pass")
        if a.expect_i2 == "pass" and r.get("void", {}).get("void_contrast_flag"):
            ok = False
    r["ok"] = bool(ok)
    a.out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k not in ("windows", "per_dim_expected_sq_diff", "count_per_group")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
