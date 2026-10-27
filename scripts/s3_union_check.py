"""GATES 3A / 3B re-plan precondition 3 (union set identity; policy venv, CPU, PERUN cpu_short).

  python scripts/s3_union_check.py --union /data/datasets/0c_s3a_s2_rl224_y15 --task-set /data/datasets/0c_s3a_place_rl224_y15 \
      --pick-set /data/datasets/0c_s2_pick_rl224_y15 --out results/s3a_union_check.json [--kept 2000]

The union export (one LeRobot dataset, `export_dataset.py --src <s3a dirs> <s2 dirs> --kept 2000 --label relabel_setpoint`)
vs the two per-task exports it must reproduce:
  1. every union episode is in exactly one of the two sets, mapped by (seed, episode_index); the union's place / stack
     episodes equal the task set's and its pick episodes equal the pick set's on every sidecar field except the shard
     dir and lerobot_episode_index; counts kept + kept;
  2. split rule per stream (stage, seed): kept rank k is val iff k % 20 == 19; meta/split_113.json agrees;
  3. every row of the union against its HDF5 source (scripts/s0c_check.columns_and_noise, as s0c_rl_check does): state,
     frame_idx, labels, split, task, frame_index exact, `action` == s1_relabel.relabel(action, action_exec); rows per
     episode = T; export counts == sidecar counts.
Exit 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_dataset import VAL_EVERY  # noqa: E402

IGNORED = ("shard", "lerobot_episode_index")


def load_side(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "meta" / "episodes_113.jsonl").read_text().splitlines()]


def by_key(side: list[dict]) -> dict[tuple[int, int], dict]:
    out = {(int(s["seed"]), int(s["episode_index"])): s for s in side}
    assert len(out) == len(side), "duplicate (seed, episode_index) in a sidecar"
    return out


def split_violations(union: list[dict]) -> list[tuple]:
    """Split rule per stream (stage, seed): rank k (episode-index order; a paired stream, task 007: the kept-group rank in
    layout_id order) is val iff k % 20 == 19. Returns [(stream, episode_index)] of the violations."""
    bad = []
    for stream in sorted({(s["file_attrs"]["stage"], int(s["seed"])) for s in union}):
        ss = sorted((s for s in union if (s["file_attrs"]["stage"], int(s["seed"])) == stream),
                    key=lambda s: int(s["episode_index"]))
        order = sorted({int(s["layout_id"]) for s in ss}) if all(s.get("paired") for s in ss) else None
        bad += [(list(stream), s["episode_index"]) for k, s in enumerate(ss)
                if s["split"] != ("val" if (order.index(int(s["layout_id"])) if order else k) % VAL_EVERY == VAL_EVERY - 1
                                  else "train")]
    return bad


def compare(union: list[dict], task: list[dict], pick: list[dict], kept: int, kept_task: int | None = None) -> dict:
    """Pure: sidecar rows -> {item: bool} + diagnostics. `kept_task` (default `kept`): the task stream's own count (task
    007: a paired s3a stream of 3-member groups keeps 3,999 while the pick stream keeps 4,000)."""
    kt = kept if kept_task is None else kept_task
    u, t, p = by_key(union), by_key(task), by_key(pick)
    diffs: dict[str, list] = {"task": [], "pick": []}
    for name, ref in (("task", t), ("pick", p)):
        for k, s in u.items():
            if k in ref:
                r = ref[k]
                bad = sorted(f for f in (set(s) | set(r)) - set(IGNORED) if s.get(f) != r.get(f))
                if bad:
                    diffs[name].append({"key": list(k), "fields": bad})
    uk, tk, pk = set(u), set(t), set(p)
    split_bad = split_violations(union)
    items = {
        "task_set_count": len(t) == kt,
        "pick_set_count": len(p) == kept,
        "union_count": len(u) == kt + kept,
        "task_and_pick_keys_disjoint": not tk & pk,
        "union_keys_eq_task_plus_pick": uk == tk | pk,
        "task_episodes_identical": uk >= tk and not diffs["task"],
        "pick_episodes_identical": uk >= pk and not diffs["pick"],
        "split_rule_per_stream": not split_bad,
        "label_relabel_setpoint": all(s.get("label") == "relabel_setpoint" for s in union),
    }
    return {"items": items, "only_in_union": sorted(uk - tk - pk)[:20], "missing_from_union": sorted((tk | pk) - uk)[:20],
            "task_field_diffs": diffs["task"][:20], "pick_field_diffs": diffs["pick"][:20],
            "split_violations": split_bad[:20], "T_by_task": {k: sorted({int(s["T"]) for s in union if s["task"] == k})
                                                              for k in sorted({s["task"] for s in union})}}


def compare_sets(union: list[dict], sets: list[list[dict]], kept: int) -> dict:
    """N per-stream sets (Stage 4: four): every union episode maps by (seed, episode_index) to exactly one set and equals it
    on every sidecar field except shard / lerobot_episode_index; counts kept per set, N * kept in the union; split rule per
    stream; label relabel_setpoint. S = option-B train windows per epoch, (T - 1) // 3 per train episode, per stream
    (stage, seed) and total."""
    u = by_key(union)
    ref = [by_key(x) for x in sets]
    keysets = [set(r) for r in ref]
    diffs: list[list] = []
    for r in ref:
        d = []
        for k, s in u.items():
            if k in r:
                bad = sorted(f for f in (set(s) | set(r[k])) - set(IGNORED) if s.get(f) != r[k].get(f))
                if bad:
                    d.append({"key": list(k), "fields": bad})
        diffs.append(d)
    uk, allk = set(u), set().union(*keysets)
    split_bad = split_violations(union)
    items = {f"set{i}_count": len(r) == kept for i, r in enumerate(ref)}
    items |= {
        "union_count": len(u) == len(sets) * kept,
        "sets_keys_pairwise_disjoint": sum(len(k) for k in keysets) == len(allk),
        "union_keys_eq_sets": uk == allk,
        **{f"set{i}_episodes_identical": uk >= k and not d for i, (k, d) in enumerate(zip(keysets, diffs))},
        "split_rule_per_stream": not split_bad,
        "label_relabel_setpoint": all(s.get("label") == "relabel_setpoint" for s in union),
    }
    train = [s for s in union if s["split"] == "train"]
    S: dict = {"total": sum((int(s["T"]) - 1) // 3 for s in train), "per_stream": {}, "per_task": {}}
    for s in train:
        key = f"{s['file_attrs']['stage']}_{int(s['seed'])}"
        S["per_stream"][key] = S["per_stream"].get(key, 0) + (int(s["T"]) - 1) // 3
        S["per_task"][s["task"]] = S["per_task"].get(s["task"], 0) + (int(s["T"]) - 1) // 3
    S["train_episodes"] = len(train)
    return {"items": items, "S_option_B_train_windows_per_epoch": S, "only_in_union": sorted(uk - allk)[:20],
            "missing_from_union": sorted(allk - uk)[:20], "set_field_diffs": [d[:20] for d in diffs],
            "split_violations": split_bad[:20]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--union", required=True, type=Path)
    p.add_argument("--task-set", type=Path)
    p.add_argument("--pick-set", type=Path)
    p.add_argument("--sets", nargs="+", type=Path, help="N per-stream sets (instead of --task-set / --pick-set)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--kept", type=int, default=2000)
    p.add_argument("--kept-task", type=int, default=None, help="the task set's own count (default --kept)")
    a = p.parse_args()
    import s0c_check as C
    from s1_relabel import relabel

    union = load_side(a.union)
    if a.sets:
        assert not (a.task_set or a.pick_set), "--sets replaces --task-set / --pick-set"
        res = compare_sets(union, [load_side(d) for d in a.sets], a.kept)
    else:
        res = compare(union, load_side(a.task_set), load_side(a.pick_set), a.kept, a.kept_task)
    exp = json.loads((a.union / "meta" / "export_113.json").read_text())
    cn = C.columns_and_noise(a.union, union, expect=lambda act, exe: relabel(act.astype("float32"), exe.astype("float32")))
    sp = json.loads((a.union / "meta" / "split_113.json").read_text())
    res["items"] |= {
        "columns_exact_action_eq_r": not cn["column_mismatch_episodes"],
        "rows_per_episode_eq_T": not cn["rows_per_episode_mismatch"],
        "export_counts_eq_sidecar": exp["episodes"] == len(union) and exp["rows"] == sum(int(s["T"]) for s in union),
        "split_json_eq_sidecar": sorted(sp["train"]) == sorted(s["lerobot_episode_index"] for s in union if s["split"] == "train")
        and sorted(sp["val"]) == sorted(s["lerobot_episode_index"] for s in union if s["split"] == "val"),
    }
    res |= {"union": str(a.union), "task_set": str(a.task_set), "pick_set": str(a.pick_set), "sets": [str(d) for d in a.sets or []],
            "kept": a.kept, "kept_task": a.kept_task,
            "columns_noise": {k: cn[k] for k in ("column_mismatch_episodes", "rows_per_episode_mismatch", "episodes_checked")},
            "pass": all(res["items"].values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("UNION_CHECK=" + json.dumps({"items": res["items"], "pass": res["pass"]}), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
