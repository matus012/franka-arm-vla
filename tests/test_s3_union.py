"""GATES 3A / 3B union set: mixed episode lengths through the exporter + option-B window sampler, the mix instrument
(scripts/s1_tools.py mix) with its first-dataset-only mutant, and scripts/s3_union_check.py (policy venv, CPU)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s1_tools  # noqa: E402
import s3_union_check as U  # noqa: E402
from synth_episodes import write_file  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402
from vla_policy.export import export, load  # noqa: E402

N_PLACE, N_PICK, T_PLACE, T_PICK, BS, EPOCHS = 6, 6, 55, 35, 8, 3  # windows (T - 1) // 3 = 18 / 11


@pytest.fixture(scope="module")
def union(tmp_path_factory):
    d = tmp_path_factory.mktemp("un")
    files = [d / "a.h5", d / "b.h5"]
    write_file(files[0], [T_PLACE] * N_PLACE, seed=0, ep0=0)
    write_file(files[1], [T_PICK] * N_PICK, seed=1, ep0=N_PLACE)

    def select(f, group, at):  # last episode of each stream is val (stand-in for the k % 20 == 19 rule)
        last = at["episode_index"] in (N_PLACE - 1, N_PLACE + N_PICK - 1)
        return at | {"split": "val" if last else "train", "task": "place" if f.name == "a.h5" else "pick"}

    export(files, "un", root=d / "ds", select=select)
    return d / "ds" / "un"


def steps_for(root: Path, seed: int = 0) -> list[dict]:
    """steps.jsonl rows exactly as the trainer writes them: WindowSampler(seed, epoch) over the train split, batches of BS."""
    ds = T.open_dataset(root, "pyav")
    is_val, _, _ = T.split_episodes(ds)
    smp = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), seed, keep=~is_val)
    spe = math.ceil(len(smp) / BS)
    out = []
    for e in range(EPOCHS):
        o = smp.order(e).tolist()
        out += [{"step": e * spe + k + 1, "epoch": e, "index": o[BS * k:BS * k + BS]} for k in range(spe)]
    return out


def args(root: Path, run: Path, mutant=None) -> argparse.Namespace:
    return argparse.Namespace(dataset=str(root), u_run=run, out=run / "mix.json", epochs=EPOCHS, mutant=mutant, batch_size=BS,
                              expect_main=(N_PLACE - 1) * ((T_PLACE - 1) // 3), expect_pick=(N_PICK - 1) * ((T_PICK - 1) // 3))


def test_mixed_T_windows_per_episode_and_sampler(union):
    ds = T.open_dataset(union, "pyav")
    ep, fi = T.column(ds, "episode_index").reshape(-1), T.column(ds, "frame_idx").reshape(-1)
    is_val, train, val = T.split_episodes(ds)
    assert len(train) == 10 and len(val) == 2
    smp = T.WindowSampler(ep, fi, 0, keep=~is_val)
    assert len(smp) == 5 * 18 + 5 * 11
    rows = smp.order(0)
    per_ep = {e: int((ep[rows] == e).sum()) for e in train}
    assert {per_ep[e] for e in train if e < N_PLACE} == {(T_PLACE - 1) // 3} and \
        {per_ep[e] for e in train if e >= N_PLACE} == {(T_PICK - 1) // 3}
    assert not is_val[rows].any() and len(set(rows.tolist())) == len(rows)
    assert math.ceil(1900 * 183 / 64 + 1900 * 116 / 64) >= 8876 and math.ceil((1900 * 183 + 1900 * 116) / 64) == 8877  # the entry's numbers


def test_mix_pass_and_mutants(union, tmp_path):
    steps = steps_for(union)
    run = tmp_path / "run"
    run.mkdir()
    (run / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    s1_tools.mix(args(union, run))
    res = json.loads((run / "mix.json").read_text())
    assert res["pass"] and res["steps_per_epoch"] == 19 and all(v["main_windows"] == 90 and v["pick_windows"] == 55
                                                                 for v in res["per_epoch"].values())
    with pytest.raises(SystemExit):  # sampler over the first stream only
        s1_tools.mix(args(union, run, "first-dataset-only"))
    m = json.loads((run / "mix.json").read_text())
    assert not m["pass"] and m["mutant"] == "first-dataset-only" and m["per_epoch"]["0"]["pick_windows"] == 0


def test_mix_flags_val_row_duplicate_and_missing_epoch(union):
    ds = T.open_dataset(union, "pyav")
    ep, fi = T.column(ds, "episode_index").reshape(-1), T.column(ds, "frame_idx").reshape(-1)
    is_val, _, _ = T.split_episodes(ds)
    is_pick = ep >= N_PLACE
    kw = dict(ep=ep, fi=fi, is_val=is_val, is_pick=is_pick, epochs=EPOCHS, expect_main=90, expect_pick=55, batch=BS)
    steps = steps_for(union)
    assert s1_tools.mix_check(steps, **kw)["pass"]
    bad = copy.deepcopy(steps)
    bad[3]["index"][0] = int(np.flatnonzero(is_val)[0])  # a val row replaces one drawn row
    r = s1_tools.mix_check(bad, **kw)
    assert not r["pass"] and r["per_epoch"][0]["val_rows"] == 1 and r["per_epoch"][0]["missing_windows"] == 1
    dup = copy.deepcopy(steps)
    dup[2]["index"][0] = dup[1]["index"][0]  # same row drawn twice in epoch 0
    assert s1_tools.mix_check(dup, **kw)["per_epoch"][0]["duplicate_windows"] == 1
    assert not s1_tools.mix_check(steps[:-1], **kw)["pass"]  # last step missing
    assert not s1_tools.mix_check(steps, **{**kw, "expect_pick": 56})["pass"]


# ----------------------------------------------------------------------------- s3_union_check.compare
def side_row(seed: int, i: int, task: str, stage: str, rank: int) -> dict:
    return {"seed": seed, "episode_index": i, "task": task, "T": 550 if task != "pick" else 350, "label": "relabel_setpoint",
            "split": "val" if rank % 20 == 19 else "train", "source": f"/d/{stage}/{seed}.h5", "file_attrs": {"stage": stage, "seed": seed},
            "spec_json": f"s{seed}-{i}", "shard": "x", "lerobot_episode_index": 0}


def synth(kept: int = 40):
    task = [side_row(32600, 2 * k, "place", "s3a", k) for k in range(kept)]
    pick = [side_row(31900, 3 * k, "pick", "s2", k) for k in range(kept)]
    union = [dict(s) | {"shard": "u", "lerobot_episode_index": j} for j, s in enumerate(reversed(task + pick))]
    return union, task, pick, kept


def test_compare_clean_and_mutants():
    union, task, pick, kept = synth()
    assert all(U.compare(union, task, pick, kept)["items"].values())
    mut = {
        "pick_episodes_identical": lambda u, t, p: u[0].update(spec_json="other") if u[0]["task"] == "pick" else u[-1].update(spec_json="other"),
        "task_episodes_identical": lambda u, t, p: [x.update(instruction="y") for x in u if x["task"] == "place"][:1],
        "union_count": lambda u, t, p: u.pop(),
        "union_keys_eq_task_plus_pick": lambda u, t, p: u[0].update(episode_index=9999),
        "split_rule_per_stream": lambda u, t, p: [x.update(split="val") for x in u if x["task"] == "pick" and x["episode_index"] == 0],
        "label_relabel_setpoint": lambda u, t, p: u[0].update(label="action"),
        "task_set_count": lambda u, t, p: t.pop(),
    }
    for item, fn in mut.items():
        u, t, p, k = copy.deepcopy(synth())
        fn(u, t, p)
        assert not U.compare(u, t, p, k)["items"][item], item
    u, t, p, k = synth()  # ignored fields may differ
    u[0]["shard"], u[0]["lerobot_episode_index"] = "zzz", 12345
    assert all(U.compare(u, t, p, k)["items"].values())


def test_s2_norm_main_on_mixed_T_union(union, tmp_path):
    """s2_norm.py as-is on the union (--init = a directory with the parent's normalizer / unnormalizer tensors)."""
    from safetensors.numpy import save_file

    import s2_norm

    init = tmp_path / "init"
    init.mkdir()
    st = {f"{k}.{s}": np.full(d, 1.0 if s == "std" else 0.0, np.float32)
          for k, d in (("observation.state", 19), ("action", 5)) for s in ("mean", "std")}
    save_file({k: v for k, v in st.items() if k.startswith("action")},
              str(init / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    save_file(st, str(init / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    out = tmp_path / "norm.json"
    s2_norm.main(["--dataset", str(union), "--init", str(init), "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["train_rows"] == 5 * T_PLACE + 5 * T_PICK and len(r["action"]) == 5 and len(r["observation.state"]) == 19


# ----------------------------------------------------------------------------- N-set generalisation (Stage 4: four sets)
def synth4(kept: int = 40):
    specs = [(32500, "pick", "s2"), (32600, "place", "s3a"), (32700, "stack", "s3b"), (32100, "mixed", "s4")]
    sets = []
    for seed, task, stage in specs:
        rows = []
        for k in range(kept):
            t = task if task != "mixed" else ("pick", "place", "stack")[k % 3]
            rows.append(side_row(seed, 2 * k, t, stage, k))
        sets.append(rows)
    union = [dict(s) | {"shard": "u", "lerobot_episode_index": j} for j, s in enumerate(reversed(sum(sets, [])))]
    return union, sets, kept


def test_compare_sets_clean_S_and_mutants():
    union, sets, kept = synth4()
    res = U.compare_sets(union, sets, kept)
    assert all(res["items"].values()), res["items"]
    S = res["S_option_B_train_windows_per_epoch"]
    n_tr = lambda ss: [s for s in ss if s["split"] == "train"]  # noqa: E731
    want = {"pick": 349 // 3, "place": 549 // 3, "stack": 549 // 3}
    assert S["total"] == sum(want[s["task"]] for ss in sets for s in n_tr(ss)) and S["train_episodes"] == 4 * 38
    assert S["per_stream"]["s2_32500"] == 38 * 116 and S["per_stream"]["s3a_32600"] == 38 * 183
    assert sum(S["per_stream"].values()) == S["total"] == sum(S["per_task"].values())
    mut = {
        "set2_episodes_identical": lambda u, ss: [x.update(spec_json="other") for x in u if x["seed"] == 32700][:1],
        "set3_episodes_identical": lambda u, ss: [x.update(instruction="y") for x in u if x["seed"] == 32100][:1],
        "union_count": lambda u, ss: u.pop(),
        "union_keys_eq_sets": lambda u, ss: u[0].update(episode_index=9999),
        "split_rule_per_stream": lambda u, ss: [x.update(split="val") for x in u if x["seed"] == 32500 and x["episode_index"] == 0],
        "label_relabel_setpoint": lambda u, ss: u[0].update(label="action"),
        "set1_count": lambda u, ss: ss[1].pop(),
        "sets_keys_pairwise_disjoint": lambda u, ss: ss[1].__setitem__(0, dict(ss[0][0])),
    }
    for item, fn in mut.items():
        u, ss, k = copy.deepcopy(synth4())
        fn(u, ss)
        assert not U.compare_sets(u, ss, k)["items"][item], item
    u, ss, k = synth4()  # ignored fields may differ
    u[0]["shard"], u[0]["lerobot_episode_index"] = "zzz", 12345
    assert all(U.compare_sets(u, ss, k)["items"].values())


def test_compare_kept_task_override():
    """Task 007: the task stream keeps its own count (3-member groups: 3,999) while the pick stream keeps --kept."""
    union, task, pick, kept = synth(40)
    task = task[:39]
    union = [dict(s) | {"shard": "u", "lerobot_episode_index": j} for j, s in enumerate(reversed(task + pick))]
    assert not U.compare(union, task, pick, kept)["items"]["task_set_count"]  # without the override: 39 != 40
    r = U.compare(union, task, pick, kept, kept_task=39)["items"]
    assert r["task_set_count"] and r["union_count"] and r["pick_set_count"]
    assert not U.compare(union[:-1], task, pick, kept, kept_task=39)["items"]["union_count"]
