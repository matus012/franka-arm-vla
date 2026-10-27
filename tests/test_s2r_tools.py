"""Task 005 tools (scripts/s2r_tools.py): the margin rule against an independent computation through LeRobot's own
chunk loader, and the steps.jsonl instrument (I2 + void-contrast flag) with its mutants. Policy venv, CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import s2r_tools as S  # noqa: E402
from synth_episodes import episode_arrays  # noqa: E402
from test_contrastive import LAYOUTS, paired_attrs  # noqa: E402
from vla_common.episode import EpisodeWriter  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402

LONG = [(["red", "blue", "green"], [55, 61, 52]), (["yellow", "red"], [58, 53])]


@pytest.fixture(scope="module")
def long_paired(tmp_path_factory):
    from vla_policy.export import export

    d = tmp_path_factory.mktemp("s2r")
    rng = np.random.default_rng(3)
    ep = 0
    with EpisodeWriter(d / "p.h5", {"git_sha": "test", "created": "test", "stage": "s2a"}) as w:
        for g, (cols, lengths) in enumerate(LONG):
            for m, (c, n) in enumerate(zip(cols, lengths)):
                w.add(**episode_arrays(n, ep, rng), attrs=paired_attrs(ep, n, g, m, c, cols))
                ep += 1
    export([d / "p.h5"], "longp", root=d / "ds")
    return d / "ds" / "longp"


def test_margin_equals_lerobot_chunk_loader(long_paired):
    r = S.margin(long_paired)
    ds = T.open_dataset(long_paired, "pyav")
    is_val, train_eps, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    mean, std = (torch.as_tensor(stats["action"][k], dtype=torch.float64) for k in ("mean", "std"))
    ep = T.column(ds, "episode_index").reshape(-1)
    first_row = {int(e): int(np.flatnonzero(ep == e)[0]) for e in train_eps}
    chunk = {e: ((torch.as_tensor(np.asarray(ds[first_row[e]]["action"]), dtype=torch.float64) - mean) / (std + 1e-8))
             for e in train_eps}  # LeRobot's own delta_timestamps chunk (50 x 5)
    assert all(c.shape == (T.CHUNK, 5) for c in chunk.values())
    members = {}
    for g, (cols, _) in enumerate(LONG):
        base = sum(len(c) for c, _ in LONG[:g])
        members.update({base + i: [base + j for j in range(len(cols)) if j != i] for i in range(len(cols))})
    want = np.mean([np.mean([float(((chunk[e] - chunk[o]) ** 2).mean()) for o in members[e]]) for e in train_eps])
    assert r["expected_sq_diff"] == pytest.approx(want, rel=1e-9)
    assert r["margin"] == pytest.approx(0.5 * want, rel=1e-9)
    assert r["ordered_pairs"] == 3 * 2 + 2 * 1 and r["train_layouts"] == 2 and r["action_dims"] == 5
    assert r["task_column_checked"]


def test_margin_refuses_short_episodes(tmp_path_factory):
    """test_contrastive's layouts (episodes of 17-31 rows) are shorter than one chunk."""
    from vla_policy.export import export

    d = tmp_path_factory.mktemp("s2r_short")
    rng = np.random.default_rng(0)
    ep = 0
    with EpisodeWriter(d / "p.h5", {"git_sha": "test", "created": "test", "stage": "s2a"}) as w:
        for g, (cols, lengths) in enumerate(LAYOUTS):
            for m, (c, n) in enumerate(zip(cols, lengths)):
                w.add(**episode_arrays(n, ep, rng), attrs=paired_attrs(ep, n, g, m, c, cols))
                ep += 1
    export([d / "p.h5"], "shortp", root=d / "ds")
    with pytest.raises(AssertionError, match="shorter than one chunk"):
        S.margin(d / "ds" / "shortp")


def write_steps(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def rows(n: int, loss: float, gap: float | None, margin: float) -> list[dict]:
    out = []
    for s in range(1, n + 1):
        r = {"step": s, "loss": loss, "lr": 1e-4, "index": [0]}
        if gap is not None:
            h = max(0.0, margin - gap)
            r |= {"gap": gap, "hinge": h, "hinge_active": float(h > 0), "l_other": loss + gap,
                  "loss_total": loss + h}
        out.append(r)
    return out


def test_steps_i2_and_void_flag(tmp_path):
    m = 0.4
    ok = S.steps_report(write_steps(tmp_path / "a.jsonl", rows(120, 0.01, 0.1, m)), m, 0.1)
    assert ok["i2"]["pass"] and not ok["void"]["void_contrast_flag"] and ok["contrastive_steps"] == 120
    assert ok["first50"]["hinge"] == pytest.approx(0.3) and ok["last50"]["hinge_active"] == 1.0
    # mutant: another init (loss 0.26 at the start, as S1 V15 on the paired set) -> I2 flags it
    bad = S.steps_report(write_steps(tmp_path / "b.jsonl", rows(120, 0.26, 0.1, m)), m, 0.1)
    assert not bad["i2"]["pass"]
    # mutant: l' = l (gap 0, hinge = m on every step) -> void flag
    void = S.steps_report(write_steps(tmp_path / "c.jsonl", rows(120, 0.01, 0.0, m)), m, 0.1)
    assert void["void"]["void_contrast_flag"]
    # a plain run (no contrastive keys) has no void section; fewer than 50 steps -> I2 cannot pass
    plain = S.steps_report(write_steps(tmp_path / "d.jsonl", rows(30, 0.01, None, m)), m, 0.1)
    assert plain["contrastive_steps"] == 0 and "void" not in plain and not plain["i2"]["pass"]


def test_steps_cli_exit_codes(tmp_path):
    m = 0.4
    good = write_steps(tmp_path / "g.jsonl", rows(60, 0.01, 0.1, m))
    bad = write_steps(tmp_path / "b.jsonl", rows(60, 0.26, 0.1, m))
    void = write_steps(tmp_path / "v.jsonl", rows(60, 0.01, 0.0, m))
    base = ["steps", "--margin", str(m), "--i2-max", "0.1", "--out"]
    assert S.main(base + [str(tmp_path / "o1.json"), "--steps", str(good)]) == 0
    assert S.main(base + [str(tmp_path / "o2.json"), "--steps", str(bad)]) == 1
    assert S.main(base + [str(tmp_path / "o3.json"), "--steps", str(bad), "--expect-i2", "fail"]) == 0
    assert S.main(base + [str(tmp_path / "o4.json"), "--steps", str(void)]) == 1


def test_runcheck_exact_margin_lambda_init_lr(tmp_path):
    mj = tmp_path / "margin.json"
    mj.write_text(json.dumps({"margin": 4.210268301913095}))
    good = ("[train] init_sha256=" + S.INIT_034440 + " (/runs/s2a1_VT/checkpoints/034440/pretrained_model)\n"
            "[train] contrastive lambda=1.0 margin=4.210268301913095 share_vision=True layouts=760 episodes_in_table=2000\n")
    (tmp_path / "g.out").write_text(good)
    summ = {"contrastive": {"lambda": 1.0, "margin": 4.210268301913095, "share_vision": True},
            "init_sha256": S.INIT_034440, "argv": {"lr": 1e-4}, "optimizer": {"lr": 1e-4}, "scheduler": {"peak_lr": 1e-4}}
    (tmp_path / "s.json").write_text(json.dumps(summ))
    assert S.runcheck(tmp_path / "g.out", mj, tmp_path / "s.json")["ok"]
    # mutants: a rounded margin, lambda 2, shared vision off, another init, another lr -> each refused
    for old, new in (("margin=4.210268301913095", "margin=4.21"), ("lambda=1.0", "lambda=2.0"),
                     ("share_vision=True", "share_vision=False"), (S.INIT_034440, "c3dcef76" + "0" * 56)):
        (tmp_path / "b.out").write_text(good.replace(old, new))
        assert not S.runcheck(tmp_path / "b.out", mj)["ok"], new
    (tmp_path / "n.out").write_text("[train] step 50\n")  # no contrastive line at all
    assert not S.runcheck(tmp_path / "n.out", mj)["ok"]
    (tmp_path / "s2.json").write_text(json.dumps(summ | {"optimizer": {"lr": 2e-4}}))
    assert not S.runcheck(tmp_path / "g.out", mj, tmp_path / "s2.json")["ok"]


def test_steps_reports_step1_loss(tmp_path):
    r = S.steps_report(write_steps(tmp_path / "a.jsonl", rows(60, 0.01, 0.1, 0.4)), 0.4, 0.1)
    assert r["loss_step1"] == 0.01


def test_margin_through_the_trainers_loss_reduction(long_paired, tmp_path):
    """Consultant pre-launch check (1): m recomputed through the trainer's own path -- LeRobot's preprocessor
    normalization, VLAFlowMatching's flow target u = eps - a and F.mse_loss(reduction="none"), then the hinge's own
    per-sample reduction (contrastive.per_sample_loss) -- with the paired other member's target u_other as the
    "prediction" against u_true (same eps, t, mask, padding). L_other_i - L_true_i must equal the margin rule's
    per-pair mean squared difference, so m = 0.5 x its mean (same unit as the hinge's gap)."""
    import torch.nn.functional as F

    from test_train_smolvla import tiny_init
    from vla_policy import contrastive as C

    r = S.margin(long_paired)
    ds = T.open_dataset(long_paired, "pyav")
    is_val, train_eps, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    policy, pre, _, cfg = T.build_policy(str(tiny_init(tmp_path)), ds, stats, 3, False, False, "cpu")
    ep = T.column(ds, "episode_index").reshape(-1)
    acts, pads = {}, {}
    for e in train_eps:
        b = torch.utils.data.default_collate([ds[int(np.flatnonzero(ep == e)[0])]])
        b = pre(T.to_float_images(b))
        acts[e] = policy.prepare_action(b).double()  # (1, 50, max_action_dim), normalized by LeRobot's processor
        pads[e] = b["action_is_pad"]
    A = cfg.action_feature.shape[0]
    torch.manual_seed(0)
    members = {}
    for g, (cols, _) in enumerate(LONG):
        base = sum(len(c) for c, _ in LONG[:g])
        members.update({base + i: [base + j for j in range(len(cols)) if j != i] for i in range(len(cols))})
    per_ep = []
    for e in train_eps:
        noise = policy.model.sample_noise(acts[e].shape, "cpu").double()
        u_true = noise - acts[e]
        gaps = []
        for o in members[e]:
            u_other = noise - acts[o]
            lt = C.per_sample_loss(F.mse_loss(u_true, u_true, reduction="none"), A, pads[e])
            lo = C.per_sample_loss(F.mse_loss(u_true, u_other, reduction="none"), A, pads[e])
            gaps.append(float(lo - lt))
            h = C.hinge_terms(lt, lo, r["margin"])
            assert float(h["hinge"]) == pytest.approx(max(0.0, r["margin"] - gaps[-1]), abs=1e-9)
        per_ep.append(np.mean(gaps))
    assert 0.5 * np.mean(per_ep) == pytest.approx(r["margin"], rel=1e-5)
    assert float(np.mean(per_ep)) == pytest.approx(r["expected_sq_diff"], rel=1e-5)


def test_early_b_on_loss_total_with_l_true_beside(tmp_path):
    m = 4.2
    rs = rows(200, 0.01, 0.0, m)  # gap 0: loss_total stays at ~m -> ratio 1 -> fires
    r = S.steps_report(write_steps(tmp_path / "a.jsonl", rs), m, 0.1, b_at=200)["early_b"]
    assert r["fires"] and r["loss_total"]["ratio"] == pytest.approx(1.0) and r["loss"]["ratio"] == pytest.approx(1.0)
    rs = rows(200, 0.01, 0.0, m)
    for x in rs[100:]:  # the gap opens past m in the last 100 steps: hinge 0 -> loss_total = L_true -> does not fire
        x.update(gap=5.0, hinge=0.0, hinge_active=0.0, loss_total=x["loss"])
    r = S.steps_report(write_steps(tmp_path / "b.jsonl", rs), m, 0.1, b_at=200)["early_b"]
    assert not r["fires"] and r["loss_total"]["ratio"] < 0.01 and r["loss"]["ratio"] == pytest.approx(1.0)


def test_agree_counts_on_a_hand_built_table():
    base = np.random.default_rng(0).normal(size=(8, 6))
    a, b, c = base.copy(), base.copy(), base.copy()
    b[3:] += 0.5  # the members share frames 0-2 exactly, then b leaves
    c[3] += 5e-5  # c is within tol at frame 3
    assert S.agree_counts([a, b, c], 1e-4) == 3 and S.agree_counts([a, c], 1e-4) == 8
    assert S.agree_counts([a, c], 1e-6) == 3 and S.agree_counts([a, b, c], 1.0) == 8
    assert S.agree_counts([a, c[:2]], 1e-4) == 2  # the shorter member bounds k
    d = S.member_diffs([a, b, c])
    assert d.shape == (8, 6) and np.all(d[:3] == 0) and np.allclose(d[3], 0.5)
    assert np.allclose(S.member_diffs([a, c])[3], 5e-5)


def test_agree_on_the_synthetic_dataset_states_differ_from_frame_0(long_paired, tmp_path):
    r = S.agree(long_paired, 1e-4)
    assert r["groups"] == 2 and r["min"] == 0 and r["max"] == 0 and r["histogram_0_to_10plus"][0] == 2
    assert len(r["median_perdim_maxdiff_frame1"]) == 19 and min(r["median_perdim_maxdiff_frame1"]) > 1e-4
    out = tmp_path / "agree.json"
    assert S.main(["agree", "--dataset", str(long_paired), "--tol", "1e-4", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["count_per_group"] == {"0": 0, "1": 0}


UNION_LAYOUTS = [(["red", "blue", "green"], [52, 55, 51]), (["yellow", "red"], [53, 54]), (["green", "blue"], [56, 50])]


def _h5(path: Path, groups: list[int], rng_seed: int = 7, perturb: int | None = None) -> Path:
    """Episodes of the given layout groups (source episode_index = global position, stable across files)."""
    rng = np.random.default_rng(rng_seed)
    arrays, ep = {}, 0
    for g, (cols, lengths) in enumerate(UNION_LAYOUTS):  # draw every group so each episode's arrays are fixed
        for m, (c, n) in enumerate(zip(cols, lengths)):
            arrays[ep] = (g, m, c, n, cols, episode_arrays(n, ep, rng))
            ep += 1
    with EpisodeWriter(path, {"git_sha": "test", "created": "test", "stage": "s2a"}) as w:
        for e, (g, m, c, n, cols, arr) in arrays.items():
            if g in groups:
                if e == perturb:
                    arr = dict(arr) | {"action": arr["action"] + 0.5}
                w.add(**arr, attrs=paired_attrs(e, n, g, m, c, cols) | {"seed": 32800})
    return path


def _export(d: Path, name: str, files: list[Path], val_groups: set[int], keep: set[int] | None = None) -> Path:
    from vla_policy.export import export

    def sel(f, grp, attrs):
        if keep is not None and int(attrs["layout_id"]) not in keep:
            return None
        return attrs | {"split": "val" if int(attrs["layout_id"]) in val_groups else "train"}
    export(files, name, root=d / "ds", select=sel)
    return d / "ds" / name


@pytest.fixture(scope="module")
def union_sets(tmp_path_factory):
    d = tmp_path_factory.mktemp("union")
    a, b = _h5(d / "a.h5", [0, 1]), _h5(d / "b.h5", [2])
    old = _export(d, "old", [a], {1})
    good = _export(d, "good", [b, a], {1})                 # union, other episode order (as a re-sharded export)
    resplit = _export(d, "resplit", [a, b], set())         # the old val group became train
    dropped = _export(d, "dropped", [a, b], {1}, keep={0, 2})  # an old group is missing
    pert = _export(d, "pert", [_h5(d / "p.h5", [0, 1], perturb=0), b], {1})  # one old episode's labels changed
    return d, old, good, resplit, dropped, pert


def test_union_only_adds_data(union_sets):
    d, old, good, resplit, dropped, pert = union_sets
    r = S.union_check(old, good, every=1)
    assert r["ok"] and r["old_episodes"] == 5 and r["new_episodes"] == 7 and r["union_val_of_old"] == 2
    assert r["label_sha_sample"] == 5 and not r["extra_groups_in_old_layout_range"]


def test_union_mutants_are_flagged(union_sets):
    d, old, good, resplit, dropped, pert = union_sets
    r = S.union_check(old, resplit, every=1)
    assert not r["ok"] and r["n_split_differs"] == 2
    r = S.union_check(old, dropped, every=1)
    assert not r["ok"] and r["n_missing"] == 2
    r = S.union_check(old, pert, every=1)
    assert not r["ok"] and r["label_sha_differs"] == [(32800, 0)]
    old02 = _export(d, "old02", [d / "a.h5", d / "b.h5"], {1}, keep={0, 2})  # group 1 absent from the old set
    r = S.union_check(old02, good, every=1)  # the union has group 1 inside the old layout range, unexplained
    assert not r["ok"] and r["unexplained_extra_groups"] == [[32800, 1]] or r["unexplained_extra_groups"] == [(32800, 1)]
    assert not r["ok"]


def test_runcheck_lr_parameter(tmp_path):
    mj = tmp_path / "margin.json"
    mj.write_text(json.dumps({"margin": 4.210268301913095}))
    log = tmp_path / "g.out"
    log.write_text("[train] init_sha256=" + "ab" * 32 + " (x)\n[train] contrastive lambda=1.0 "
                   "margin=4.210268301913095 share_vision=True layouts=1 episodes_in_table=2\n")
    summ = {"contrastive": {"lambda": 1.0, "margin": 4.210268301913095, "share_vision": True}, "init_sha256": "ab" * 32,
            "argv": {"lr": 5e-5}, "optimizer": {"lr": 5e-5}, "scheduler": {"peak_lr": 5e-5}}
    (tmp_path / "s.json").write_text(json.dumps(summ))
    assert S.runcheck(log, mj, tmp_path / "s.json", init_sha="ab" * 32, lr=5e-5)["ok"]
    assert not S.runcheck(log, mj, tmp_path / "s.json", init_sha="ab" * 32)["ok"]  # the default 1e-4 refuses it


def test_union_with_another_stream_only_adds_data(union_sets, tmp_path):
    """Task 007: a union of the old stream with ANOTHER data seed (layout ids colliding with the old range) is not an
    unexplained extra group; a missing old group is still flagged."""
    import h5py

    d, old, *_ = union_sets
    other = _h5(tmp_path / "o.h5", [0, 1, 2], rng_seed=9)
    with h5py.File(other, "r+") as f:
        for n in f:
            f[n].attrs["seed"] = 33000
    u = _export(tmp_path, "u2", [d / "a.h5", other], {1})
    r = S.union_check(old, u, every=1)
    assert r["ok"] and r["new_episodes"] == 5 + 7 and not r["extra_groups_in_old_layout_range"], r
    u3 = _export(tmp_path, "u3", [d / "a.h5", other], {1}, keep={0, 2})  # old group 1 dropped (both seeds)
    r = S.union_check(old, u3, every=1)
    assert not r["ok"] and r["n_missing"] == 2
