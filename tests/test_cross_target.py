"""Fix candidate for task 005's one diagnosed fix (consultant 113, 2026-10-25; NOT registered, no GATES entry): the
cross-instruction flow target t^2 ||v(x_t, t | o, l') - u'||^2 with u' = (x_t - a_other) / t, on frame-0 rows
(vla_policy.contrastive.cross_terms, train_smolvla --cross-lambda). Consultant's unit tests:
 (i)   lam' = 0 (with lam = 0) is bit-equal to the plain trainer (one step: loss, gradients, RNG; and a whole run);
 (ii)  with x_t built from a_other, u' equals the plain flow target u_other (and the term is 0 at v' = u_other);
 (iii) at the symmetric point v' = v = u_true the term's gradient is nonzero, 2 t (a_other - a_true) / n -- where the
       hinge's gradient wrt v' is exactly 0."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

from test_contrastive import MARGIN, NoSched, Snap, built, common, paired, steps, weights  # noqa: E402,F401
from test_s2r_tools import long_paired  # noqa: E402,F401
from vla_policy import contrastive as C  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402

A, CH, D = 5, 50, 32


def first_chunks(dsdir) -> dict:
    """Every episode's first chunk through LeRobot's own loader (padded by clamping on the short fixture), normalized as
    the trainer's preprocessor does."""
    ds = T.open_dataset(dsdir, "pyav")
    is_val, _, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    mu, sd = (torch.as_tensor(stats["action"][k], dtype=torch.float64) for k in ("mean", "std"))
    ep = T.column(ds, "episode_index").reshape(-1)
    return {int(e): ((torch.as_tensor(np.asarray(ds[int(np.flatnonzero(ep == e)[0])]["action"]), dtype=torch.float64)
                      - mu) / (sd + 1e-8)).float() for e in np.unique(ep)}


def one_step(policy, pre, batch, contrast):
    torch.manual_seed(123)
    opt = Snap(policy)
    loss, _ = T.train_step(policy, pre, copy.deepcopy(batch), opt, NoSched(), 1e9, torch.device("cpu"),
                           contrast=contrast)
    return loss, opt.grads, torch.get_rng_state()


def test_i_lambda_prime_0_bit_equal_to_plain_one_step(built, paired):
    policy, pre, batch, tab = built
    assert int(batch["frame_index"].reshape(-1)[0]) == 0  # the batch holds a frame-0 row, so the term is computed
    first = first_chunks(paired[1])
    l0, g0, r0 = one_step(policy, pre, batch, None)
    c = T.Contrast(tab, 0.0, MARGIN, 0, True, 0.0, first)
    c.step = 1
    l1, g1, r1 = one_step(policy, pre, batch, c)
    assert l1 == l0 and torch.equal(r0, r1) and g0.keys() == g1.keys()
    assert not [n for n in g0 if not torch.equal(g0[n], g1[n])]
    assert c.last["cross_rows"] >= 1 and np.isfinite(c.last["cross"])
    c2 = T.Contrast(tab, 0.0, MARGIN, 0, True, 1.0, first)  # lam' > 0 alone changes the gradients
    c2.step = 1
    _, g2, _ = one_step(policy, pre, batch, c2)
    assert any(not torch.equal(g0[n], g2[n]) for n in g0)


def test_i_lambda_prime_0_run_equals_plain_run(long_paired, tmp_path):
    from test_train_smolvla import tiny_init

    d, dsdir, init = tmp_path, long_paired, tiny_init(tmp_path)  # episodes >= one chunk (the trainer's table asserts it)
    plain = d / "XP"
    T.main(common(dsdir, init) + ["--out", str(plain)])
    T.main(common(dsdir, init) + ["--out", str(d / "XZ"), "--contrastive-lambda", "0", "--contrastive-margin",
                                  str(MARGIN), "--cross-lambda", "0"])
    T.main(common(dsdir, init) + ["--out", str(d / "XO"), "--contrastive-lambda", "1", "--contrastive-margin",
                                  str(MARGIN), "--cross-lambda", "1"])
    assert [r["loss"] for r in steps(plain / "steps.jsonl")] == [r["loss"] for r in steps(d / "XZ" / "steps.jsonl")]
    assert weights(plain) == weights(d / "XZ") and weights(plain) != weights(d / "XO")
    for r in steps(d / "XO" / "steps.jsonl"):
        assert {"cross", "cross_rows"} <= r.keys() and np.isfinite(r["cross"])
        assert r["loss_total"] == pytest.approx(r["loss"] + r["hinge"] + r["cross"], abs=1e-9)


def test_cross_lambda_needs_the_contrastive_flag(paired):
    d, dsdir, init = paired
    with pytest.raises(AssertionError, match="needs --contrastive-lambda"):
        T.main(common(dsdir, init) + ["--out", str(d / "XN"), "--cross-lambda", "1"])


def test_ii_target_equals_plain_flow_target_when_xt_is_built_from_a_other():
    g = torch.Generator().manual_seed(0)
    eps = torch.randn(3, CH, D, generator=g, dtype=torch.float64)
    a_o = torch.randn(3, CH, D, generator=g, dtype=torch.float64)
    t = torch.tensor([0.9, 0.31, 0.002], dtype=torch.float64)
    x_t = t[:, None, None] * eps + (1 - t[:, None, None]) * a_o
    u_prime = (x_t - a_o) / t[:, None, None]
    torch.testing.assert_close(u_prime, eps - a_o, rtol=1e-9, atol=1e-9)  # = VLAFlowMatching's u for a_other
    ct = C.cross_terms(eps - a_o, x_t, t, a_o, A, torch.ones(3, dtype=torch.bool))
    assert float(ct["per"].abs().max()) < 1e-12 and float(ct["n"]) == 3


def test_iii_gradient_nonzero_at_the_symmetric_point_where_the_hinge_gradient_is_zero():
    g = torch.Generator().manual_seed(1)
    eps = torch.randn(2, CH, D, generator=g, dtype=torch.float64)
    a_t = torch.randn(2, CH, D, generator=g, dtype=torch.float64)
    a_o = a_t + torch.randn(2, CH, D, generator=g, dtype=torch.float64)
    t = torch.tensor([0.7, 0.2], dtype=torch.float64)
    x_t = t[:, None, None] * eps + (1 - t[:, None, None]) * a_t
    u = eps - a_t
    v_o = u.clone().requires_grad_(True)  # v' = v = u_true: the language-ignoring fit
    ct = C.cross_terms(v_o, x_t, t, a_o, A, torch.ones(2, dtype=torch.bool))
    (gc,) = torch.autograd.grad(ct["sum"], v_o)
    want = 2 * t[:, None, None] * (a_o - a_t) / (CH * A)
    want[:, :, A:] = 0
    torch.testing.assert_close(gc, want, rtol=1e-9, atol=1e-12)
    assert float(gc[:, :, :A].abs().min()) > 0
    v_o2 = u.clone().requires_grad_(True)
    lo = C.per_sample_loss((v_o2 - u).pow(2), A, None)
    lt = C.per_sample_loss(torch.zeros_like(u), A, None)
    (gh,) = torch.autograd.grad(C.hinge_terms(lt, lo, MARGIN)["hinge"], v_o2)
    assert float(gh.abs().max()) == 0.0  # the hinge gives nothing to push v' away at this point


def test_cross_rows_only_frame_0_and_pick_other_eps_names_the_member(paired):
    tab = C.read_table(paired[1])
    eps_ = [0, 1, 2, 3, 4]
    tasks = [tab[e]["instruction"] for e in eps_]
    for step in range(1, 50):
        instr, oth = C.pick_other_eps(tab, eps_, tasks, [10 * e for e in eps_], 0, step)
        assert instr == C.pick_other(tab, eps_, tasks, [10 * e for e in eps_], 0, step)
        for e, i, o in zip(eps_, instr, oth):
            assert tab[o]["layout"] == tab[e]["layout"] and o != e and tab[o]["target"] in i
    rows = torch.tensor([True, False, True])
    ct = C.cross_terms(torch.zeros(3, CH, D), torch.ones(3, CH, D), torch.full((3,), 0.5), torch.zeros(3, CH, D), A, rows)
    assert float(ct["n"]) == 2 and float(ct["sum"]) == pytest.approx(float(ct["per"][[0, 2]].sum()))


# ---------------------------------------------------------------- --cross-frame0-extra K (frame-0 rows added to the batch)

def run_args(d, dsdir, init, name: str, *extra: str) -> list[str]:
    return common(dsdir, init) + ["--out", str(d / name), "--contrastive-lambda", "1", "--contrastive-margin",
                                  str(MARGIN), "--cross-lambda", "1", *extra]


def test_extra_k0_equals_the_cross_lambda_run_and_k_adds_frame0_rows(long_paired, tmp_path):
    from test_train_smolvla import tiny_init

    d, dsdir, init = tmp_path, long_paired, tiny_init(tmp_path)
    T.main(run_args(d, dsdir, init, "K0"))
    T.main(run_args(d, dsdir, init, "K0b", "--cross-frame0-extra", "0"))
    s0, s0b = steps(d / "K0" / "steps.jsonl"), steps(d / "K0b" / "steps.jsonl")
    assert [(r["loss"], r["cross"], r["index"]) for r in s0] == [(r["loss"], r["cross"], r["index"]) for r in s0b]
    assert weights(d / "K0") == weights(d / "K0b") and not any("extra_index" in r for r in s0)
    K = 3
    summ = T.main(run_args(d, dsdir, init, "K3", "--cross-frame0-extra", str(K)))
    s3 = steps(d / "K3" / "steps.jsonl")
    assert [r["index"] for r in s3] == [r["index"] for r in s0]  # the registered main rows, untouched
    ds = T.open_dataset(dsdir, "pyav")
    fi, ep = T.column(ds, "frame_index").reshape(-1), T.column(ds, "episode_index").reshape(-1)
    is_val, train_eps, _ = T.split_episodes(ds)
    pool = np.flatnonzero((fi == 0) & ~is_val)
    sets = []
    for r in s3:
        x = r["extra_index"]
        assert r["extra_rows"] == K and len(x) == K == len(set(x)) and (fi[x] == 0).all()
        assert set(ep[x].tolist()) <= set(train_eps) and not is_val[x].any()
        assert x == np.random.default_rng([1, r["step"], 240]).choice(pool, K, replace=False).tolist()  # f(seed, step)
        assert r["cross_rows"] >= K and np.isfinite([r["cross"], r["loss"], r["hinge"]]).all()
        sets.append(tuple(sorted(x)))
    assert len(set(sets)) > 1
    assert summ["cross_frame0_extra"]["extra_samples_seen_this_run"] == K * 4 and summ["samples_seen_this_run"] == 16
    assert weights(d / "K3") != weights(d / "K0")
    with pytest.raises(AssertionError, match="needs --cross-lambda"):
        T.main(common(dsdir, init) + ["--out", str(d / "KN"), "--cross-frame0-extra", "2"])


def test_extra_rows_leave_l_true_and_hinge_to_the_main_rows_and_feed_the_cross_term(built, paired):
    policy, pre, batch, tab = built
    d, dsdir, init = paired
    first = first_chunks(dsdir)
    ds = T.open_dataset(dsdir, "pyav")
    fi, ep = T.column(ds, "frame_index").reshape(-1), T.column(ds, "episode_index").reshape(-1)
    xs = [int(np.flatnonzero((ep == e) & (fi == 0))[0]) for e in (1, 3)]  # frame-0 rows of two other episodes
    rows = [0, 1, 2, 3, *xs]  # main = rows 0-3 (episode 0, frames 0-3; frame 0 included), then the 2 extras
    full = next(iter(torch.utils.data.DataLoader(ds, batch_sampler=[rows])))
    assert full["frame_index"].reshape(-1).tolist()[4:] == [0, 0]

    def go(b, n_extra):
        c = T.Contrast(tab, 1.0, MARGIN, 0, True, 1.0, first)
        c.step = 1
        torch.manual_seed(123)
        opt = Snap(policy)
        loss, _ = T.train_step(policy, pre, copy.deepcopy(b), opt, NoSched(), 1e9, torch.device("cpu"), autocast=False,
                               contrast=c, n_extra=n_extra)
        return loss, c.last

    l0, c0 = go(full, 2)
    assert c0["cross_rows"] >= 2 + 1  # both extras + the main frame-0 row
    mut = copy.deepcopy(full)
    mut["action"][4:] = 0.0  # mutant: the extra rows' actions
    l1, c1 = go(mut, 2)
    assert l1 == pytest.approx(l0, rel=1e-6) and c1["hinge"] == pytest.approx(c0["hinge"], rel=1e-6, abs=1e-9)
    assert c1["gap"] == pytest.approx(c0["gap"], rel=1e-5, abs=1e-8) and c1["cross"] != pytest.approx(c0["cross"], rel=1e-3)
    mut2 = copy.deepcopy(full)
    mut2["action"][:4] = 0.0  # a main-row mutant does move L_true (the mutant above is not vacuous)
    assert go(mut2, 2)[0] != pytest.approx(l0, rel=1e-3)
    l2, c2 = go(full, 0)  # without the mask the extras enter L_true and the hinge
    assert l2 != l0 and c2["cross_rows"] == c0["cross_rows"]


@pytest.fixture(scope="module")
def even_paired(tmp_path_factory):
    """long_paired with one longer episode so the window count is even (the DDP sampler needs an even partial batch)."""
    from synth_episodes import episode_arrays
    from test_contrastive import paired_attrs
    from test_s2r_tools import LONG
    from vla_common.episode import EpisodeWriter
    from vla_policy.export import export

    d = tmp_path_factory.mktemp("evenp")
    rng, ep = np.random.default_rng(3), 0
    with EpisodeWriter(d / "p.h5", {"git_sha": "test", "created": "test", "stage": "s2a"}) as w:
        for g, (cols, lengths) in enumerate(LONG):
            for m, (c, n) in enumerate(zip(cols, lengths)):
                n += 3 * (ep == 4)
                w.add(**episode_arrays(n, ep, rng), attrs=paired_attrs(ep, n, g, m, c, cols))
                ep += 1
    export([d / "p.h5"], "evenp", root=d / "ds")
    return d / "ds" / "evenp"


def test_extra_ddp_world2_k4_runs_and_rank_slices_partition_the_extras(even_paired, tmp_path):
    import ddp_toy
    import test_ddp
    from test_train_smolvla import tiny_init

    long_paired = even_paired
    d, dsdir, init = tmp_path, long_paired, tiny_init(tmp_path)
    args = run_args(d, dsdir, init, "DX", "--cross-frame0-extra", "4") + ["--batch-size", "4", "--max-steps", "3",
                                                                          "--distributed", "gloo"]
    if test_ddp.GLOO:
        ddp_toy.main_threads(2, args)
    else:
        torch.multiprocessing.spawn(ddp_toy.trainer_worker, args=(2, test_ddp.free_port(), args), nprocs=2)
    sd = steps(d / "DX" / "steps.jsonl")
    assert len(sd) == 3
    for r in sd:
        assert len(r["index"]) == 4 and len(r["extra_index"]) == 4 == r["extra_rows"] and r["cross_rows"] >= 4
        assert np.isfinite([r["loss"], r["hinge"], r["cross"], r["loss_total"]]).all()
    assert json.loads((d / "DX" / "train_summary.json").read_text())["world_size"] == 2
    ds = T.open_dataset(long_paired, "pyav")
    fi = T.column(ds, "frame_index").reshape(-1)
    smp = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), 1, keep=~T.split_episodes(ds)[0])
    ss = [T.ExtraBatchSampler(smp, 4, r, 2, np.flatnonzero(fi == 0), 4, 1) for r in (0, 1)]
    for step in (1, 2, 3):
        a, b = (s.global_extra(step) for s in ss)
        assert a == b == sd[step - 1]["extra_index"]
    for s in ss:
        s.epoch = 0
    b0, b1 = (next(iter(s)) for s in ss)
    x0, x1 = b0[-2:], b1[-2:]
    assert not set(x0) & set(x1) and x0 + x1 == sd[0]["extra_index"] and b0[:2] + b1[:2] == sd[0]["index"]
