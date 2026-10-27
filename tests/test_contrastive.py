"""Task 005 (GATES "Stage 2R"): the instruction-contrastive flow loss (vla_policy.contrastive + train_smolvla
`--contrastive-lambda`). Policy venv, CPU, the tiny SmolVLA of tests/test_train_smolvla.py on a tiny PAIRED dataset
(two layouts: 3 and 2 members, distinct target colours, colour-named instructions).

Spec items (task 005 "Unit tests first"):
 (i)   lam = 0: loss and gradients equal the plain trainer bit for bit on a fixed batch (also the RNG state after the
       step, with and without the shared vision tower), and a whole lam = 0 run equals a plain run (steps + weights);
 (ii)  l' is never l and always names another member target of the same layout (and the guard catches violations);
 (iii) a mutant with l' = l makes the hinge constant (= m, gap 0) on every step and `step_flags` flags it;
 (iv)  both branches see the same x_t / t / eps (recorded at embed_suffix and asserted; mutants raise).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

from synth_episodes import LATENCY, PERIOD, episode_arrays  # noqa: E402
from test_train_smolvla import tiny_init  # noqa: E402
from vla_common.episode import EpisodeWriter  # noqa: E402
from vla_policy import contrastive as C  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402

LAYOUTS = [(["red", "blue", "green"], [20, 31, 17]), (["yellow", "red"], [25, 22])]
MARGIN = 0.5


def paired_attrs(ep: int, T_: int, layout: int, member: int, target: str, cubes: list[str]) -> dict:
    spec = {"task": "pick", "target": [target, "small"], "dest": None, "cubes": [[c, "small"] for c in cubes],
            "pads": ["red", "green", "blue", "yellow"]}
    return {"instruction": f"pick up the {target} cube", "split": "train", "task": "pick", "spec_json": spec,
            "labels": [0, ep % 4, 0, 12], "seed": 32800, "env_id": ep, "success": True, "metrics_json": {},
            "period": PERIOD, "latency": LATENCY, "image_size": 48, "visual_mode": "nominal",
            "action_noise_sigma": 0.1, "expert_version": "synthetic", "episode_index": ep, "T": T_,
            "n_action_steps": 10, "detector_version": "none", "paired": True, "layout_id": layout, "member": member}


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    from vla_policy.export import export

    d = tmp_path_factory.mktemp("contrastive")
    rng = np.random.default_rng(0)
    ep = 0
    with EpisodeWriter(d / "p.h5", {"git_sha": "test", "created": "test", "stage": "s2a"}) as w:
        for g, (cols, lengths) in enumerate(LAYOUTS):
            for m, (c, n) in enumerate(zip(cols, lengths)):
                w.add(**episode_arrays(n, ep, rng), attrs=paired_attrs(ep, n, g, m, c, cols))
                ep += 1
    export([d / "p.h5"], "tinyp", root=d / "ds")
    init = tiny_init(d)
    return d, d / "ds" / "tinyp", init


def common(dsdir: Path, init: Path) -> list[str]:
    return ["--dataset", str(dsdir), "--init", str(init), "--device", "cpu", "--n-action-steps", "3", "--epochs", "1",
            "--max-steps", "4", "--batch-size", "4", "--seed", "1", "--video-backend", "pyav"]


def steps(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


# ---------------------------------------------------------------- (ii) the l' choice

def test_layout_table_from_the_exported_sidecar(paired):
    tab = C.read_table(paired[1])
    assert sorted(tab) == [0, 1, 2, 3, 4]
    assert tab[0] == {"instruction": "pick up the red cube", "target": "red", "layout": 0, "others": ["blue", "green"],
                      "group": ("s2a", 32800, 0)}  # + the (stage, seed, layout_id) group key (task 007)
    assert tab[4]["others"] == ["yellow"] and tab[3]["layout"] == 1


def test_pick_other_never_l_always_another_member_uniform_deterministic(paired):
    tab = C.read_table(paired[1])
    eps = [0, 1, 2, 3, 4]
    tasks = [tab[e]["instruction"] for e in eps]
    seen = {e: set() for e in eps}
    for step in range(1, 200):
        o = C.pick_other(tab, eps, tasks, [10 * e for e in eps], seed=0, step=step)
        assert o == C.pick_other(tab, eps, tasks, [10 * e for e in eps], seed=0, step=step)  # pure function
        for e, t, x in zip(eps, tasks, o):
            assert x != t
            (col,) = C.COLOUR_RE.findall(x)
            assert col in tab[e]["others"] and x == t.replace(tab[e]["target"], col)  # same template
            seen[e].add(col)
    assert all(seen[e] == set(tab[e]["others"]) for e in eps)  # every other member is drawn


def test_guard_catches_l_equals_l_foreign_colour_and_misaligned_episode(paired):
    tab = C.read_table(paired[1])
    with pytest.raises(AssertionError, match="contrast is void"):
        C.check_others(tab, [0], ["pick up the red cube"], ["pick up the red cube"])
    with pytest.raises(AssertionError, match="does not name exactly one other cube"):
        C.check_others(tab, [0], ["pick up the red cube"], ["pick up the yellow cube"])  # yellow is not in layout 0
    with pytest.raises(AssertionError, match="sidecar instruction"):
        C.pick_other(tab, [0], ["pick up the blue cube"], [0], 0, 1)


def test_layout_table_rejects_bad_groups():
    def row(ep, layout, member, target, cubes, paired=True):
        return {"lerobot_episode_index": ep, "layout_id": layout, "member": member, "paired": paired,
                "instruction": f"pick up the {target} cube",
                "spec_json": json.dumps({"target": [target, "small"], "cubes": [[c, "small"] for c in cubes]})}
    with pytest.raises(ValueError, match="distinct targets"):
        C.layout_table([row(0, 0, 0, "red", ["red", "blue"]), row(1, 0, 1, "red", ["red", "blue"])])
    with pytest.raises(ValueError, match="distinct targets"):
        C.layout_table([row(0, 0, 0, "red", ["red", "blue"])])
    with pytest.raises(ValueError, match="another scene"):
        C.layout_table([row(0, 0, 0, "red", ["red", "blue"]), row(1, 0, 1, "blue", ["green", "blue"])])
    with pytest.raises(ValueError, match="not a paired"):
        C.layout_table([row(0, 0, 0, "red", ["red", "blue"], paired=False)])


# ---------------------------------------------------------------- (i) lam = 0 is the plain trainer

class Snap:
    """A stand-in optimizer: step() records the clipped gradients, zero_grad() clears them (train_step's order)."""

    def __init__(self, policy):
        self.policy, self.grads = policy, None
        self.param_groups = [{"lr": 0.0}]

    def step(self):
        self.grads = {n: p.grad.detach().clone() for n, p in self.policy.named_parameters() if p.grad is not None}

    def zero_grad(self, set_to_none=True):
        self.policy.zero_grad(set_to_none=set_to_none)


class NoSched:
    def step(self):
        pass


@pytest.fixture(scope="module")
def built(paired):
    d, dsdir, init = paired
    ds = T.open_dataset(dsdir, "pyav")
    is_val, train_eps, _ = T.split_episodes(ds)
    stats, _ = T.norm_stats(ds, ~is_val)
    policy, pre, post, cfg = T.build_policy(str(init), ds, stats, 3, False, False, "cpu")
    T.set_train(policy, False)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(dl))
    return policy, pre, batch, C.read_table(dsdir)


def one_step(policy, pre, batch, contrast, autocast: bool = True) -> tuple[float, dict, torch.Tensor]:
    torch.manual_seed(123)
    opt = Snap(policy)
    loss, _ = T.train_step(policy, pre, copy.deepcopy(batch), opt, NoSched(), 1e9, torch.device("cpu"),
                           autocast=autocast, contrast=contrast)
    return loss, opt.grads, torch.get_rng_state()


@pytest.mark.parametrize("share", [True, False])
def test_lambda0_loss_grads_rng_bit_identical_to_plain(built, share):
    policy, pre, batch, tab = built
    l0, g0, r0 = one_step(policy, pre, batch, None)
    c = T.Contrast(tab, 0.0, MARGIN, 0, share)
    c.step = 1
    l1, g1, r1 = one_step(policy, pre, batch, c)
    assert l1 == l0
    assert g0.keys() == g1.keys() and len(g0) > 0
    bad = [n for n in g0 if not torch.equal(g0[n], g1[n])]
    assert not bad, bad[:5]
    assert torch.equal(r0, r1)  # the same RNG draws: noise / time from LeRobot's samplers, nothing else
    assert c.last["loss_total"] == l0 and 0 <= c.last["hinge_active"] <= 1
    # lam > 0 does change the gradients (the hinge is live on this batch: the untrained model ignores the words)
    c2 = T.Contrast(tab, 1.0, MARGIN, 0, share)
    c2.step = 1
    l2, g2, _ = one_step(policy, pre, batch, c2)
    assert l2 == l0 and c2.last["hinge_active"] > 0
    assert any(not torch.equal(g0[n], g2[n]) for n in g0)


@pytest.mark.parametrize("autocast", [False, True])
def test_shared_vision_equals_separate_and_runs_the_tower_once(built, autocast):
    """fp32: the gradients agree to float32 summation order (measured global relative difference 1.6e-7). bf16
    autocast (the training setting): the vision-tower gradient is accumulated once vs twice in bf16, so the two differ
    by bf16 rounding (measured worst 0.74 % of a tensor's max |grad|, ~ one bf16 epsilon) -- tolerance 2 bf16 eps."""
    policy, pre, batch, tab = built
    calls = {"n": 0}
    f = policy.model.vlm_with_expert.embed_image
    out = {}
    for share in (True, False):
        calls["n"] = 0

        def counted(x):
            calls["n"] += 1
            return f(x)
        policy.model.vlm_with_expert.embed_image = counted
        try:
            c = T.Contrast(tab, 1.0, MARGIN, 0, share)
            c.step = 1
            loss, g, _ = one_step(policy, pre, batch, c, autocast)
        finally:
            del policy.model.vlm_with_expert.embed_image
        out[share] = (loss, g, dict(c.last), calls["n"])
    assert out[True][3] == 2 and out[False][3] == 4  # 2 cameras: once per camera vs once per camera per branch
    assert out[True][0] == out[False][0]
    for k in ("hinge", "gap", "l_other"):
        assert out[True][2][k] == pytest.approx(out[False][2][k], abs=1e-6)
    g1, g2 = out[True][1], out[False][1]
    assert g1.keys() == g2.keys()
    num = sum(float(((g1[n] - g2[n]) ** 2).sum()) for n in g1) ** 0.5
    den = sum(float((g2[n] ** 2).sum()) for n in g1) ** 0.5
    if not autocast:
        assert num / den < 1e-5, num / den
    else:
        for n in g1:
            assert (g1[n] - g2[n]).abs().max() <= 2 * 2 ** -7 * g2[n].abs().max() + 1e-12, n


# ---------------------------------------------------------------- (iv) same x_t / t / eps

def test_branches_share_xt_t_eps_and_mutants_raise(built):
    policy, pre, batch, tab = built
    raw = T.to_float_images(copy.deepcopy(batch))
    c = T.Contrast(tab, 1.0, MARGIN, 0, True)
    c.step = 1
    bt, bo = pre(dict(raw)), pre(dict(raw) | {"task": c.others(raw)})
    assert not torch.equal(bt["observation.language.tokens"], bo["observation.language.tokens"])
    torch.manual_seed(5)
    r = C.flow_pair(policy, bt, bo)
    torch.manual_seed(5)
    noise = policy.model.sample_noise(r["x_t"].shape, "cpu")
    t = policy.model.sample_time(r["x_t"].shape[0], "cpu")
    a = policy.prepare_action(bt)
    torch.testing.assert_close(r["x_t"], t[:, None, None] * noise + (1 - t[:, None, None]) * a, atol=0, rtol=0)
    assert torch.equal(r["t"], t)
    # mutant 1: the second branch draws its own noise -> different x_t -> raises
    f = policy.forward
    n = {"k": 0}

    def own_noise(b, noise=None, time=None, **kw):
        n["k"] += 1
        return f(b, noise=None if n["k"] == 2 else noise, time=time, **kw)
    policy.forward = own_noise
    try:
        with pytest.raises(AssertionError, match="different x_t"):
            C.flow_pair(policy, bt, bo)
    finally:
        del policy.forward
    # mutant 2: another scene / chunk in the second branch -> raises before any forward
    bo2 = dict(bo) | {"action": bo["action"] + 1}
    with pytest.raises(AssertionError, match="branch inputs differ at action"):
        C.flow_pair(policy, bt, bo2)


# ---------------------------------------------------------------- whole runs: lam = 0 == plain; logs; (iii) mutant

@pytest.fixture(scope="module")
def runs(paired):
    d, dsdir, init = paired
    base = common(dsdir, init)
    plain = T.main(base + ["--out", str(d / "P")])
    zero = T.main(base + ["--out", str(d / "Z"), "--contrastive-lambda", "0", "--contrastive-margin", str(MARGIN)])
    one = T.main(base + ["--out", str(d / "O"), "--contrastive-lambda", "1", "--contrastive-margin", str(MARGIN)])
    return d, plain, zero, one


def weights(run_dir: Path) -> bytes:
    last = (run_dir / "checkpoints" / "last.txt").read_text().strip()
    return (run_dir / "checkpoints" / last / "pretrained_model" / "model.safetensors").read_bytes()


def test_lambda0_run_equals_plain_run(runs):
    d, plain, zero, one = runs
    sp, sz = steps(d / "P" / "steps.jsonl"), steps(d / "Z" / "steps.jsonl")
    assert [r["loss"] for r in sp] == [r["loss"] for r in sz] and [r["index"] for r in sp] == [r["index"] for r in sz]
    assert weights(d / "P") == weights(d / "Z")
    assert weights(d / "O") != weights(d / "P")


def test_contrastive_run_logs_and_summary(runs):
    d, plain, zero, one = runs
    so = steps(d / "O" / "steps.jsonl")
    assert len(so) == 4
    for r in so:
        assert {"hinge", "gap", "hinge_active", "l_other", "loss_total"} <= r.keys()
        assert r["loss_total"] == pytest.approx(r["loss"] + 1.0 * r["hinge"], abs=1e-9)
        assert 0 <= r["hinge"] <= MARGIN + max(0.0, -r["gap"]) + 1e-6 and 0 <= r["hinge_active"] <= 1
    assert one["contrastive"]["lambda"] == 1.0 and one["contrastive"]["margin"] == MARGIN
    assert one["contrastive"]["layouts"] == 2 and one["contrastive"]["last50_mean"]["hinge"] >= 0
    assert "contrastive" not in plain
    assert not C.step_flags(so, MARGIN)["void_contrast_flag"]
    with pytest.raises(AssertionError, match="resume with different settings"):  # run_key carries the settings
        T.main(common(d / "ds" / "tinyp", d / "init") + ["--out", str(d / "O"), "--resume", "--contrastive-lambda",
                                                         "2", "--contrastive-margin", str(MARGIN)])


def test_mutant_l_prime_equals_l_gives_constant_hinge_and_is_flagged(paired, monkeypatch):
    d, dsdir, init = paired
    monkeypatch.setattr(C, "pick_other", lambda table, episodes, tasks, rows, seed, step: list(tasks))
    T.main(common(dsdir, init) + ["--out", str(d / "M"), "--contrastive-lambda", "1", "--contrastive-margin",
                                  str(MARGIN)])
    sm = steps(d / "M" / "steps.jsonl")
    assert all(r["gap"] == 0.0 and r["hinge"] == MARGIN and r["hinge_active"] == 1.0 for r in sm)
    assert C.step_flags(sm, MARGIN)["void_contrast_flag"]
    # and the trainer's own path refuses l' == l (pick_other's guard), so the mutant only runs with the guard bypassed
    tab = C.read_table(dsdir)
    with pytest.raises(AssertionError, match="contrast is void"):
        C.check_others(tab, [0], [tab[0]["instruction"]], [tab[0]["instruction"]])


def test_ddp_contrastive_runs_and_logs_global_values(paired):
    """3 data-parallel ranks (threads with patched collectives, or gloo processes): the contrastive step under
    --distributed runs, rank 0 logs the global batch and finite hinge / gap / active values."""
    import ddp_toy
    import test_ddp

    d, dsdir, init = paired
    args = common(dsdir, init) + ["--batch-size", "6", "--max-steps", "3", "--distributed", "gloo", "--out",
                                  str(d / "DD"), "--contrastive-lambda", "1", "--contrastive-margin", str(MARGIN)]
    if test_ddp.GLOO:
        ddp_toy.main_threads(3, args)
    else:
        torch.multiprocessing.spawn(ddp_toy.trainer_worker, args=(3, test_ddp.free_port(), args), nprocs=3)
    sd = steps(d / "DD" / "steps.jsonl")
    assert len(sd) == 3 and all(len(r["index"]) == 6 for r in sd)
    assert all(np.isfinite([r["hinge"], r["gap"], r["hinge_active"], r["loss_total"]]).all() for r in sd)
    s = json.loads((d / "DD" / "train_summary.json").read_text())
    assert s["world_size"] == 3 and s["contrastive"]["lambda"] == 1.0
