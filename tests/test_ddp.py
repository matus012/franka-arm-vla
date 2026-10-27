"""Opt-in data parallel in the SmolVLA trainer (`--distributed`): rank slicing of the single-GPU global batches, and a
2-process gloo run (CPU, tiny regression model through the trainer's own train_step) against the single process."""

from __future__ import annotations

import socket

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

import ddp_toy  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_rank_slice_contiguous_and_partial_batch():
    b = list(range(64))
    assert [T.rank_slice(b, r, 4) for r in range(4)] == [b[0:16], b[16:32], b[32:48], b[48:64]]
    part = list(range(100, 148))  # the 48-row last batch of the 220,400-row epoch
    got = [T.rank_slice(part, r, 4) for r in range(4)]
    assert all(len(g) == 12 for g in got) and sum(got, []) == part
    with pytest.raises(AssertionError):
        T.rank_slice(list(range(50)), 0, 4)


def test_rank_batch_sampler_is_the_single_gpu_order_cut_and_sliced():
    n, gbs, world = 3443 * 64 + 48, 64, 4  # the Stage 1 / 2a epoch: 3,443 full batches and one of 48
    ep = np.repeat(np.arange(n // 4 + 1), 4)[:n]
    smp = T.WindowSampler(ep, None, seed=3)
    o = smp.order(0)
    spe = -(-len(o) // gbs)
    assert (len(o), spe, len(o) - (spe - 1) * gbs) == (n, 3444, 48)
    ranks = [T.RankBatchSampler(smp, gbs, r, world) for r in range(world)]
    its = [list(iter(x)) for x in ranks]
    assert all(len(i) == spe for i in its)
    for k in range(spe):
        g = o[k * gbs:(k + 1) * gbs].tolist()
        assert sum((its[r][k] for r in range(world)), []) == g  # ranks together = the global batch, in order
        assert ranks[0].global_rows(k + 1) == g
    assert ranks[0].global_rows(spe + 1) == smp.order(1)[:gbs].tolist()  # next epoch
    with pytest.raises(AssertionError):  # a global batch that world_size does not divide
        T.RankBatchSampler(smp, 64, 0, 3)
    with pytest.raises(AssertionError):  # divisible batch, but the partial last batch (48 rows) is not
        T.RankBatchSampler(T.WindowSampler(ep[:n - 1], None, seed=3), 64, 0, 4)


def test_rank_batch_sampler_resume_position():
    ep = np.arange(21)
    smp = T.WindowSampler(ep, None, seed=1)
    full = list(iter(T.RankBatchSampler(smp, 6, 1, 3)))  # epoch 0, rank 1
    rbs = T.RankBatchSampler(smp, 6, 1, 3)
    rbs.epoch, rbs.start = 0, 2 * 6  # resumed after two global batches
    assert list(iter(rbs)) == full[2:]


def gloo_works() -> str | None:
    """None if torch.distributed gloo can form a group on this machine, else the reason."""
    import torch.distributed as dist
    try:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{free_port()}", rank=0, world_size=1)
        dist.destroy_process_group()
    except RuntimeError as e:
        return str(e).splitlines()[0]
    return None


GLOO = gloo_works()


@pytest.fixture(params=["gloo", "threads"])
def run_ddp(request, tmp_path):
    """2 data-parallel ranks of ddp_toy.train: real gloo processes where torch has gloo, else (and always, second param)
    threads sharing an all_reduce stand-in -- the trainer's own train_step / RankBatchSampler either way."""
    if request.param == "gloo" and GLOO:
        pytest.skip(f"no working gloo in this torch ({GLOO})")

    def run(world: int = 2, **kw) -> list[dict]:
        if request.param == "threads":
            return ddp_toy.train_threads(world, **kw)
        torch.multiprocessing.spawn(ddp_toy.worker, args=(world, free_port(), str(tmp_path / "r"), *kw.values()), nprocs=world)
        return [torch.load(f"{tmp_path / 'r'}.{r}") for r in range(world)]
    return run


def test_ddp_equals_single_process_2_ranks(run_ddp):
    single = ddp_toy.train(0, 1)
    ranks = run_ddp()
    steps = len(single["losses"])
    assert steps == 6 and ddp_toy.N % ddp_toy.GBS == 6  # 3 steps per epoch, the last on the partial batch of 6
    for r in ranks:  # one identical model on every rank
        torch.testing.assert_close(r["params"], ranks[0]["params"], atol=0, rtol=0)
    d = (ranks[0]["params"] - single["params"]).abs().max().item()
    assert d <= 1e-6, d
    np.testing.assert_allclose(ranks[0]["losses"], single["losses"], atol=1e-6)  # the logged loss is the global one
    for k in range(steps):  # the ranks' slices are the single-process batch, in order
        assert ranks[0]["rows"][k] + ranks[1]["rows"][k] == single["rows"][k]


def test_ddp_mutants_are_caught(run_ddp):
    single = ddp_toy.train(0, 1)
    plain = run_ddp(weight_valid=False)  # plain mean of rank means != global valid-weighted loss
    assert (plain[0]["params"] - single["params"]).abs().max().item() > 1e-6
    noreduce = ddp_toy.train(0, 1)  # mutant: rank 0 alone on its slice (no all-reduce) is not the global update
    smp = T.WindowSampler(np.arange(ddp_toy.N), None, seed=5)
    rbs = T.RankBatchSampler(smp, ddp_toy.GBS, 0, 2)
    pol, d = ddp_toy.Toy(), ddp_toy.data()
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-2, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + float(torch.cos(torch.tensor(3.14159 * s / 10)))))
    for _ in range(ddp_toy.EPOCHS):
        for rows in rbs:
            T.train_step(pol, lambda b: b, ddp_toy.batch_of(d, rows), opt, sched, 10.0, torch.device("cpu"), autocast=False)
    p = torch.cat([x.detach().reshape(-1) for x in pol.parameters()])
    assert (p - noreduce["params"]).abs().max().item() > 1e-6
