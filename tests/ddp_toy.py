"""Workers for tests/test_ddp.py (spawned processes import this module): a tiny regression 'policy' trained with the
trainer's own train_step / RankBatchSampler / rank_slice, single process or gloo data parallel."""

from __future__ import annotations

import os
import threading

import torch

from vla_policy import train_smolvla as T

N, D, H, GBS = 22, 8, 4, 8  # 22 rows: global batches 8, 8, 6 per epoch (partial last batch kept)
EPOCHS = 2
LOCK = threading.Lock()  # Toy() seeds the global RNG: serialise it when the ranks are threads


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(torch.nn.Linear(D, 16), torch.nn.Tanh(), torch.nn.Linear(16, H))

    def forward(self, b: dict) -> tuple[torch.Tensor, dict]:
        """LeRobot's loss shape: sum of squared errors over the valid entries / number of valid entries."""
        err = (self.net(b["x"]) - b["y"]) ** 2
        valid = ~b["action_is_pad"]
        return (err * valid).sum() / valid.sum().clamp_min(1), {}


def data() -> dict:
    g = torch.Generator().manual_seed(1)
    pad = torch.rand(N, H, generator=g) < 0.4  # uneven valid counts between rows (and so between rank slices)
    return {"x": torch.randn(N, D, generator=g), "y": torch.randn(N, H, generator=g), "action_is_pad": pad}


def batch_of(d: dict, rows: list[int]) -> dict:
    b = {k: v[rows].clone() for k, v in d.items()}
    return b | {k: torch.zeros(len(rows), 1) for k in T.RENAME_MAP}  # to_float_images wants the two camera keys


def train(rank: int, world: int, weight_valid: bool = True) -> dict:
    d = data()
    with LOCK:
        policy = Toy()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + float(torch.cos(torch.tensor(3.14159 * s / 10)))))
    smp = T.WindowSampler(torch.arange(N).numpy(), None, seed=5)
    rbs = T.RankBatchSampler(smp, GBS, rank, world)
    losses, seen = [], []
    for _ in range(EPOCHS):
        for rows in rbs:
            seen.append(rows)
            loss, gn = T.train_step(policy, lambda b: b, batch_of(d, rows), opt, sched, 10.0, torch.device("cpu"),
                                    autocast=False, world=world, weight_valid=weight_valid)
            losses.append(loss)
    return {"params": torch.cat([p.detach().reshape(-1) for p in policy.parameters()]), "losses": losses, "rows": seen}


def worker(rank: int, world: int, port: int, out: str, weight_valid: bool = True) -> None:
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    r = train(rank, world, weight_valid)
    dist.destroy_process_group()
    torch.save(r, f"{out}.{rank}")


def trainer_worker(rank: int, world: int, port: int, argv: list[str]) -> None:
    """vla_policy.train_smolvla.main under --distributed gloo (the torchrun env, set by hand)."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))
    T.main(argv)


class ThreadGroup:
    """Stand-in for the torch.distributed calls the trainer makes, between `world` threads of this process (for machines
    whose torch has no working gloo, e.g. the Windows cu130 wheel: "makeDeviceForHostname(): unsupported gloo device").
    all_reduce is SUM in the same rank order on every rank; the calling thread's rank is `ThreadGroup.rank.v`."""

    rank = threading.local()

    def __init__(self, world: int):
        self.world, self.barrier_, self.buf = world, threading.Barrier(world), [None] * world

    def _exchange(self, x):
        self.buf[self.rank.v] = x
        self.barrier_.wait()
        out = list(self.buf)
        self.barrier_.wait()
        return out

    def all_reduce(self, t: torch.Tensor, *a, **k) -> None:
        xs = self._exchange(t.detach().clone())
        t.copy_(sum(xs[1:], xs[0]))

    def all_gather_object(self, out: list, obj) -> None:
        out[:] = self._exchange(obj)

    def barrier(self, *a, **k) -> None:
        self.barrier_.wait()


class ThreadEnviron(dict):
    """os.environ for the trainer's main(): RANK / WORLD_SIZE / LOCAL_RANK of the calling thread."""

    def __init__(self, world: int):
        super().__init__()
        self.world = world

    def __getitem__(self, k: str) -> str:
        return str({"RANK": ThreadGroup.rank.v, "LOCAL_RANK": ThreadGroup.rank.v, "WORLD_SIZE": self.world}[k])


def run_threads(world: int, fn, **kw) -> list:
    """fn(rank, world, **kw) on `world` threads with torch.distributed's collectives (and the trainer's os.environ) patched."""
    import types
    from concurrent.futures import ThreadPoolExecutor

    import torch.distributed as dist

    g = ThreadGroup(world)
    names = ("init_process_group", "destroy_process_group", "all_reduce", "all_gather_object", "barrier")
    orig, ostar = {n: getattr(dist, n) for n in names}, T.os
    for n in names:
        setattr(dist, n, getattr(g, n, lambda *a, **k: None))
    T.os = types.SimpleNamespace(environ=ThreadEnviron(world))

    def one(r: int):
        ThreadGroup.rank.v = r
        return fn(r, world, **kw)
    try:
        with ThreadPoolExecutor(world) as ex:
            return list(ex.map(one, range(world)))
    finally:
        T.os = ostar
        for n, f in orig.items():
            setattr(dist, n, f)


def train_threads(world: int, **kw) -> list[dict]:
    return run_threads(world, train, **kw)


def main_threads(world: int, argv: list[str]) -> list[dict]:
    """vla_policy.train_smolvla.main(argv) as `world` data-parallel ranks (argv carries --distributed gloo)."""
    real = T.build_policy
    barrier = threading.Barrier(world)

    def build(*a, **k):  # from_pretrained flips the PROCESS-wide default dtype: build one at a time, then wait for the rest
        with LOCK:
            out = real(*a, **k)
        barrier.wait()
        return out
    T.build_policy = build
    try:
        return run_threads(world, lambda r, w: T.main(argv))
    finally:
        T.build_policy = real
