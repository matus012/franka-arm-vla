"""GATES 0A.3 faithfulness tests of the chunked expert (no simulator): the expert's information flow is tested against a
small kinematic stand-in world (TCP follows the commanded target one step late, the target cube rides with the TCP once
the claw is closed around it). (i) open loop: teleporting the target cube 5 cm at a step strictly between two queries
leaves every action of the running chunk bit-identical; (ii) staleness: a teleport in (frame step, t] leaves the chunk
computed at t unchanged, a teleport at or before the frame step changes it; (iii) N = 1 with period 1 / latency 0
reproduces the fresh (unchunked) expert bit-identically on 10 episodes; (iv) the chunk at t uses the action term's
commanded target of the frame step (GATES 0A.3 attempt 1). Mutants: scripts/detector_mutants.py --set expert."""

from __future__ import annotations

import math

import torch

from vision_lift.frames import frame_step
from lift_rl.task_space import BOX_HI, BOX_LO, LEAD_MAX, STEP_POS, STEP_YAW
from vla_tabletop.expert import Expert, Snapshot

E = torch.tensor([0.048, 0.060] * 4)
TZ = -0.003
PERIOD, LATENCY = 3, 1


class FakeWorld:
    """N envs, one target cube each (index target), pads on the table, a bottom cube for stack."""

    def __init__(self, task, target, dest, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.N = n = len(task)
        self.target = torch.tensor(target)
        self.cube = torch.zeros(n, 8, 3)
        self.cube[..., 0] = 15.0
        self.cube[..., 2] = -1.0
        ar = torch.arange(n)
        self.cube[ar, self.target] = torch.stack([0.40 + 0.2 * torch.rand(n, generator=g),
                                                  -0.2 + 0.4 * torch.rand(n, generator=g), TZ + E[self.target] / 2], 1)
        d = torch.tensor(dest)
        st = torch.tensor(task) == 2
        bot = torch.stack([torch.full((n,), 0.62), torch.full((n,), 0.25), TZ + E[d] / 2], 1)
        self.cube[ar[st], d[st]] = bot[st]
        self.yaw = (torch.rand(n, 8, generator=g) - 0.5) * 2
        self.pad = torch.tensor([[0.42, 0.22, TZ + 0.001], [0.62, -0.22, TZ + 0.001], [0.40, -0.05, TZ + 0.001],
                                 [0.62, 0.05, TZ + 0.001]]).expand(n, 4, 3).clone()
        self.tcp = torch.tensor([0.46, 0.0, 0.38]).expand(n, 3).clone()
        self.hyaw = torch.zeros(n)
        self.claw = torch.zeros(n)
        self.grasped = torch.zeros(n, dtype=torch.bool)
        self.term_target = self.tcp.clone()  # the action term's commanded target / yaw (stand-in, with the 5 cm lead clamp)
        self.term_yaw = torch.zeros(n)

    def snap(self) -> Snapshot:
        ar = torch.arange(self.N)
        f = torch.zeros(self.N, 8)
        f[ar, self.target] = torch.where(self.grasped, 6.0, 0.0)
        w = torch.where(self.claw > 0.5, E[self.target], torch.full((self.N,), 0.08))
        return Snapshot(cube_pos=self.cube.clone(), cube_yaw=self.yaw.clone(), pad_pos=self.pad.clone(), lf=f.clone(),
                        rf=f.clone(), tcp=self.tcp.clone(), hand_yaw=self.hyaw.clone(), width=w,
                        finger_vel=torch.zeros(self.N, 2), cmd_target=self.term_target.clone(),
                        cmd_yaw=self.term_yaw.clone())

    def step(self, ex: Expert, exe: torch.Tensor) -> None:
        ar = torch.arange(self.N)
        lo, hi = torch.tensor(BOX_LO), torch.tensor(BOX_HI)
        tt = torch.maximum(torch.minimum(self.term_target + exe[:, :3] * STEP_POS, hi), lo)
        lead = tt - self.tcp
        self.term_target = self.tcp + lead * torch.clamp(LEAD_MAX / torch.linalg.norm(lead, dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        self.term_yaw = self.term_yaw + exe[:, 3] * STEP_YAW
        d = self.term_target - self.tcp  # the arm follows the term's target at <= 1.2 cm per step
        new_tcp = self.tcp + d * torch.clamp(0.012 / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        near = torch.linalg.norm(self.tcp - self.cube[ar, self.target], dim=-1) < 0.012
        self.claw = exe[:, 4]
        self.grasped = (self.claw > 0.5) & (near | self.grasped)
        moved = new_tcp - self.tcp
        self.cube[ar, self.target] += torch.where(self.grasped[:, None], moved, torch.zeros_like(moved))
        self.tcp = new_tcp
        self.hyaw = self.term_yaw.clone()


TASKS = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
TARGETS = [0, 1, 2, 3, 4, 5, 6, 7, 0, 3]
DESTS = [0, 2, 5, 0, 1, 4, 0, 3, 3, 0]


def run(fresh: bool, n_query: int = 10, period: int = PERIOD, latency: int = LATENCY, T: int = 300,
        teleport: tuple[int, float] | None = None, cmd_shift: tuple[int, float] | None = None) -> torch.Tensor:
    w = FakeWorld(TASKS, TARGETS, DESTS)
    ex = Expert(torch.tensor(TASKS), torch.tensor(TARGETS), torch.tensor(DESTS), E, TZ, 0.002, 0.02, fresh=fresh,
                n_query=n_query, period=period, latency=latency)
    ex.reset(w.snap())
    out = []
    for t in range(T):
        if teleport is not None and t == teleport[0]:
            ar = torch.arange(w.N)
            w.cube[ar, w.target, 0] += teleport[1]
        if cmd_shift is not None and t == cmd_shift[0]:
            w.term_target[:, 0] += cmd_shift[1]
        label, exe = ex.act(t, w.snap)
        out.append(exe.clone())
        w.step(ex, exe)
    return torch.stack(out, 1)  # (N, T, 5)


def test_fresh_expert_completes_tasks_in_fake_world():
    a = run(True)
    assert (a[..., 4] > 0.5).any(1).all()  # every env closes the claw at some point
    assert torch.isfinite(a).all() and a[..., :4].abs().max() <= 1.0


def test_n1_period1_latency0_reproduces_fresh():
    assert torch.equal(run(True, T=400), run(False, n_query=1, period=1, latency=0, T=400))


def test_open_loop_within_chunk():
    N = 10
    base = run(False, n_query=N)
    for q in (60, 100, 150):
        for k in (q + 1, q + 4, q + N - 1):
            tel = run(False, n_query=N, teleport=(k, 0.05))
            assert torch.equal(base[:, q:q + N], tel[:, q:q + N]), (q, k)
            assert torch.equal(base[:, :k], tel[:, :k])


def test_staleness_of_query_snapshot():
    N = 10
    base = run(False, n_query=N)
    changed = 0
    for q in (10, 20, 30, 40, 150):
        fs = frame_step(q, PERIOD, LATENCY)
        for k in range(fs + 1, q + 1):  # teleport after the frame step, up to the query: the chunk at q is unchanged
            tel = run(False, n_query=N, teleport=(k, 0.05))
            assert torch.equal(base[:, q:q + N], tel[:, q:q + N]), (q, k, fs)
        tel = run(False, n_query=N, teleport=(fs, 0.05))  # at the frame step: the chunk at q sees it
        changed += int(not torch.equal(base[:, q:q + N], tel[:, q:q + N]))
    assert changed >= 4  # during the approach every action depends on where the cube is


def test_commanded_target_from_frame_step():
    """(iv): overwriting the action term's commanded target at a step in (frame step, t] leaves the chunk computed at t
    unchanged; at the frame step it changes it."""
    N = 10
    base = run(False, n_query=N)
    changed = 0
    for q in (10, 20, 30, 40, 150):
        fs = frame_step(q, PERIOD, LATENCY)
        for k in range(fs + 1, q + 1):
            tel = run(False, n_query=N, cmd_shift=(k, 0.03))
            assert torch.equal(base[:, q:q + N], tel[:, q:q + N]), (q, k, fs)
        tel = run(False, n_query=N, cmd_shift=(fs, 0.03))
        changed += int(not torch.equal(base[:, q:q + N], tel[:, q:q + N]))
    assert changed == 5


def test_frame_step_rule():
    assert [frame_step(t, PERIOD, LATENCY) for t in (0, 2, 3, 5, 6, 10, 20)] == [0, 0, 0, 0, 3, 6, 15]
    assert math.isclose(0.0, 0.0)
