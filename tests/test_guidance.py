"""Task 004 instruction-guidance probe (GATES "Inference-time instruction-guidance probe (004)"), CPU only, no model:
other instructions, row expansion, the guided velocity, the patched denoise step, the guided chunk on a fake policy,
the server guide / reset / act contract, the bridge and executor wiring, the eval-policy server argv."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vla_common.catalog import COLORS
from vla_common.guidance import other_instructions
from vla_common.tasks import TEMPLATES, instruction, matches, sample_spec
from vla_policy import guidance as G
from vla_policy.server import NoiseBank, SmolVLAServer, build, make_handler
from vla_tabletop.bridge import ChunkExecutor, PolicyBridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("eval_policy", ROOT / "scripts" / "eval_policy.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


# --------------------------------------------------------------------------- 1. other_instructions
@pytest.mark.parametrize("split", ["train", "paraphrase"])
def test_other_instructions_every_pick_template(split):
    table = ["red", "green", "blue", "yellow"]
    for tpl in TEMPLATES[split]["pick"]:
        for colour in COLORS:
            for size in ("", "small "):
                text = tpl.format(t=f"the {size}{colour} cube")
                got = other_instructions(text, colour, table)
                want = [text.replace(colour, c) for c in table if c != colour]
                assert got == want and len(got) == 3
                assert all((("small" in g) == bool(size)) for g in got)


def test_other_instructions_follows_table_order_and_dedups():
    assert other_instructions("pick up the red cube", "red", ["blue", "red", "green"]) == [
        "pick up the blue cube", "pick up the green cube"]


def test_other_instructions_errors():
    with pytest.raises(ValueError):
        other_instructions("pick up the cube", "red", ["red", "blue"])  # 0 colour words
    with pytest.raises(ValueError):
        other_instructions("put the red cube on the blue pad", "red", ["red", "blue"])  # 2 colour words
    with pytest.raises(ValueError):
        other_instructions("pick up the red cube", "blue", ["red", "blue"])  # target != found
    with pytest.raises(ValueError):
        other_instructions("pick up the red cube", "red", ["red"])  # no other colour
    with pytest.raises(ValueError):
        other_instructions("pick up the red cube", "red", ["red", "red"])


def test_other_instructions_on_real_s2a_specs():
    for i in range(200):
        rng = np.random.default_rng([21640, i])
        spec = sample_spec("s2a", "train", rng)
        text = instruction(spec, rng)
        colours = [c[0] for c in spec.cubes]
        outs = other_instructions(text, spec.target[0], colours)
        assert len(outs) == len(spec.cubes) - 1
        want = [c for c in spec.cubes if c[0] != spec.target[0]]
        assert [m for o in outs for m in matches(spec, o)] == want
        assert all(len(matches(spec, o)) == 1 for o in outs)


# --------------------------------------------------------------------------- 2. expand_rows
def test_expand_rows_layout():
    texts, eor, tr = G.expand_rows(["t0", "t1", "t2"], [["a", "b"], ["c"], ["d", "e", "f"]])
    assert texts == ["t0", "a", "b", "t1", "c", "t2", "d", "e", "f"]
    assert eor.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2] and eor.dtype == np.int64
    assert tr.tolist() == [0, 3, 5] and tr.dtype == np.int64


def test_expand_rows_errors():
    with pytest.raises(ValueError):
        G.expand_rows(["t0", "t1"], [["a"], []])
    with pytest.raises(ValueError):
        G.expand_rows(["t0", "t1"], [["a"]])


# --------------------------------------------------------------------------- 3. combine
def _rows():
    _, eor, tr = G.expand_rows(["t"] * 3, [["a", "b"], ["c"], ["d", "e", "f"]])
    return torch.from_numpy(eor), torch.from_numpy(tr)


def test_combine_formulas():
    eor, tr = _rows()
    v = torch.randn(9, 50, 32, generator=torch.Generator().manual_seed(0))
    assert torch.equal(G.combine(v, eor, tr, 1.0), v[tr][eor])
    for w in (2.0, 4.0):
        out = G.combine(v, eor, tr, w)
        assert out.shape == v.shape
        for e in range(3):
            rows = (eor == e).nonzero().flatten()
            vt, vo = v[tr[e]], v[rows[1:]].mean(0)
            want = vo + w * (vt - vo)
            if w == 2.0:
                assert torch.allclose(want, 2 * vt - vo, atol=1e-6)
            for r in rows:
                assert torch.allclose(out[r], want, atol=1e-5)
            assert all(torch.equal(out[rows[0]], out[r]) for r in rows)


# --------------------------------------------------------------------------- 4. guided_denoise
class _Fake(nn.Module):
    def denoise_step(self, x_t, timestep=None):
        return x_t * 2.0


def test_guided_denoise_patches_and_restores():
    eor, tr = _rows()
    m = _Fake()
    x = torch.randn(9, 4)
    with G.guided_denoise(m, 2.0, eor, tr) as calls:
        out = m.denoise_step(x_t=x)
        m.denoise_step(x_t=x)
        assert calls["n"] == 2
        assert torch.equal(out, G.combine(x * 2.0, eor, tr, 2.0))
    assert m.denoise_step.__func__ is _Fake.denoise_step
    with pytest.raises(RuntimeError):
        with G.guided_denoise(m, 2.0, eor, tr):
            raise RuntimeError("boom")
    assert m.denoise_step.__func__ is _Fake.denoise_step
    assert "denoise_step" not in m.__dict__


# --------------------------------------------------------------------------- 5. guided_chunk on a fake policy
def _tid(texts: list[str]) -> torch.Tensor:
    return torch.tensor([float(sum(map(ord, t)) % 11) for t in texts])


class _Model(nn.Module):
    def denoise_step(self, x_t=None, timestep=None, tid=None, **kw):
        return 0.3 * x_t + 0.1 * tid[:, None, None] + 0.01 * timestep[:, None, None]


class _Policy:
    def __init__(self):
        self.model = _Model()
        self.config = type("C", (), {"num_steps": 10})()
        self.noise_seen = None

    def predict_action_chunk(self, obs, noise):
        self.noise_seen = noise
        x, dt = noise, -1.0 / 10
        t = torch.ones(x.shape[0])
        for _ in range(10):
            x = x + dt * self.model.denoise_step(x_t=x, timestep=t, tid=obs["tid"])
            t = t + dt
        return x[:, :, :7]


def _pre(batch):
    return {"tid": _tid(batch["task"]), "obs": batch["obs"]}


def test_guided_chunk_fake_policy():
    B, chunk, D = 3, 6, 9
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(B, chunk, D, generator=g)
    raw = {"obs": torch.randn(B, 5, generator=g), "task": ["pick up the red cube", "pick up the blue cube",
                                                          "pick up the green cube"]}
    others = [["pick up the blue cube", "pick up the green cube"], ["pick up the red cube"],
              ["pick up the red cube", "pick up the blue cube", "pick up the yellow cube"]]
    pol = _Policy()
    plain = pol.predict_action_chunk(_pre(raw), noise)
    out1 = G.guided_chunk(pol, _pre, raw, others, noise, 1.0)
    assert out1.shape == (B, chunk, 7) and torch.equal(out1, plain)
    eor = G.expand_rows(raw["task"], others)[1]  # all rows of an env got the env's noise
    assert torch.equal(pol.noise_seen, noise[torch.from_numpy(eor)])
    out2 = G.guided_chunk(pol, _pre, raw, others, noise, 2.0)
    assert out2.shape == (B, chunk, 7) and not torch.allclose(out2, plain)
    assert "denoise_step" not in pol.model.__dict__
    for w in (2.0, 4.0, 8.0):  # lead F7: hand-written reference integration of the guided ODE, env by env
        ref = []
        for i in range(B):
            x, dt, t = noise[i:i + 1].clone(), -1.0 / 10, torch.ones(1)
            tt, to = _tid([raw["task"][i]]), _tid(others[i])
            for _ in range(10):
                vt = pol.model.denoise_step(x_t=x, timestep=t, tid=tt)
                vo = torch.stack([pol.model.denoise_step(x_t=x, timestep=t, tid=to[k:k + 1])
                                  for k in range(len(others[i]))]).mean(0)
                x = x + dt * (vo + w * (vt - vo))
                t = t + dt
            ref.append(x[:, :, :7])
        assert torch.allclose(G.guided_chunk(pol, _pre, raw, others, noise, w), torch.cat(ref), atol=1e-5)


# --------------------------------------------------------------------------- 6. server without a model
def _srv(w: float | None = 2.0) -> SmolVLAServer:
    s = object.__new__(SmolVLAServer)
    s.guidance_w, s.others, s.noise = w, {}, NoiseBank((50, 32))
    s.sampling, s.debug, s.image_size = "default", False, 8
    s.cfg = type("C", (), {"n_action_steps": 5})()
    s.pre = lambda b: b
    return s


def _guide_msg(ids=(0, 1), others=None, **extra) -> dict:
    return {"op": "guide", "env_ids": np.array(ids, np.int32), "others": others or [["a"], ["b", "c"]]} | extra


def test_guide_accepts_and_stores():
    s = _srv()
    s.guide(_guide_msg())
    assert s.others == {0: ["a"], 1: ["b", "c"]}


@pytest.mark.parametrize("msg", [
    _guide_msg() | {"env_ids": np.array([0, 1], np.int64)},
    _guide_msg(extra=1),
    _guide_msg(others=[["a"], []]),
    _guide_msg(others=[["a"], [1]]),
    _guide_msg(others=[["a"]]),
])
def test_guide_rejects(msg):
    s = _srv()
    with pytest.raises(ValueError):
        s.guide(msg)
    assert s.others == {}


def test_guide_rejects_without_guidance():
    with pytest.raises(ValueError):
        _srv(None).guide(_guide_msg())


def test_reset_forgets_only_reset_envs():
    s = _srv()
    s.guide(_guide_msg(ids=(0, 1, 2), others=[["a"], ["b"], ["c"]]))
    s.reset({"env_ids": np.array([1], np.int32), "seeds": np.array([5], np.int64)})
    assert set(s.others) == {0, 2}


def test_act_refuses_envs_without_others():
    s = _srv()
    s.noise.reset(np.arange(3, dtype=np.int32), np.arange(3, dtype=np.int64))
    s.guide(_guide_msg(ids=(0,), others=[["a"]]))
    msg = {"env_ids": np.array([0, 1, 2], np.int32), "side": np.zeros((3, 8, 8, 3), np.uint8),
           "wrist": np.zeros((3, 8, 8, 3), np.uint8), "state": np.zeros((3, 19), np.float32),
           "instruction": ["x"] * 3}
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        s.act(msg)


def test_handler_routes_guide():
    s = _srv()
    h = make_handler(s, 1, 0)
    assert h(_guide_msg()) == {"ok": True}
    assert s.others[1] == ["b", "c"]


def test_build_rejects_guidance_for_baseline():
    with pytest.raises(ValueError):
        build("baseline", "x", 5, "cpu", None, image_size=8, guidance_w=2.0)


# --------------------------------------------------------------------------- 7. bridge / executor
class _Client:
    def __init__(self):
        self.msgs = []

    def call(self, msg):
        self.msgs.append(msg)
        return {"ok": True}


def test_bridge_guide_message():
    b = object.__new__(PolicyBridge)
    b.client = _Client()
    b.guide(np.array([0, 1], np.int64), [("a",), ["b", "c"]])
    m = b.client.msgs[0]
    assert m["op"] == "guide" and m["env_ids"].dtype == np.int32 and m["others"] == [["a"], ["b", "c"]]


class _Bridge:
    def __init__(self):
        self.calls = []

    def reset(self, ids, seeds):
        self.calls.append(("reset", ids.tolist()))

    def guide(self, ids, others):
        self.calls.append(("guide", ids.tolist(), others))


def test_executor_sends_guide_after_reset():
    br = _Bridge()
    others = [["a"], ["b"]]
    ex = ChunkExecutor(br, 5, 1, 0, ["x", "y"], others=others)
    ex.reset(np.array([1, 2], np.int64))
    assert br.calls == [("reset", [0, 1]), ("guide", [0, 1], others)]


def test_executor_without_others_never_guides():
    br = _Bridge()
    ChunkExecutor(br, 5, 1, 0, ["x", "y"]).reset(np.array([1, 2], np.int64))
    assert br.calls == [("reset", [0, 1])]


# --------------------------------------------------------------------------- 8. server_args
def test_server_args_guidance_flag():
    c = {"policy": "smolvla", "ckpt": "c", "n": 5, "dataset": None}
    base = E.server_args(c)
    assert "--guidance-w" not in base
    got = E.server_args(c | {"guidance_w": 2.0})
    i = got.index("--guidance-w")
    assert got[i + 1] == "2.0" and got[:i] + got[i + 2:] == base