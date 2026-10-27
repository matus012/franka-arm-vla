"""Instruction swap instruments (policy venv, CPU, no model): the donor permutation, the paired bootstrap and the
`vloss --task-swap` loop against a stub policy whose loss depends on the instruction text."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s1_tools  # noqa: E402
from vla_common.swap import donors, swap_report  # noqa: E402


def test_donors_permutation_and_targets():
    tg = [["red", "small"]] * 5 + [["blue", "large"]] * 4 + [["green", "small"]] * 3
    ins = [f"pick up {t[0]} {t[1]} cube {i % 2}" for i, t in enumerate(tg)]
    d = donors(ins, tg)
    assert sorted(d) == list(range(12)) and (d != np.arange(12)).all()
    r = swap_report(ins, tg, d)
    assert r["frac_different_target"] == 1.0 and r["frac_same_instruction"] == 0.0
    assert (donors(ins, tg) == d).all()


def test_donors_scene_and_unavoidable():
    tg = [("a", 1), ("b", 1), ("c", 1), ("d", 1)]
    cubes = [[("a", 1), ("b", 1)], [("b", 1), ("c", 1)], [("c", 1), ("d", 1)], [("d", 1), ("a", 1)]]
    d = donors(list("wxyz"), tg, cubes)
    assert swap_report(list("wxyz"), tg, d, cubes)["frac_donor_target_absent_from_scene"] == 1.0
    d1 = donors(["p", "p", "q"], [("a", 1)] * 3)  # one target only: still no own / equal instruction
    assert d1[2] != 2 and d1[0] == 2 and d1[1] == 2 or sorted(d1) == [0, 1, 2]
    assert [i for i in range(3) if d1[i] == i] == []


def test_paired_bootstrap_ci():
    d = np.random.default_rng(1).normal(0.5, 1.0, 400)
    lo, hi = s1_tools.paired_bootstrap(d)
    assert lo < d.mean() < hi and hi - lo < 0.4
    assert s1_tools.paired_bootstrap(d) == [lo, hi]
    lo0, hi0 = s1_tools.paired_bootstrap(np.zeros(50) + 0.25, np.arange(50) // 5)
    assert lo0 == hi0 == pytest.approx(0.25)


class StubDS:
    def __init__(self, root: Path, eps: list[int], text: dict[int, str]):
        self.root, self.items = root, []
        for e in eps:
            for t in range(40):
                self.items.append({"index": len(self.items), "episode_index": e, "frame_index": t, "task": text[e],
                                   "action_is_pad": np.zeros(50, bool),
                                   **{k: np.zeros((3, 4, 4), np.uint8) for k in ("observation.images.side", "observation.images.wrist")}})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class StubPolicy:
    config = types.SimpleNamespace(action_feature=types.SimpleNamespace(shape=(5,)))
    prepare_images = prepare_state = prepare_action = staticmethod(lambda b: (b["obs"], b["obs"])) 

    def __init__(self):
        self.model = types.SimpleNamespace(forward=self.fwd)

    def fwd(self, images, masks, tok, tm, state, actions, noise, time):  # noqa: D102
        v = tok[:, :1].float() + 0.01 * torch.rand(1).item()  # seeded draw: same for every variant of a batch
        return v[:, :, None].expand(-1, 50, 8).clone()

    def forward(self, b):
        ls = self.fwd(None, None, b["observation.language.tokens"], None, None, None, None, None)[:, :, :5] * (~b["action_is_pad"])[..., None]
        return ls.mean(), {}


import torch  # noqa: E402


@pytest.fixture
def stub_run(tmp_path, monkeypatch):
    from vla_policy import train_smolvla as T
    import vla_policy.server as S

    eps = [3, 5, 8, 9]
    text = {3: "pick up the red small cube", 5: "pick up the blue small cube", 8: "pick up the red small cube please", 9: "pick up the green large cube"}
    tgt = {3: ["red", "small"], 5: ["blue", "small"], 8: ["red", "small"], 9: ["green", "large"]}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "episodes_113.jsonl").write_text("".join(json.dumps(
        {"lerobot_episode_index": e, "instruction": text[e], "spec_json": json.dumps({"target": tgt[e]})}) + "\n" for e in [3, 5, 8, 9]))
    ds = StubDS(tmp_path, eps, text)
    ep = np.array([x["episode_index"] for x in ds.items])
    monkeypatch.setattr(T, "open_dataset", lambda *a: ds)
    monkeypatch.setattr(T, "resolve_dataset", lambda *a: tmp_path)
    monkeypatch.setattr(T, "split_episodes", lambda d: (np.ones(len(d), bool), [], eps))
    monkeypatch.setattr(T, "column", lambda d, n: ep if n == "episode_index" else np.zeros(len(d), int))
    monkeypatch.setattr(T, "WindowSampler", lambda *a, **k: types.SimpleNamespace(order=lambda e: np.arange(len(ds))))
    monkeypatch.setattr(T, "to_float_images", lambda b: b)
    monkeypatch.setattr(T, "git_sha", lambda: "stub")

    class Srv:
        ckpt_sha = "x"
        policy = StubPolicy()
        seen: list = []

        def __init__(self, *a):
            pass

        def pre(self, b):
            Srv.seen.append(list(b["task"]))
            tok = torch.tensor([[float(len(t))] for t in b["task"]])  # instruction-dependent "tokens"
            return {"observation.language.tokens": tok, "observation.language.attention_mask": tok, "obs": 0, "action_is_pad": b["action_is_pad"]}

    monkeypatch.setattr(S, "SmolVLAServer", Srv)
    monkeypatch.setattr(torch.utils.data, "DataLoader", lambda sub, batch_size, shuffle, num_workers: (
        torch.utils.data.default_collate([sub[i] for i in range(j, min(j + batch_size, len(sub)))])
        for j in range(0, len(sub), batch_size)))
    monkeypatch.setattr(torch.utils.data, "Subset", lambda d, rows: [d[i] for i in rows])
    Srv.seen = []

    def run(swap, per_sample=False):
        Srv.seen.clear()
        out = tmp_path / f"o_{swap}.json"
        s1_tools.vloss(argparse.Namespace(dataset="x", ckpt="c", device="cpu", action_override=None, task_swap=swap,
                                          per_sample=per_sample, approach_steps=30, out=out))
        return json.loads(out.read_text()), list(Srv.seen)
    return run, text, tgt


def test_vloss_none_matches_legacy_scalar(stub_run):
    run, _, _ = stub_run
    r, _ = run("none")
    r2, _ = run("none", per_sample=True)
    assert r["task_swap"] == "none" and r["samples"] == 160 and r["val_loss"] == pytest.approx(r2["val_loss"])


def test_vloss_compare(stub_run):
    run, text, tgt = stub_run
    r, seen = run("compare", per_sample=True)
    assert r["samples"] == 160 and r["shuffle_frac_samples_different_target"] == 1.0
    assert r["blank_task"] == "" and set(r["loss"]) == {"none", "shuffle", "blank"}
    assert r["approach"]["samples"] == 4 * 31 and r["all"]["samples"] == 160
    ps = r["per_sample"]
    assert np.mean(ps["none"]) == pytest.approx(r["loss"]["none"])
    assert r["all"]["shuffle_minus_none"]["mean"] == pytest.approx(np.mean(ps["shuffle"]) - np.mean(ps["none"]))
    lo, hi = r["all"]["shuffle_minus_none"]["ci95_sample_bootstrap"]
    assert lo <= r["all"]["shuffle_minus_none"]["mean"] <= hi
    # the strings given to the preprocessor: none = own, blank = "", shuffle = another episode's text, never own
    for b in range(0, len(seen), 3):
        none, shuf, blank = seen[b:b + 3]
        assert blank == [""] * len(none) and all(s != n for s, n in zip(shuf, none))
        assert all(n in text.values() for n in none) and all(s in text.values() for s in shuf)
