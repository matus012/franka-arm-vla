"""CPU-only parts of scripts/eval_policy.py: smart-length seed-stream continuity, aggregation math, the GATES smart
length decision, failure typing, label injection for the baseline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("eval_policy", ROOT / "scripts" / "eval_policy.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


# --------------------------------------------------------------------------- stream continuity (--extend-to)
def test_extension_continues_the_same_stream():
    from vla_common.tasks import sample_spec
    from vla_tabletop.bridge import episode_seed
    from vla_tabletop.rollout import make_stream

    mk = lambda rng: sample_spec("s4", "paraphrase", rng)  # noqa: E731
    full = make_stream(31, 0, 120, mk, split="paraphrase")
    first, rest = make_stream(31, 0, 40, mk, split="paraphrase"), make_stream(31, 40, 80, mk, split="paraphrase")
    for a, b in zip(full, first + rest):
        assert (a.index, a.spec.to_json(), a.instruction, a.layout) == (b.index, b.spec.to_json(), b.instruction, b.layout)
    # policy noise seeds depend on (seed, episode) only
    assert [episode_seed(31, e.index) for e in full] == [episode_seed(31, e.index) for e in first + rest]
    assert len({episode_seed(31, i) for i in range(1000)}) == 1000


def test_fixed_batch_layout_single_task():
    """Protocol: episode i runs in round i // B, slot i % B, all B slots always run; an extension reuses the layout."""
    task_of = lambda i: "pick"  # noqa: E731
    first = E.plan_rounds(task_of, 0, 300, 128)
    assert [r["round"] for r in first] == [0, 1, 2]
    assert all(len(r["indices"]) == 128 for r in first)
    assert first[2]["indices"] == list(range(256, 384)) and sum(first[2]["keep"]) == 44
    ext = E.plan_rounds(task_of, 300, 1000, 128)
    assert ext[0]["round"] == 2 and ext[0]["indices"] == first[2]["indices"]  # same composition as in the first run
    assert sum(ext[0]["keep"]) == 84
    kept = sorted(i for r in ext for i, k in zip(r["indices"], r["keep"]) if k)
    assert kept == list(range(300, 1000))
    for r in first + ext:
        assert all(i // 128 == r["round"] and r["indices"].index(i) == i % 128 for i in r["indices"])


def test_fixed_batch_layout_mixed_stream():
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import make_stream

    cache = {}

    def task_of(i):
        if i not in cache:
            cache[i] = make_stream(7, i, 1, lambda rng: sample_spec("s4", "train", rng), split="train")[0].spec.task
        return cache[i]

    B = 16
    a = E.plan_rounds(task_of, 0, 60, B)
    b = E.plan_rounds(task_of, 60, 200, B)
    for plan, lo, hi in ((a, 0, 60), (b, 60, 200)):
        kept = sorted(i for r in plan for i, k in zip(r["indices"], r["keep"]) if k)
        assert kept == list(range(lo, hi))
        assert all(len(r["indices"]) == B and all(task_of(i) == r["task"] for i in r["indices"]) for r in plan)
    # the slot of an episode never depends on the requested range
    slot = lambda plan: {i: (r["task"], r["round"], j) for r in plan for j, i in enumerate(r["indices"])}  # noqa: E731
    sa, sb = slot(a), slot(b)
    assert all(sb[i] == sa[i] for i in sa.keys() & sb.keys())
    full = slot(E.plan_rounds(task_of, 0, 200, B))
    assert all(full[i] == sa[i] for i in range(60)) and all(full[i] == sb[i] for i in range(60, 200))


# --------------------------------------------------------------------------- aggregation
def rec(i: int, task: str = "pick", **kw) -> dict:
    r = {k: False for k in E.RECORD_KEYS} | {"index": i, "task": task, "completion_step": -1, "max_jam_run": 0,
                                             "max_hand_table_or_pad_force": 0.0, "target": "red_small",
                                             "n_cubes": 2 + i % 2}
    return r | kw


def test_aggregate_counts_and_ci():
    rs = [rec(i, success=i < 7, completion_step=100 if i < 7 else -1) for i in range(10)]
    rs += [rec(10 + i, "stack", success=i < 2, stack_release_rest_seen=i < 3, stack_stable_after_release=i < 2,
               release_failure=i == 2, slam=i == 3) for i in range(4)]
    a = E.aggregate(rs)
    assert a["pick"]["success"]["count"] == 7 and a["pick"]["success"]["n"] == 10
    assert a["pick"]["success"]["rate"] == pytest.approx(0.7)
    lo, hi = a["pick"]["success"]["ci95"]
    assert lo < 0.7 < hi
    assert a["stack"]["stack_stability"] == {"count": 2, "n": 3, "rate": 2 / 3, "ci95": a["stack"]["stack_stability"]["ci95"]}
    assert a["all"]["n"] == 14 and a["all"]["success"]["count"] == 9 and a["all"]["slam"]["count"] == 1
    assert "stack_stability" not in a["pick"]
    assert a["pick"]["completion_s_p50"] == pytest.approx(2.0)


def test_breakdowns_and_frame_choice():
    rs = [rec(i, success=i % 3 > 0) for i in range(12)] + [rec(12, target="blue_large", success=True)]
    a = E.aggregate(rs)["pick"]
    assert a["success_by_target"]["blue_large"]["n"] == 1 and a["success_by_target"]["red_small"]["n"] == 12
    assert a["success_by_n_cubes"]["2"]["n"] + a["success_by_n_cubes"]["3"]["n"] == 13
    rs[3]["slam"] = rs[6]["drop"] = True  # failures: 0 other, 3 slam, 6 drop, 9 other
    assert E.pick_for_frames(rs, 2) == [0, 1, 2, 3, 6]  # first fail_other, 2 successes, first slam, first drop


def test_fail_types():
    assert E.fail_types(rec(0, success=True)) == ["success"]
    assert E.fail_types(rec(0)) == ["fail_other"]
    assert E.fail_types(rec(0, slam=True, drop=True)) == ["slam", "drop"]


# --------------------------------------------------------------------------- smart length (GATES conventions)
def stream(n: int, p_success: float, wrong: float = 0.0) -> list[dict]:
    g = np.random.default_rng(0)
    s = g.permutation(n) < round(n * p_success)
    w = g.permutation(n) < round(n * wrong)
    return [rec(i, success=bool(s[i]), wrong_grasp=bool(w[i])) for i in range(n)]


BARS = E.parse_bars("pick.success>=0.85,pick.wrong_grasp<0.05")


def test_parse_bars():
    assert BARS == [("pick", "success", ">=", 0.85), ("pick", "wrong_grasp", "<", 0.05)]


def test_decide_fail_at_300_on_any_metric():
    assert E.decide(stream(300, 0.70), BARS)["decision"] == "FAIL decided at 300"
    assert E.decide(stream(300, 0.99, wrong=0.12), BARS)["decision"] == "FAIL decided at 300"


def test_decide_pass_at_300_needs_every_metric_clear():
    only = E.parse_bars("pick.success>=0.85")
    assert E.decide(stream(300, 0.95), only)["decision"] == "PASS decided at 300"
    # "< 5 %" can never be > 5 points clear -> always extends (GATES note)
    assert E.decide(stream(300, 0.99), BARS)["decision"] == "extend to 1000"


def test_decide_at_1000_uses_first_1000():
    r = stream(1000, 0.88, wrong=0.03)
    d = E.decide(r, BARS)
    assert d["decision"] == "PASS decided at 1000" and d["pass"] is True
    assert E.decide(stream(1000, 0.84), BARS)["decision"] == "FAIL decided at 1000"
    assert E.decide(r[:500], BARS)["decision"] == "extend to 1000"
    assert E.decide(r[:200], BARS)["decision"] == "incomplete"


def test_decide_requires_contiguous_prefix():
    r = stream(300, 0.9)
    with pytest.raises(AssertionError):
        E.decide(r[1:], BARS)


# --------------------------------------------------------------------------- baseline labels
def test_labelled_client_adds_labels_by_env_id():
    sent = []

    class Fake:
        def call(self, msg):
            sent.append(msg)
            return {}

    c = E.Labelled(Fake())
    c.labels = np.arange(12, dtype=np.int64).reshape(3, 4)
    c.call({"op": "act", "env_ids": np.array([2, 0], np.int32)})
    c.call({"op": "reset"})
    assert np.array_equal(sent[0]["labels"], [[8, 9, 10, 11], [0, 1, 2, 3]])
    assert "labels" not in sent[1]


def test_labels_sent_follow_the_instruction_swap_donors():
    from vla_common.tasks import labels as task_labels
    from vla_common.tasks import sample_spec
    from vla_tabletop.rollout import make_stream

    eps = make_stream(41, 0, 60, lambda rng: sample_spec("s2", "train", rng), split="train")
    byi = {e.index: e for e in eps}
    episode = lambda i: byi[i]  # noqa: E731
    _, rep = E.plan_instruction_swap(eps, "shuffle")
    own = E.labels_sent(eps, None, episode, task_labels)
    assert own.dtype == np.int64 and own.shape == (60, 4) and (own == np.stack([task_labels(e.spec) for e in eps])).all()
    sw = E.labels_sent(eps, rep, episode, task_labels)
    d = rep["donor_of"]
    for k, e in enumerate(eps):
        assert (sw[k] == task_labels(byi[d[str(e.index)]].spec)).all()  # the DONOR's labels
    assert (sw != own).any(axis=1).mean() > 0.9  # a different target for (nearly) every episode (the donors differ)
    # mutant: swap ignored for labels -> own labels sent -> caught above ((sw == own) everywhere)
    assert not (own == sw).all()
    # a slot with no donor entry (discarded slot past --episodes) keeps its own labels
    part = E.labels_sent(eps, {"donor_of": {k: v for k, v in list(d.items())[:10]}}, episode, task_labels)
    assert (part[10:] == own[10:]).all()


def test_server_args_and_expected_info_per_policy():
    from vla_common import timing as TM

    base = dict(policy="baseline", ckpt="/r/ckpt_000100", n=10, image_size=224, dataset="")
    a = E.server_args(base)
    assert a[a.index("--image-size") + 1] == "224" and a[a.index("--n-action-steps") + 1] == "10"
    assert E.expected_info(base) == {"policy": "baseline", "n_action_steps": 10, "period": TM.PERIOD, "latency": TM.LATENCY,
                                     "image_size": 224, "state_dim": 19, "action_dim": 5, "chunk_size": 10}
    smol = dict(base, policy="smolvla")
    assert "--image-size" not in E.server_args(smol) and E.expected_info(smol)["chunk_size"] == 50


# --------------------------------------------------------------------------- per-episode visual record (table value for s4b_report)
@pytest.mark.parametrize("mode", ["s4r_train", "s4r_heldout", "s4r_nominal"])
def test_visual_record_is_what_the_randomizer_applies(mode):
    """The record of slot j is computed from row j of the params handed to VR.apply (env j <- row j), and equals an
    independent re-sample of that episode's generator; light is the round's (row 0)."""
    import torch
    from collect import visual_params
    from vla_tabletop import visual_rand as VR

    idx = [7, 3, 11, 0, 5, 9]
    vis = visual_params(mode, idx, 21)
    for j, i in enumerate(idx):
        r = E.visual_record(vis, j)
        g = torch.Generator().manual_seed(int(np.random.SeedSequence([21, i, 1]).generate_state(1, np.uint64)[0] >> 1))
        p = VR.sample_params(mode, 1, g)
        hsv = VR.rgb_to_hsv(p["table_rgb"][0])
        assert r["table_value"] == pytest.approx(float(hsv[2]), abs=1e-6)
        assert r["table_sat"] == pytest.approx(float(hsv[1]), abs=1e-6)
        assert r["visual_mode"] == mode
        assert r["light_elev_deg"] == pytest.approx(VR.light_elevation_deg(vis["light"]), abs=1e-3)
        assert r["table_value"] == pytest.approx(float(VR.rgb_to_hsv(vis["table_rgb"][j])[2]), abs=1e-6)
    lo, hi = {"s4r_train": (0.35, 0.75), "s4r_heldout": (0.0, 1.0), "s4r_nominal": (0.5, 0.5)}[mode]
    assert all(lo - 1e-6 <= E.visual_record(vis, j)["table_value"] <= hi + 1e-6 for j in range(len(idx)))
    if mode == "s4r_nominal":
        assert all(E.visual_record(vis, j)["table_sat"] == 0.0 for j in range(len(idx)))


# --------------------------------------------------------------------------- --paired-targets plan
def test_paired_stream_plans_into_full_rounds():
    """The paired probe stream (groups 0..N-1 + B spare groups) fills every round with existing episodes; keep = exactly the
    first N groups' episodes, each once; defaults of the config merge are unpaired."""
    from vla_tabletop.rollout import make_paired_stream

    N, B = 150, 128
    pep, _ = make_paired_stream(21645, 0, N + B, "s2a", "train", 15.0)
    by = {e.index: e for e in pep}
    stop = sum(e.layout_id < N for e in pep)
    rounds = E.plan_rounds(lambda i: by[i].spec.task, 0, stop, B)
    kept = [i for r in rounds for i, k in zip(r["indices"], r["keep"]) if k]
    assert kept == list(range(stop)) and all(len(r["indices"]) == B for r in rounds)
    assert all(i in by for r in rounds for i in r["indices"])
    assert all(by[i].layout_id < N for i in kept) and len(rounds) == -(-stop // B)
    assert E.CONFIG_LEGACY["paired_targets"] is False and E.CONFIG_LEGACY["groups"] == 0
    assert {"paired_targets", "groups"} <= set(E.CONFIG_KEYS)


# --------------------------------------------------------------------------- --dump-slots (instrument D1)
class _Ex:
    """Stands in for ChunkExecutor: delayed frames (index, (side, wrist)), labels, the chunk of the latest query."""

    def __init__(self, B=5, H=8, N=3, labels=True):
        rng = np.random.default_rng(0)
        self.frames = type("F", (), {"current": lambda s: (0, (self.side, self.wrist))})()
        self.side, self.wrist = (rng.integers(0, 255, (B, H, H, 3), dtype=np.uint8) for _ in range(2))
        self.labels = rng.integers(0, 2, (B, 4)).astype(np.int64) if labels else None
        self.chunk = rng.normal(size=(B, N, 5)).astype(np.float32)


def test_slot_recorder_keeps_first_k_slots_and_every_step():
    ex, K, T = _Ex(), 2, 4
    rec = E.SlotRecorder(K)
    rng = np.random.default_rng(1)
    st = [rng.normal(size=(5, 22)).astype(np.float32) for _ in range(T)]  # obs_buf is wider than 19: only 19 are kept
    ac = [rng.normal(size=(5, 5)).astype(np.float32) for _ in range(T)]
    for t in range(T):
        rec.record(t, st[t], ac[t], ex)
    batch = [type("Ep", (), {"index": 10 + i, "spec": type("S", (), {"to_json": lambda s, i=i: f"spec{i}"})()})() for i in range(5)]
    d = rec.arrays(batch, 777)
    assert d["states"].shape == (K, T, 19) and d["actions"].shape == (K, T, 5)
    assert np.array_equal(d["states"][1, 2], st[2][1, :19]) and np.array_equal(d["actions"][0, 3], ac[3][0])
    assert np.array_equal(d["side0"], ex.side[:K]) and np.array_equal(d["labels0"], ex.labels[:K])
    assert np.array_equal(d["chunk0"], ex.chunk[:K]) and np.array_equal(d["state0"], st[0][:K, :19])
    assert int(d["seed"]) == 777 and d["episode_index"].tolist() == [10, 11] and d["spec_json"].tolist() == ["spec0", "spec1"]
    assert not np.shares_memory(d["side0"], ex.side)  # copies: later frames cannot overwrite the dump


def test_slot_recorder_without_labels_and_first_query_only():
    ex = _Ex(labels=False)
    rec = E.SlotRecorder(1)
    rec.record(0, np.zeros((5, 19), np.float32), np.zeros((5, 5), np.float32), ex)
    ex.chunk[:] = 99.0  # a later query changes the chunk: the dump keeps the FIRST one
    rec.record(1, np.zeros((5, 19), np.float32), np.zeros((5, 5), np.float32), ex)
    assert "labels0" not in rec.first and rec.first["chunk0"].max() < 99.0


def test_dump_slots_flag_is_guarded_and_off_by_default(monkeypatch):
    import inspect

    src = inspect.getsource(E.main)
    # off (default 0): no recorder exists, the only extra statement per step is the local read of the state it already passed
    assert '"--dump-slots", type=int, default=0' in src
    assert "SlotRecorder(args.dump_slots) if dump_round else None" in src
    assert "dump_round = bool(args.dump_slots) and rd is rounds[0]" in src
    monkeypatch.setattr(sys, "argv", ["eval_policy", "--policy", "expert", "--dump-slots", "4", "--envs", "8",
                                      "--episodes", "8", "--stage", "s2", "--split", "train", "--seed", "1"])
    with pytest.raises(AssertionError, match="--dump-slots"):
        E.main()
    monkeypatch.setattr(sys, "argv", ["eval_policy", "--policy", "baseline", "--dump-slots", "9", "--envs", "8",
                                      "--episodes", "8", "--stage", "s2", "--split", "train", "--seed", "1"])
    with pytest.raises(AssertionError, match="--dump-slots"):
        E.main()  # K > --envs
