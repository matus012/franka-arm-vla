"""scripts/s4b_underfit.py on hand-computed values and a tiny synthetic split dataset (48 px, N = 3)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")
from test_train_smolvla import runs  # noqa: E402,F401  (fixture: the module fixtures need it)

from vla_policy import train_baseline as TB  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402
from vla_policy.baseline import BaselineConfig, BaselinePolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("s4b_underfit", ROOT / "scripts" / "s4b_underfit.py")
U = importlib.util.module_from_spec(sp)
sp.loader.exec_module(U)
N = 3
LEN = [20, 31, 17, 25, 22, 18]
TASK = [0, 1, 2, 0, 1, 2]  # per episode: labels[0]
VAL = {1, 4}


def test_reference_arithmetic_by_hand():
    # task 0: episode A (3 steps, dim0 1,2,3, claw 0,0,1), episode B (2 steps, dim0 5,7, claw 0,1); N = 2
    a = np.zeros((5, 5), np.float32)
    a[:, 0], a[:, 4] = [1, 2, 3, 5, 7], [0, 0, 1, 0, 1]
    ep = np.array([0, 0, 0, 1, 1])
    j, L = U.layout(ep)
    assert j.tolist() == [0, 1, 2, 0, 1] and L.tolist() == [3, 3, 3, 2, 2]
    tgt, pad = U.chunks(a, j, L, np.arange(5), 2)
    assert tgt[2, :, 0].tolist() == [3, 3] and pad[2].tolist() == [False, True] and pad[4].tolist() == [False, True]
    assert tgt[3, :, 0].tolist() == [5, 7] and tgt[4, :, 0].tolist() == [7, 7]
    task = np.zeros(5, np.int64)
    ref = U.reference_tables(a, task, j, L, np.ones(5, bool), 2)
    assert ref["M"]["arm"][0] == pytest.approx(3.6) and ref["M"]["claw"] is False  # claw mean 0.4
    w = ref["T"]["w"]
    assert w == 3
    arm = ref["T"]["arm"][:, :, 0]
    assert arm[0].tolist() == [3.0, 4.5]  # (task 0, t 0): mean(1,5), mean(2,7)
    assert arm[1].tolist() == [4.5, 3.0]  # t 1: mean(2,7); position 1 only A is unpadded -> 3
    assert arm[2, 0] == 3.0 and ref["T"]["count"][2].tolist() == [1, 0]
    assert ref["T"]["claw"][0].tolist() == [False, False]  # position 1: claws 0 and 1 = tie -> 0
    assert ref["T"]["claw"][1].tolist() == [False, True] and ref["T"]["claw"][2, 0]
    m, t, fb = U.reference_preds(ref, task, j, np.arange(5), 2, pad)
    assert fb == 0 and (m[..., 0] == 3.6).all() and t[2, 1, 0] == pytest.approx(3.6)  # the padded fallback position
    assert t[0, 1, 0] == 4.5 and t[3, 0, 0] == 3.0 and t[4, 0, 0] == 4.5
    s = U.score(m, tgt, pad, np.ones(5, bool))
    keep = ~pad
    want = (np.abs(3.6 - tgt[..., 0]) * keep).sum() / keep.sum()
    assert s["arm_mae"][0] == pytest.approx(want) and s["claw_acc"] == pytest.approx(
        ((tgt[..., 4] > 0.5) == False)[keep].mean())  # noqa: E712
    assert keep.sum() == 8


def test_claw_change_near_and_task_split():
    a = np.zeros((40, 5), np.float32)
    a[15:, 4] = 1  # change at row 15, one episode
    ep = np.zeros(40, int)
    near = U.claw_change_near(a, ep)
    assert np.flatnonzero(near).tolist() == list(range(5, 26))
    ep2 = np.r_[np.zeros(20, int), np.ones(20, int)]  # an episode boundary is not a claw change
    a2 = np.zeros((40, 5), np.float32)
    a2[20:, 4] = 1
    assert not U.claw_change_near(a2, ep2).any()


@pytest.fixture(scope="module")
def uds(tmp_path_factory, runs):
    from synth_episodes import attrs, episode_arrays
    from vla_common.episode import EpisodeWriter
    from vla_policy.export import export

    d = tmp_path_factory.mktemp("uf")
    rng = np.random.default_rng(0)
    with EpisodeWriter(d / "s.h5", {"git_sha": "test", "created": "test", "stage": "0B"}) as w:
        for e, n in enumerate(LEN):
            arr = episode_arrays(n, e, rng, 48)
            arr["action"][:, :4] = rng.uniform(-0.9, 0.9, (n, 4)) + 0.05 * TASK[e]
            arr["action"][:, 4] = (np.arange(n) >= 8 + e).astype(np.float32)  # claw closes at a fixed step
            arr["action_exec"] = arr["action"].copy()
            at = attrs(e, n, 48, env_id=e) | {"split": "val" if e in VAL else "train", "labels": [TASK[e], e % 4, e % 2, 12]}
            w.add(**arr, attrs=at)
    export([d / "s.h5"], "uf", root=d / "ds")
    root = d / "ds" / "uf"
    meta = TB.make_ds(root, None, N, "pyav")
    return TB.make_ds(root, None, N, "pyav", meta.fps), d


def head_model(ds, claw_majority: bool, arm_mean: np.ndarray | None = None) -> BaselinePolicy:
    """Head zeroed; arm output = a_mean, claw logit = +-10."""
    is_val, _, _ = T.split_episodes(ds)
    st, _ = T.norm_stats(ds, ~is_val)
    torch.manual_seed(0)
    m = BaselinePolicy(BaselineConfig(n_action_steps=N))
    m.set_norm(torch.zeros(19), torch.ones(19), torch.as_tensor(st["action"]["mean"][:4]), torch.ones(4))
    with torch.no_grad():
        m.head[-1].weight.zero_()
        m.head[-1].bias.copy_(torch.tensor([0, 0, 0, 0, 10.0 if claw_majority else -10.0]).repeat(N))
    return m


def test_zeroed_head_at_train_mean_scores_M(uds, tmp_path):
    ds, _ = uds
    action = T.column(ds, "action")
    is_val, _, _ = T.split_episodes(ds)
    maj = bool(action[~is_val, 4].mean() > 0.5)
    m = head_model(ds, maj)
    m.save(tmp_path / "ck")
    for split in ("train", "val"):
        r = U.run(ds, [tmp_path / "ck"], split, max_rows=40, seed=1, bs=16)
        c = r["ckpts"][str(tmp_path / "ck")]
        for g in ("overall", "pick", "place", "stack"):
            for s in U.SUBSETS:
                a, b = c["metrics"][g][s], r["references"]["M"][g][s]
                assert a["n_rows"] == b["n_rows"]
                if a["n_rows"]:
                    np.testing.assert_allclose(a["arm_mae"], b["arm_mae"], atol=1e-5)
                    assert a["claw_acc"] == b["claw_acc"]
                    np.testing.assert_allclose(c["ratio_vs_M"][g][s]["arm_mae"], 1.0, atol=1e-4)
        assert r["n_t0_rows"] >= len({e for e in range(len(LEN)) if (e in VAL) == (split == "val")})
        # ratio fields: model / T = M / T
        o, t = c["metrics"]["overall"]["all"], r["references"]["T"]["overall"]["all"]
        np.testing.assert_allclose(c["ratio_vs_T"]["overall"]["all"]["arm_mae"], np.array(o["arm_mae"]) / np.array(t["arm_mae"]),
                                   rtol=1e-6)
        assert c["ratio_vs_T"]["overall"]["all"]["arm_mae_mean"] == pytest.approx(o["arm_mae_mean"] / t["arm_mae_mean"])
        json.dumps(r)


def test_t_reference_beats_m_on_time_structured_claw_and_t0_rows_forced(uds, tmp_path):
    ds, _ = uds
    m = head_model(ds, False)
    m.save(tmp_path / "ck")
    r = U.run(ds, [tmp_path / "ck"], "val", max_rows=5, seed=3, bs=8)
    assert r["n_t0_rows"] == 2 and r["n_rows_sampled"] >= 5  # both val episodes' t=0 rows are in
    # train episodes of one task share the claw schedule only per episode (closes at 8 + e): T and M differ, both finite
    t = r["references"]["T"]["overall"]["all"]
    assert t["arm_mae_mean"] is not None and 0 <= t["claw_acc"] <= 1


def test_ratio_none_on_zero_reference():
    z = {"overall": {"all": {"arm_mae": [0.0, 1.0, 2.0, 2.0], "arm_mae_mean": 0.0}}}
    n = {"overall": {"all": {"arm_mae": [1.0, 1.0, 1.0, 1.0], "arm_mae_mean": 1.0}}}
    r = U.ratio(n, z)["overall"]["all"]
    assert r["arm_mae"] == [None, 1.0, 0.5, 0.5] and r["arm_mae_mean"] is None


def test_cli(uds, tmp_path, monkeypatch):
    ds, d = uds
    m = head_model(ds, False)
    m.save(tmp_path / "ck")
    out = tmp_path / "o" / "u.json"
    monkeypatch.setattr(sys, "argv", ["s4b_underfit", "--dataset", str(d / "ds" / "uf"), "--ckpt", str(tmp_path / "ck"),
                                      "--split", "val", "--max-rows", "10", "--out", str(out)])
    U.main()
    j = json.loads(out.read_text())
    assert j["split"] == "val" and str(tmp_path / "ck") in j["ckpts"] and j["N"] == N
