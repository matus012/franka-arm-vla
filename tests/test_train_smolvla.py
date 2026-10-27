"""0B: SmolVLA trainer (policy venv, CPU). Window sampler (GATES 0B row option B), exact resume, normalisation stats
re-estimated from the fine-tuning dataset, and the checkpoint loading in vla_policy.server.

The training tests use a TINY randomly initialised SmolVLA (real LeRobot 0.6.1 classes; the SmolVLM2 config shrunk to
2 text layers of width 128 and a 1-layer width-32 vision tower, real tokenizer; the tiny VLM saved in bf16 and loaded
with load_vlm_weights=True, so the dtype layout is smolvla_base's: VLM + expert bf16, projections fp32) saved as an
`--init` dir, so the real trainer code path runs on CPU in seconds. Data: synthetic contract episodes -> vla_policy.export (pyav).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from synth_episodes import PERIOD, LATENCY, write_file  # noqa: E402
from vision_lift.frames import frame_index  # noqa: E402
from vla_common.episode import read_episodes  # noqa: E402
from vla_policy import train_smolvla as T  # noqa: E402

LENGTHS = [20, 31, 17]  # not multiples of PERIOD; one episode shorter than the 50-step chunk


def windows_of(lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    ep = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)])
    fi = np.concatenate([[frame_index(t, PERIOD, LATENCY) for t in range(n)] for n in lengths])
    return ep, fi


# ---------------------------------------------------------------- window sampler (no model)

def test_window_sampler_one_row_per_window_covers_all_and_is_boundary_safe():
    ep, fi = windows_of(LENGTHS)
    s = T.WindowSampler(ep, fi, seed=3)
    n_win = len({(e, f) for e, f in zip(ep, fi)})
    assert len(s) == n_win
    for _ in range(5):
        rows = list(iter(s))
        assert len(rows) == n_win and len(set(rows)) == n_win
        keys = {(ep[r], fi[r]) for r in rows}
        assert len(keys) == n_win  # one row per window, every window covered
    # a window never spans an episode: the same frame_idx in two episodes are two windows
    assert len(s) == sum(len(set(fi[ep == e])) for e in range(len(LENGTHS)))


def test_window_sampler_every_row_of_a_window_is_drawn():
    ep, fi = windows_of(LENGTHS)
    s = T.WindowSampler(ep, fi, seed=0)
    seen = set()
    for e in range(200):
        seen |= set(s.order(e).tolist())
    assert seen == set(range(len(ep)))


def test_window_sampler_deterministic_per_seed_and_epoch_and_resumable():
    ep, fi = windows_of(LENGTHS)
    a, b = T.WindowSampler(ep, fi, seed=7), T.WindowSampler(ep, fi, seed=7)
    np.testing.assert_array_equal(a.order(4), b.order(4))
    assert not np.array_equal(a.order(4), a.order(5))
    assert not np.array_equal(a.order(4), T.WindowSampler(ep, fi, seed=8).order(4))
    first = [list(iter(a)) for _ in range(3)]  # epochs 0, 1, 2 via __iter__
    b.epoch, b.start = 1, 5
    assert list(iter(b)) == first[1][5:] and list(iter(b)) == first[2]


def test_all_rows_mode_is_a_permutation_of_every_row():
    ep, _ = windows_of(LENGTHS)
    s = T.WindowSampler(ep, None, seed=0)
    rows = list(iter(s))
    assert len(s) == len(ep) and sorted(rows) == list(range(len(ep)))


def test_norm_stats_near_constant_dim_handled():
    import datasets

    class Stub:
        hf_dataset = datasets.Dataset.from_dict({
            "observation.state": [[1.0, 0.25, float(i)] for i in range(10)],
            "action": [[float(i % 2), 0.5] for i in range(10)]})

        def __len__(self):
            return 10

    stats, rep = T.norm_stats(Stub())
    assert rep["observation.state"]["near_constant_dims"] == [0, 1] and rep["action"]["near_constant_dims"] == [1]
    np.testing.assert_allclose(stats["observation.state"]["std"][:2], 1.0)
    np.testing.assert_allclose(stats["observation.state"]["mean"], [1.0, 0.25, 4.5])
    np.testing.assert_allclose(stats["action"]["std"][0], 0.5)


# ---------------------------------------------------------------- tiny model end to end (CPU)

def tiny_init(d: Path) -> Path:
    from huggingface_hub import snapshot_download
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from transformers import AutoConfig, AutoModelForImageTextToText
    import torch

    src = Path(snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", allow_patterns=["*.json", "*.txt"]))
    vlm = d / "vlm"
    vlm.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, vlm / f.name)
    c = json.loads((vlm / "config.json").read_text())
    c["text_config"].update(hidden_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                            intermediate_size=256, num_hidden_layers=2)
    c["vision_config"].update(hidden_size=32, num_attention_heads=2, intermediate_size=64, num_hidden_layers=1)
    (vlm / "config.json").write_text(json.dumps(c))
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(vlm)).to(torch.bfloat16).save_pretrained(vlm)
    img = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 48))
    cfg = SmolVLAConfig(vlm_model_name=str(vlm), load_vlm_weights=True, num_vlm_layers=2, device="cpu",
                        input_features={"observation.images.camera1": img, "observation.images.camera2": img,
                                        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(19,))},
                        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))})
    unit = {"observation.state": {"mean": [0.0] * 19, "std": [1.0] * 19}, "action": {"mean": [0.0] * 5, "std": [1.0] * 5}}
    pre, post = make_smolvla_pre_post_processors(cfg, unit)
    out = d / "init"
    SmolVLAPolicy(cfg).save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    from vla_policy.export import export

    d = tmp_path_factory.mktemp("smolvla")
    write_file(d / "a.h5", LENGTHS, seed=0)
    export([d / "a.h5"], "tiny", root=d / "ds")
    init = tiny_init(d)
    common = ["--dataset", str(d / "ds" / "tiny"), "--init", str(init), "--device", "cpu", "--n-action-steps", "3",
              "--epochs", "2", "--batch-size", "4", "--save-every-steps", "5", "--seed", "1", "--video-backend", "pyav"]
    a = T.main(common + ["--out", str(d / "A")])  # uninterrupted: 21 windows / bs 4 = 6 steps per epoch, 12 steps
    # interruption right after the step-5 checkpoint (mid epoch 0): a fresh run dir holding only that checkpoint
    (d / "B" / "checkpoints").mkdir(parents=True)
    shutil.copytree(d / "A" / "checkpoints" / "000005", d / "B" / "checkpoints" / "000005")
    (d / "B" / "checkpoints" / "last.txt").write_text("000005")
    lines = (d / "A" / "steps.jsonl").read_text().splitlines()
    (d / "B" / "steps.jsonl").write_text("".join(x + "\n" for x in lines if json.loads(x)["step"] <= 5))
    b = T.main(common + ["--out", str(d / "B"), "--resume"])
    u = T.main([x for x in common if x not in ("--save-every-steps", "5")] + [
        "--out", str(d / "U"), "--max-steps", "3", "--freeze-vision-encoder", "false", "--train-expert-only", "false",
        "--refreeze-non-vision-vlm", "true"])
    return d, a, b, u


def steps(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_uninterrupted_run_shape(runs):
    d, a, _, _ = runs
    assert a["final_step"] == a["total_steps"] == 12 and a["steps_per_epoch"] == 6 and a["samples_per_epoch"] == 21
    s = steps(d / "A" / "steps.jsonl")
    assert [r["step"] for r in s] == list(range(1, 13))
    for e in (0, 1):  # every epoch = one row per window, all windows
        rows = [i for r in s if r["epoch"] == e for i in r["index"]]
        assert len(rows) == 21 and len(set(rows)) == 21
    assert all(np.isfinite(r["loss"]) for r in s)
    import hashlib
    assert a["init_sha256"] == hashlib.sha256((d / "init" / "model.safetensors").read_bytes()).hexdigest()


def test_resume_equals_uninterrupted(runs):
    d, _, b, _ = runs
    assert b["resumed_from_step"] == 5 and b["final_step"] == 12
    sa, sb = steps(d / "A" / "steps.jsonl"), steps(d / "B" / "steps.jsonl")
    assert [r["step"] for r in sb] == list(range(1, 13))
    for ra, rb in zip(sa[5:], sb[5:]):  # steps 6..12 cross the epoch boundary at step 6
        assert ra["index"] == rb["index"], (ra["step"], ra["index"], rb["index"])
        assert ra["epoch"] == rb["epoch"]
        assert abs(ra["loss"] - rb["loss"]) <= 1e-4 * max(1.0, abs(ra["loss"])), (ra["step"], ra["loss"], rb["loss"])
        assert ra["lr"] == pytest.approx(rb["lr"], rel=1e-9)


def test_norm_stats_recomputed_from_dataset(runs):
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs import PreTrainedConfig

    d, a, _, _ = runs
    src = [arr for _, arr, _ in read_episodes(d / "a.h5")]
    ckpt = a["final_ckpt"]
    pre, post = make_pre_post_processors(PreTrainedConfig.from_pretrained(ckpt), pretrained_path=ckpt)
    norm = next(s for s in pre.steps if isinstance(s, NormalizerProcessorStep))
    for key, col in (("observation.state", "state"), ("action", "action")):
        x = np.concatenate([e[col] for e in src]).astype(np.float64)
        st = {k: np.asarray(v, dtype=np.float64) for k, v in norm.stats[key].items()}
        np.testing.assert_allclose(st["mean"], x.mean(0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(st["std"], x.std(0), rtol=1e-5, atol=1e-5)
        assert not np.allclose(st["std"], 1.0)  # the --init dir's unit stats were NOT inherited
    unnorm = post.steps[0]
    np.testing.assert_allclose(np.asarray(unnorm.stats["action"]["mean"], dtype=np.float64),
                               np.concatenate([e["action"] for e in src]).mean(0), rtol=1e-5, atol=1e-5)


def test_checkpoint_loads_in_server(runs):
    from vla_policy.server import SmolVLAServer

    _, a, _, _ = runs
    srv = SmolVLAServer(a["final_ckpt"], n_action_steps=3, device="cpu")
    rng = np.random.default_rng(0)
    ids = np.arange(2, dtype=np.int32)
    srv.reset({"env_ids": ids, "seeds": np.array([5, 6], np.int64)})  # contract v2 b
    msg = {"env_ids": ids, "side": rng.integers(0, 255, (2, 48, 48, 3), dtype=np.uint8),
           "wrist": rng.integers(0, 255, (2, 48, 48, 3), dtype=np.uint8),
           "state": rng.normal(size=(2, 19)).astype(np.float32), "instruction": ["pick up the cube"] * 2}
    out = srv.act(msg)
    assert out.shape == (2, 3, 5) and np.isfinite(out).all() and set(np.unique(out[..., 4])) <= {0.0, 1.0}


def _legacy_act(srv, msg: dict) -> np.ndarray:
    """SmolVLAServer.act verbatim as of 8dee2ef (before --sampling): the reference for "default is unchanged"."""
    import torch
    B, N, H = len(msg["instruction"]), srv.cfg.n_action_steps, srv.image_size
    noise = srv.noise.draw(msg["env_ids"]).to(srv.device)
    batch = {"observation.images.side": torch.from_numpy(msg["side"]).permute(0, 3, 1, 2).float() / 255.0,
             "observation.images.wrist": torch.from_numpy(msg["wrist"]).permute(0, 3, 1, 2).float() / 255.0,
             "observation.state": torch.from_numpy(msg["state"]),
             "task": list(msg["instruction"])}
    chunk = srv.policy.predict_action_chunk(srv.pre(batch), noise=noise)
    a = srv.post(chunk)[:, :N].float().cpu().numpy()
    a[..., 4] = (a[..., 4] > 0.5).astype(np.float32)
    return a


def test_server_sampling_default_bit_identical_zero_and_mean_k(runs):
    """vla_policy.server --sampling (Stage 1 fix diagnosis): default = bit-identical to the pre-flag act over several
    queries; zero = deterministic, no NoiseBank draw; meanK = the mean of k seeded chunks, claw thresholded after."""
    import torch
    from vla_policy.server import SmolVLAServer

    _, a, _, _ = runs
    rng = np.random.default_rng(1)
    ids, seeds = np.arange(3, dtype=np.int32), np.array([5, 6, 7], np.int64)
    msgs = [{"env_ids": ids, "side": rng.integers(0, 255, (3, 48, 48, 3), dtype=np.uint8),
             "wrist": rng.integers(0, 255, (3, 48, 48, 3), dtype=np.uint8),
             "state": rng.normal(size=(3, 19)).astype(np.float32), "instruction": ["pick up the cube"] * 3}
            for _ in range(3)]
    srv = {m: SmolVLAServer(a["final_ckpt"], n_action_steps=3, device="cpu", sampling=m, k=4)
           for m in ("default", "zero", "meanK")}
    ref = SmolVLAServer(a["final_ckpt"], n_action_steps=3, device="cpu")
    for s in (*srv.values(), ref):
        s.reset({"env_ids": ids, "seeds": seeds})
    for m in msgs:  # several queries: the NoiseBank query counter advances identically
        assert np.array_equal(srv["default"].act(m), _legacy_act(ref, m))
    assert srv["default"].info()["sampling"] == "default" and srv["default"].info()["k"] == 1

    z = srv["zero"]
    out = [z.act(m) for m in msgs[:2]]
    assert all(q == 0 for q in z.noise.q.values())  # zero mode never draws
    assert np.array_equal(z.act(msgs[0]), out[0])  # deterministic
    assert not np.array_equal(out[0], _legacy_act(ref, msgs[0]))

    mk = srv["meanK"]
    got = mk.act(msgs[0])
    assert set(mk.noise.q.values()) == {4}  # k draws per query
    man = SmolVLAServer(a["final_ckpt"], n_action_steps=3, device="cpu")
    man.reset({"env_ids": ids, "seeds": seeds})
    m0 = msgs[0]
    batch = {"observation.images.side": torch.from_numpy(m0["side"]).permute(0, 3, 1, 2).float() / 255.0,
             "observation.images.wrist": torch.from_numpy(m0["wrist"]).permute(0, 3, 1, 2).float() / 255.0,
             "observation.state": torch.from_numpy(m0["state"]), "task": list(m0["instruction"])}
    chunks = [man.post(man.policy.predict_action_chunk(man.pre(dict(batch)), noise=man.noise.draw(ids)))[:, :3]
              .float().numpy() for _ in range(4)]
    want = np.mean(chunks, 0)
    np.testing.assert_allclose(got[..., :4], want[..., :4], rtol=1e-5, atol=1e-6)
    assert np.array_equal(got[..., 4], (want[..., 4] > 0.5).astype(np.float32))


# ---------------------------------------------------------------- arms (GATES Stage 1) + fp32 master weights

ARMS = {"F": (True, True, False), "U": (False, False, True),
        "mutant_fve_only": (False, True, False), "mutant_no_refreeze": (False, False, False)}


@pytest.fixture(scope="module")
def built(runs):
    from vla_policy.export import load

    d = runs[0]
    ds = load(d / "ds" / "tiny")
    stats, _ = T.norm_stats(ds)
    out = {}
    for k, (fve, teo, rf) in ARMS.items():
        pol = T.build_policy(str(d / "init"), ds, stats, 3, fve, teo, "cpu", rf, True)[0]
        T.set_train(pol, rf)
        out[k] = pol
    return out


def trainable(pol) -> set[str]:
    return {n for n, p in pol.named_parameters() if p.requires_grad}


def vision_names(pol) -> set[str]:
    return {n for n, _ in pol.named_parameters() if ".vlm.model.vision_model." in n}


def u_is_f_plus_vision(u: set[str], f: set[str], vis: set[str]) -> bool:
    return f < u and u - f == vis


def test_arm_u_trains_exactly_f_plus_vision_encoder(built):
    f, u, vis = trainable(built["F"]), trainable(built["U"]), vision_names(built["F"])
    assert vis and not (f & vis)
    assert u_is_f_plus_vision(u, f, vis)
    for m in ("mutant_fve_only", "mutant_no_refreeze"):  # the mutants are caught by the same check
        assert not u_is_f_plus_vision(trainable(built[m]), f, vis), m
    rep = T.param_report(built["U"])
    assert rep["vision_params_trainable"] == sum(p.numel() for n, p in built["U"].named_parameters() if n in vis)


def test_arm_u_eval_modes_and_flag_guard(built, runs):
    vwe = built["U"].model.vlm_with_expert
    vm = vwe.get_vlm_model()
    assert vm.vision_model.training and not vm.text_model.training and not vm.connector.training
    assert not vwe.vlm.lm_head.training and vwe.lm_expert.training
    with pytest.raises(AssertionError):
        from vla_policy.export import load
        ds = load(runs[0] / "ds" / "tiny")
        T.build_policy(str(runs[0] / "init"), ds, T.norm_stats(ds)[0], 3, False, True, "cpu", True, True)


def test_trainable_fp32_frozen_bf16(built):
    import torch

    for k in ("F", "U"):
        for n, p in built[k].named_parameters():
            if p.requires_grad:
                assert p.dtype == torch.float32, (k, n, p.dtype)
            elif ".vlm." in n:
                assert p.dtype == torch.bfloat16, (k, n, p.dtype)


def test_non_trainable_parts_bit_identical_after_training(runs):
    from safetensors.torch import load_file
    import torch

    d, a, _, u = runs
    init = load_file(str(d / "init" / "model.safetensors"))
    for summ, vision_should_move in ((a, False), (u, True)):
        assert summ["flags"]["trainable_fp32"] is True
        sd = load_file(str(Path(summ["final_ckpt"]) / "model.safetensors"))
        moved = []
        for k, v in init.items():
            if ".vlm." not in k:
                continue
            same = torch.equal(sd[k].to(v.dtype), v) and sd[k].dtype == v.dtype
            if ".vlm.model.vision_model." in k:
                moved.append(not same)
            else:
                assert same, (k, "non-vision VLM changed")
        assert moved and (any(moved) if vision_should_move else not any(moved)), (vision_should_move, moved)
    assert u["params"]["vision_params_trainable"] > 0 and a["params"]["vision_params_trainable"] == 0


# ---------------------------------------------------------------- train split only (GATES 0C)

def test_sampler_with_split_never_yields_val_rows_and_mutant_does():
    ep, fi = windows_of(LENGTHS)
    is_val = ep == 1  # episode 1 is a val episode
    s = T.WindowSampler(ep, fi, seed=2, keep=~is_val)
    n_train_win = len({(e, f) for e, f in zip(ep[~is_val], fi[~is_val])})
    assert len(s) == n_train_win
    for _ in range(50):
        rows = np.array(list(iter(s)))
        assert not is_val[rows].any() and len({(ep[r], fi[r]) for r in rows}) == n_train_win
    mutant = T.WindowSampler(ep, fi, seed=2)  # split ignored
    assert is_val[np.array(list(iter(mutant)))].any()


@pytest.fixture(scope="module")
def split_ds(tmp_path_factory, runs):
    from synth_episodes import attrs, episode_arrays
    from vla_common.episode import EpisodeWriter
    from vla_policy.export import export

    d = tmp_path_factory.mktemp("split")
    rng = np.random.default_rng(0)
    with EpisodeWriter(d / "s.h5", {"git_sha": "test", "created": "test", "stage": "0B"}) as w:
        for e, n in enumerate(LENGTHS):
            at = attrs(e, n, 48, env_id=e) | {"split": "val" if e == 1 else "train"}
            arr = episode_arrays(n, e, rng, 48)
            if e == 1:  # val rows shifted, so train-only stats differ visibly from all-row stats
                arr["state"] += 5.0
                arr["action"][:, :4] += 5.0
            w.add(**arr, attrs=at)
    ds = export([d / "s.h5"], "split", root=d / "ds")
    (d / "ds" / "split" / "meta" / "split_113.json").write_text(json.dumps({"train": [0, 2], "val": [1]}))
    return d / "ds" / "split", runs[0] / "init"


def run_split(dsdir: Path, init: Path, out: Path) -> tuple[dict, list[int]]:
    s = T.main(["--dataset", str(dsdir), "--init", str(init), "--device", "cpu", "--n-action-steps", "3", "--epochs",
                "2", "--batch-size", "4", "--seed", "0", "--out", str(out)])
    return s, [i for r in steps(out / "steps.jsonl") for i in r["index"]]


def source_stats(dsdir: Path, train_only: bool) -> dict:
    eps = [(at["split"], a) for _, a, at in read_episodes(dsdir.parents[1] / "s.h5")]
    out = {}
    for key, col in (("observation.state", "state"), ("action", "action")):
        x = np.concatenate([a[col] for sp, a in eps if sp == "train" or not train_only]).astype(np.float64)
        out[key] = {"mean": x.mean(0), "std": x.std(0)}
    return out


def stats_match(got: dict, want: dict) -> bool:
    return all(np.allclose(np.asarray(got[k][f], np.float64), want[k][f], rtol=1e-5, atol=1e-5)
               for k in want for f in ("mean", "std"))


def test_trainer_trains_on_train_split_only(split_ds, tmp_path, monkeypatch):
    from vla_policy.export import load

    dsdir, init = split_ds
    ds = load(dsdir)
    is_val = T.column(ds, "split").reshape(-1).astype(str) == "val"
    assert is_val.any()
    s, rows = run_split(dsdir, init, tmp_path / "ok")
    assert (s["n_train_episodes"], s["n_val_episodes"]) == (2, 1) and s["train_rows"] == int((~is_val).sum())
    assert rows and not is_val[rows].any()
    # norm stats: train rows only (val rows are shifted, so all-row stats differ)
    tr, al = source_stats(dsdir, True), source_stats(dsdir, False)
    assert stats_match(s["norm_stats"], tr) and not stats_match(s["norm_stats"], al)
    assert (s["stats_rows"], s["stats_split"]) == (int((~is_val).sum()), "train")
    # mutant: split ignored (every episode reported as train) -> val rows reach training, the check above catches it
    real = T.split_episodes
    monkeypatch.setattr(T, "split_episodes", lambda d: (np.zeros(len(d), bool), *real(d)[1:]))
    m, rows = run_split(dsdir, init, tmp_path / "mutant")
    assert is_val[rows].any()
    assert not stats_match(m["norm_stats"], tr)  # val rows in the stats are caught by the same check


def test_split_json_must_match_column(split_ds):
    from vla_policy.export import load

    dsdir, _ = split_ds
    ds = load(dsdir)
    assert T.split_episodes(ds)[1:] == ([0, 2], [1])
    f = dsdir / "meta" / "split_113.json"
    good = f.read_text()
    f.write_text(json.dumps({"train": [0, 1, 2], "val": []}))
    try:
        with pytest.raises(AssertionError):
            T.split_episodes(ds)
    finally:
        f.write_text(good)


# ---------------------------------------------------------------- baseline trainer: option B + dataset val split

def run_baseline(dsdir: Path, out: Path, monkeypatch, *extra: str) -> dict:
    import sys
    from vla_policy import train_baseline

    monkeypatch.setattr(sys, "argv", ["train_baseline", "--root", str(dsdir), "--out", str(out), "--epochs", "2",
                                      "--n-action-steps", "3", "--batch-size", "4", "--seed", "0", "--device", "cpu",
                                      *extra])
    train_baseline.main()
    return json.loads((out / "train_summary.json").read_text())


def test_baseline_rows_match_smolvla_and_val_is_the_val_split(split_ds, tmp_path, monkeypatch):
    from vla_policy.export import load

    dsdir, init = split_ds
    ds = load(dsdir)
    is_val = T.column(ds, "split").reshape(-1).astype(str) == "val"
    ep, fi = T.column(ds, "episode_index").reshape(-1), T.column(ds, "frame_idx").reshape(-1)
    b = run_baseline(dsdir, tmp_path / "bl", monkeypatch)
    s, _ = run_split(dsdir, init, tmp_path / "vla")  # same seed 0, batch 4, 2 epochs
    assert (b["n_train_episodes"], b["n_val_episodes"]) == (2, 1)
    tr, al = source_stats(dsdir, True), source_stats(dsdir, False)
    tr4 = {"observation.state": tr["observation.state"]}  # the baseline uses all state dims and arm dims 0-3
    assert stats_match(b["norm_stats"], tr4) and not stats_match(b["norm_stats"], {"observation.state": al["observation.state"]})
    np.testing.assert_allclose(b["norm_stats"]["action"]["mean"][:4], tr["action"]["mean"][:4], rtol=1e-5, atol=1e-5)
    assert (b["stats_rows"], b["stats_split"]) == (int((~is_val).sum()), "train")
    assert b["train_windows_per_epoch"] == s["samples_per_epoch"] == len({(e, f) for e, f in zip(ep[~is_val], fi[~is_val])})
    sb, sv = steps(tmp_path / "bl" / "steps.jsonl"), steps(tmp_path / "vla" / "steps.jsonl")
    for e in (0, 1):  # parity: the baseline's batches = SmolVLA's (baseline drops the last partial batch)
        rb = [i for r in sb if r["epoch"] == e for i in r["index"]]
        rv = [i for r in sv if r["epoch"] == e for i in r["index"]]
        assert rb and rb == rv[:len(rb)], (e, rb, rv)
        assert not is_val[rb].any()
    # validation loss: computed, on val-split rows only, one per val window
    val_rows = np.array(b["val_rows_seen"])
    assert b["last_val"] and len(val_rows) == b["val_windows"] == len({(e, f) for e, f in zip(ep[is_val], fi[is_val])})
    assert is_val[val_rows].all()


def test_baseline_mutant_train_rows_in_val_is_caught(split_ds, tmp_path, monkeypatch):
    from vla_policy import train_baseline

    dsdir, _ = split_ds
    monkeypatch.setattr(train_baseline, "val_sampler", lambda ds, is_val, seed: T.WindowSampler(
        T.column(ds, "episode_index"), T.column(ds, "frame_idx"), seed))  # split ignored: train rows reach validation
    with pytest.raises(AssertionError, match="split violation"):
        run_baseline(dsdir, tmp_path / "mut", monkeypatch)


def test_action_override_reproduces_lerobot_chunks(runs, tmp_path):
    """--action-override (Stage 1 fix diagnosis D2): with R = column + 1 every chunk equals LeRobot's own chunk + 1
    (incl. the episode-end clamp, where action_is_pad is set); a misaligned R is refused by the bit check."""
    import torch

    d, _, _, _ = runs
    ds = T.open_dataset(d / "ds" / "tiny", "pyav")
    col = T.column(ds, "action").astype(np.float32).reshape(len(ds), -1)
    np.savez(tmp_path / "r.npz", action=col + 1)
    ovr = T.ActionOverride(tmp_path / "r.npz", ds)
    seen_pad = False
    for batch in torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False):
        want = batch["action"] + 1
        seen_pad |= bool(batch["action_is_pad"].any())
        assert torch.equal(ovr(batch)["action"], want)
    assert seen_pad  # one episode is shorter than the chunk: the clamp was exercised
    ovr.orig = np.roll(ovr.orig, 1, 0)  # mutant: off-by-one row alignment
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)))
    with pytest.raises(AssertionError, match="LeRobot chunk"):
        ovr(batch)


def test_baseline_step_checkpoints_vloss_and_abort_numbers(split_ds, tmp_path, monkeypatch):
    from vla_policy.baseline import BaselinePolicy

    dsdir, _ = split_ds
    out = tmp_path / "fr"
    b = run_baseline(dsdir, out, monkeypatch, "--save-fractions", "0.25,0.5,0.75,1.0")
    total = b["total_steps"]
    assert total == b["epochs"] * b["steps_per_epoch"] == b["final_step"] and b["S"] == b["train_windows_per_epoch"]
    steps_ = sorted(round(f * total) for f in (0.25, 0.5, 0.75, 1.0))
    assert sorted(int(k) for k in b["save_steps"]) == steps_
    for k in steps_:
        m = BaselinePolicy.load(out / f"ckpt_{k:06d}")  # the server's --ckpt format
        assert m.cfg.n_action_steps == 3
        vj = json.loads((out / f"vloss_{k:06d}.json").read_text())
        assert vj["step"] == k and np.isfinite(vj["loss"]) and b["val_at_checkpoints"][str(k)]["loss"] == vj["loss"]
    assert not list(out.glob("epoch_*")) and not (out / "last.pt").exists()  # per-epoch saving is optional
    eb, ec = b["early_abort"]["b"], b["early_abort"]["c"]
    L = [r["loss"] for r in steps(out / "steps.jsonl")]
    s25 = steps_[0]
    assert eb["mean_loss_steps_1_50"] == pytest.approx(np.mean(L[:50])) and eb["step_25"] == s25
    assert eb["mean_loss_last_100_before_25"] == pytest.approx(np.mean(L[max(0, s25 - 100):s25]))
    assert ec["val_loss_25"] == b["val_at_checkpoints"][str(s25)]["loss"] and ec["val_loss_50"] == b["val_at_checkpoints"][str(steps_[1])]["loss"]
    assert b["early_abort"]["a_nan"] is False and b["last_val"]["loss"] == b["val_at_checkpoints"][str(total)]["loss"]
    assert {"t", "w", "c"} <= set(steps(out / "steps.jsonl")[0])


def test_baseline_max_steps_bench_mode(split_ds, tmp_path, monkeypatch):
    dsdir, _ = split_ds
    b = run_baseline(dsdir, tmp_path / "mx", monkeypatch, "--epochs", "6", "--max-steps", "5", "--save-fractions", "1.0")
    assert b["final_step"] == 5 == len(steps(tmp_path / "mx" / "steps.jsonl")) and b["total_steps"] > 5
    assert not list((tmp_path / "mx").glob("ckpt_*")) and b["last_val"] == {} and b["timing"]["timed_steps"] == 0


def test_baseline_expect_windows_is_checked_before_step_one(split_ds, tmp_path, monkeypatch):
    dsdir, _ = split_ds
    S = run_baseline(dsdir, tmp_path / "ok", monkeypatch, "--max-steps", "1")["S"]
    with pytest.raises(SystemExit, match="expect-windows"):
        run_baseline(dsdir, tmp_path / "bad", monkeypatch, "--expect-windows", str(S + 1))
    assert not (tmp_path / "bad" / "steps.jsonl").exists()  # exited before any step
    run_baseline(dsdir, tmp_path / "eq", monkeypatch, "--max-steps", "1", "--expect-windows", str(S))  # equal: runs


def test_baseline_nan_loss_stops_with_a_clear_message(split_ds, tmp_path, monkeypatch, capsys):
    from vla_policy import train_baseline as tb

    real, n = tb.loss_fn, [0]

    def nan_at_3(model, batch):
        loss, parts = real(model, batch)
        n[0] += 1
        return (loss * float("nan") if n[0] == 3 else loss), parts

    monkeypatch.setattr(tb, "loss_fn", nan_at_3)
    with pytest.raises(SystemExit) as e:
        run_baseline(dsdir := split_ds[0], tmp_path / "nan", monkeypatch)
    assert e.value.code == 3 and "EARLY ABORT (a)" in capsys.readouterr().out
    j = json.loads((tmp_path / "nan" / "train_summary.json").read_text())
    assert j["aborted"] == "a_nan" and j["early_abort"]["a_nan"] and j["final_step"] == 2


# ---------------------------------------------------------------- data parallel (--distributed) + non-DDP row order

def epoch_rows(path: Path) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r in steps(path):
        out.setdefault(r["epoch"], []).extend(r["index"])
    return out


def test_single_process_batches_are_the_sampler_order_cut_in_batches(runs):
    """Non-DDP mode: step k of epoch e logs exactly WindowSampler(seed).order(e)[bs*k : bs*(k+1)] (partial last batch kept)."""
    d, a, _, _ = runs
    ds = T.open_dataset(d / "ds" / "tiny", "pyav")
    smp = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), 1)
    s = steps(d / "A" / "steps.jsonl")
    assert len(s) == 12 and [len(r["index"]) for r in s[:6]] == [4, 4, 4, 4, 4, 1]
    for r in s:
        e, k = (r["step"] - 1) // 6, (r["step"] - 1) % 6
        assert r["index"] == smp.order(e)[4 * k:4 * k + 4].tolist(), r["step"]
    assert "distributed" in a["argv"] and a["argv"]["distributed"] == "off" and "world_size" not in a
    assert a["trainable_parts"]["vision"] == 0 and a["trainable_parts"]["text_model"] == 0 and a["trainable_parts"]["expert"] > 0


@pytest.fixture(scope="module")
def ddp(runs):
    """3 data-parallel ranks (21 windows, global batch 6: 3 full + one of 3, 4 steps per epoch, 8 steps): real gloo
    processes where torch has gloo, else threads with patched collectives; then an interruption after the step-3
    checkpoint and a resume."""
    import ddp_toy
    import test_ddp

    d = runs[0]
    common = ["--dataset", str(d / "ds" / "tiny"), "--init", str(d / "init"), "--device", "cpu", "--n-action-steps", "3",
              "--epochs", "2", "--batch-size", "6", "--save-every-steps", "3", "--seed", "1", "--video-backend", "pyav",
              "--distributed", "gloo"]

    def run(extra: list[str]) -> None:
        if test_ddp.GLOO:
            ddp_toy.main_threads(3, common + extra)
        else:
            torch.multiprocessing.spawn(ddp_toy.trainer_worker, args=(3, test_ddp.free_port(), common + extra), nprocs=3)
    import torch
    run(["--out", str(d / "D")])
    (d / "E" / "checkpoints").mkdir(parents=True)
    shutil.copytree(d / "D" / "checkpoints" / "000003", d / "E" / "checkpoints" / "000003")
    (d / "E" / "checkpoints" / "last.txt").write_text("000003")
    lines = (d / "D" / "steps.jsonl").read_text().splitlines()
    (d / "E" / "steps.jsonl").write_text("".join(x + "\n" for x in lines if json.loads(x)["step"] <= 3))
    run(["--out", str(d / "E"), "--resume"])
    return d


def test_ddp_global_order_is_the_single_gpu_one_and_rank0_writes(ddp, runs):
    d = ddp
    sd = steps(d / "D" / "steps.jsonl")
    assert [r["step"] for r in sd] == list(range(1, 9)) and [len(r["index"]) for r in sd[:4]] == [6, 6, 6, 3]
    assert epoch_rows(d / "D" / "steps.jsonl") == epoch_rows(d / "A" / "steps.jsonl")  # same epochs' rows, same order
    s = json.loads((d / "D" / "train_summary.json").read_text())
    assert (s["world_size"], s["per_rank_batch"], s["backend"], s["rank_seeds"]) == (3, 2, "gloo", [1, 2, 3])
    assert s["final_step"] == 8 and s["samples_seen_this_run"] == 42 and s["init_sha256"] == runs[1]["init_sha256"]
    assert all(np.isfinite(r["loss"]) for r in sd) and s["trainable_parts"]["expert"] > 0
    assert sorted(p.name for p in (d / "D" / "checkpoints").iterdir()) == ["000003", "000006", "000008", "last.txt"]
    import torch
    ts = torch.load(d / "D" / "checkpoints" / "000006" / "training_state.pt", map_location="cpu", weights_only=False)
    assert len(ts["rng_ranks"]) == 3 and ts["run_key"]["batch_size"] == 6
    assert (d / "D" / "checkpoints" / "000008" / "pretrained_model" / "model.safetensors").exists()


def test_ddp_resume(ddp):
    d = ddp
    s = json.loads((d / "E" / "train_summary.json").read_text())
    assert s["resumed_from_step"] == 3 and s["final_step"] == 8 and s["world_size"] == 3
    sd, se = steps(d / "D" / "steps.jsonl"), steps(d / "E" / "steps.jsonl")
    assert [r["step"] for r in se] == list(range(1, 9))
    for rd, re_ in zip(sd, se):
        assert rd["index"] == re_["index"] and rd["epoch"] == re_["epoch"] and rd["lr"] == pytest.approx(re_["lr"], rel=1e-9)


# ---------------------------------------------------------------- arm VT (Stage 2 redesign #2): vision + text + expert

def test_arm_vt_trainable_parts_and_lerobot_frozen_names(built):
    vt, f, u = (T.trainable_parts(built[k]) for k in ("mutant_no_refreeze", "F", "U"))  # no refreeze, both flags false = VT
    n = sum(p.numel() for p in built["mutant_no_refreeze"].parameters() if p.requires_grad)
    assert sum(vt[k] for k in ("vision", "text_model", "connector", "expert", "other")) == n
    assert vt["vision"] > 0 and vt["text_model"] > 0 and vt["connector"] > 0 and vt["expert"] > 0
    # LeRobot's own freezing (SmolVLMWithExpertModel.set_requires_grad): lm_head, text_model.norm.weight, last text layer(s)
    fr = vt["frozen_names_prefixes"]
    pre = "model.vlm_with_expert.vlm."
    assert set(fr) == {pre + "lm_head", pre + "model.text_model.norm", pre + "model.text_model.layers.1"}, fr
    assert (f["vision"], f["text_model"], f["connector"]) == (0, 0, 0) and set(f["frozen_names_prefixes"]) == {"model.vlm_with_expert.vlm"}
    assert u["vision"] > 0 and (u["text_model"], u["connector"]) == (0, 0) and u["expert"] == f["expert"]
    assert vt["expert"] == f["expert"] and vt["vision"] == u["vision"]


def test_param_report_accepts_f_u_vt_counts_only():
    base = {"params_total": 450_046_176, "vision_params_trainable": 0, "by_dtype": {}}
    for k in (99_880_992, 186_314_016, 392_904_096):  # F, U, VT
        T.assert_expected(base | {"params_trainable": k})
    for k in (392_904_097, 392_904_096 - 47_308_800, 450_046_176):  # off by one; VT without embeddings; everything
        with pytest.raises(AssertionError):
            T.assert_expected(base | {"params_trainable": k})
    T.assert_expected({"params_total": 13_298_656, "params_trainable": 1})  # the tiny test model: unchecked


def test_rank_batch_sampler_with_dataloader_workers(runs):
    """Regression (PERUN smoke 98121): with num_workers > 0 the DataLoader calls iter(batch_sampler) twice and drops the
    first iterator, so an __iter__ that advanced the epoch eagerly delivered epoch 1's rows at step 1. The rows the real
    LeRobot dataset path delivers (persistent workers, two epochs) must be this rank's slice of the global batches."""
    import torch

    d = runs[0]
    ds = T.open_dataset(d / "ds" / "tiny", "pyav")
    smp = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), 1)
    for workers in (0, 2):
        rbs = T.RankBatchSampler(smp, 6, 1, 3)
        dl = torch.utils.data.DataLoader(ds, batch_sampler=rbs, num_workers=workers, persistent_workers=workers > 0,
                                         generator=torch.Generator().manual_seed(1))
        got = [b["index"].tolist() for _ in range(2) for b in dl]  # two epochs: 4 steps each
        want = [T.rank_slice(rbs.global_rows(k), 1, 3) for k in range(1, 9)]
        assert got == want, (workers, got, want)
