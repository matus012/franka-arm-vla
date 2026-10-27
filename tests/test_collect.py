"""CPU-only parts of scripts/collect.py (chunk plan, part naming / rollover / rename, sidecar schema, manifest,
resumability) and the training-set audit of scripts/export_dataset.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("collect")
X = _load("export_dataset")


def _episode(T: int = 7, H: int = 4, seed: int = 0, big: int = 0) -> tuple[dict, dict]:
    g = np.random.default_rng(seed)
    F = (T + 2) // 3
    arrays = {"side_rgb": g.integers(0, 255, (F + big, H, H, 3), dtype=np.uint8),
              "wrist_rgb": g.integers(0, 255, (F + big, H, H, 3), dtype=np.uint8),
              "state": g.normal(size=(T, 19)).astype(np.float32), "action": np.zeros((T, 5), np.float32),
              "action_exec": np.zeros((T, 5), np.float32),
              "frame_idx": np.array([max(t // 3 - 1, 0) for t in range(T)], np.int32)}
    attrs = {"instruction": "pick up the red cube", "split": "train", "task": "pick", "spec_json": "{}",
             "labels": [0, 0, 0, 12], "seed": 1, "env_id": 0, "success": True, "metrics_json": {}, "period": 3,
             "latency": 1, "image_size": H, "visual_mode": "nominal", "action_noise_sigma": 0.1, "expert_version": "x",
             "episode_index": seed, "T": T, "n_action_steps": None, "detector_version": "x"}
    return arrays, attrs


FA = {"git_sha": "abc", "created": "now", "stage": "s2"}


# --------------------------------------------------------------------------- chunk plan
def test_plan_groups_per_task_in_stream_order():
    tasks = ["pick", "place", "pick", "stack", "pick", "place", "pick"]
    plan = C.plan_chunks(tasks, 2)
    assert [c["chunk"] for c in plan] == ["pick_e000000", "pick_e000004", "place_e000001", "stack_e000003"]
    assert [c["indices"] for c in plan] == [[0, 2], [4, 6], [1, 5], [3]]
    assert sorted(i for c in plan for i in c["indices"]) == list(range(len(tasks)))
    assert all(len(c["indices"]) <= 2 for c in plan)
    off = C.plan_chunks(tasks, 2, start=100)  # a later batch of the same dataset: absolute indices, unique ids
    assert [c["indices"] for c in off] == [[100, 102], [104, 106], [101, 105], [103]]
    assert off[0]["chunk"] == "pick_e000100"


def test_plan_prefix_stable_when_stream_grows():
    g = np.random.default_rng(0)
    tasks = [("pick", "place", "stack")[k] for k in g.integers(0, 3, 500)]
    short, long = C.plan_chunks(tasks[:300], 16), C.plan_chunks(tasks, 16)
    full = [c for c in short if len(c["indices"]) == 16]
    assert all(c in long for c in full)


# --------------------------------------------------------------------------- parts: naming, rollover, rename
def test_part_writer_tmp_then_rename(tmp_path):
    from vla_common.episode import read_episodes

    w = C.PartWriter(tmp_path, "s2_train_s1", "pick_c00000", FA, part_bytes=1)  # every episode starts a new part
    for i in range(3):
        a, at = _episode(seed=i)
        w.add(a, at)
        assert (tmp_path / C.part_name("s2_train_s1", "pick_c00000", i, tmp=True)).exists()
        assert not list(tmp_path.glob("*.h5"))  # nothing final before close
    names = w.close()
    assert names == [f"s2_train_s1_pick_c00000_p{j}.h5" for j in range(3)]
    assert not list(tmp_path.glob("*.tmp"))
    got = [at["episode_index"] for n in names for _, _, at in read_episodes(tmp_path / n)]
    assert got == [0, 1, 2]


def test_part_writer_rolls_at_size(tmp_path):
    from vla_common.episode import file_attrs, read_episodes

    a, at = _episode(H=32, big=30)
    w = C.PartWriter(tmp_path, "p", "pick_c00000", FA, part_bytes=500_000)
    for i in range(8):
        w.add(a, at | {"episode_index": i})
    names = w.close()
    assert len(names) > 1
    sizes = [(tmp_path / n).stat().st_size for n in names]
    assert all(s < 500_000 + 2 * 300_000 for s in sizes)  # a part closes at the first add after it passes the size
    assert sum(len(list(read_episodes(tmp_path / n))) for n in names) == 8
    assert file_attrs(tmp_path / names[1])["part"] == 1


def test_part_writer_nothing_kept(tmp_path):
    assert C.PartWriter(tmp_path, "p", "pick_c00000", FA).close() == []


# --------------------------------------------------------------------------- sidecar + resumability
def _outcomes(n: int) -> dict:
    o = {k: np.zeros(n, bool) for k in ("success", *C.FLAG_KEYS)}
    o["success"][:] = [True, False, True, False][:n]
    o["slam"][1] = True
    o["wrong_object"][1] = o["wrong_grasp"][1] = True
    return o


def test_sidecar_schema_and_counts():
    chunk = {"chunk": "pick_e000000", "task": "pick", "indices": [0, 2, 4, 6]}
    d = C.build_sidecar(chunk, _outcomes(4), ["a.h5"], 350, 1e-9, 1.0, "sha", {"x": 1})
    assert tuple(d) == C.SIDECAR_KEYS
    assert d["n"] == 4 and d["n_success"] == 2 and d["success_rate"] == 0.5
    assert d["kept_indices"] == [0, 4] and d["failing_indices"] == [2, 6]
    assert d["failures"]["slam"] == 1 and d["failures"]["wrong_object"] == 1 and d["failed_unflagged"] == 1
    json.dumps(d)


def test_resume_logic(tmp_path):
    pre = "s2_train_s1"
    chunk = {"chunk": "pick_c00000", "task": "pick", "indices": [0, 1]}
    assert not C.chunk_done(tmp_path, pre, chunk)
    # a crashed run leaves tmp + finished parts but no sidecar -> not done, leftovers removed
    (tmp_path / C.part_name(pre, "pick_c00000", 0)).write_bytes(b"x")
    (tmp_path / C.part_name(pre, "pick_c00000", 1, tmp=True)).write_bytes(b"x")
    (tmp_path / C.part_name(pre, "pick_c00001", 0)).write_bytes(b"other chunk")
    assert not C.chunk_done(tmp_path, pre, chunk)
    assert C.clean_chunk(tmp_path, pre, "pick_c00000") == [f"{pre}_pick_c00000_p0.h5", f"{pre}_pick_c00000_p1.h5.tmp"]
    assert (tmp_path / C.part_name(pre, "pick_c00001", 0)).exists()
    # finished: parts + sidecar
    (tmp_path / C.part_name(pre, "pick_c00000", 0)).write_bytes(b"x")
    sc = C.build_sidecar(chunk, _outcomes(2), [C.part_name(pre, "pick_c00000", 0)], 350, 1, 1.0, "s", {})
    C.sidecar_path(tmp_path, pre, "pick_c00000").write_text(json.dumps(sc))
    assert C.chunk_done(tmp_path, pre, chunk)
    with pytest.raises(RuntimeError, match="plan says"):
        C.chunk_done(tmp_path, pre, chunk | {"indices": [0, 2]})
    (tmp_path / C.part_name(pre, "pick_c00000", 0)).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        C.chunk_done(tmp_path, pre, chunk)


def test_run_config_must_match(tmp_path):
    cfg = {k: 1 for k in C.RUN_KEYS} | {"episodes": 10}
    C.check_run_config(tmp_path, cfg)
    C.check_run_config(tmp_path, cfg | {"episodes": 20})  # a longer range: allowed
    with pytest.raises(RuntimeError, match="envs"):
        C.check_run_config(tmp_path, cfg | {"envs": 2})


def test_run_config_cube_yaw(tmp_path):
    """A RUN.json written before --cube-yaw-max-deg existed counts as 180; a resume with another range fails."""
    cfg = {k: 1 for k in C.RUN_KEYS} | {"cube_yaw_max_deg": 180.0}
    legacy = {k: v for k, v in cfg.items() if k != "cube_yaw_max_deg"}
    (tmp_path / "RUN.json").write_text(json.dumps(legacy))
    C.check_run_config(tmp_path, cfg)
    (tmp_path / "RUN.json").write_text(json.dumps(legacy))
    with pytest.raises(RuntimeError, match="cube_yaw_max_deg"):
        C.check_run_config(tmp_path, cfg | {"cube_yaw_max_deg": 15.0})


def test_run_config_table_palette(tmp_path):
    """A RUN.json written before --table-palette existed counts as 112; a resume with s4r fails."""
    cfg = {k: 1 for k in C.RUN_KEYS} | {"table_palette": "112"}
    legacy = {k: v for k, v in cfg.items() if k != "table_palette"}
    (tmp_path / "RUN.json").write_text(json.dumps(legacy))
    C.check_run_config(tmp_path, cfg)
    (tmp_path / "RUN.json").write_text(json.dumps(legacy))
    with pytest.raises(RuntimeError, match="table_palette"):
        C.check_run_config(tmp_path, cfg | {"table_palette": "s4r"})


def test_manifest(tmp_path):
    (tmp_path / "b.h5").write_bytes(b"hello")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "c.h5.tmp").write_bytes(b"partial")
    lines = C.write_manifest(tmp_path).read_text().splitlines()
    assert lines == [f"{hashlib.sha256(b'{}').hexdigest()}  a.json", f"{hashlib.sha256(b'hello').hexdigest()}  b.h5"]
    assert X.verify_manifest(tmp_path) == 2
    (tmp_path / "b.h5").write_bytes(b"tampered")
    with pytest.raises(AssertionError, match="mismatch"):
        X.verify_manifest(tmp_path)


def test_noise_table_per_episode():
    a = C.noise_table(31800, [5, 6, 7], 350, 0.1)
    b = C.noise_table(31800, [7, 5], 350, 0.1)  # other batch composition / order
    assert a.shape == (3, 350, 4) and a.dtype == np.float32
    assert np.array_equal(a[2], b[0]) and np.array_equal(a[0], b[1])
    assert abs(float(a.std()) - 0.1) < 0.005 and not np.array_equal(a[0], a[1])
    assert not C.noise_table(31800, [5], 350, 0.0).any()


# --------------------------------------------------------------------------- export audit (held-out sets)
def _attrs(spec, instr: str, split: str = "train") -> dict:
    from vla_common.tasks import labels

    return {"spec_json": spec.to_json(), "instruction": instr, "split": split, "task": spec.task, "seed": 31800,
            "labels": json.dumps(labels(spec).tolist()), "visual_mode": "nominal", "visual_json": '{"mode": "nominal"}'}


def test_audit_clean_on_sampled_train_specs():
    from vla_common.tasks import instruction, sample_spec

    for stage in ("s1", "s2", "s3a", "s3b", "s4"):
        rng = np.random.default_rng(5)
        for _ in range(500):
            s = sample_spec(stage, "train", rng)
            assert X.audit_episode(_attrs(s, instruction(s, rng, "train"))) == []


def test_audit_catches_every_heldout_kind():
    from vla_common.tasks import (HELD_OUT_PLACE_PAIRS, HELD_OUT_STACK_PAIRS, HELD_OUT_TARGETS, TaskSpec,
                                  instruction, sample_spec)
    from vla_common.catalog import COLORS

    rng = np.random.default_rng(1)
    t = HELD_OUT_TARGETS[0]
    s = TaskSpec("pick", t, None, [t, ("red", "small")], list(COLORS))
    assert any("held-out target" in v for v in X.audit_episode(_attrs(s, instruction(s, rng))))
    (c, pad) = HELD_OUT_PLACE_PAIRS[0]
    s = TaskSpec("place", c, pad, [c, ("blue", "small")], list(COLORS))
    assert any("place pair" in v for v in X.audit_episode(_attrs(s, instruction(s, rng))))
    (top, bot) = HELD_OUT_STACK_PAIRS[0]
    s = TaskSpec("stack", top, bot, [top, bot], list(COLORS))
    assert any("stack pair" in v for v in X.audit_episode(_attrs(s, instruction(s, rng))))
    s = sample_spec("s2", "train", rng)
    v = X.audit_episode(_attrs(s, instruction(s, rng, "paraphrase")))
    assert any("paraphrase" in x for x in v) and any("no train template" in x for x in v)
    assert any("split" in x for x in X.audit_episode(_attrs(s, instruction(s, rng), split="heldout_comp")))
    hc = [sample_spec("s3b", "heldout_comp", rng) for _ in range(200)]
    assert all(X.audit_episode(_attrs(s, instruction(s, rng))) for s in hc)


def test_summarize_and_wilson():
    from lift_rl.metrics import wilson_ci

    a = C.build_sidecar({"chunk": "pick_e000000", "task": "pick", "indices": [0, 1, 2, 3]}, _outcomes(4), [], 350, 0.0,
                        1.0, "s", {})
    b = C.build_sidecar({"chunk": "place_e000004", "task": "place", "indices": [4, 5]}, _outcomes(2), [], 550, 0.0,
                        1.0, "s", {})
    s = C.summarize([a, b])
    assert list(s) == ["pick", "place"] and s["pick"]["attempted"] == 4 and s["pick"]["kept"] == 2
    assert s["place"]["failures"]["slam"] == 1 and s["pick"]["failed_unflagged"] == 1
    for k, n in ((0, 10), (7, 10), (300, 300), (97, 100)):
        assert np.allclose(C.wilson(k, n), wilson_ci(k, n))
