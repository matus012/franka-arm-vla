"""Stage 0.4 (3/3): dataset write/read round trip (every array written comes back identical; depth within fp16)."""

from __future__ import annotations

import h5py
import numpy as np
import torch

from vision_lift.dataset import EpisodeWriter, StepDataset

E, T, F, RES, PD = 2, 9, 3, 16, 27


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "rt.h5"
    w = EpisodeWriter(path, E, T, F, RES, PD, {"world": "P6", "seed": 5, "cfg": {"a": 1}})
    frames = []
    for k in range(F):
        fr = {"side_rgb": rng.integers(0, 256, (E, RES, RES, 3), dtype=np.uint8),
              "wrist_rgb": rng.integers(0, 256, (E, RES, RES, 3), dtype=np.uint8),
              "side_depth": rng.uniform(0.05, 3.0, (E, RES, RES)).astype(np.float32),
              "wrist_depth": rng.uniform(0.05, 3.0, (E, RES, RES)).astype(np.float32),
              "proprio": rng.normal(size=(E, PD)).astype(np.float32)}
        cube = rng.normal(size=(E, 3)).astype(np.float32)
        w.add_frame(k, **fr, priv={"cube_pos": cube})
        frames.append((fr, cube))
    steps = []
    for t in range(T):
        d = {"proprio": rng.normal(size=(E, PD)).astype(np.float32),
             "teacher_arm": rng.uniform(-1, 1, (E, 4)).astype(np.float32),
             "claw_label": rng.integers(0, 2, E).astype(np.uint8),
             "exec_action": rng.normal(size=(E, 5)).astype(np.float32),
             "driver": np.ones(E, np.uint8)}
        pr = {"phase": rng.integers(0, 6, E).astype(np.int8)}
        w.add_step(d, pr)
        steps.append((d, pr))
    fos = np.array([max(t // 3 - 1, 0) for t in range(T)])
    w.close(fos, {"success": np.array([True, False]), "dist": np.array([0.01, 0.2], np.float32)})

    with h5py.File(path, "r") as h:
        assert h.attrs["world"] == "P6" and int(h.attrs["seed"]) == 5 and h.attrs["cfg"] == '{"a": 1}'
        for k, (fr, cube) in enumerate(frames):
            for key in ("side_rgb", "wrist_rgb", "proprio"):
                np.testing.assert_array_equal(h[f"frames/{key}"][:, k], fr[key])
            for key in ("side_depth", "wrist_depth"):
                np.testing.assert_allclose(h[f"frames/{key}"][:, k], fr[key], rtol=1e-3)
            np.testing.assert_array_equal(h["privileged/frame_cube_pos"][:, k], cube)
        for t, (d, pr) in enumerate(steps):
            for key, v in d.items():
                np.testing.assert_array_equal(h[f"steps/{key}"][:, t], v)
            np.testing.assert_array_equal(h["privileged/step_phase"][:, t], pr["phase"])
        np.testing.assert_array_equal(h["steps/frame"][:], fos)
        np.testing.assert_array_equal(h["episodes/success"][:], [True, False])
        assert h["frames/side_rgb"].compression == "lzf" and h["frames/side_rgb"].chunks == (1, 1, RES, RES, 3)

    ds = StepDataset([path], depth=True)
    for i in range(len(ds)):
        s = ds[i]
        _, e, t, k = s["_where"].tolist()
        assert k == fos[t]
        fr, _ = frames[k]
        assert torch.equal(s["side_rgb"], torch.from_numpy(fr["side_rgb"][e]).permute(2, 0, 1))
        np.testing.assert_allclose(s["wrist_depth"][0].numpy(), fr["wrist_depth"][e], rtol=1e-3)
        np.testing.assert_array_equal(s["proprio"].numpy(), steps[t][0]["proprio"][e])
        np.testing.assert_array_equal(s["teacher_arm"].numpy(), steps[t][0]["teacher_arm"][e])
