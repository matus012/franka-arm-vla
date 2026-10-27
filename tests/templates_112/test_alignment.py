"""Stage 0.4 (1/3): frame/action alignment with synthetic frames that encode their own index.

The real Recorder (camera ticks + latency buffer + disk writer, used by every collection) is driven by a fake camera
whose frames encode the control step they were captured on; per-step labels encode their own step. For every step we
check (a) the frame the POLICY was fed at runtime, (b) the frame the DATASET returns for that step, (c) the label the
dataset returns, against the timing rule of vision_lift.frames (period 3, latency 1)."""

from __future__ import annotations

import numpy as np
import torch

from vision_lift.collect import Recorder
from vision_lift.dataset import EpisodeWriter, StepDataset
from vision_lift.frames import LATENCY, PERIOD, frame_step, n_frames

E, T, RES, PD = 3, 40, 8, 27


def _frame(t: int) -> dict:
    img = torch.full((E, RES, RES, 3), t % 256, dtype=torch.uint8)
    img[:, 0, 0, 0] = t // 256
    img[:, 0, 0, 1] = torch.arange(E, dtype=torch.uint8)  # env id
    depth = torch.full((E, RES, RES), float(t))
    return {"side_rgb": img, "wrist_rgb": img.clone(), "side_depth": depth, "wrist_depth": depth.clone(),
            "proprio": torch.full((E, PD), float(t))}


def _decode(img: torch.Tensor) -> int:  # (3, H, W)
    return int(img[0, 1, 1]) + 256 * int(img[0, 0, 0])


def test_frame_step_rule():
    assert PERIOD == 3 and LATENCY == 1
    assert [frame_step(t) for t in range(10)] == [0, 0, 0, 0, 0, 0, 3, 3, 3, 6]


def test_recorder_runtime_and_dataset_agree(tmp_path):
    path = tmp_path / "ep.h5"
    rec = Recorder(EpisodeWriter(path, E, T, n_frames(T), RES, PD, {"test": 1}))
    fed = []
    for t in range(T):
        k, fr = rec.frame(t, lambda t=t: _frame(t))
        fed.append(_decode(fr["side_rgb"][0].permute(2, 0, 1)))
        lab = torch.full((E, 4), float(t))
        rec.step({"proprio": torch.full((E, PD), float(t)), "teacher_arm": lab,
                  "claw_label": torch.full((E,), t % 2, dtype=torch.uint8),
                  "exec_action": torch.zeros(E, 5), "driver": torch.ones(E, dtype=torch.uint8)}, {})
    rec.close({"success": np.ones(E, bool)})
    assert fed == [frame_step(t) for t in range(T)], "runtime feed violates the timing rule"
    ds = StepDataset([path], depth=True)
    assert len(ds) == E * T
    for i in range(len(ds)):
        s = ds[i]
        _, e, t, _ = s["_where"].tolist()
        for cam in ("side", "wrist"):
            assert _decode(s[f"{cam}_rgb"]) == frame_step(t), (e, t)
            assert int(s[f"{cam}_rgb"][1, 0, 0]) == e  # the right env's frame
            assert float(s[f"{cam}_depth"][0, 3, 3]) == frame_step(t)
        assert float(s["teacher_arm"][0]) == t and float(s["proprio"][0]) == t
        assert float(s["claw_label"]) == t % 2
        assert _decode(s["side_rgb"]) == fed[t]  # dataset sample == what the policy saw at runtime
