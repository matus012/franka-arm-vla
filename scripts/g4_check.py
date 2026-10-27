"""Task 004 instrument proof (laptop, policy venv): the guided sampler on a real SmolVLA checkpoint.

  (a) w = 1 through vla_policy.guidance == the plain chunk of the SAME expanded batch's true rows (bit for bit);
  (b) w = 1 guided vs plain on the B true rows alone (batch-composition level, fp32 serving ~1e-5);
  (c) others = the true instruction itself at w = 8 -> the plain chunk (v_other = v_true);
  (d) w = 2 / 8 with real other colours move the chunk (nonzero), the denoiser is called num_steps times per chunk.

  python scripts/g4_check.py --ckpt checkpoints/s1a1_U/pretrained_model --out results/s2a1_guidance/g4_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_common.guidance import other_instructions  # noqa: E402
from vla_policy.guidance import expand_rows, guided_chunk  # noqa: E402
from vla_policy.server import NoiseBank, SmolVLAServer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA unavailable -- STOP"
    srv = SmolVLAServer(a.ckpt, n_action_steps=10, device=a.device)
    pol, H = srv.policy, srv.image_size
    rng = np.random.default_rng(0)
    instr = ["pick up the red cube", "pick the small blue cube up", "please pick up the yellow cube"]
    tables = [["blue", "red", "green"], ["blue", "yellow"], ["green", "yellow", "red", "blue"]]
    tgt = ["red", "blue", "yellow"]
    others = [other_instructions(t, c, cols) for t, c, cols in zip(instr, tgt, tables)]
    B = len(instr)
    raw = {"observation.images.side": torch.from_numpy(rng.integers(0, 256, (B, 3, H, H), dtype=np.uint8)).float() / 255,
           "observation.images.wrist": torch.from_numpy(rng.integers(0, 256, (B, 3, H, H), dtype=np.uint8)).float() / 255,
           "observation.state": torch.from_numpy(rng.normal(0, 0.3, (B, 19)).astype(np.float32)), "task": instr}
    bank = NoiseBank((pol.config.chunk_size, pol.config.max_action_dim))
    bank.reset(np.arange(B), np.array([11, 22, 33]))
    noise = bank.draw(np.arange(B)).to(a.device)

    def plain(rw: dict, nz: torch.Tensor) -> torch.Tensor:
        return pol.predict_action_chunk(srv.pre(dict(rw)), noise=nz)

    res: dict = {"ckpt": a.ckpt, "ckpt_sha": srv.ckpt_sha, "image_size": H, "instructions": instr, "others": others}
    t = time.time()
    g1 = guided_chunk(pol, srv.pre, raw, others, noise, 1.0)
    res["guided_s"] = round(time.time() - t, 2)
    texts, eor, tr = expand_rows(instr, others)
    r = torch.from_numpy(eor)
    big = {k: v[r] for k, v in raw.items() if k != "task"} | {"task": texts}
    pe = plain(big, noise[r.to(a.device)])[torch.from_numpy(tr).to(a.device)]
    res["a_w1_vs_plain_same_batch_maxabs"] = float((g1 - pe).abs().max())
    t = time.time()
    p0 = plain(raw, noise)
    res["plain_s"] = round(time.time() - t, 2)
    res["b_w1_vs_plain_B_rows_maxabs"] = float((g1 - p0).abs().max())
    self_others = [[s] for s in instr]
    res["c_self_others_w8_vs_plain_B_rows_maxabs"] = float((guided_chunk(pol, srv.pre, raw, self_others, noise, 8.0)
                                                           - p0).abs().max())
    for w in (2.0, 8.0):
        g = guided_chunk(pol, srv.pre, raw, others, noise, w)
        res[f"d_w{int(w)}_vs_w1_arm_l2_per_env"] = [float(x) for x in (g - g1)[..., :4].flatten(1).norm(dim=1)]
    res["plain_noise_q1_vs_q0_arm_l2_per_env"] = [float(x) for x in (plain(raw, bank.draw(np.arange(B)).to(a.device))
                                                                     - p0)[..., :4].flatten(1).norm(dim=1)]
    res["pass"] = bool(res["a_w1_vs_plain_same_batch_maxabs"] == 0.0 and res["b_w1_vs_plain_B_rows_maxabs"] < 1e-3
                       and res["c_self_others_w8_vs_plain_B_rows_maxabs"] < 1e-3
                       and min(res["d_w2_vs_w1_arm_l2_per_env"]) > 0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("G4_CHECK=" + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
