"""PERUN smoke (policy venv): CUDA, lerobot import, load lerobot/smolvla_base, one batched bf16 forward on random inputs.
  python hpc/smoke_policy.py --prefetch     # login node (has internet): fills $HF_HOME, CPU only
  python hpc/smoke_policy.py [--batch 16]   # GPU node
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--prefetch", action="store_true")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--out", default="")
a = p.parse_args()

import lerobot  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

dev = "cpu" if a.prefetch else "cuda"
if not a.prefetch:
    assert torch.cuda.is_available(), "CUDA unavailable -- STOP"
    print("CUDA OK", torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, "lerobot", lerobot.__version__, flush=True)
t0 = time.perf_counter()
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(dev).eval()
print(f"loaded smolvla_base on {next(policy.parameters()).device} in {time.perf_counter() - t0:.1f}s", flush=True)
if a.prefetch:
    raise SystemExit(0)

cfg, B = policy.config, a.batch
batch: dict[str, torch.Tensor] = {}
for k, f in cfg.image_features.items():
    batch[k] = torch.rand(B, *f.shape, device=dev)
batch["observation.state"] = torch.randn(B, cfg.robot_state_feature.shape[0], device=dev)
L = cfg.tokenizer_max_length
batch["observation.language.tokens"] = torch.randint(5, 1000, (B, L), device=dev)
batch["observation.language.attention_mask"] = torch.ones(B, L, dtype=torch.bool, device=dev)
ts = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(a.reps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
assert torch.isfinite(chunk).all()
res = dict(gpu=torch.cuda.get_device_name(0), batch=B, chunk_shape=list(chunk.shape), first_s=round(ts[0], 3),
           forward_s_median=round(sorted(ts[1:])[len(ts[1:]) // 2], 4), peak_vram_mib=int(torch.cuda.max_memory_allocated() / 2**20),
           images=list(cfg.image_features), hf_offline=os.environ.get("HF_HUB_OFFLINE"))
print("SMOLVLA_FORWARD", json.dumps(res), flush=True)
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=1))
