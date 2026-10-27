"""0B.6 training throughput: GPU-seconds per sample of SmolVLA fine-tuning, frozen vs unfrozen, on a real exported
dataset at its image size, one GPU (policy venv; Windows pyav or Linux torchcodec).

  python scripts/bench_train.py --dataset <dir> --batch-size 64 --workers 8 --video-backend torchcodec \
      --out results/0b6_throughput_bs64.json [--configs frozen unfrozen] [--warmup 50 --steps 300]

Configs (LeRobot 0.6.1 flags): frozen = freeze_vision_encoder true + train_expert_only true (smolvla_base default);
unfrozen = GATES Stage 1 arm U: both flags false + the non-vision VLM re-frozen (only the SigLIP vision encoder trains
beyond F). (freeze_vision_encoder=false alone changes nothing: train_expert_only=true freezes the whole VLM.)
Trainable params are fp32 master weights in both (trainer default --trainable-fp32 true).
Per config, the real trainer step (vla_policy.train_smolvla.train_step: preprocessor, bf16 autocast, backward, clip,
AdamW, scheduler) runs `--warmup` steps, then `--steps` timed steps fed by the real DataLoader (end to end, data wait
measured), then `--steps` timed steps cycling 8 cached batches (pure GPU time; the GPU-s/sample used for the plan).
Every timed step ends with torch.cuda.synchronize. Rows sampled with the window sampler (GATES 0B row option B).
"""

from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_policy import train_smolvla as T  # noqa: E402

# (fve, train_expert_only, refreeze, trainable_fp32, bf16 autocast): frozen / unfrozen = our trainer (arms F / U);
# frozen_lerobot = arm F as lerobot-train runs SmolVLA (bf16-loaded weights trained directly, no autocast; GATES 0B.6
# "LeRobot default dtype", recorded next to ours).
CONFIGS = {"frozen": (True, True, False, True, True), "unfrozen": (False, False, True, True, True),
           "frozen_lerobot": (True, True, False, False, False)}
N_CACHED = 8


def timed(fn, n: int, dev: torch.device) -> tuple[float, float]:
    """(total wall s, data-wait s) of n calls of fn() -> data-wait seconds."""
    torch.cuda.synchronize(dev)
    t0, wait = time.perf_counter(), 0.0
    for _ in range(n):
        wait += fn()
    torch.cuda.synchronize(dev)
    return time.perf_counter() - t0, wait


def bench(a: argparse.Namespace, name: str, ds, stats: dict, dev: torch.device) -> dict:
    fve, teo, refreeze, fp32, amp = CONFIGS[name]
    policy, pre, _, cfg = T.build_policy(a.init, ds, stats, a.n_action_steps, fve, teo, a.device, refreeze, fp32)
    assert next(policy.parameters()).device.type == "cuda"
    ocfg = cfg.get_optimizer_preset()
    opt = ocfg.build(policy.get_optim_params())
    sched = cfg.get_scheduler_preset().build(opt, a.warmup + 2 * a.steps)
    sampler = T.WindowSampler(T.column(ds, "episode_index"), T.column(ds, "frame_idx"), a.seed,
                              keep=~T.split_episodes(ds)[0])  # train split only, as the trainer
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    it, cache = iter(dl), []

    def next_batch() -> dict:
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    def loader_step() -> float:
        t = time.perf_counter()
        b = next_batch()
        w = time.perf_counter() - t
        cache.append(copy.copy(b))
        del cache[:-N_CACHED]
        T.train_step(policy, pre, b, opt, sched, ocfg.grad_clip_norm, dev, amp)
        return w

    k = 0

    def cached_step() -> float:
        nonlocal k
        T.train_step(policy, pre, copy.copy(cache[k % len(cache)]), opt, sched, ocfg.grad_clip_norm, dev, amp)
        k += 1
        return 0.0

    T.set_train(policy, refreeze)
    torch.cuda.reset_peak_memory_stats(dev)
    warm_s, _ = timed(loader_step, a.warmup, dev)
    e2e_t0 = time.time()
    e2e_s, wait_s = timed(loader_step, a.steps, dev)
    gpu_t0 = time.time()
    gpu_s, _ = timed(cached_step, a.steps, dev)
    gpu_t1 = time.time()
    n = a.steps * a.batch_size
    r = {"freeze_vision_encoder": cfg.freeze_vision_encoder, "train_expert_only": cfg.train_expert_only,
         "refreeze_non_vision_vlm": refreeze, "trainable_fp32": fp32, "bf16_autocast": amp, "workers": a.workers,
         "phase_epoch_s": {"e2e": [e2e_t0, gpu_t0], "gpu_cached": [gpu_t0, gpu_t1]},  # join with nvidia-smi samples
         **T.param_report(policy), "warmup_steps": a.warmup, "timed_steps": a.steps, "warmup_s": warm_s,
         "gpu_s_per_sample": gpu_s / n, "e2e_s_per_sample": e2e_s / n, "data_wait_frac_e2e": wait_s / e2e_s,
         "samples_per_s_gpu": n / gpu_s, "samples_per_s_e2e": n / e2e_s,
         "peak_mem_allocated_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
         "peak_mem_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30}
    print(f"[bench] {name}: {json.dumps(r)}", flush=True)
    del policy, pre, opt, sched, dl, it, cache
    gc.collect()
    torch.cuda.empty_cache()
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--init", default="lerobot/smolvla_base")
    p.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=T.WARMUP_STEPS)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    p.add_argument("--seed", type=int, default=21500)  # GATES seed registry: 0B throughput
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    assert dev.type == "cuda" and torch.cuda.is_available(), "bench needs CUDA. STOP."
    torch.manual_seed(a.seed)
    dpath = T.resolve_dataset(a.dataset, a.root)
    ds = T.open_dataset(dpath, a.video_backend)
    stats, _ = T.norm_stats(ds)
    gpu = torch.cuda.get_device_name(dev)
    print(f"[bench] {gpu}; dataset {dpath} rows={len(ds)}; bs={a.batch_size} workers={a.workers}", flush=True)
    t0 = time.time()
    out = {"git_sha": T.git_sha(), "gpu": gpu, "torch": torch.__version__, "lerobot": __import__("lerobot").__version__,
           "dataset": str(dpath), "rows": len(ds), "episodes": ds.meta.total_episodes,
           "image_shape": list(ds.meta.features["observation.images.side"]["shape"]), "batch_size": a.batch_size,
           "workers": a.workers, "video_backend": a.video_backend, "dtype": "per config (bf16_autocast, trainable_fp32)",
           "row_sampling": "window",
           "init": a.init, "n_action_steps": a.n_action_steps, "seed": a.seed,
           "configs": {c: bench(a, c, ds, stats, dev) for c in a.configs}}
    out["wall_s"] = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {a.out} in {out['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
