"""Fine-tune SmolVLA on a local LeRobot v3.0 dataset (policy venv). A thin explicit loop around LeRobot 0.6.1's own
SmolVLAPolicy, its pre/post processor pipelines and LeRobotDataset (not a fork of lerobot-train).

  policy\\.venv\\Scripts\\python.exe -m vla_policy.train_smolvla --dataset 0b_overfit --out runs/0b3_overfit \\
      --init lerobot/smolvla_base --n-action-steps 10 --epochs 24 --max-steps 5000 --batch-size 8 --lr 1e-4 \\
      --row-sampling window --freeze-vision-encoder true --train-expert-only true --video-backend pyav

What it reproduces from lerobot-train 0.6.1 (scripts/lerobot_train.py), explicitly:
  - policy: SmolVLAPolicy.from_pretrained(--init) with input features set to ours (camera1 = side, camera2 = wrist via
    the rename map, state 19, action 5; padded to 32 inside the model), chunk_size 50, n_action_steps N;
  - processors: make_pre_post_processors(cfg, pretrained_path=--init, overrides) = rename -> batch -> newline ->
    tokenizer (<= 48 tokens) -> device -> normalizer (MEAN_STD state / action, IDENTITY images); stats are ALWAYS
    re-estimated here from the fine-tuning dataset's parquet columns (never inherited from --init);
  - action chunk: delta_timestamps {"action": t + i / fps, i < 50}; `action_is_pad` masks the loss past the episode end;
  - optimizer / scheduler / clip: SmolVLAConfig presets (AdamW betas 0.9/0.95, wd 1e-10, eps 1e-8; cosine decay with
    warmup 1000 / decay 30000 -> 2.5e-6, auto-scaled by LeRobot when the run is shorter; grad-norm clip 10);
  - images uint8 -> float /255 before the preprocessor; bf16 autocast (CUDA, and CPU for the tests). lerobot-train runs
    SmolVLA without autocast (no `dtype` field, use_amp=False): the VLM / expert compute in their loaded bf16, the
    projections in fp32. Autocast here is the task-file choice, recorded.
Deviations (recorded in train_summary.json): `--trainable-fp32 true` (default) = fp32 master weights for every trainable
parameter (LeRobot updates the bf16-loaded expert / vision weights directly); `--refreeze-non-vision-vlm true` = GATES
Stage 1 arm U (both freeze flags false, then only the SigLIP vision encoder is trainable beyond arm F).
`--action-override <npz>` (Stage 1 fix diagnosis D2, scripts/s1_relabel.py build): every action chunk is re-read from
the npz's per-row `action` (same rows, same episode-end clamp as LeRobot's chunk) after checking that the dataset's own
column reproduces LeRobot's chunk bit for bit; the action normalisation stats come from the override.
Rows: `--row-sampling window` = GATES 0B option B (one random row per camera-frame window = the rows of one episode
that share `frame_idx`, i.e. show the same image), `all` = option A (every row). An epoch = one pass over the sampled
rows. The order is a pure function of (seed, epoch), so resume restarts mid-epoch sample-exactly.
Checkpoints: <out>/checkpoints/<step>/pretrained_model (policy.save_pretrained + policy_preprocessor/postprocessor,
loadable by `vla_policy.server --policy smolvla --ckpt <that dir>`) + training_state.pt; <out>/checkpoints/last.txt.
Logs: TensorBoard <out>/tb, per-step <out>/steps.jsonl (loss, lr, batch row indices), <out>/train_summary.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RENAME_MAP = {"observation.images.side": "observation.images.camera1",
              "observation.images.wrist": "observation.images.camera2"}  # contract v2 g (= vla_policy.server)
CHUNK = 50  # SmolVLA chunk_size (brief section 6)
STD_FLOOR = 1e-6  # GATES 0B.3 (b): a dim with std below this is near-constant and handled explicitly (std := 1)
WARMUP_STEPS = 50  # GATES 0B.6: steps excluded from the steady-state GPU-s/sample
PRINT_EVERY = 50


def str2bool(s: str) -> bool:
    if s.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError(f"expected true|false, got {s!r}")
    return s.lower() == "true"


def resolve_dataset(dataset: str, root: Path | None) -> Path:
    d = root / dataset if root else Path(dataset) if Path(dataset).is_dir() else ROOT / "datasets" / dataset
    if not (d / "meta" / "info.json").exists():
        raise FileNotFoundError(f"no local LeRobot dataset at {d} (the Hub is never used for data)")
    return d.resolve()


def open_dataset(path: Path, backend: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    fps = json.loads((path / "meta" / "info.json").read_text())["fps"]
    return LeRobotDataset(repo_id=f"local/{path.name}", root=path, video_backend=backend, return_uint8=True,
                          delta_timestamps={"action": [i / fps for i in range(CHUNK)]})


def column(ds, name: str) -> np.ndarray:
    return np.asarray(ds.hf_dataset.with_format("numpy")[name])


def norm_stats(ds, keep: np.ndarray | None = None, action: np.ndarray | None = None) -> tuple[dict, dict]:
    """MEAN_STD stats for state and action from the dataset's own rows (images are IDENTITY). `keep` (bool per row):
    only those rows count -- the trainers pass the train split (val episodes never shape the normalisation).
    Dims with std < STD_FLOOR get std = 1 (mean-centred only) and are listed in the report."""
    stats, report = {}, {}
    for k in ("observation.state", "action"):
        x = (column(ds, k) if k != "action" or action is None else action).astype(np.float64).reshape(len(ds), -1)
        x = x if keep is None else x[np.asarray(keep, bool)]
        mean, std = x.mean(0), x.std(0)
        flat = np.flatnonzero(std < STD_FLOOR)
        std[flat] = 1.0
        stats[k] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
                    "min": x.min(0).astype(np.float32), "max": x.max(0).astype(np.float32),
                    "count": np.array([len(x)])}
        report[k] = {"near_constant_dims": flat.tolist(), "std_floor": STD_FLOOR, "stats_rows": len(x)}
    return stats, report


def split_episodes(ds) -> tuple[np.ndarray, list[int], list[int]]:
    """(is_val per row, train episodes, val episodes) from the per-row `split` column the exporters write (export.py:
    all "train"; scripts/export_dataset.py: kept rank k % 20 == 19 -> "val", GATES 0C), cross-checked against
    meta/split_113.json when present. Only "train" and "val" are accepted."""
    sp = column(ds, "split").reshape(-1).astype(str)
    ep = column(ds, "episode_index").reshape(-1)
    assert set(np.unique(sp)) <= {"train", "val"}, np.unique(sp)
    is_val = sp == "val"
    val, train = sorted(set(ep[is_val].tolist())), sorted(set(ep[~is_val].tolist()))
    assert not set(val) & set(train), "an episode has rows of both splits"
    f = Path(ds.root) / "meta" / "split_113.json"
    if f.exists():
        j = json.loads(f.read_text())
        assert sorted(j["train"]) == train and sorted(j["val"]) == val, "split column != meta/split_113.json"
    return is_val, train, val


class ActionOverride:
    """Per-row replacement actions R (len(ds), 5): chunk of row i = R[min(i + j, episode end - 1)], j < CHUNK (LeRobot
    clamps out-of-episode timestamps to the episode's last row and sets action_is_pad; the pad mask is kept)."""

    def __init__(self, path: Path, ds):
        self.R = np.load(path)["action"].astype(np.float32)
        self.orig = column(ds, "action").astype(np.float32).reshape(len(ds), -1)
        assert self.R.shape == self.orig.shape, (self.R.shape, self.orig.shape)
        ep = column(ds, "episode_index").reshape(-1)
        ends = np.r_[np.flatnonzero(ep[1:] != ep[:-1]) + 1, len(ep)]
        self.end = ends[np.searchsorted(ends, np.arange(len(ep)), side="right")]

    def __call__(self, batch: dict) -> dict:
        i = batch["index"].numpy().reshape(-1)
        rows = np.minimum(i[:, None] + np.arange(CHUNK), self.end[i][:, None] - 1)
        assert np.array_equal(batch["action"].numpy(), self.orig[rows]), "LeRobot chunk != dataset column rows"
        batch["action"] = torch.from_numpy(self.R[rows])
        return batch


class WindowSampler(torch.utils.data.Sampler[int]):
    """Per epoch: one row drawn uniformly at random from every window, windows in shuffled order. Window = rows of one
    episode with the same `frame_idx` (row option B), or a single row (`frame_idx=None`, option A). The order is a
    pure function of (seed, epoch); `epoch` / `start` position a resume. Each `__iter__` advances the epoch.
    `keep` (bool per row): only those rows are ever sampled (the train split, GATES 0C); yields dataset positions."""

    def __init__(self, episode_index: np.ndarray, frame_idx: np.ndarray | None, seed: int,
                 keep: np.ndarray | None = None):
        ep = np.asarray(episode_index).reshape(-1)
        self.rows = np.arange(len(ep)) if keep is None else np.flatnonzero(keep)
        if frame_idx is None:
            self.window = np.arange(len(self.rows))
        else:
            key = np.stack([ep[self.rows], np.asarray(frame_idx).reshape(-1)[self.rows]], 1)
            self.window = np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)
        self.n = int(self.window.max()) + 1
        self.seed, self.epoch, self.start = seed, 0, 0

    def order(self, epoch: int) -> np.ndarray:
        g = np.random.default_rng([self.seed, epoch])
        o = np.lexsort((g.random(len(self.window)), self.window))  # by window, then a random key
        w = self.window[o]
        last = np.r_[w[1:] != w[:-1], True]  # the row with the largest random key in each window
        pick = np.empty(self.n, np.int64)
        pick[w[last]] = o[last]
        return self.rows[pick[g.permutation(self.n)]]

    def __iter__(self):
        e, s = self.epoch, self.start
        self.epoch, self.start = e + 1, 0
        return iter(self.order(e)[s:].tolist())

    def __len__(self) -> int:
        return self.n


def build_policy(init: str, ds, stats: dict, n_action_steps: int, fve: bool, teo: bool, device: str,
                 refreeze: bool = False, trainable_fp32: bool = True):
    """SmolVLA + processors exactly as lerobot-train builds them for `--policy.path=<init> --rename_map=...`, with
    the input features set to ours (as vla_policy.server does) and the stats from `stats`. Then (GATES Stage 1 arms /
    context.md 2026-10-01): `refreeze` re-freezes the non-vision VLM; `trainable_fp32` casts trainable params to fp32."""
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(init)
    cfg.n_action_steps, cfg.device = n_action_steps, device
    cfg.freeze_vision_encoder, cfg.train_expert_only = fve, teo
    f = ds.meta.features
    img = lambda k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *f[k]["shape"][:2]))  # noqa: E731
    cfg.input_features = {RENAME_MAP[k]: img(k) for k in RENAME_MAP}
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=tuple(f["observation.state"]["shape"]))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(f["action"]["shape"]))}
    assert cfg.chunk_size == CHUNK, cfg.chunk_size
    with torch.device(device):  # build the VLM on the target device: laptop CPU-RAM peak 3.6 -> 2.7 GiB, 4x faster
        policy = SmolVLAPolicy.from_pretrained(init, config=cfg)
    if refreeze:
        refreeze_non_vision_vlm(policy)
    if trainable_fp32:
        fp32_trainable(policy, init)
    feats = {**cfg.input_features, **cfg.output_features}
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=init,
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": RENAME_MAP},
                                "normalizer_processor": {"features": feats, "norm_map": cfg.normalization_mapping,
                                                         "stats": stats}},
        postprocessor_overrides={"unnormalizer_processor": {"features": cfg.output_features,
                                                            "norm_map": cfg.normalization_mapping, "stats": stats}})
    return policy.to(device), pre, post, cfg


def refreeze_non_vision_vlm(policy) -> None:
    """GATES Stage 1 arm U: with both freeze flags false, every `vlm_with_expert.vlm` parameter outside
    `vlm.model.vision_model` (text model incl. embeddings, connector, lm_head) goes back to requires_grad=False, so the
    only trainable part beyond arm F is the SigLIP vision encoder. Eval mode for them: `set_train`."""
    c = policy.config
    assert not c.freeze_vision_encoder and not c.train_expert_only, "--refreeze-non-vision-vlm needs both freeze flags false"
    vwe = policy.model.vlm_with_expert
    vis = {id(p) for p in vwe.get_vlm_model().vision_model.parameters()}
    for p in vwe.vlm.parameters():
        if id(p) not in vis:
            p.requires_grad = False


def set_train(policy, refreeze: bool) -> None:
    """policy.train() (LeRobot keeps its own frozen parts in eval); with `refreeze`, every VLM submodule outside the
    vision encoder that holds no trainable parameter is put back in eval mode."""
    policy.train()
    if refreeze:
        vwe = policy.model.vlm_with_expert
        vis = vwe.get_vlm_model().vision_model
        for m in vwe.vlm.modules():
            ps = list(m.parameters())
            if ps and not any(p.requires_grad for p in ps):
                m.eval()
        vis.train()


def fp32_trainable(policy, init: str) -> None:
    """fp32 master weights (orchestrator decision, context.md 2026-10-01; a deviation from LeRobot, which trains the
    bf16-loaded expert / vision weights directly): every TRAINABLE parameter -> fp32, frozen ones keep their dtype.
    When `init` is one of our checkpoints, the fp32 values it saved are re-read, since loading into the bf16 model
    rounded them (exact resume)."""
    names = set()
    for n, p in policy.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            names.add(n)
    f = Path(init) / "model.safetensors"
    if f.exists():
        from safetensors.torch import load_file
        sd = load_file(str(f), device=str(next(policy.parameters()).device))
        assert names <= sd.keys(), sorted(names - sd.keys())[:5]
        policy.load_state_dict({k: sd[k] for k in names}, strict=False)


# Trainable counts for smolvla_base (F, U) and arm VT (Stage 2 redesign #2: both freeze flags false, no refreeze; measured
# on CPU 2026-10-04: LeRobot itself freezes lm_head, text_model.norm.weight and text_model.layers.15 -> 392,904,096).
EXPECTED_TRAINABLE = {450_046_176: (99_880_992, 186_314_016, 392_904_096)}


def assert_expected(r: dict) -> None:
    if r["params_total"] in EXPECTED_TRAINABLE:
        assert r["params_trainable"] in EXPECTED_TRAINABLE[r["params_total"]], r  # only arm F, U or VT is valid


def trainable_parts(policy) -> dict:
    """Trainable parameter counts per part (vision encoder, text model, connector, expert = lm_expert, other = the
    action / state projections) and `frozen_names_prefixes`: the shortest module prefixes whose parameters are all
    frozen, with their parameter counts (arm VT: lm_head, text_model.norm, the last text layer)."""
    c = {"vision": 0, "text_model": 0, "connector": 0, "expert": 0, "other": 0}
    for n, p in policy.named_parameters():
        if p.requires_grad:
            c[next((k for k, s in (("vision", ".vision_model."), ("text_model", ".text_model."), ("connector", ".connector."),
                                   ("expert", ".lm_expert.")) if s in n), "other")] += p.numel()
    frozen = {n for n, p in policy.named_parameters() if not p.requires_grad}
    pre: dict[str, int] = {}
    for n, m in policy.named_modules():
        own = [f"{n}.{k}" if n else k for k, _ in m.named_parameters()]
        if own and all(x in frozen for x in own) and not any(n == q or n.startswith(q + ".") for q in pre):
            pre[n] = sum(p.numel() for p in m.parameters())
    return c | {"frozen_names_prefixes": pre}


def param_report(policy) -> dict:
    vis = policy.model.vlm_with_expert.get_vlm_model().vision_model
    dt: dict[str, int] = {}
    for p in policy.parameters():
        k = f"{'trainable' if p.requires_grad else 'frozen'}_{str(p.dtype).replace('torch.', '')}"
        dt[k] = dt.get(k, 0) + p.numel()
    r = {"params_total": sum(p.numel() for p in policy.parameters()),
         "params_trainable": sum(p.numel() for p in policy.parameters() if p.requires_grad),
         "vision_params_trainable": sum(p.numel() for p in vis.parameters() if p.requires_grad), "by_dtype": dt}
    assert_expected(r)
    return r


def to_float_images(batch: dict) -> dict:
    for k in RENAME_MAP:
        batch[k] = batch[k].float() / 255.0
    return batch


def rank_slice(batch: list, rank: int, world: int) -> list:
    """Rows [rank * k, (rank + 1) * k) of one global batch, k = len(batch) / world (also for the partial last batch)."""
    assert len(batch) % world == 0, f"global batch of {len(batch)} rows not divisible by world_size {world}"
    k = len(batch) // world
    return batch[rank * k:(rank + 1) * k]


class RankBatchSampler:
    """Data-parallel batches (opt-in `--distributed`): the single-GPU order `sampler.order(epoch)` cut into consecutive
    `gbs`-row global batches (partial last batch kept); this rank yields its `rank_slice` of each. `epoch` / `start`
    (row offset) position a resume exactly as WindowSampler's do; each `__iter__` advances the epoch."""

    def __init__(self, sampler: WindowSampler, gbs: int, rank: int, world: int):
        self.sampler, self.gbs, self.rank, self.world = sampler, gbs, rank, world
        self.epoch, self.start, self._cache = 0, 0, (-1, None)
        self.spe = math.ceil(len(sampler) / gbs)
        assert gbs % world == 0 and (len(sampler) % gbs) % world == 0, (gbs, world, len(sampler) % gbs)

    def global_rows(self, step: int) -> list[int]:
        """The global batch of optimizer step `step` (1-based) -- the rows logged to steps.jsonl."""
        e, k = (step - 1) // self.spe, (step - 1) % self.spe
        if self._cache[0] != e:
            self._cache = (e, self.sampler.order(e))
        return self._cache[1][k * self.gbs:(k + 1) * self.gbs].tolist()

    def __iter__(self):
        """A generator FUNCTION on purpose: the epoch advances when the first batch is pulled, not when iter() is called.
        With num_workers > 0 the DataLoader calls iter(batch_sampler) twice and drops the first iterator unused."""
        e, s = self.epoch, self.start
        self.epoch, self.start = e + 1, 0
        o = self.sampler.order(e)[s:]
        for i in range(0, len(o), self.gbs):
            yield rank_slice(o[i:i + self.gbs].tolist(), self.rank, self.world)

    def __len__(self) -> int:
        return self.spe


def allreduce_mean_grads(params: list, world: int) -> None:
    """In-place mean over ranks of every gradient (None -> zeros, so all ranks reduce the same tensors); one flat
    all_reduce(SUM) per dtype, then / world (SUM + divide: gloo has no AVG)."""
    import torch.distributed as dist
    for dt in {p.dtype for p in params}:
        ps = [p for p in params if p.dtype == dt]
        gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in ps]
        flat = torch.cat([g.reshape(-1) for g in gs])
        dist.all_reduce(flat)
        flat /= world
        for p, g in zip(ps, flat.split([p.numel() for p in ps])):
            p.grad = g.view_as(p).clone()


class ExtraBatchSampler(RankBatchSampler):
    """Fix candidate (`--cross-frame0-extra K`, NOT registered): RankBatchSampler's main rows (the registered global batch,
    unchanged and in the same order) PLUS K extra rows per global step drawn from `pool` (dataset positions of the
    frame-0 rows of the paired train episodes): uniform without replacement within a step, a pure function of
    default_rng([seed, step, 240]). This rank yields rank_slice(main) + its contiguous K / world slice of the extras, so
    the DataLoader workers load the extras. `step` = absolute optimizer step (1-based), epoch * spe + batch position."""

    def __init__(self, sampler: WindowSampler, gbs: int, rank: int, world: int, pool: np.ndarray, k: int, seed: int):
        super().__init__(sampler, gbs, rank, world)
        assert 0 < k <= len(pool) and k % world == 0, (k, len(pool), world)
        self.pool, self.k, self.seed = np.asarray(pool), k, seed

    def global_extra(self, step: int) -> list[int]:
        return np.random.default_rng([self.seed, step, 240]).choice(self.pool, self.k, replace=False).tolist()

    def __iter__(self):
        e, s = self.epoch, self.start
        self.epoch, self.start = e + 1, 0
        o = self.sampler.order(e)[s:]
        for b, i in enumerate(range(0, len(o), self.gbs)):
            ex = self.global_extra(e * self.spe + s // self.gbs + b + 1)
            yield rank_slice(o[i:i + self.gbs].tolist(), self.rank, self.world) + rank_slice(ex, self.rank, self.world)


class Contrast:
    """Task 005 `--contrastive-lambda`: settings + the per-step l' choice (vla_policy.contrastive). The loop sets `step`
    (the optimizer step being computed) before train_step; train_step leaves the global means of the step in `last`."""

    def __init__(self, table: dict, lam: float, margin: float, seed: int, share_vision: bool,
                 cross_lambda: float | None = None, first_chunk: dict | None = None):
        self.table, self.lam, self.margin, self.seed, self.share_vision = table, lam, margin, seed, share_vision
        self.cross_lambda, self.first_chunk = cross_lambda, first_chunk  # fix candidate (not registered); None = off
        self.step, self.last = 0, {}

    def others(self, batch: dict) -> list[str]:
        from vla_policy.contrastive import pick_other
        return pick_other(self.table, batch["episode_index"].reshape(-1).tolist(), list(batch["task"]),
                          batch["index"].reshape(-1).tolist(), self.seed, self.step)

    def others_eps(self, batch: dict) -> tuple[list[str], list[int]]:
        from vla_policy.contrastive import pick_other_eps
        return pick_other_eps(self.table, batch["episode_index"].reshape(-1).tolist(), list(batch["task"]),
                              batch["index"].reshape(-1).tolist(), self.seed, self.step)


def train_step(policy, pre, batch: dict, opt, sched, clip: float, dev: torch.device,
               autocast: bool = True, world: int = 1, weight_valid: bool = True,
               contrast: Contrast | None = None, n_extra: int = 0) -> tuple[float, float]:
    """autocast=False = lerobot-train's SmolVLA default (no AMP; only scripts/bench_train.py uses it, GATES 0B.6).
    world > 1 (data parallel, each rank holds its slice of the global batch): the per-rank loss is LeRobot's mean over
    the rank's valid (non-padded) action entries; `weight_valid` scales it by valid_r / mean_r(valid) so that the mean
    over ranks of the gradients equals the gradient of the single-process loss on the global batch (sum / total valid);
    False = the plain mean of the rank means. The gradients are all-reduced as a mean before the clip. The returned
    loss is the global one (same value on every rank).
    `contrast` (task 005): loss = L_true + lam * hinge, L_true = the loss above, hinge = mean over the rank's rows of
    max(0, m - (L_other_i - L_true_i)) (ranks hold equal row counts, so the mean of rank means is the global mean). The
    returned loss stays L_true (the flow loss every instrument reads); the step's global hinge / gap / active fraction /
    L_other / total go to `contrast.last`. `contrast.cross_lambda` (fix candidate, not registered; None = off): + lam' x
    the cross-instruction target term (contrastive.cross_terms) averaged over the global batch's frame-0 rows.
    `n_extra` (fix candidate, `--cross-frame0-extra`; needs `contrast.cross_lambda`): the LAST n_extra rows of the rank's
    batch are extra frame-0 rows; L_true (LeRobot's reduction recomputed from the true branch's per-sample sums), the
    hinge and the DDP weight use the main rows only, the cross term uses every frame-0 row."""
    hinge = cross = None
    if contrast is None:
        batch = pre(to_float_images(batch))
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast):  # also on CPU (tests)
            loss, _ = policy.forward(batch)
    else:
        from vla_policy.contrastive import cross_terms, flow_pair, hinge_terms
        raw = to_float_images(batch)
        others, other_eps = contrast.others_eps(raw) if contrast.cross_lambda is not None else (contrast.others(raw), None)
        batch, b_other = pre(dict(raw)), pre(dict(raw) | {"task": others})
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast):
            r = flow_pair(policy, batch, b_other, contrast.share_vision)
            ht = hinge_terms(r["lt"][:r["lt"].shape[0] - n_extra], r["lo"][:r["lo"].shape[0] - n_extra], contrast.margin)
        loss, hinge = r["l_true"], ht["hinge"]
        if n_extra:  # LeRobot's scalar = sum / valid entries, over the main rows only
            assert contrast.cross_lambda is not None
            m, pad, A = r["lt"].shape[0] - n_extra, batch.get("action_is_pad"), policy.config.action_feature.shape[0]
            nv = (torch.full((m,), batch["action"].shape[1]) if pad is None else (~pad[:m]).sum(1)).to(r["lt"].device) * A
            loss = (r["lt"][:m] * nv).sum() / nv.sum()
        if contrast.cross_lambda is not None:
            x_t = r["x_t"].float()
            a_o = torch.stack([contrast.first_chunk[int(o)] for o in other_eps]).to(x_t.device, torch.float32)
            a_o = torch.nn.functional.pad(a_o, (0, x_t.shape[-1] - a_o.shape[-1]))  # max_action_dim padding (zeros)
            rows = raw["frame_index"].reshape(-1).to(x_t.device) == 0
            ct = cross_terms(r["v_other"].float(), x_t, r["t"].float(), a_o, policy.config.action_feature.shape[0], rows)
            n_glob = ct["n"].detach().clone()
            if world > 1:
                import torch.distributed as dist
                dist.all_reduce(n_glob)
            cross = ct["sum"] * (world / n_glob.clamp_min(1.0))  # mean over ranks of the grads = the global row mean
    aux = None if hinge is None else contrast.lam * hinge + (0.0 if cross is None else contrast.cross_lambda * cross)
    if world > 1:
        import torch.distributed as dist
        w = 1.0
        if weight_valid:
            v = (~batch["action_is_pad"][:batch["action_is_pad"].shape[0] - n_extra]).sum().double().reshape(1)
            tot = v.clone()
            dist.all_reduce(tot)
            w = (v * world / tot).item()
        (loss * w if aux is None else loss * w + aux).backward()
        allreduce_mean_grads([p for p in policy.parameters() if p.requires_grad], world)
        g = (loss.detach() * w).double().reshape(1) / world
        dist.all_reduce(g)
        loss_value = g.item()
    elif aux is None:
        loss.backward()
    else:
        (loss + aux).backward()
    if hinge is not None:
        s = torch.stack([hinge.detach(), ht["gap"], ht["active"], ht["lo"]]).double()
        if world > 1:
            dist.all_reduce(s)
            s /= world
        h, gap, act, lo = s.tolist()
        lt = loss_value if world > 1 else loss.item()
        contrast.last = {"hinge": h, "gap": gap, "hinge_active": act, "l_other": lo,
                         "loss_total": lt + contrast.lam * h}
        if cross is not None:
            c = (cross.detach().double() / world).reshape(1)
            if world > 1:
                dist.all_reduce(c)
            contrast.last |= {"cross": c.item(), "cross_rows": int(n_glob.item()),
                              "loss_total": contrast.last["loss_total"] + contrast.cross_lambda * c.item()}
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    return (loss_value if world > 1 else loss.item()), gn.item()


def rng_state(dev: torch.device) -> dict:
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(s: dict, dev: torch.device) -> None:
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state(s["cuda"], dev)
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def save_ckpt(out: Path, step: int, policy, pre, post, opt, sched, dev: torch.device, run_key: dict,
              rng_ranks: list | None = None) -> Path:
    d = out / "checkpoints" / f"{step:06d}"
    policy.save_pretrained(d / "pretrained_model")
    pre.save_pretrained(d / "pretrained_model")
    post.save_pretrained(d / "pretrained_model")
    torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "rng": rng_state(dev),
                "run_key": run_key} | ({} if rng_ranks is None else {"rng_ranks": rng_ranks}), d / "training_state.pt")
    (out / "checkpoints" / "last.txt").write_text(d.name)
    return d / "pretrained_model"


def git_sha() -> str:
    """HEAD of the repo; on PERUN the repo is a `git archive` with the sha in GIT_HEAD (hpc/push_repo.sh)."""
    if (ROOT / "GIT_HEAD").exists():
        return (ROOT / "GIT_HEAD").read_text().strip()
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, help="local dataset dir, or a name under --root (default datasets/)")
    p.add_argument("--root", type=Path, default=None, help="parent dir of --dataset")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", default="lerobot/smolvla_base", help="Hub id or a pretrained_model dir of ours")
    p.add_argument("--freeze-vision-encoder", type=str2bool, default=True)
    p.add_argument("--train-expert-only", type=str2bool, default=True)
    p.add_argument("--refreeze-non-vision-vlm", type=str2bool, default=False,
                   help="GATES Stage 1 arm U: with both freeze flags false, train only the vision encoder beyond arm F")
    p.add_argument("--trainable-fp32", type=str2bool, default=True, help="fp32 master weights for trainable params")
    p.add_argument("--n-action-steps", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=None, help="cap on optimizer steps (also the scheduler horizon)")
    p.add_argument("--row-sampling", choices=("window", "all"), default="window")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="peak lr (default: SmolVLAConfig.optimizer_lr = 1e-4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every-steps", type=int, default=0, help="0 = only at the end")
    p.add_argument("--resume", action="store_true", help="continue <out> from checkpoints/last.txt")
    p.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (per rank under --distributed)")
    p.add_argument("--distributed", choices=("off", "nccl", "gloo"), default="off",
                   help="data parallel over the torchrun processes (RANK / WORLD_SIZE / LOCAL_RANK env): --batch-size "
                        "stays the GLOBAL batch (world x per-rank), the global order is the single-GPU one")
    p.add_argument("--ddp-grad-weight", choices=("valid", "rows"), default="valid",
                   help="distributed: rank losses weighted by valid action entries (exact global loss) or plain rank mean")
    p.add_argument("--device", default="cuda")
    p.add_argument("--action-override", type=Path, default=None,
                   help="npz with a per-row `action` replacing the labels (scripts/s1_relabel.py build)")
    p.add_argument("--cosine-to-total", action="store_true",
                   help="GATES Stage 1 redesign: LeRobot's own short-run scaling (warmup x total / decay, decay = total) "
                        "applied also when total > the preset decay steps (LeRobot only scales shorter runs)")
    p.add_argument("--contrastive-lambda", type=float, default=None,
                   help="task 005: instruction-contrastive hinge weight (off when omitted; needs a paired-scene dataset)")
    p.add_argument("--contrastive-margin", type=float, default=None, help="task 005: hinge margin m (GATES 005 rule)")
    p.add_argument("--contrastive-share-vision", type=str2bool, default=True,
                   help="task 005: the l' branch reuses the l branch's image embeddings (vision tower runs once)")
    p.add_argument("--cross-lambda", type=float, default=None,
                   help="fix candidate (consultant 113, NOT registered): weight of the cross-instruction flow target "
                        "||t v(x_t | l') - (x_t - a_other)||^2 on frame-0 rows (needs --contrastive-lambda)")
    p.add_argument("--cross-frame0-extra", type=int, default=0,
                   help="fix candidate (NOT registered): K extra frame-0 rows of paired train episodes per global step, "
                        "added to the unchanged main batch (needs --cross-lambda; K %% world == 0)")
    p.add_argument("--overfit-abort", action="store_true",
                   help="GATES 0B.3 early abort: stop if mean loss over steps 951-1000 >= 50%% of mean over steps 1-50")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    a = parse(argv)
    rank, world = 0, 1
    if a.distributed != "off":  # data parallel under torchrun: one process per GPU, device cuda:LOCAL_RANK
        import torch.distributed as dist
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        dist.init_process_group(a.distributed)
        if a.distributed == "nccl":
            a.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
            torch.cuda.set_device(a.device)
        assert a.batch_size % world == 0, f"global batch {a.batch_size} not divisible by world_size {world}"
    say = print if rank == 0 else lambda *x, **k: None
    dev = torch.device(a.device)
    if dev.type == "cuda":
        assert torch.cuda.is_available(), "--device cuda but torch.cuda.is_available() is False. STOP."
    say(f"[train] device={dev} cuda_available={torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(dev) if dev.type == 'cuda' else platform.processor()}"
        + (f" distributed={a.distributed} world_size={world} per_rank_batch={a.batch_size // world} "
           f"rank_seeds={[a.seed + r for r in range(world)]}" if world > 1 else ""), flush=True)
    random.seed(a.seed + rank), np.random.seed(a.seed + rank), torch.manual_seed(a.seed + rank)  # rank 0 = single
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    dpath = resolve_dataset(a.dataset, a.root)
    ds = open_dataset(dpath, a.video_backend)
    frame_idx = column(ds, "frame_idx") if a.row_sampling == "window" else None
    is_val, train_eps, val_eps = split_episodes(ds)
    ovr = ActionOverride(a.action_override, ds) if a.action_override else None
    stats, stats_report = norm_stats(ds, ~is_val, None if ovr is None else ovr.R)  # train split only
    sampler = WindowSampler(column(ds, "episode_index"), frame_idx, a.seed, keep=~is_val)  # train split only
    spe = math.ceil(len(sampler) / a.batch_size)
    total = a.epochs * spe if a.max_steps is None else min(a.epochs * spe, a.max_steps)
    run_key = {"dataset": str(dpath), "rows": len(ds), "n_train_episodes": len(train_eps), "samples_per_epoch": len(sampler), "batch_size": a.batch_size,
               "seed": a.seed, "row_sampling": a.row_sampling, "epochs": a.epochs, "max_steps": a.max_steps}
    if a.cosine_to_total:
        run_key["cosine_to_total"] = True
    if ovr is not None:
        run_key["action_override"] = str(a.action_override)
    contrast = None
    K = a.cross_frame0_extra
    assert K == 0 or (a.cross_lambda is not None and K > 0 and K % world == 0), \
        "--cross-frame0-extra K needs --cross-lambda and K % world_size == 0"
    assert a.cross_lambda is None or a.contrastive_lambda is not None, "--cross-lambda needs --contrastive-lambda"
    if a.contrastive_lambda is not None:  # task 005
        from vla_policy.contrastive import read_table
        assert a.contrastive_margin is not None and a.contrastive_margin > 0, "--contrastive-margin m > 0 required"
        table = read_table(dpath)
        missing = sorted(set(train_eps) - set(table))
        assert not missing, f"train episodes without a layout group: {missing[:10]}"
        first = None
        if a.cross_lambda is not None:  # every table episode's first chunk, normalized as the preprocessor does
            act = column(ds, "action").astype(np.float64).reshape(len(ds), -1)
            epi = column(ds, "episode_index").reshape(-1)
            mu, sd = (stats["action"][k].astype(np.float64) for k in ("mean", "std"))
            start = {int(e): int(i) for e, i in zip(*np.unique(epi, return_index=True))}
            first = {e: torch.as_tensor((act[start[e]:start[e] + CHUNK] - mu) / (sd + 1e-8), dtype=torch.float32)
                     for e in table}
            assert all(v.shape == (CHUNK, act.shape[1]) and np.all(epi[start[e]:start[e] + CHUNK] == e)
                       for e, v in first.items()), "an episode is shorter than one chunk"
            run_key["cross_lambda"] = a.cross_lambda
            if K:
                run_key["cross_frame0_extra"] = K
        contrast = Contrast(table, a.contrastive_lambda, a.contrastive_margin, a.seed, a.contrastive_share_vision,
                            a.cross_lambda, first)
        run_key["contrastive"] = {"lambda": a.contrastive_lambda, "margin": a.contrastive_margin,
                                  "share_vision": a.contrastive_share_vision}
        say(f"[train] contrastive lambda={a.contrastive_lambda} margin={a.contrastive_margin} "
            f"share_vision={a.contrastive_share_vision} layouts={len({r['group'] for r in table.values()})} "
            f"episodes_in_table={len(table)}", flush=True)
    say(f"[train] dataset {dpath} episodes={ds.meta.total_episodes} (train {len(train_eps)}, val {len(val_eps)} "
        f"never sampled) rows={len(ds)} samples/epoch={len(sampler)} "
        f"steps/epoch={spe} total_steps={total}", flush=True)

    init_sha256 = None
    if Path(a.init).is_dir():  # GATES Stage 2: the init weights' sha256 is recorded (parent checkpoint identity)
        init_sha256 = hashlib.sha256((Path(a.init) / "model.safetensors").read_bytes()).hexdigest()
        say(f"[train] init_sha256={init_sha256} ({a.init})", flush=True)
    step, init, ts = 0, a.init, None
    last = a.out / "checkpoints" / "last.txt"
    if a.resume and last.exists():
        ck = a.out / "checkpoints" / last.read_text().strip()
        ts = torch.load(ck / "training_state.pt", map_location="cpu", weights_only=False)
        assert ts["run_key"] == run_key, f"resume with different settings: {ts['run_key']} != {run_key}"
        step, init = ts["step"], str(ck / "pretrained_model")
    policy, pre, post, cfg = build_policy(init, ds, stats, a.n_action_steps, a.freeze_vision_encoder,
                                          a.train_expert_only, a.device, a.refreeze_non_vision_vlm, a.trainable_fp32)
    pdev = next(policy.parameters()).device
    assert pdev.type == dev.type, f"policy on {pdev}, expected {dev}"
    params = param_report(policy)
    parts = trainable_parts(policy)
    say(f"[train] policy on {pdev}; {params}; freeze_vision_encoder={cfg.freeze_vision_encoder} "
        f"train_expert_only={cfg.train_expert_only}; trainable_parts={parts}", flush=True)
    if not cfg.freeze_vision_encoder and cfg.train_expert_only and not a.refreeze_non_vision_vlm:
        say("[train] WARNING: train_expert_only=true freezes the whole VLM incl. the vision encoder, so "
              "freeze_vision_encoder=false has no effect (LeRobot 0.6.1 SmolVLMWithExpertModel.set_requires_grad)")

    if a.lr is not None:
        cfg.optimizer_lr = a.lr
    ocfg, scfg = cfg.get_optimizer_preset(), cfg.get_scheduler_preset()
    if a.cosine_to_total:
        scfg.num_warmup_steps = int(scfg.num_warmup_steps * total / scfg.num_decay_steps)
        scfg.num_decay_steps = total
    opt = ocfg.build(policy.get_optim_params())
    sched = scfg.build(opt, total)
    if ts is not None:
        opt.load_state_dict(ts["optimizer"]), sched.load_state_dict(ts["scheduler"])
        if world == 1:
            set_rng_state(ts["rng"], dev)
        elif len(ts.get("rng_ranks", [])) == world:  # a data-parallel checkpoint: every rank's own state
            set_rng_state(ts["rng_ranks"][rank], dev)
        elif rank == 0:  # a single-process one: rank 0 continues it, the others keep their seed + rank streams
            set_rng_state(ts["rng"], dev)
        say(f"[train] resumed at step {step} from {init}", flush=True)
    sampler.epoch, sampler.start = step // spe, (step % spe) * a.batch_size
    if K:  # main rows = the registered batch, then this rank's slice of K extra frame-0 rows (loaded by the workers)
        f0 = column(ds, "frame_index").reshape(-1) == 0
        pool = np.flatnonzero(f0 & ~is_val & np.isin(column(ds, "episode_index").reshape(-1), list(table)))
        rbs = ExtraBatchSampler(sampler, a.batch_size, rank, world, pool, K, a.seed)
        rbs.epoch, rbs.start = sampler.epoch, sampler.start
        dl = torch.utils.data.DataLoader(ds, batch_sampler=rbs, num_workers=a.workers, pin_memory=dev.type == "cuda",
                                         persistent_workers=a.workers > 0, generator=torch.Generator().manual_seed(a.seed))
    elif world == 1:
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                                         pin_memory=dev.type == "cuda", drop_last=False,
                                         persistent_workers=a.workers > 0,
                                         generator=torch.Generator().manual_seed(a.seed))  # keeps the global RNG clean
    else:  # this rank's slice of every global batch; the global order is the single-GPU one
        rbs = RankBatchSampler(sampler, a.batch_size, rank, world)
        rbs.epoch, rbs.start = sampler.epoch, sampler.start
        dl = torch.utils.data.DataLoader(ds, batch_sampler=rbs, num_workers=a.workers, pin_memory=dev.type == "cuda",
                                         persistent_workers=a.workers > 0, generator=torch.Generator().manual_seed(a.seed))

    slog = a.out / "steps.jsonl"
    if rank == 0:  # under --distributed rank 0 alone writes the logs and the checkpoints
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(a.out / "tb"), purge_step=step)
        kept = [x for x in slog.read_text().splitlines() if json.loads(x)["step"] <= step] if ts is not None else []
        slog.write_text("".join(x + "\n" for x in kept))  # drop steps logged after the checkpoint we resume from
        log = slog.open("a")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    set_train(policy, a.refreeze_non_vision_vlm)
    losses: list[float] = []
    closs: list[dict] = []
    steady_s, steady_n, steady_x, data_s, ckpt = 0.0, 0, 0, 0.0, None
    start_step, aborted = step, None
    t_loop = time.time()
    while step < total:
        t0 = time.perf_counter()
        for batch in dl:
            t1 = time.perf_counter()
            idx = batch["index"].tolist()
            xidx = None
            if K:  # main rows (index in steps.jsonl) + extras (extra_index); both asserted against the sampler
                kx = K // world
                gidx, xidx = rbs.global_rows(step + 1), rbs.global_extra(step + 1)
                assert idx[:-kx] == rank_slice(gidx, rank, world) and idx[-kx:] == rank_slice(xidx, rank, world), (step + 1, rank)
                idx = gidx
            elif world > 1:  # steps.jsonl / sample counts carry the GLOBAL batch; this rank got exactly its slice of it
                gidx = rbs.global_rows(step + 1)
                assert idx == rank_slice(gidx, rank, world), (step + 1, rank)
                idx = gidx
            assert not is_val[idx].any(), f"val rows sampled: {[i for i in idx if is_val[i]]}"  # GATES 0C
            if ovr is not None:
                batch = ovr(batch)
            if contrast is not None:
                contrast.step = step + 1
            loss, gn = train_step(policy, pre, batch, opt, sched, ocfg.grad_clip_norm, dev, world=world,
                                  weight_valid=a.ddp_grad_weight == "valid", contrast=contrast,
                                  n_extra=K // world)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t2 = time.perf_counter()
            step += 1
            losses.append(loss)
            lr = opt.param_groups[0]["lr"]
            if step - start_step > WARMUP_STEPS:
                steady_s, steady_n, steady_x, data_s = steady_s + t2 - t0, steady_n + len(idx), steady_x + K, data_s + t1 - t0
            cl = contrast.last if contrast is not None else {}
            if cl:
                closs.append(cl)
            if rank == 0:
                for k, v in (("loss", loss), ("grad_norm", gn), ("lr", lr), ("update_s", t2 - t1), ("data_s", t1 - t0),
                             *cl.items()):
                    tb.add_scalar(f"train/{k}", v, step)
                log.write(json.dumps({"step": step, "epoch": (step - 1) // spe, "loss": loss, "lr": lr, **cl,
                                      "index": idx, **({"extra_index": xidx, "extra_rows": K} if K else {})}) + "\n")
            if step % PRINT_EVERY == 0 or step == total:
                if rank == 0:
                    log.flush()
                say(f"[train] step {step}/{total} epoch {(step - 1) // spe} loss {loss:.4f} lr {lr:.2e} "
                    f"gn {gn:.2f} update {t2 - t1:.3f}s data {t1 - t0:.3f}s"
                    + (f" total {cl['loss_total']:.4f} hinge {cl['hinge']:.4f} gap {cl['gap']:.4f} "
                       f"active {cl['hinge_active']:.3f}" if cl else ""), flush=True)
            if a.overfit_abort and step == 1000 and start_step == 0:
                ref, now = float(np.mean(losses[:50])), float(np.mean(losses[950:1000]))
                aborted = {"loss_mean_1_50": ref, "loss_mean_951_1000": now, "abort": now >= 0.5 * ref}
                say(f"[train] overfit early-abort check: {aborted}", flush=True)
            if (a.save_every_steps and step % a.save_every_steps == 0) or step == total or (aborted or {}).get("abort"):
                rngs = None
                if world > 1:
                    rngs = [None] * world
                    dist.all_gather_object(rngs, rng_state(dev))
                if rank == 0:
                    ckpt = save_ckpt(a.out, step, policy, pre, post, opt, sched, dev, run_key, rngs)
                if world > 1:
                    dist.barrier()
            t0 = time.perf_counter()
            if step >= total or (aborted or {}).get("abort"):
                break
        if (aborted or {}).get("abort"):
            break
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return {}
    log.close()
    tb.close()
    wall = time.time() - t_start
    with (a.out / "steps.jsonl").open() as fh:
        rows = [json.loads(x) for x in fh]
    samples = sum(len(r["index"]) for r in rows if r["step"] > start_step)
    summary = {
        "git_sha": git_sha(), "argv": vars(a) | {"out": str(a.out), "root": str(a.root)}, "init": a.init, "init_sha256": init_sha256,
        "resumed_from_step": start_step if ts is not None else None, "final_step": step, "total_steps": total,
        "samples_seen_this_run": samples, "samples_per_epoch": len(sampler), "steps_per_epoch": spe,
        "epochs_done": step / spe, "dataset": str(dpath), "episodes": ds.meta.total_episodes, "rows": len(ds),
        "n_train_episodes": len(train_eps), "n_val_episodes": len(val_eps), "train_rows": int((~is_val).sum()),
        "fps": ds.fps, "image_shape": list(ds.meta.features["observation.images.side"]["shape"]),
        "wall_s": wall, "loop_s": time.time() - t_loop,
        "steady_state": {"warmup_steps_excluded": WARMUP_STEPS, "samples": steady_n, "s": steady_s, **({"extra_samples": steady_x} if K else {}),
                         "gpu_s_per_sample": steady_s * world / steady_n if steady_n else None,
                         "data_wait_frac": data_s / steady_s if steady_s else None},
        "peak_mem_allocated_gib": torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else None,
        "peak_mem_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30 if dev.type == "cuda" else None,
        "device": str(pdev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "dtype": "bf16 autocast",
        "flags": {"freeze_vision_encoder": cfg.freeze_vision_encoder, "train_expert_only": cfg.train_expert_only,
                  "train_state_proj": cfg.train_state_proj, "chunk_size": cfg.chunk_size,
                  "n_action_steps": cfg.n_action_steps, "tokenizer_max_length": cfg.tokenizer_max_length,
                  "normalization_mapping": {k: str(v.value) for k, v in cfg.normalization_mapping.items()},
                  "rename_map": RENAME_MAP, "row_sampling": a.row_sampling,
                  "refreeze_non_vision_vlm": a.refreeze_non_vision_vlm, "trainable_fp32": a.trainable_fp32},
        "deviations_from_lerobot_default": ["bf16 autocast (lerobot-train: fp32 for SmolVLA)"]
        + (["trainable params cast to fp32 master weights (LeRobot trains bf16 weights directly)"] if a.trainable_fp32 else [])
        + (["non-vision VLM re-frozen (GATES Stage 1 arm U)"] if a.refreeze_non_vision_vlm else []),
        "params": params, "trainable_parts": parts, "optimizer": vars(ocfg), "scheduler": vars(scfg),
        "norm_stats": {k: {s: v.tolist() for s, v in d.items() if s in ("mean", "std")} for k, d in stats.items()},
        "norm_report": stats_report, "stats_rows": int((~is_val).sum()), "stats_split": "train", "loss_first": losses[:1], "loss_last50_mean": float(np.mean(losses[-50:])) if losses else None,
        "overfit_abort_check": aborted, "final_ckpt": str(ckpt) if ckpt else None,
        "torch": torch.__version__, "lerobot": __import__("lerobot").__version__,
    }
    if contrast is not None:
        last = closs[-50:]
        summary["contrastive"] = run_key["contrastive"] | {
            "layouts": len({r["group"] for r in contrast.table.values()}),  # (stage, seed, layout_id) groups
            "last50_mean": {k: float(np.mean([c[k] for c in last])) for k in last[0]} if last else None,
            "first50_mean": {k: float(np.mean([c[k] for c in closs[:50]])) for k in closs[0]} if closs else None}
        summary["deviations_from_lerobot_default"].append(
            "task 005 instruction-contrastive hinge added to the flow loss (steps.jsonl `loss` = the flow loss L_true)")
    if K:
        summary["cross_frame0_extra"] = {"k": K, "extra_samples_seen_this_run": K * sum(r["step"] > start_step for r in rows),
                                         "pool_episodes": len(pool)}
    if world > 1:
        summary |= {"world_size": world, "per_rank_batch": a.batch_size // world, "backend": a.distributed,
                    "rank_seeds": [a.seed + r for r in range(world)], "global_batch": a.batch_size,
                    "ddp_grad_weight": a.ddp_grad_weight}
    (a.out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    say(f"[train] done: {json.dumps({k: summary[k] for k in ('final_step', 'samples_seen_this_run', 'wall_s', 'steady_state', 'peak_mem_allocated_gib', 'final_ckpt')}, default=str)}", flush=True)
    if (aborted or {}).get("abort"):
        raise SystemExit("[train] GATES 0B.3 early abort fired: loss at step 1000 not below 50% of the step 1-50 mean")
    return summary


if __name__ == "__main__":
    main()
