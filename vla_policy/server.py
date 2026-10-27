"""Batched policy server (docs/CONTRACT.md "Bridge" + v2 b/c), policy venv. One `act` = one forward for all envs in it.

  python -m vla_policy.server --policy smolvla --ckpt lerobot/smolvla_base --dataset datasets/<name> \
      --n-action-steps 10 --period 3 --latency 1 --device cuda [--port 50731]
  python -m vla_policy.server --policy smolvla --ckpt outputs/train/<run>/checkpoints/last/pretrained_model ...
  python -m vla_policy.server --policy baseline --ckpt <run>/epoch_XXX --image-size 128 --n-action-steps 20 ...

The shared secret comes from the VLA_AUTHKEY environment variable (vla_common.protocol). PERIOD / LATENCY / N are
passed explicitly (vla_common/timing.py is their single source; the sim client passes the same values and checks
`info`).

SmolVLA (brief section 6 [consultant] mapping, contract v2 g):
  - cameras: observation.images.side -> camera1, observation.images.wrist -> camera2 (RENAME_MAP, the rename
    lerobot-train applies with --rename_map); no third / empty camera (empty_cameras = 0);
  - state 19 -> padded to max_state_dim 32, action 5 -> padded to max_action_dim 32 (inside SmolVLA);
  - chunk_size 50, n_action_steps = N (the first N actions of each predicted chunk are returned);
  - MEAN_STD for state and action, IDENTITY for images (smolvla_base normalization_mapping);
  - instructions tokenized to at most 48 tokens, padded to max_length (batch composition cannot change the tokens);
  - the claw (action dim 4) is thresholded at 0.5 after un-normalising;
  - served in fp32 (LeRobot loads the VLM + expert in bf16; see SmolVLAServer).
Stateless (v2 b): `predict_action_chunk` (no LeRobot action queue); the flow-matching start noise of env i at its q-th
query since `reset` comes from a generator seeded with (reset seed of i, q), so an env's chunks do not depend on batch
size, batch order or the other envs in the batch.
--sampling (Stage 1 fix diagnosis, validation only; default unchanged): `default` = one chunk from the seeded start
noise above; `zero` = start noise 0 (no draw); `meanK` = the mean of --k chunks per query, each from the next seeded
draw of the env's NoiseBank (its query counter advances k per query), averaged in normalized space before
un-normalising (MEAN_STD is affine, so = the mean of the un-normalised chunks), the claw thresholded after averaging.
--guidance-w w (task 004 guidance probe, validation only; default off = everything above unchanged): the guided
velocity of vla_policy.guidance at every denoising step; the env's "other" instructions arrive with op `guide`
{env_ids, others} after its `reset` (reset forgets them); `act` refuses an env without them. Seeded start noise as
`default`; w = 1 = the plain policy through the guided path.
With --dataset (base / untrained weights): policy and processors are built the way lerobot-train builds them for
`--policy.path=<ckpt> --rename_map=...` (normalizer stats from the dataset), with the input features set to ours
(camera1, camera2, state 19). Without --dataset, --ckpt must be a fine-tuned `pretrained_model` dir whose saved
processors already carry the rename map and our stats.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch

from vla_common import protocol as P

RENAME_MAP = {"observation.images.side": "observation.images.camera1",
              "observation.images.wrist": "observation.images.camera2"}
CLAW = 4
SAMPLING = ("default", "zero", "meanK")

log = logging.getLogger("vla_policy.server")


class NoiseBank:
    """Per-env start noise: draw(env) at query q since reset = randn(shape) from a generator seeded by (seed, q)."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.seed: dict[int, int] = {}
        self.q: dict[int, int] = {}

    def reset(self, env_ids: np.ndarray, seeds: np.ndarray) -> None:
        for i, s in zip(env_ids.tolist(), seeds.tolist()):
            self.seed[i], self.q[i] = s, 0

    def draw(self, env_ids: np.ndarray) -> torch.Tensor:
        missing = [i for i in env_ids.tolist() if i not in self.seed]
        if missing:
            raise ValueError(f"envs {missing} were never reset (reset carries their noise seeds)")
        out = []
        for i in env_ids.tolist():
            s = int(np.random.SeedSequence([self.seed[i], self.q[i]]).generate_state(1, np.uint64)[0] >> 1)
            out.append(torch.randn(self.shape, generator=torch.Generator().manual_seed(s)))
            self.q[i] += 1
        return torch.stack(out)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def weights_path(ckpt: str) -> Path:
    if Path(ckpt).is_dir():
        return Path(ckpt) / "model.safetensors"
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(ckpt, "model.safetensors"))


class SmolVLAServer:
    LABELS = False  # VLAs get the instruction text; labels in a request are banned

    def __init__(self, ckpt: str, n_action_steps: int, device: str, dataset: str | None = None,
                 sampling: str = "default", k: int = 8, guidance_w: float | None = None):
        from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.n_action_steps, cfg.device = n_action_steps, device
        pre_over: dict = {"device_processor": {"device": device}}
        post_over: dict = {}
        if dataset is not None:
            from vla_policy.export import load

            meta = load(dataset).meta
            H = meta.features["observation.images.side"]["shape"][0]
            img = PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, H))
            cfg.input_features = {RENAME_MAP["observation.images.side"]: img,
                                  RENAME_MAP["observation.images.wrist"]: img,
                                  "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(P.STATE_DIM,))}
            cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(P.ACTION_DIM,))}
            feats = {**cfg.input_features, **cfg.output_features}
            pre_over |= {"rename_observations_processor": {"rename_map": RENAME_MAP},
                         "normalizer_processor": {"features": feats, "norm_map": cfg.normalization_mapping,
                                                  "stats": meta.stats}}
            post_over = {"unnormalizer_processor": {"features": cfg.output_features,
                                                    "norm_map": cfg.normalization_mapping, "stats": meta.stats}}
        self.cfg, self.device = cfg, device
        assert sampling in SAMPLING and k >= 1, (sampling, k)
        self.sampling, self.k = sampling, k
        assert guidance_w is None or (sampling == "default" and guidance_w >= 1.0), (sampling, guidance_w)
        self.guidance_w, self.others = guidance_w, {}  # others: env id -> its "other" instructions (op `guide`)
        self.image_size = int(cfg.input_features[RENAME_MAP["observation.images.side"]].shape[-1])
        # LeRobot 0.6.1 builds SmolVLA's VLM and action expert in bfloat16 (474 / 500 tensors); bf16 matmuls round
        # differently per batch shape (B = 3 vs B = 1: 0.17 in normalized action on CPU). Served in strict fp32
        # (weights upcast exactly, TF32 off) the same check gives 8e-6, so the server is batch invariant (GATES 0B.2).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg).to(device).float().eval()  # upcast on device
        gc.collect()  # drop the host copies of the weights (laptop RAM is shared with the simulator)
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides=pre_over,
                                                       postprocessor_overrides=post_over)
        self.noise = NoiseBank((cfg.chunk_size, cfg.max_action_dim))
        self.debug, self.last_tokens = False, None  # instrument (op `debug`): decoded tokens of the next act
        self.ckpt_sha = file_sha256(weights_path(ckpt))
        dev = next(self.policy.parameters()).device
        assert dev.type == torch.device(device).type, f"policy on {dev}, expected {device}"
        log.info("smolvla on %s: inputs %s, outputs %s, n_action_steps %d, chunk %d, freeze_vision_encoder %s, "
                 "train_expert_only %s, weights sha256 %s, sampling %s (k %d), guidance_w %s", dev, list(cfg.input_features),
                 list(cfg.output_features), cfg.n_action_steps, cfg.chunk_size, cfg.freeze_vision_encoder,
                 cfg.train_expert_only, self.ckpt_sha, sampling, k, guidance_w)

    def info(self) -> dict:
        c = self.cfg
        return {"policy": "smolvla", "n_action_steps": c.n_action_steps, "chunk_size": c.chunk_size,
                "image_size": self.image_size, "state_dim": P.STATE_DIM, "action_dim": P.ACTION_DIM,
                "ckpt_sha": self.ckpt_sha, "rename_map": RENAME_MAP,
                "input_features": {k: list(v.shape) for k, v in c.input_features.items()},
                "max_state_dim": c.max_state_dim, "max_action_dim": c.max_action_dim,
                "tokenizer_max_length": c.tokenizer_max_length, "freeze_vision_encoder": c.freeze_vision_encoder,
                "train_expert_only": c.train_expert_only, "dtype": "float32", "sampling": self.sampling,
                "k": self.k if self.sampling == "meanK" else 1, "guidance_w": self.guidance_w,
                "device": str(next(self.policy.parameters()).device)}

    def reset(self, msg: dict) -> None:
        self.noise.reset(msg["env_ids"], msg["seeds"])
        for i in msg["env_ids"].tolist():
            self.others.pop(i, None)

    def guide(self, msg: dict) -> None:
        """Op `guide` (task 004): {env_ids int32 (B,), others: B lists of >= 1 str}."""
        if self.guidance_w is None:
            raise ValueError("op guide: server started without --guidance-w")
        ids, others = msg["env_ids"], msg["others"]
        if not (isinstance(ids, np.ndarray) and ids.dtype == np.int32 and ids.ndim == 1) or set(msg) != {
                "op", "env_ids", "others"}:
            raise ValueError("guide: {env_ids int32 (B,), others}")
        if not (isinstance(others, list) and len(others) == len(ids) and all(
                isinstance(o, list) and o and all(isinstance(x, str) for x in o) for o in others)):
            raise ValueError("guide: others must be B non-empty lists of str")
        self.others.update(zip(ids.tolist(), others))

    def decode_tokens(self, obs: dict) -> list[str]:
        """The text the model sees: the preprocessor's token ids (attention mask applied) decoded by its tokenizer step."""
        from lerobot.processor import TokenizerProcessorStep

        tz = next(s for s in self.pre.steps if isinstance(s, TokenizerProcessorStep)).input_tokenizer
        tok, msk = obs["observation.language.tokens"], obs["observation.language.attention_mask"]
        return [tz.decode(t[m.bool()].tolist()) for t, m in zip(tok, msk)]

    @torch.no_grad()
    def act(self, msg: dict) -> np.ndarray:
        B, N, H = len(msg["instruction"]), self.cfg.n_action_steps, self.image_size
        if B == 0:
            return np.zeros((0, N, P.ACTION_DIM), np.float32)
        if msg["side"].shape[1] != H:
            raise ValueError(f"image size {msg['side'].shape[1]} != policy image size {H}")
        batch = {"observation.images.side": torch.from_numpy(msg["side"]).permute(0, 3, 1, 2).float() / 255.0,
                 "observation.images.wrist": torch.from_numpy(msg["wrist"]).permute(0, 3, 1, 2).float() / 255.0,
                 "observation.state": torch.from_numpy(msg["state"]),
                 "task": list(msg["instruction"])}
        obs = self.pre(batch)
        if self.debug:
            self.debug, self.last_tokens = False, self.decode_tokens(obs)
        if self.sampling == "zero":
            noise = torch.zeros((B, *self.noise.shape), device=self.device)
            chunk = self.policy.predict_action_chunk(obs, noise=noise)
        elif self.guidance_w is not None:
            from vla_policy.guidance import guided_chunk

            missing = [i for i in msg["env_ids"].tolist() if i not in self.others]
            if missing:
                raise ValueError(f"guidance: envs {missing} have no other instructions (op guide after reset)")
            others = [self.others[i] for i in msg["env_ids"].tolist()]
            noise = self.noise.draw(msg["env_ids"]).to(self.device)
            chunk = guided_chunk(self.policy, self.pre, batch, others, noise, self.guidance_w)
        elif self.sampling == "meanK":
            # a fresh dict per call: LeRobot's _get_action_chunk re-assigns batch entries from its obs queues
            chunk = torch.stack([self.policy.predict_action_chunk(dict(obs), noise=self.noise.draw(msg["env_ids"]).to(
                self.device)) for _ in range(self.k)]).mean(0)
        else:
            noise = self.noise.draw(msg["env_ids"]).to(self.device)
            chunk = self.policy.predict_action_chunk(obs, noise=noise)  # (B, chunk_size, 5), normalized
        a = self.post(chunk)[:, :N].float().cpu().numpy()  # un-normalized
        a[..., CLAW] = (a[..., CLAW] > 0.5).astype(np.float32)
        return a


class BaselineServer:
    """The no-language label baseline (vla_policy.baseline): deterministic, takes `labels` (B, 4) instead of text.
    Image-size agnostic network; `image_size` is what the sim must send (checked through `info`)."""

    LABELS = True

    def __init__(self, ckpt: str, n_action_steps: int, device: str, image_size: int):
        from vla_policy.baseline import BaselinePolicy

        self.pol = BaselinePolicy.load(ckpt, device)
        if self.pol.cfg.n_action_steps != n_action_steps:
            raise ValueError(f"--n-action-steps {n_action_steps} != checkpoint n_action_steps "
                             f"{self.pol.cfg.n_action_steps}")
        self.image_size = image_size
        self.ckpt_sha = file_sha256(Path(ckpt) / "weights.pt")
        self.debug, self.last_labels = False, None  # instrument B-I1 (op `debug`): labels of the next act, as received

    def info(self) -> dict:
        return {"policy": "baseline", "n_action_steps": self.pol.cfg.n_action_steps,
                "chunk_size": self.pol.cfg.n_action_steps, "image_size": self.image_size, "state_dim": P.STATE_DIM,
                "action_dim": P.ACTION_DIM, "ckpt_sha": self.ckpt_sha,
                "device": str(next(self.pol.parameters()).device)}

    def reset(self, msg: dict) -> None:
        pass  # deterministic: no per-env state

    def act(self, msg: dict) -> np.ndarray:
        if self.debug:
            self.debug, self.last_labels = False, msg["labels"].copy()
        if msg["side"].shape[0] and msg["side"].shape[1] != self.image_size:
            raise ValueError(f"image size {msg['side'].shape[1]} != {self.image_size}")
        if not msg["state"].shape[0]:
            return np.zeros((0, self.pol.cfg.n_action_steps, P.ACTION_DIM), np.float32)
        return self.pol.predict(msg["side"], msg["wrist"], msg["state"], msg["labels"])


def build(policy: str, ckpt: str, n_action_steps: int, device: str, dataset: str | None,
          image_size: int | None = None, sampling: str = "default", k: int = 8, guidance_w: float | None = None):
    if policy != "smolvla" and (sampling != "default" or guidance_w is not None):
        raise ValueError(f"--sampling {sampling} / --guidance-w {guidance_w}: smolvla only")
    if policy == "smolvla":
        return SmolVLAServer(ckpt, n_action_steps, device, dataset, sampling, k, guidance_w)
    if policy == "baseline":
        if image_size is None:
            raise ValueError("--policy baseline needs --image-size (the network itself is size agnostic)")
        return BaselineServer(ckpt, n_action_steps, device, image_size)
    raise NotImplementedError("--policy pi05: not served yet (waits for the gated PaliGemma tokenizer)")


def make_handler(srv, period: int, latency: int):
    def handle(msg: dict) -> dict:
        op = msg.get("op")
        if op == "info":
            return srv.info() | {"period": period, "latency": latency}
        if op == "reset":
            srv.reset(msg)
            return {"ok": True}
        if op == "guide":
            srv.guide(msg)
            return {"ok": True}
        if op == "debug":
            srv.debug = True
            return {"ok": True}
        if op == "last_labels":
            return {"labels": srv.last_labels}
        if op == "last_tokens":
            return {"tokens": srv.last_tokens}
        if op == "act":
            t = time.perf_counter()
            a = srv.act(msg)  # ends with .cpu(): synchronous
            return {"actions": a, "server_s": time.perf_counter() - t}
        raise ValueError(f"unknown op {op!r}")

    return handle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, choices=("smolvla", "pi05", "baseline"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-action-steps", type=int, required=True)
    p.add_argument("--period", type=int, required=True)
    p.add_argument("--latency", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", default=None, help="LeRobot dataset dir for normalization stats (base weights)")
    p.add_argument("--image-size", type=int, default=None, help="baseline only: the image size the sim sends")
    p.add_argument("--port", type=int, default=P.DEFAULT_PORT)
    p.add_argument("--sampling", choices=SAMPLING, default="default",
                   help="smolvla flow-matching start noise: seeded draw (default) | zero | meanK (mean of --k chunks)")
    p.add_argument("--k", type=int, default=8, help="--sampling meanK: chunks averaged per query")
    p.add_argument("--guidance-w", type=float, default=None,
                   help="task 004: instruction guidance weight w (op guide carries the other instructions)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[server] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but torch.cuda.is_available() is False. STOP.")
    t0 = time.time()
    srv = build(args.policy, args.ckpt, args.n_action_steps, args.device, args.dataset, args.image_size,
                args.sampling, args.k, args.guidance_w)
    import psutil

    log.info("loaded in %.1f s, host RSS %.0f MiB", time.time() - t0, psutil.Process().memory_info().rss / 2 ** 20)
    P.serve(make_handler(srv, args.period, args.latency), args.port,
            on_listen=lambda: log.info("listening on %s:%d", P.HOST, args.port), labels=srv.LABELS)
    log.info("client closed; exiting")


if __name__ == "__main__":
    main()
