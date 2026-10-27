"""No-language baseline (113 brief §6): the 112 student network, fed task labels instead of text, chunked head.

Inputs: side + wrist RGB (any square or non-square H x W), state (19, MEAN_STD), labels int64 (4):
(task 0-2, colour 0-3, size 0-1, destination 0-12) as learned embeddings. No prev_action, no target.
Output: chunk (n_action_steps, 5): 4 arm deltas (trained with MSE on MEAN_STD-normalised targets, un-normalised and
clipped to [-1, 1] at inference) + claw logit (BCE; executed as sigmoid > 0.5).
Copies the 112 ConvEncoder (no vision_lift import: that package pulls in the sim stack).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LABEL_SIZES = (3, 4, 2, 13)  # task, colour, size, destination
STATE_DIM = 19
FEAT_HW = 16  # encoder feature map is adaptively pooled to 16 x 16 -> same network for any input size


class SpatialSoftmax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = torch.softmax(x.flatten(2), dim=-1).view(B, C, H, W)
        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], 1)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, out_dim: int = 128, kp: int = 32):
        super().__init__()

        def blk(i: int, o: int, k: int, s: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

        self.net = nn.Sequential(blk(in_ch, 32, 5, 2), blk(32, 64, 3, 2), blk(64, 64, 3, 1), blk(64, 128, 3, 2),
                                 blk(128, 128, 3, 1), nn.AdaptiveAvgPool2d(FEAT_HW))
        self.kp = nn.Conv2d(128, kp, 1)
        self.ss = SpatialSoftmax()
        self.out = nn.Sequential(nn.Linear(2 * kp + 128, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.net(x)
        return self.out(torch.cat([self.ss(self.kp(f)), f.mean((2, 3))], 1))


@dataclass
class BaselineConfig:
    n_action_steps: int = 10
    feat: int = 128
    label_dim: int = 16


class BaselinePolicy(nn.Module):
    def __init__(self, cfg: BaselineConfig | None = None):
        super().__init__()
        self.cfg = cfg or BaselineConfig()
        f, n = self.cfg.feat, self.cfg.n_action_steps
        self.side, self.wrist = ConvEncoder(3, f), ConvEncoder(3, f)
        self.prop = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU(inplace=True), nn.Linear(128, f), nn.ReLU(inplace=True))
        self.emb = nn.ModuleList([nn.Embedding(k, self.cfg.label_dim) for k in LABEL_SIZES])
        self.lab = nn.Sequential(nn.Linear(4 * self.cfg.label_dim, f), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(4 * f, 256), nn.ReLU(inplace=True), nn.Linear(256, 256),
                                  nn.ReLU(inplace=True), nn.Linear(256, n * 5))
        self.register_buffer("s_mean", torch.zeros(STATE_DIM))
        self.register_buffer("s_std", torch.ones(STATE_DIM))
        self.register_buffer("a_mean", torch.zeros(4))
        self.register_buffer("a_std", torch.ones(4))

    def set_norm(self, s_mean: torch.Tensor, s_std: torch.Tensor, a_mean: torch.Tensor, a_std: torch.Tensor) -> None:
        self.s_mean.copy_(s_mean)
        self.s_std.copy_(s_std.clamp_min(1e-3))
        self.a_mean.copy_(a_mean)
        self.a_std.copy_(a_std.clamp_min(1e-3))

    @staticmethod
    def _img(x: torch.Tensor) -> torch.Tensor:
        """(B,H,W,3) uint8 or (B,3,H,W) float in [0,1] -> (B,3,H,W) float in [0,1]."""
        if x.dtype == torch.uint8:
            return x.permute(0, 3, 1, 2).float() / 255.0
        return x.float()

    def forward(self, side: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor,
                labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (arm (B,N,4) in normalised units, claw logit (B,N))."""
        e = torch.cat([m(labels[:, i]) for i, m in enumerate(self.emb)], 1)
        z = torch.cat([self.side(self._img(side)), self.wrist(self._img(wrist)),
                       self.prop((state.float() - self.s_mean) / self.s_std), self.lab(e)], 1)
        o = self.head(z).view(-1, self.cfg.n_action_steps, 5)
        return o[..., :4], o[..., 4]

    @torch.no_grad()
    def predict(self, side: np.ndarray, wrist: np.ndarray, state: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """uint8 (B,H,H,3) x2, float (B,19), int64 (B,4) -> float32 (B,N,5); arm clipped to [-1,1], claw in {0,1}."""
        was = self.training
        self.eval()
        dev = self.s_mean.device
        arm, logit = self(torch.as_tensor(side, device=dev), torch.as_tensor(wrist, device=dev),
                          torch.as_tensor(state, device=dev), torch.as_tensor(labels, device=dev).long())
        out = torch.cat([(arm * self.a_std + self.a_mean).clamp(-1, 1), (torch.sigmoid(logit) > 0.5).float()[..., None]], -1)
        self.train(was)
        return out.cpu().numpy().astype(np.float32)

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), d / "weights.pt")
        norm = {k: getattr(self, k).tolist() for k in ("s_mean", "s_std", "a_mean", "a_std")}
        (d / "config.json").write_text(json.dumps({"config": asdict(self.cfg), "norm": norm}, indent=1))

    @classmethod
    def load(cls, d: str | Path, device: str | torch.device = "cpu") -> "BaselinePolicy":
        d = Path(d)
        j = json.loads((d / "config.json").read_text())
        m = cls(BaselineConfig(**j["config"]))
        m.load_state_dict(torch.load(d / "weights.pt", map_location="cpu"))
        m.set_norm(*(torch.tensor(j["norm"][k]) for k in ("s_mean", "s_std", "a_mean", "a_std")))
        return m.to(device).eval()


def loss_fn(model: BaselinePolicy, batch: dict, w_claw: float = 1.0) -> tuple[torch.Tensor, dict]:
    """batch: side, wrist, state, labels, action (B,N,5) raw, pad (B,N) bool (True = padded, masked out)."""
    arm, logit = model(batch["side"], batch["wrist"], batch["state"], batch["labels"])
    a = batch["action"].float()
    keep = (~batch["pad"]).float()
    tgt = (a[..., :4] - model.a_mean) / model.a_std
    mse = (((arm - tgt) ** 2).mean(-1) * keep).sum() / keep.sum().clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logit, a[..., 4], reduction="none") * keep).sum() / keep.sum().clamp_min(1)
    return mse + w_claw * bce, {"mse": mse.item(), "bce": bce.item()}
