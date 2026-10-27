"""Student policy (brief section 5): a small CNN per camera + proprioception MLP -> MLP head.

Outputs: 4 arm deltas (tanh, same range as the teacher's clipped actions) and 1 claw logit (close if logit > 0).
Inputs: side + wrist images (RGB, or RGB-D with depth in metres as a 4th channel) and the 27-dim student proprio
(joint pos 9, joint vel 9, gripper width 1, target 3, previous action 5), normalized with dataset statistics stored in
the checkpoint. Loss: Huber on arm deltas + w_claw x BCE on the claw (weights fixed in GATES.md per stage).
The Stage-1 perception estimator reuses the same encoders with a cube-pose head (`CubeEstimator`).
"""

from __future__ import annotations

import torch
from torch import nn

DEPTH_MAX = 2.0  # m, depth clipped to [0, DEPTH_MAX] and scaled to [0, 1]


class SpatialSoftmax(nn.Module):
    """Per-channel expected (x, y) image coordinate of a softmax over the feature map -> (B, 2C)."""

    def forward(self, x):
        B, C, H, W = x.shape
        p = torch.softmax(x.flatten(2), dim=-1).view(B, C, H, W)
        ys = torch.linspace(-1, 1, H, device=x.device)
        xs = torch.linspace(-1, 1, W, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], 1)


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int = 128, kp: int = 32):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.ReLU(inplace=True))

        self.net = nn.Sequential(blk(in_ch, 32, 5, 2), blk(32, 64, 3, 2), blk(64, 64, 3, 1), blk(64, 128, 3, 2),
                                 blk(128, 128, 3, 1))  # 128x128 -> 16x16
        self.kp = nn.Conv2d(128, kp, 1)
        self.ss = SpatialSoftmax()
        self.out = nn.Sequential(nn.Linear(2 * kp + 128, out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        f = self.net(x)
        return self.out(torch.cat([self.ss(self.kp(f)), f.mean((2, 3))], 1))


def image_tensor(rgb: torch.Tensor, depth: torch.Tensor | None) -> torch.Tensor:
    """(B,3,H,W) uint8 [+ (B,1,H,W) metres] -> float input in [0, 1] (depth clipped at DEPTH_MAX)."""
    x = rgb.float() / 255.0
    if depth is not None:
        d = torch.nan_to_num(depth.float(), nan=DEPTH_MAX, posinf=DEPTH_MAX)
        d = torch.where(d <= 0, torch.full_like(d, DEPTH_MAX), d).clamp(0, DEPTH_MAX) / DEPTH_MAX  # 0 = no hit
        x = torch.cat([x, d], 1)
    return x


class _Backbone(nn.Module):
    def __init__(self, depth: bool, proprio_dim: int, feat: int = 128):
        super().__init__()
        self.depth = depth
        c = 4 if depth else 3
        self.side, self.wrist = ConvEncoder(c, feat), ConvEncoder(c, feat)
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, feat),
                                  nn.ReLU(inplace=True))
        self.register_buffer("p_mean", torch.zeros(proprio_dim))
        self.register_buffer("p_std", torch.ones(proprio_dim))

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.p_mean.copy_(mean)
        self.p_std.copy_(std.clamp_min(1e-3))

    def features(self, inp: dict) -> torch.Tensor:
        sd = inp.get("side_depth") if self.depth else None
        wd = inp.get("wrist_depth") if self.depth else None
        fs = self.side(image_tensor(inp["side_rgb"], sd))
        fw = self.wrist(image_tensor(inp["wrist_rgb"], wd))
        fp = self.prop((inp["proprio"].float() - self.p_mean) / self.p_std)
        return torch.cat([fs, fw, fp], 1)


class StudentPolicy(_Backbone):
    def __init__(self, depth: bool = False, proprio_dim: int = 27):
        super().__init__(depth, proprio_dim)
        self.head = nn.Sequential(nn.Linear(3 * 128, 256), nn.ReLU(inplace=True), nn.Linear(256, 256),
                                  nn.ReLU(inplace=True), nn.Linear(256, 5))

        self.aux_head = nn.Linear(3 * 128, 2)  # training-only auxiliary head (3.2 attempt 3); unused when acting

    def forward(self, inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.head(self.features(inp))
        return torch.tanh(o[:, :4]), o[:, 4]  # arm deltas in [-1, 1], claw logit

    def forward_aux(self, inp: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training only: the policy outputs + the auxiliary (sin, cos) prediction from the same features."""
        f = self.features(inp)
        o = self.head(f)
        return torch.tanh(o[:, :4]), o[:, 4], self.aux_head(f)

    @torch.no_grad()
    def act(self, inp: dict) -> torch.Tensor:
        arm, logit = self(inp)
        return torch.cat([arm, (logit > 0).float()[:, None]], 1)


class CubeEstimator(_Backbone):
    """Stage 1 probe / Stage 5 modular baseline: cube position (env frame, m) + (sin, cos) of 4 x yaw."""

    def __init__(self, depth: bool = False, proprio_dim: int = 27):
        super().__init__(depth, proprio_dim)
        self.head = nn.Sequential(nn.Linear(3 * 128, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.register_buffer("pos_mean", torch.tensor([0.5, 0.0, 0.1]))

    def forward(self, inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.head(self.features(inp))
        return o[:, :3] + self.pos_mean, o[:, 3:5]
