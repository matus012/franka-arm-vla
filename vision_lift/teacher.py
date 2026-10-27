"""The frozen 111 phase-6 expert as a plain torch module (brief section 4).

Actor = MLP 48 -> 256 -> 128 -> 64 -> 4, ELU, no observation normalizer; the deterministic action is the MLP mean
(distribution.std_param is unused at evaluation). Never retrained or modified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TEACHER_PT = ROOT / "teacher" / "phase6_full_twist.pt"
TEACHER_SHA256 = "E8FD5A3981A8FF8A72D6E49623FA9639A843599B72A809CDD0CA6E592275F352"
TEACHER_TASK = "LiftRL-Phased-P6-TS-v0"
OBS_DIM, ACT_DIM = 48, 4

# 48-dim teacher observation layout (lift_rl/phased_cfg.py PhasedObservationsCfg, concatenated in this order)
OBS_SLICES: dict[str, slice] = {
    "joint_pos": slice(0, 9),
    "joint_vel": slice(9, 18),
    "last_arm_action": slice(18, 22),
    "tcp": slice(22, 25),
    "tcp_to_cube": slice(25, 28),
    "cube_to_target": slice(28, 31),
    "target": slice(31, 34),
    "orientation": slice(34, 37),  # cos tilt, sin/cos 4 x (hand yaw - cube yaw)
    "gripper": slice(37, 41),  # width, left/right pad force / 20 N, palm flag
    "phase": slice(41, 48),  # one-hot of 6 phases + grasp latch
}


def sha256(path: Path = TEACHER_PT) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def verify_checksum(path: Path = TEACHER_PT) -> None:
    got = sha256(path)
    if got != TEACHER_SHA256:
        raise RuntimeError(f"teacher checksum mismatch: {got} != {TEACHER_SHA256}. STOP.")


class TeacherActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                                 nn.Linear(128, 64), nn.ELU(), nn.Linear(64, ACT_DIM))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_teacher(device: str | torch.device = "cuda:0", path: Path = TEACHER_PT) -> TeacherActor:
    verify_checksum(path)
    sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    actor = TeacherActor()
    actor.load_state_dict({k: v for k, v in sd.items() if k.startswith("mlp.")})
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor.to(device).eval()
