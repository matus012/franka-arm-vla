"""Task 004 (GATES "Inference-time instruction-guidance probe (004)"), policy venv. Diagnosis on validation only.

Guided flow-matching velocity at every denoising step: v = v_other + w * (v_true - v_other), computed as
v_true + (w - 1) * (v_true - v_other) so that w = 1 returns v_true bit for bit. v_true = velocity with the env's own
instruction, v_other = mean velocity over the env's "other" instructions (vla_common.guidance.other_instructions), same
observation and start noise. The batch is expanded to M rows (per env: its true row, then its K others); every row of
an env gets the env's guided velocity, so all branches see the same x_t at every step. No empty-prompt branch (the
model was never trained with dropped language).
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


def expand_rows(instructions: list[str], others: list[list[str]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """-> (texts (M,), env_of_row (M,) int64, true_rows (B,) int64): env i's rows = [true_i, *others_i]."""
    if len(instructions) != len(others):
        raise ValueError(f"{len(instructions)} instructions vs {len(others)} other-lists")
    texts: list[str] = []
    env_of_row: list[int] = []
    true_rows: list[int] = []
    for i, (t, o) in enumerate(zip(instructions, others)):
        if not o:
            raise ValueError(f"env {i}: no other instruction (guidance needs >= 1)")
        true_rows.append(len(texts))
        texts += [t, *o]
        env_of_row += [i] * (1 + len(o))
    return texts, np.asarray(env_of_row, np.int64), np.asarray(true_rows, np.int64)


def combine(v: torch.Tensor, env_of_row: torch.Tensor, true_rows: torch.Tensor, w: float) -> torch.Tensor:
    """v (M, ...) per-row velocities -> (M, ...) guided velocity of each row's env."""
    B = true_rows.shape[0]
    other = torch.ones(v.shape[0], dtype=torch.bool, device=v.device)
    other[true_rows] = False
    eo = env_of_row[other]
    cnt = torch.zeros(B, dtype=v.dtype, device=v.device).index_add_(0, eo, torch.ones_like(eo, dtype=v.dtype))
    assert bool((cnt > 0).all()), "an env without other rows"
    v_other = torch.zeros((B, *v.shape[1:]), dtype=v.dtype, device=v.device).index_add_(0, eo, v[other])
    v_other = v_other / cnt.view(B, *([1] * (v.dim() - 1)))
    v_true = v[true_rows]
    g = v_true + (w - 1.0) * (v_true - v_other)
    return g[env_of_row]


@contextmanager
def guided_denoise(model, w: float, env_of_row: torch.Tensor, true_rows: torch.Tensor):
    """Patch `model.denoise_step` (VLAFlowMatching; sample_actions calls self.denoise_step per Euler step)."""
    orig = model.denoise_step
    calls = {"n": 0}

    def step(*args, **kwargs):
        calls["n"] += 1
        return combine(orig(*args, **kwargs), env_of_row, true_rows, w)

    model.denoise_step = step
    try:
        yield calls
    finally:
        del model.denoise_step  # back to the class method
        assert model.denoise_step.__func__ is type(model).denoise_step


@torch.no_grad()
def guided_chunk(policy, pre, raw: dict, others: list[list[str]], noise: torch.Tensor, w: float) -> torch.Tensor:
    """raw = the server's un-preprocessed batch (B rows, "task" = true instructions); noise (B, chunk, max_action_dim)
    on the policy device. Returns the normalized chunk (B, chunk, action_dim) of the true rows."""
    texts, eor, tr = expand_rows(list(raw["task"]), others)
    r = torch.from_numpy(eor)
    batch = {k: v[r] for k, v in raw.items() if k != "task"} | {"task": texts}
    obs = pre(batch)
    dev = noise.device
    eor_d, tr_d = r.to(dev), torch.from_numpy(tr).to(dev)
    with guided_denoise(policy.model, w, eor_d, tr_d) as calls:
        chunk = policy.predict_action_chunk(obs, noise=noise[eor_d])
    assert calls["n"] == policy.config.num_steps, (calls["n"], policy.config.num_steps)
    return chunk[tr_d]
