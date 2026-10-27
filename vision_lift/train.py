"""Behaviour cloning of the student on recorded episodes (BC and every DAgger refit use this).

Loss = Huber(arm deltas, teacher label; delta HUBER_DELTA) + W_CLAW x BCE(claw logit, teacher claw label).
Images: uint8 -> [0, 1] (+ depth channel for RGB-D). No image augmentation by default: with fixed cameras a 4-px
random shift = ~3 cm of apparent cube position in the side view (`augment=True` keeps the DrQ-style shift available).
Proprio normalized with training-set mean/std (stored in the checkpoint). bf16 autocast on CUDA.
Offline validation loss on held-out episodes is logged (TensorBoard) and saved per epoch; the checkpoint used for a
gate is picked by closed-loop validation rollouts (GATES.md), not by this loss.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import StepDataset, model_inputs
from .student import StudentPolicy

HUBER_DELTA = 0.1
W_CLAW = 1.0
W_AUX = 0.0  # auxiliary relative-yaw loss weight; set per fit (3.2 attempt 3 on: 1.0), see GATES.md
PREV_ACTION = slice(22, 27)  # student proprio: previous 4 arm deltas + claw command (vision_lift.envs.StudentObsCfg)


def load_student(path, device) -> tuple[StudentPolicy, bool]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = StudentPolicy(depth=ck["depth"])
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("aux_head.") for k in missing), (missing, unexpected)
    return m.to(device).eval(), ck["depth"]


def save_student(path, model: StudentPolicy, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "depth": model.depth, "meta": meta}, path)


def random_shift(x: torch.Tensor, pad: int = 4, gen: torch.Generator | None = None) -> torch.Tensor:
    """DrQ-style: replicate-pad by `pad` px and crop back at a random offset per sample (integer shifts)."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=gen)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=gen)
    rows = (oy[:, None] + torch.arange(H, device=x.device)[None])  # (B, H)
    cols = (ox[:, None] + torch.arange(W, device=x.device)[None])  # (B, W)
    idx_r = rows[:, None, :, None].expand(B, C, H, W + 2 * pad)
    xr = xp.gather(2, idx_r)
    idx_c = cols[:, None, None, :].expand(B, C, H, W)
    return xr.gather(3, idx_c)


def _augment(inp: dict, gen) -> dict:
    out = dict(inp)
    for cam in ("side", "wrist"):
        rgb = inp[f"{cam}_rgb"].float()
        if f"{cam}_depth" in inp:
            both = random_shift(torch.cat([rgb, inp[f"{cam}_depth"].float()], 1), gen=gen)
            out[f"{cam}_rgb"], out[f"{cam}_depth"] = both[:, :3], both[:, 3:]
        else:
            out[f"{cam}_rgb"] = random_shift(rgb, gen=gen)
    return out


def bc_loss(model, batch: dict, device, augment: bool, gen=None, w_aux: float | None = None,
            p_prev_drop: float = 0.0) -> tuple[torch.Tensor, dict]:
    inp = {k: v.to(device, non_blocking=True) for k, v in model_inputs(batch).items()}
    if p_prev_drop > 0:  # training only (3.2 attempt 4): zero the previous-action input on a random fraction of samples
        pr = inp["proprio"].clone()
        drop = torch.rand(pr.shape[0], device=device, generator=gen) < p_prev_drop
        pr[drop, PREV_ACTION] = 0.0
        inp["proprio"] = pr
    if augment:
        inp = _augment(inp, gen)
    arm_t = batch["teacher_arm"].to(device).float()
    claw_t = batch["claw_label"].to(device).float()
    w_aux = W_AUX if w_aux is None else w_aux
    use_aux = w_aux > 0 and "aux_rel_yaw_deg" in batch
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if use_aux:
            arm, logit, aux = model.forward_aux(inp)
        else:
            arm, logit = model(inp)
    arm, logit = arm.float(), logit.float()
    l_arm = F.huber_loss(arm, arm_t, delta=HUBER_DELTA)
    l_claw = F.binary_cross_entropy_with_logits(logit, claw_t)
    loss = l_arm + W_CLAW * l_claw
    l_aux = torch.zeros((), device=device)
    if use_aux:
        ang = torch.deg2rad(batch["aux_rel_yaw_deg"].to(device).float()) * 4
        l_aux = F.mse_loss(aux.float(), torch.stack([torch.sin(ang), torch.cos(ang)], 1))
        loss = loss + w_aux * l_aux
    with torch.no_grad():
        stats = {"loss": float(loss), "huber_arm": float(l_arm), "bce_claw": float(l_claw), "aux_mse": float(l_aux),
                 "arm_mae": float((arm - arm_t).abs().mean()),
                 "claw_acc": float(((logit > 0).float() == claw_t).float().mean())}
    return loss, stats


DEGENERATE_STD = 1e-2  # a proprio dim with a stored std below this was (near-)constant in the parent's data


def refresh_degenerate_norm(model, new_mean: torch.Tensor, new_std: torch.Tensor) -> list[int]:
    """Fine-tune fix (3.3 attempt 2): dims that were constant in the parent's training data (std at the floor, e.g. the
    target in the fixed-target stages) get mean/std from the new training data. Other dims keep the parent's stats.
    Returns the refreshed dim indices."""
    dims = [i for i in range(model.p_std.numel()) if float(model.p_std[i]) < DEGENERATE_STD
            and float(new_std[i]) >= DEGENERATE_STD]
    for i in dims:
        model.p_mean[i] = new_mean[i]
        model.p_std[i] = new_std[i]
    return dims


def proprio_stats(ds: StepDataset, n: int = 20000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    import h5py
    import numpy as np

    rows = []
    for f in ds.files:
        with h5py.File(f, "r") as h:
            rows.append(h["steps/proprio"][:].reshape(-1, h["steps/proprio"].shape[-1]))
    x = torch.from_numpy(np.concatenate(rows))
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(len(x), generator=g)[:n]]
    return x.mean(0), x.std(0)


def train_bc(train_files, val_files, out_dir: Path, depth: bool, epochs: int, batch_size: int = 256, lr: float = 3e-4,
             seed: int = 0, workers: int = 4, init: str | None = None, max_minutes: float = 110.0,
             augment: bool = False, w_aux: float = 0.0, p_prev_drop: float = 0.0, log=print) -> dict:
    """Train (or fine-tune from `init`) and save a checkpoint per epoch. Returns a summary dict."""
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    assert torch.cuda.is_available(), "CUDA not available: STOP"
    tr = StepDataset(train_files, depth=depth)
    va = StepDataset(val_files, depth=depth) if val_files else None
    model = StudentPolicy(depth=depth)
    if init:
        m0, d0 = load_student(init, "cpu")  # tolerates checkpoints without the aux head (fresh aux head)
        assert d0 == depth
        model.load_state_dict(m0.state_dict())
        refreshed = refresh_degenerate_norm(model, *proprio_stats(tr, seed=seed))
        log(f"[train] fine-tune from {init}: proprio dims with refreshed normalization (degenerate in the parent): {refreshed}")
    else:
        model.set_norm(*proprio_stats(tr, seed=seed))
    model.to(device)
    assert next(model.parameters()).device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(tr) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.05)
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                                     generator=g, persistent_workers=workers > 0, pin_memory=True,
                                     prefetch_factor=4 if workers > 0 else None)
    vdl = torch.utils.data.DataLoader(va, batch_size=512, shuffle=False, num_workers=workers,
                                      persistent_workers=workers > 0) if va else None
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(out_dir / "tb"))
    ggpu = torch.Generator(device=device).manual_seed(seed)
    hist, t0, it = [], time.time(), 0
    for ep in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        nb, te = 0, time.time()
        for batch in dl:
            loss, st = bc_loss(model, batch, device, augment=augment, gen=ggpu, w_aux=w_aux, p_prev_drop=p_prev_drop)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
            it += 1
            if it % 100 == 0:
                for k, v in st.items():
                    tb.add_scalar(f"train/{k}", v, it)
        row = {"epoch": ep, **{f"train_{k}": v / max(1, nb) for k, v in agg.items()},
               "minutes": round((time.time() - t0) / 60, 2), "samples_per_s": round(nb * batch_size / (time.time() - te), 1)}
        if vdl is not None:
            model.eval()
            vagg: dict[str, float] = {}
            vn = 0
            with torch.no_grad():
                for batch in vdl:
                    _, st = bc_loss(model, batch, device, augment=False, w_aux=w_aux)
                    for k, v in st.items():
                        vagg[k] = vagg.get(k, 0.0) + v
                    vn += 1
            row |= {f"val_{k}": v / max(1, vn) for k, v in vagg.items()}
            for k, v in vagg.items():
                tb.add_scalar(f"val/{k}", v / max(1, vn), ep)
        hist.append(row)
        save_student(out_dir / f"epoch_{ep:03d}.pt", model, {"epoch": ep, "row": row, "depth": depth})
        log(json.dumps(row))
        if (time.time() - t0) / 60 > max_minutes:
            log(f"[train] time cap {max_minutes} min reached after epoch {ep}: stopping")
            break
    tb.close()
    (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
    return {"epochs_run": len(hist), "last": hist[-1], "train_samples": len(tr), "val_samples": len(va) if va else 0}
