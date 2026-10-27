# Container build (2026-10-01)
- Method: `apptainer build --fakeroot` ON the PERUN login node (apptainer 1.4.5; "User not listed in /etc/subuid, trying
  root-mapped namespace" works). Context `hpc/` (def + `sim/` + `policy/` Linux-restricted pyproject/uv.lock), build tmp on
  node-local `/tmp/$USER-apt`. No WSL, no upload. Final build wall **123 s** (1790856892 -> 1790857015 epoch), image **4,205,043,712 B (3.9 GiB)**.
- sha256 `f76eaf8e4802a2ca408d4a7703a0c57550c8e3aeeb376d7c4f93581edd116c1c` -> `/mnt/project/<project-id>/113_vla_tabletop/vla_tabletop.sif` (+ `.sha256`).
- Recipe: 112's (plain `ubuntu:24.04`, uv 0.12.19, py3.12, usd-core reinstalled last) + apt `ffmpeg` + second venv `/opt/policy`.
  `/opt/venv` = sim (hpc/sim), `/opt/policy` = lerobot 0.6.1 (hpc/policy). Both torch 2.10.0+cu130.
- First attempt failed `%test` (23 s of build): torchcodec 0.11.1 (what policy/uv.lock resolved) needs torch 2.11
  (`undefined symbol torch_dtype_float4_e2m1fn_x2` against torch 2.10; FFmpeg 6.1 libs were found). Fix in the Linux copy only:
  override `torchcodec>=0.10.0,<0.11.0` -> locks 0.10.0. `policy/pyproject.toml` (Windows) is unchanged and does not carry it;
  anyone relocking the Linux copy from `policy/` must keep the 0.10 pin.
- `%test` output: `sim torch 2.10.0+cu130 warp 1.13.0 newton 1.2.1` / `policy torch 2.10.0+cu130 lerobot 0.6.1 torchcodec 0.10.0`.
- Policy lock source: working-tree `policy/` (untracked at build time); Linux copy = same lock re-resolved with `environments=[linux x86_64]`.
- Compute nodes (<node>) have internet (huggingface.co 200). HF cache prefetched on the login node into `$PROJ/cache/hf` (2.8 GB: smolvla_base + SmolVLM2-500M backbone).
