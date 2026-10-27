# hpc/ — PERUN prep for 112_vision_lift (Stage 0.6)

**Status: prepared, never run on the cluster.** Owner has not connected PERUN; nothing here touched it.

| file | what |
|---|---|
| `pyproject.toml`, `uv.lock` | copy of the repo's project restricted to `linux x86_64` (`uv lock`); Windows lock untouched. Same pins: py3.12, torch 2.10.0+cu130, torchvision 0.25.0, isaaclab 3.0.0b2.post1, newton 1.2.1, mujoco-warp 3.8.1, warp-lang 1.13.0, scipy 1.16.2, usd-core 25.11, usd-exchange 3.0.0, h5py 3.16.0 |
| `Dockerfile`, `vision_lift.def` | image recipes (same content). Base `nvidia/cuda:13.0.1-runtime-ubuntu24.04`, uv 0.12.19, `uv sync --frozen`. Repo code is bind-mounted, not baked in |
| `render_check.py` | headless Newton-Warp camera check + images/s (runs on the laptop too; tested there: 16 envs 128x128 -> 537 img/s on the 4060, frame looked at) |
| `slurm/env.sh` | paths + `vl_exec` (`apptainer exec --nv --containall`, binds `/workspace /data /runs /cache`) |
| `slurm/render_check.sbatch`, `job.sbatch`, `seeds_array.sbatch` | render check / generic command / 3-seed array (`--array=0-2`) |
| `SYNC_PLAN.md`, `BUILD_TEST.md` | dataset sync plan; WSL container build test |

Build: see `BUILD_TEST.md` (`apptainer build vision_lift.sif hpc/vision_lift.def` from `hpc/`).

## First cluster job checklist (only after the owner says "PERUN connected")
1. Build/copy the `.sif`, `sha256sum`, rsync per `SYNC_PLAN.md`, verify checksum on PERUN, unpack `git archive HEAD` into `$PROJ/repo`.
2. On the login node: `which apptainer` / `module avail apptainer` (PLACEHOLDER in `env.sh`); confirm `gpu_short` max walltime; confirm outbound network from compute nodes is unavailable/available (assets: see below).
3. `sbatch hpc/slurm/render_check.sbatch` (1 H200, <= 0.35 GPU-h). Check `runs/logs/*.out` for `CUDA OK`, `RENDER_CHECK {...}` (images_per_s, frame_std) and look at `runs/render_check_*/frame0.png` (look-at-the-artifact).
4. Only then: a smoke run of the real job via `job.sbatch`, measure its wall, set `--time`, then `seeds_array.sbatch` chained `--dependency=afterok:<smoke>`; check the first running unit against its wall right after submit.
Rules: max 4 queued jobs/user; `sacct` does not answer (done = absent from `squeue`); **never `sbatch --export`**.

## Logging rule (every job, no exceptions)
After each job group append ONE row to `<home>\A_MAIN\ws\000_infra\hpc_usage.md` (date, project, job IDs, GPU-h, CPU-h, source of number,
verified y/n) and update its running total (commit only that file in the 000_infra repo), AND one row per job in this repo's `ops/gpu_log.csv`.
GPU-h = GPUs x wall hours. **Budget: 50 H200 GPU-hours total for 112** (brief section 9, hard stop); the launch line must print the running total before submit. No rows are written by this prep task.

## Known issues / decisions for the owner
- **Assets:** the task env loads DexCube from Nucleus at runtime (S3 over https). If compute nodes have no internet, `render_check.py` (which builds the real task env) will fail there; check step 2 and, if needed, pre-cache the asset into `$PROJ/cache` or run once on the login node. Not solved here.
- Apptainer on PERUN and gpu_short limits are unverified placeholders.
