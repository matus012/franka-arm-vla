#!/bin/bash
#SBATCH --job-name=vla_job4
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=00:30:00                  # SET PER JOB (sbatch --time=...; gpu_short MaxTime is 2 days)
#SBATCH --gres=gpu:4                     # GPU-h = 4 x wall; 113 budget <= 100 H200-h
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.err
# 4-GPU data-parallel trainer unit. Usage: sbatch hpc/slurm/job4.sbatch <unit> <vla_policy.train_smolvla args...>
#   e.g. sbatch hpc/slurm/job4.sbatch s2a_VT --dataset /data/datasets/... --out /runs/s2a_VT --batch-size 64 --workers 6 ...
# The trainer is started under torch.distributed.run (4 processes, one per GPU) with --distributed nccl added here;
# --batch-size is the GLOBAL batch (64 = 4 x 16), --workers are PER RANK (32 cpus / 4 ranks = 8 each). Positional args
# only, never `--export`. Idempotent: exits 0 if /runs/<unit>/DONE exists. Outputs go under /data or /runs.
set -euo pipefail
source /mnt/project/<project-id>/113_vla_tabletop/repo/hpc/slurm/env.sh
UNIT=$1; shift
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=== $UNIT on $(hostname) $GPU start $(date -Is)"
echo "NET_PROBE huggingface.co http_code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' https://huggingface.co 2>&1 | tail -c 60)"
pol_exec -c "import torch; n = torch.cuda.device_count(); assert torch.cuda.is_available() and n == 4, f'need 4 CUDA devices, found {n} -- STOP'; print('CUDA OK', n, torch.cuda.get_device_name(0))"
SLURM_CPUS_PER_TASK=8 pol_exec -m torch.distributed.run --standalone --nproc_per_node=4 -m vla_policy.train_smolvla --distributed nccl "$@"  # 8 OMP threads per rank
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT on $(hostname) $GPU done $(date -Is)"
