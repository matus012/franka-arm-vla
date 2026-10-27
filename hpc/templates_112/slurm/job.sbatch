#!/bin/bash
#SBATCH --job-name=vl_job
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=02:00:00                  # PLACEHOLDER: set from the measured smoke wall (measure-the-batch-before-it-dies)
#SBATCH --gres=gpu:1                     # GPU-h charged = GPUs x wall; budget 50 H200-h total for 112
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # PLACEHOLDER
#SBATCH --output=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.err
# Generic job: runs ONE command inside the container from the repo root.
# Usage:  sbatch hpc/slurm/job.sbatch "<unit-name>" python scripts/s01_collect.py --out /data/s01 --episodes 1000
# Idempotent: exits 0 if /runs/<unit-name>/DONE exists. The command must write its outputs under /data or /runs.
set -euo pipefail
source /mnt/project/<project-id>/112_vision_lift/repo/hpc/slurm/env.sh
UNIT=$1; shift
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
echo "=== $UNIT on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start $(date -Is)"
vl_exec python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable -- STOP'"
vl_exec "$@"
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT done $(date -Is) (SLURM wall: see squeue/logs; log GPU-h in 000_infra/hpc_usage.md + ops/gpu_log.csv)"
