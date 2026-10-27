#!/bin/bash
#SBATCH --job-name=vla_job
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=00:30:00                  # SET PER JOB (edit before sbatch; gpu_short MaxTime is 2 days)
#SBATCH --gres=gpu:1                     # GPU-h = GPUs x wall; 113 budget <= 100 H200-h
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.err
# Generic unit. Usage: sbatch hpc/slurm/job.sbatch <unit> <sim|pol> <python args...>
#   e.g. sbatch job.sbatch smoke_pol pol hpc/smoke_policy.py --batch 16
# Idempotent: exits 0 if /runs/<unit>/DONE exists. Outputs go under /data or /runs.
set -euo pipefail
source /mnt/project/<project-id>/113_vla_tabletop/repo/hpc/slurm/env.sh
UNIT=$1; VENV=$2; shift 2
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
echo "=== $UNIT on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start $(date -Is)"
echo "NET_PROBE huggingface.co http_code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' https://huggingface.co 2>&1 | tail -c 60)"
"${VENV}_exec" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable -- STOP'; print('CUDA OK', torch.cuda.get_device_name(0))"
"${VENV}_exec" "$@"
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT done $(date -Is)"
