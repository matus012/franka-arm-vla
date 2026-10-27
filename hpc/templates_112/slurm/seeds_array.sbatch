#!/bin/bash
#SBATCH --job-name=vl_seeds
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=04:00:00                  # PLACEHOLDER: from the smoke wall
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                        # PLACEHOLDER
#SBATCH --array=0-2                      # 3 training seeds: 0,1,2  (3 x wall x 1 GPU = GPU-h; check vs budget BEFORE submit)
#SBATCH --output=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%A_%a.out
#SBATCH --error=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%A_%a.err
# 3-seed array. Usage: sbatch [--dependency=afterok:<smoke_id>] hpc/slurm/seeds_array.sbatch <unit-prefix> <command...>
# The command gets `--seed <SLURM_ARRAY_TASK_ID>` and `--out /runs/<unit-prefix>_s<seed>` appended.
# Site cap: >4 queued jobs per user are refused; array tasks count -> 3 is fine, do not stack more.
set -euo pipefail
source /mnt/project/<project-id>/112_vision_lift/repo/hpc/slurm/env.sh
PFX=$1; shift
SEED=$SLURM_ARRAY_TASK_ID
UNIT=${PFX}_s${SEED}
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
echo "=== $UNIT on $(hostname) start $(date -Is)"
vl_exec python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable -- STOP'"
vl_exec "$@" --seed "$SEED" --out "/runs/$UNIT"
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT done $(date -Is)"
