#!/bin/bash
#SBATCH --job-name=vl_stage4
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.err
# Stage 4 (visual randomization, not a gate) as ONE job: scripts/pipeline.py --stage s4 (estimated ~40 min on 1 H200).
# Arg $1 = the 3.4 pipeline name whose final pick is fine-tuned (default s34).
set -euo pipefail
source /mnt/project/<project-id>/112_vision_lift/repo/hpc/slurm/env.sh
echo "=== stage4 on $(hostname) start $(date -Is)"
vl_exec python -u scripts/pipeline.py --stage s4 --prev_stage "${1:-s34a2}" --tag "${2:-s4r2}"
echo "=== stage4 done $(date -Is)"
