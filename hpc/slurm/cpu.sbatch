#!/bin/bash
#SBATCH --job-name=vla_cpu
#SBATCH --partition=cpu_short
#SBATCH --account=<project-id>
#SBATCH --time=01:00:00                  # SET PER JOB (sbatch --time=...; cpu_short MaxTime 2 days)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.err
# CPU-only unit (no GPU requested, 0 GPU-h), e.g. the 0C export. Same contract as job.sbatch:
#   sbatch hpc/slurm/cpu.sbatch <unit> <sim|pol> <python args...>
# Idempotent: exits 0 if /runs/<unit>/DONE exists. apptainer --nv (from env.sh) only warns on a node without GPUs.
set -euo pipefail
source /mnt/project/<project-id>/113_vla_tabletop/repo/hpc/slurm/env.sh
UNIT=$1; VENV=$2; shift 2
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
echo "=== $UNIT on $(hostname) cpu x${SLURM_CPUS_PER_TASK:-?} start $(date -Is)"
"${VENV}_exec" "$@"
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT done $(date -Is)"
