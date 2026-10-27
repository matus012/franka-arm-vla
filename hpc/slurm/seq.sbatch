#!/bin/bash
#SBATCH --job-name=vla_seq
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=00:30:00                  # SET PER JOB (sbatch --time=...)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/113_vla_tabletop/runs/logs/%x_%j.err
# Several steps in one GPU job: sbatch hpc/slurm/seq.sbatch <unit> <steps file in the repo>
# Steps file: one step per line `<sim|pol|sh> <args...>` (# comments, blank lines skipped); `sh` runs on the host.
# Each step is timestamped (=== step k start/done epoch) and the job fails at the first failing step.
# Idempotent per unit (/runs/<unit>/DONE). GPU utilisation sampled every 1 s to /runs/<unit>/gpu_util.csv.
set -euo pipefail
source /mnt/project/<project-id>/113_vla_tabletop/repo/hpc/slurm/env.sh
UNIT=$1; STEPS=$REPO/$2
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
echo "=== $UNIT on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start $(date -Is) repo $(cat $REPO/GIT_HEAD)"
echo "PARTITION $SLURM_JOB_PARTITION $(scontrol show partition $SLURM_JOB_PARTITION | grep -o 'MaxTime=[^ ]*') job TimeLimit $(squeue -h -j $SLURM_JOB_ID -o %l) cpus ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$RUNS/$UNIT/gpu_util.csv" &
SMI=$!
pol_exec -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable -- STOP'; print('CUDA OK', torch.cuda.get_device_name(0))"
k=0
while read -r V ARGS; do
  V=${V%$'\r'}; ARGS=${ARGS%$'\r'}  # steps files are CRLF in a Windows checkout
  [[ -z "$V" || "$V" == \#* ]] && continue
  k=$((k+1))
  echo "=== step $k start $(date +%s.%N) $V $ARGS"
  if [ "$V" = sh ]; then eval "$ARGS" < /dev/null; else eval "${V}_exec $ARGS" < /dev/null; fi
  echo "=== step $k done $(date +%s.%N)"
done < "$STEPS"
kill $SMI || true
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT done $(date -Is)"
