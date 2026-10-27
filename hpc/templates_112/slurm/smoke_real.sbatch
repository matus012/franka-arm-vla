#!/bin/bash
#SBATCH --job-name=vl_smoke_real
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.err
# Real-job smoke (<= 0.5 GPU-h): the same scripts as the laptop stage cycle, timed per step for a like-for-like
# comparison: teacher collection (P6, 512 envs, 512 episodes), BC fine-tune (400 k samples, 4 workers), closed-loop
# student evaluation (P6, 256 envs x 256 episodes). Laptop references: collect 512@512 = 88.9 s; training 2,360
# samples/s (RGB); eval 256@256 ~ 45 s.
set -euo pipefail
source /mnt/project/<project-id>/112_vision_lift/repo/hpc/slurm/env.sh
echo "=== node $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
t() { local s=$(date +%s.%N); "$@"; local e=$(date +%s.%N); echo "STEP_WALL $(python3 -c "print(round($e-$s,1))") s :: $*"; }
t vl_exec python -u scripts/collect.py --world P6 --driver teacher --action_sigma 0.1 --episodes 512 --num_envs 512 \
    --seed 99001 --out_dir datasets/smoke_perun --summary runs_smoke_collect.json
t vl_exec python -u scripts/train_bc.py --data datasets/smoke_perun --out checkpoints/smoke_perun --depth 0 \
    --samples 400000 --lr 1e-4 --init checkpoints_ref/s33a2_pick.pt --w_aux 1.0 --workers 8
t vl_exec python -u scripts/eval_students.py --world P6 --ckpts checkpoints/smoke_perun/epoch_001.pt --episodes 256 \
    --num_envs 256 --seed 99002 --json runs_smoke_eval.json
echo "SMOKE DONE"
