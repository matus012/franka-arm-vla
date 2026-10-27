#!/bin/bash
#SBATCH --job-name=vl_render_check
#SBATCH --partition=gpu_short
#SBATCH --account=<project-id>
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.out
#SBATCH --error=/mnt/project/<project-id>/112_vision_lift/runs/logs/%x_%j.err
# FIRST cluster job. 1x H200, <= 0.35 GPU-h. Verifies headless Isaac Lab + Newton Warp camera rendering
# and prints images/s. Arg $1 = num_envs (default 256), $2 = steps (default 300).
# Submit: sbatch hpc/slurm/render_check.sbatch [256 300]      (from the login node, no --export)
set -euo pipefail
source /mnt/project/<project-id>/112_vision_lift/repo/hpc/slurm/env.sh
echo "=== node $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
N=${1:-256}; S=${2:-300}
# asset server reachability from the compute node (the env downloads DexCube / table / Franka USDs at runtime)
echo "NET_PROBE http_code=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd 2>&1)"
vl_exec python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable on the GPU node -- STOP'; x=torch.ones(2048,2048,device='cuda')@torch.ones(2048,2048,device='cuda'); print('CUDA OK', torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, float(x[0,0]))"
for RES in 128 224; do
  vl_exec python hpc/render_check.py --num_envs "$N" --steps "$S" --width $RES --height $RES --out /runs/render_check_${SLURM_JOB_ID}_$RES
done
vl_exec python hpc/render_check.py --num_envs "$N" --steps "$S" --width 128 --height 128 --depth --out /runs/render_check_${SLURM_JOB_ID}_128d
echo "RENDER CHECK DONE"
