# Sourced by every 113 sbatch. All PERUN-side paths in one place (112 pattern; apptainer is /usr/bin/apptainer, no module).
export PROJ=/mnt/project/<project-id>/113_vla_tabletop
export SIF=$PROJ/vla_tabletop.sif             # built on the login node (hpc/BUILD.md), sha256 in $SIF.sha256
export REPO=$PROJ/repo                        # `git archive HEAD` unpacked by hpc/push_repo.sh (bound at /workspace)
export DATA=$PROJ/data
export RUNS=$PROJ/runs
export CACHE=$PROJ/cache                      # warp/mpl/tmp + HF cache (/cache/hf)
mkdir -p "$RUNS/logs" "$CACHE/warp" "$CACHE/mpl" "$CACHE/tmp" "$CACHE/hf"
# Never `sbatch --export=...` (job goes HELD): pass values as positional args / edit the script.
_exec() {  # _exec <python> <args...>
  local py=$1; shift
  apptainer exec --nv --containall \
    --bind "$REPO:/workspace","$DATA:/data","$RUNS:/runs","$CACHE:/cache" \
    --env WARP_CACHE_PATH=/cache/warp,MPLCONFIGDIR=/cache/mpl,TMPDIR=/cache/tmp,XDG_CACHE_HOME=/cache,HF_HOME=/cache/hf,WANDB_MODE=disabled,HF_HUB_DISABLE_TELEMETRY=1,PYTHONPATH=/workspace,PYTHONUNBUFFERED=1,OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    --pwd /workspace "$SIF" "$py" "$@"
}
sim_exec() { _exec /opt/venv/bin/python "$@"; }
pol_exec() { _exec /opt/policy/bin/python "$@"; }
