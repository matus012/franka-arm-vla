# Sourced by every 112 sbatch template. All PERUN-side paths in one place.
# Verified in the hpc skill (110_lockon, 2026-09-05): account, partitions, /mnt/project storage root.
# apptainer on PERUN: /usr/bin/apptainer, no module needed (reported by the consultant 30 Sep 2026 ~20:15; re-verify on first use).
export PROJ=/mnt/project/<project-id>/112_vision_lift
export SIF=$PROJ/vision_lift.sif              # built off-cluster (WSL), rsynced, sha256-checked (see SYNC_PLAN.md)
export REPO=$PROJ/repo                        # `git archive HEAD` bundle unpacked here (bind-mounted at /workspace)
export DATA=$PROJ/data                        # HDF5 episode chunks (see SYNC_PLAN.md)
export RUNS=$PROJ/runs                        # logs, checkpoints, result JSONs
export CACHE=$PROJ/cache                      # warp / torch / matplotlib kernel + font caches (persist across jobs)
# (no `module load` needed for apptainer on PERUN)
mkdir -p "$RUNS/logs" "$CACHE/warp" "$CACHE/mpl" "$CACHE/tmp"
# Always: run inside the container, --nv exposes the H200 driver, --containall keeps $HOME out of the image.
# Never `sbatch --export=...` (job goes HELD on PERUN): pass values as positional args / edit the script.
vl_exec() {
  apptainer exec --nv --containall \
    --bind "$REPO:/workspace","$DATA:/data","$RUNS:/runs","$CACHE:/cache" \
    --env WARP_CACHE_PATH=/cache/warp,MPLCONFIGDIR=/cache/mpl,TMPDIR=/cache/tmp,XDG_CACHE_HOME=/cache,OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    --pwd /workspace "$SIF" "$@"
}
