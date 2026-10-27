# Dataset / artifact sync plan (112_vision_lift <-> PERUN)

**rsync/scp/ssh are NOT to be run until the owner says PERUN is connected.** Nothing here was executed. Host/user/key come from
`~/.ssh/config` (`<cluster-host>`, `ssh -o BatchMode=yes`; skill `hpc` section 1). `<PERUN>` below = that ssh alias.

## Where things live on PERUN
Project storage `/mnt/project/<project-id>/112_vision_lift/` (the only storage the hpc skill documents; no separate scratch is
documented -> **PLACEHOLDER: ask the owner whether a faster scratch FS exists; until then use project storage for everything**).

| path | content | direction |
|---|---|---|
| `vision_lift.sif` (+ `.sha256`) | Apptainer image (~6-10 GB, see BUILD_TEST.md) | laptop/WSL -> PERUN, once (re-sync only when `hpc/uv.lock` changes) |
| `repo/` | `git archive HEAD` of this repo (code, teacher `.pt` 1.3 MB, `hpc/`) | laptop -> PERUN each time code changes; small |
| `data/<stage>/*.h5` | HDF5 episode chunks | see below |
| `runs/<unit>/` | logs, checkpoints, result JSONs, `DONE` marker | PERUN -> laptop (pull small artifacts only) |
| `cache/` | warp kernel/torch caches | stays on PERUN |

## Size estimate (to be replaced by Stage 0.5's measured number)
Raw ~33 MB/episode (2 cams x 128x128, RGB uint8 + fp16 depth, 25 Hz, 8 s). Chunks are HDF5 with gzip(4)+shuffle: synthetic
renders compress well, estimate **2-3x -> ~11-16 MB/episode, ~11-16 GB per 1,000 episodes** (assumption, not measured).
Total = episodes needed (0.5 estimate) x that; e.g. 50k episodes ~ 0.55-0.8 TB -> exceeds a sensible laptop->cluster upload
(at 100 Mbit/s 1 TB ~ 22 h). **Rule: generate data ON PERUN when the dataset is > ~50 GB; upload only seeds/configs.** Uploading
is for small seed sets (Stage 1/2 pilot data) and for the laptop-collected DAgger seeds if any.

## Chunk contract (what makes resume + verification cheap)
- One HDF5 file = N episodes (target 100-200 MB), name `<stage>_<seed>_<idx:05d>.h5`. Writers write `*.h5.tmp` and `os.rename` to `.h5`
  only after closing -> rsync of `*.h5` never sees a half-written chunk.
- A manifest `data/<stage>/MANIFEST.sha256` (`sha256sum *.h5`) is written by the producer after the last chunk.

## Commands (run only after "connected")
```bash
# 0. image + repo (laptop/WSL -> PERUN)
sha256sum vision_lift.sif > vision_lift.sif.sha256
rsync -av --partial --progress vision_lift.sif vision_lift.sif.sha256 <PERUN>:/mnt/project/<project-id>/112_vision_lift/
ssh -o BatchMode=yes <PERUN> 'cd /mnt/project/<project-id>/112_vision_lift && sha256sum -c vision_lift.sif.sha256'
git archive --format=tar.gz -o repo.tgz HEAD && rsync -av repo.tgz <PERUN>:/mnt/project/<project-id>/112_vision_lift/
ssh -o BatchMode=yes <PERUN> 'cd /mnt/project/<project-id>/112_vision_lift && rm -rf repo && mkdir repo && tar xzf repo.tgz -C repo'

# 1. dataset up (small pilot sets only)
rsync -av --partial --partial-dir=.rsync-partial --include='*.h5' --exclude='*' data/s01/ <PERUN>:/mnt/project/<project-id>/112_vision_lift/data/s01/
rsync -av data/s01/MANIFEST.sha256 <PERUN>:/mnt/project/<project-id>/112_vision_lift/data/s01/
ssh -o BatchMode=yes <PERUN> 'cd /mnt/project/<project-id>/112_vision_lift/data/s01 && sha256sum -c MANIFEST.sha256'

# 2. results down (small artifacts ONLY: result JSONs, curves, chosen checkpoints, logs; never full data dirs)
rsync -av --partial --include='*/' --include='*.json' --include='*.csv' --include='*.log' --include='*.out' --include='*.err' \
      --include='model_best.pt' --include='DONE' --exclude='*' <PERUN>:/mnt/project/<project-id>/112_vision_lift/runs/ runs_pulled/
# then verify: each pulled checkpoint has a sha256 in its result JSON (trainer writes it) -> `sha256sum` and compare
```
- **Resume:** `--partial` keeps an interrupted file; rerunning the identical command continues. Whole-file `--checksum` is NOT used
  on the big sync (slow); integrity is the sha256 manifest check after the transfer. Re-run the manifest check, re-rsync any file
  that fails (`rsync -av --checksum` on that file only).
- **Never** delete on either side (`--delete` is not used).
- **Throttle:** the site cap is 4 queued jobs, not transfers, but keep transfers to one rsync at a time on the login node
  (`--bwlimit=20000` if the owner asks).
- **Partner data:** none of this is partner/client data (simulated images only); still check the manifest contents before upload (hpc skill section 0).
- Each job's results are ledgered per the README logging rule.
