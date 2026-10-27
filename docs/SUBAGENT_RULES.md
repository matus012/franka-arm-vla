# Standing rules pasted into every 113 subagent prompt

- Never close, kill or restart any process the orchestrator did not start itself — including the owner's apps,
  browsers and other Claude sessions. If laptop RAM is short, move the run to PERUN or wait. (Consultant rule,
  2026-10-02, after a subagent closed the owner's browser.)
- No AskUserQuestion. No Hugging Face Hub pushes, no W&B, nothing public, no force-push, no repo/branch deletes, no spending.
- Write only inside the repo and under /mnt/project/<project-id>/113_vla_tabletop/ on PERUN.
- Laptop GPU work only through `ops/run_job.ps1` (GPU lock, 2 h kill, 20 h total, 1 GB free-RAM guard).
- PERUN: never `sbatch --export`; max 4 queued jobs; sacct does not work; log every job in `hpc/JOBS.md` + `ops/gpu_log.csv`.
- Gates: follow GATES.md to the letter; an entry is committed before its run; never move a threshold; look at every
  frame sheet you cite.
- Ledger (consultant 113, 2026-10-03): edit the 113 row in `000_infra\hpc_usage.md`, but never `git commit` in
  000_infra; those edits stay uncommitted for the consultant / owner.
