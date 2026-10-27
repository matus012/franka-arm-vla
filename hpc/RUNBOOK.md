# PERUN runbook — 0C collect, export, train, evaluate (113)

Login node `<cluster-host>`, project `$PROJ=/mnt/project/<project-id>/113_vla_tabletop` (all paths in
`hpc/slurm/env.sh`). Container `$PROJ/vla_tabletop.sif` (`hpc/BUILD.md`): `/opt/venv` = sim, `/opt/policy` = policy.
Inside a job: repo at `/workspace` (= `$PROJ/repo`, a `git archive HEAD`, sha in `GIT_HEAD`), `/data` = `$PROJ/data`,
`/runs` = `$PROJ/runs`, HF cache `/cache/hf`.

Rules: never `sbatch --export` (job goes HELD) — values are positional args, `--time` / `--job-name` / `--dependency`
on the sbatch line are fine. Units are idempotent (`/runs/<unit>/DONE`). Datasets never leave PERUN; only JSON,
sidecars, sheets and checkpoints are synced back. Every job gets a row in `hpc/JOBS.md` (+ `000_infra/hpc_usage.md`)
and `ops/gpu_log.csv` (`perun_<jobid>`), GPU-h from the job log's start / done timestamps.

## 0. Push the committed code
```bash
bash hpc/push_repo.sh                      # git archive HEAD -> $PROJ/repo, sha256-checked; uncommitted work is NOT sent
ssh <cluster-host> "cat $PROJ/repo/GIT_HEAD"
```
A push replaces `$PROJ/repo` wholesale: never write outputs under `/workspace` (use `/data`, `/runs`).

## 1. 0C collect (GPU, `hpc/slurm/job.sbatch`, sim venv)
One job = one GATES 0C batch = a contiguous episode range of one dataset (>= 200 episodes), its own `--out` dir.
Data seeds / samplers (GATES 0C table): s1 31700, s2 31800, s3a 31900, s3b 32000, s4r 32100 (`--visual train_rand`,
`--mix` = the plan's s2 : s3a : s3b counts). H, N, counts and batch sizes come from the 0A.4 / 0A.3 RESULTs and the
`## GPU-h plan` entry; `<...>` below are those values.
```bash
cd $PROJ/repo
sbatch --time=01:00:00 --job-name=c0_s2_b000 hpc/slurm/job.sbatch c0_s2_b000 sim \
  scripts/collect.py --stage s2 --split train --start 0 --episodes 400 --seed 31800 --envs <B> \
  --noise-sigma 0.1 --image-size <H> --visual nominal --out /data/0c/s2/b000
sbatch --time=01:00:00 --job-name=c0_s2_b001 hpc/slurm/job.sbatch c0_s2_b001 sim \
  scripts/collect.py --stage s2 --split train --start 400 --episodes 400 --seed 31800 ... --out /data/0c/s2/b001
```
Resume after a kill: resubmit the same line with a new unit name (finished chunks are skipped). Read back per batch:
`/data/0c/<ds>/b<k>/collect_summary.json` (attempted / kept / rate + CI / failure classes / eps per h) and the chunk
sidecars; early abort per GATES 0C (first batch < ceiling - 10 points, or eps/H200-h projecting > 1.2 x plan).

## 2. Export + audit (CPU, `hpc/slurm/cpu.sbatch`, policy venv; 0 GPU-h)
```bash
sbatch --time=02:00:00 --job-name=x0_s2 --dependency=afterok:<collect job ids, colon-separated> \
  hpc/slurm/cpu.sbatch x0_s2 pol scripts/export_dataset.py \
  --src /data/0c/s2/b000 /data/0c/s2/b001 ... --name 0c_s2_pick --root /data/datasets --kept <plan count> --workers 16
sbatch --time=00:30:00 --job-name=a0_s2 --dependency=afterok:<export id> \
  hpc/slurm/cpu.sbatch a0_s2 pol tests/test_data_audit.py --root /data/datasets/0c_s2_pick
```
The export verifies every MANIFEST, audits the attributes (exit 1, nothing written, on any held-out item), keeps the
first `--kept` successes per stream, marks kept rank k % 20 == 19 as `val`, shards over `--workers` processes and
merges with LeRobot's `aggregate_datasets`. `tests/test_data_audit.py --root` is the GATES 0C item-2 audit on the
exported data (columns, split rule, test / validation stream collisions; ~1 min for the stream index).

## 3. Train SmolVLA (GPU, policy venv)
```bash
sbatch --time=<from plan> --job-name=t1_F hpc/slurm/job.sbatch t1_F pol -m vla_policy.train_smolvla \
  --dataset /data/datasets/0c_s1_pick --out /runs/s1_frozen --init lerobot/smolvla_base --n-action-steps <N> \
  --epochs <E1> --batch-size 64 --row-sampling window --freeze-vision-encoder true --train-expert-only true \
  --video-backend torchcodec --workers 8 --save-every-steps <25 % of steps> --seed 0
```
Chained jobs: resubmit the same line with `--resume` (continues from `checkpoints/last.txt`). Checkpoints:
`/runs/<run>/checkpoints/<step>/pretrained_model` (the eval `--ckpt`). Sync back only checkpoints + `train_summary.json`
+ `steps.jsonl` (`scp -r <cluster-host>:$PROJ/runs/<run>/checkpoints/<step> checkpoints/<run>/`).

## 4. Evaluate (GPU, sim venv; spawns the policy server with `/opt/policy/bin/python`)
```bash
sbatch --time=01:00:00 --job-name=e1_F_val hpc/slurm/job.sbatch e1_F_val sim scripts/eval_policy.py \
  --policy smolvla --ckpt /runs/s1_frozen/checkpoints/<step>/pretrained_model --stage s1 --split train \
  --visual nominal --episodes 300 --seed 21600 --envs 128 --image-size <H> --n <N> \
  --policy-python /opt/policy/bin/python --out /runs/e1_F_val/s1_F_val_s21600.json \
  --bars "pick.success>=0.90,pick.slam<0.05,pick.jam<0.02"
# smart length: same line + --extend-to 1000 instead of --episodes 300 (same --out; new unit name)
# close-up sheets (protocol: replay of the rounds that hold the chosen episodes):
sbatch --time=00:30:00 --job-name=f1_F_val hpc/slurm/job.sbatch f1_F_val sim scripts/eval_policy.py \
  --replay /runs/e1_F_val/s1_F_val_s21600.json --sheets 3 --videos 0 --policy-python /opt/policy/bin/python
# chunked-expert ceiling on the same stream: --policy expert (no --ckpt)
```
`--envs 128` is the protocol's fixed layout (episode i of a single-task stream in round i // 128, slot i % 128; all
128 slots always run, slots past the range are discarded): keep it identical across policies and checkpoints that are
compared. Base (untrained) weights need `--dataset <exported dataset>` for the normalization stats.
Pull back: `scp <cluster-host>:$PROJ/runs/<unit>/*.json results/` and `.../frames/<stem>/*.png
results/frames/<stem>/`; look at a sheet before writing any RESULT.

## 5. Log the job
```bash
ssh <cluster-host> "grep -h '^===' $PROJ/runs/logs/<name>_<jobid>.out"   # start / done timestamps
sacct -j <jobid> --format=JobID,Elapsed,AllocTRES%40,State
```
GPU-h = (done - start) x GPUs; a row per job in `hpc/JOBS.md` and `ops/gpu_log.csv` (`perun_<jobid>,start,end,minutes,...`).
