# context — 113_vla_tabletop

Fine-tune a vision-language-action model (SmolVLA, `lerobot/smolvla_base`) on a simulated Franka tabletop:
pick / place / stack among 8 cubes (4 colours x 2 sizes) and 4 coloured pads, from side + wrist cameras, joint
readings and a typed instruction. Compared with pi0.5 (1 seed) and a no-language baseline that gets perfect labels.
Brief: `113_project.md` (the plan; `[consultant]` lines are part of it). Task file: `_todo/113_vla_tabletop/001-vla-tabletop.md`.
Gates: `GATES.md` (each committed before its run). Interfaces: `docs/CONTRACT.md`. Report: `REPORT.md`.

## Layout
- `lift_rl/` frozen copy of 111 (via 112), `vision_lift/` frozen copy of 112 — never edited, never imported from 111/112.
- `vla_common/` (stdlib + numpy; both venvs), `vla_tabletop/` (sim side), `vla_policy/` (LeRobot side).
- `scripts/` stage scripts; `scripts/templates_112/`, `tests/templates_112/`, `hpc/templates_112/` = 112 originals to adapt.
- `ops/run_job.ps1` — every laptop GPU job (GPU lock, 2 h kill, 20 h total) -> `ops/gpu_log.csv`.
- `hpc/` PERUN container + SLURM; every job logged in `000_infra/hpc_usage.md` + `ops/gpu_log.csv` (`perun_<jobid>`).

## Rules
- Subagent standing rules: `docs/SUBAGENT_RULES.md`, pasted into every subagent prompt. Incl. (consultant 2026-10-02):
  never close, kill or restart any process the orchestrator did not start (owner's apps, browsers, other Claude
  sessions); if laptop RAM is short, move the run to PERUN or wait.

## Venvs
- Sim: `.venv` (uv, `pyproject.toml`/`uv.lock`): Python 3.12, torch 2.10.0+cu130, isaaclab 3.0.0b2.post1 (Newton, lightweight).
- Policy: `policy/.venv` (`policy/pyproject.toml`): Python 3.12, lerobot[smolvla,pi]==0.6.1, torch 2.10.0+cu130.
- Sim and policy talk over localhost (`vla_common/protocol.py`).

## Facts verified
- 2026-10-01: PERUN `gpu_short` MaxTime = 2 days (`scontrol show partition`), `gpu_long` 4 days; 26 nodes x 8 H200.
  112's "gpu_short <= 2 h" was 112's own limit, not the site's -> no chained jobs needed for long fine-tunes.
- 2026-10-01: no Hugging Face token on the laptop (`~/.cache/huggingface/token` absent) -> pi0.5 tokenizer blocked (Needs you).

- 2026-10-01 (trainer agent, lerobot 0.6.1 source): `freeze_vision_encoder=false` alone does nothing while
  `train_expert_only=true` (the whole VLM incl. vision tower stays frozen). Stage 1's "unfrozen" arm must be defined in
  GATES.md Stage 1 (both flags false, optionally re-freezing the non-vision VLM layers so only the vision encoder is the
  change). lerobot-train runs SmolVLA in fp32 (no AMP); our trainer uses bf16 autocast (recorded in train_summary).
- 2026-10-01: laptop SmolVLA frozen bs8 128 px: ~0.08-0.09 GPU-s/sample, 3.0 GB VRAM; unfrozen bs8 does not fit 8 GB.
- 2026-10-01 (0B part 1): LeRobot's default video codec (libsvtav1 4:2:0 crf 30) gives min PSNR 30.6 dB on real
  128 px sim frames (< 0B.1's 35 dB) -> exporter uses h264 4:4:4 crf 14 (min 38.2 / mean 43.0 dB). Streaming encoder
  queue unbounded (it silently dropped frames); parallel encoding off on Windows.
- 2026-10-01: smolvla_base loads mostly bf16; bf16 serving is not batch-invariant (0.17 diff B=3 vs B=1) -> server runs
  strict fp32 (TF32 off; ~2e-6). Laptop fp32 query: 1.15 s (B=1), 2.3 s (B=16), 7.3 s (B=64) -> closed-loop evals
  of SmolVLA run on PERUN. Only `--rows control` exists; GATES option B = `FrameWindowSampler`.

## Deviation log
- 2026-10-01 (interpretation, GATES conventions): Stage 1 cube colour/size sampled per episode from the 6 non-held-out cubes.
- 2026-10-01 (contract v2): labels = clean clipped expert action, `action_exec` = noisy executed action (112's
  "labels as executed" lesson was about clipping; with demo noise, training on the noise would teach a flow-matching
  policy to reproduce it). Pads in every scene (lead finding). One policy request per query step (every N control
  steps), not per camera frame (brief §7 wording) — with chunked execution only query-step frames are used.
- 2026-10-01 (0A dev, before the first counted 0A run, commit 3963e51; audit 2026-10-01): pads are dynamic 2 mm mats of
  0.5 kg (kinematic pads made the claw drop every lifted cube: 0/32 holds vs 30/32); the expert waits for still fingers
  before lifting and freezes its hand yaw from the close on; the IK cube-weight feed-forward uses the episode's target
  cube mass. 0A.1 close-up frames come from the validation-seed camera check (21100), not the test stream.
- 2026-10-01: sim episode outcomes are not bit-reproducible across batch composition (6 of 8 replayed 0A.2 failures
  succeeded on replay) — relevant to 0B.3's replay-ceiling prerequisite.
- 2026-10-01 (orchestrator, before the 0B.3 counted run): trainable parameters are cast to fp32 (master weights) with
  bf16 autocast compute; LeRobot's default trains the bf16 weights directly, where late low-lr AdamW updates fall below
  bf16 resolution. Same for every arm and stage; recorded per run in train_summary.json. Unfrozen arm = vision encoder
  only (GATES Stage 1: both flags false + non-vision VLM re-frozen; 186.3 M vs 99.9 M trainable).
- 2026-10-01 (randomizer sim check, 295456c): `visual_rand.apply()` works in sim; cubes/pads keep their named colours
  (chroma-nearest 100 % in every mode; CIELAB misses = shaded yellow-large reading olive). The 112 table-hue train bands
  (0.00-0.35, 0.50-1.00) include red/yellow/green tables -> some low table/cube contrast. Kept (brief: the 112
  randomizer with its tested split); reported as a Stage 4 finding.
- 2026-10-01: `scripts/demo.py` (7b3a3ed) = the one-command demo (instruction + seed -> video); verified with the expert.
- 2026-10-02 ~07:00 — **Stage 1 one fix = relabel (authority: consultant 113, under owner standing approval; reversible
  by the owner at ~09:00).** Quote: "the D2 jitter clause is MET by its derivation. Approve (b) as Stage 1's one fix
  ... The clause was 'white jitter at the σ = 0 level'. The 0.014–0.045 range was only an estimate of that level from
  the 10 overfit demos ... The derivation-faithful reference is the noise-free expert on the same 128 validation
  episodes: 0.018 / 0.035 / 0.046 / 0.043. Relabelled: 0.022 / 0.037 / 0.050 / 0.043." The original criterion text
  (reports/2026-10-02-s1-fix-criteria.md) is not edited. Relabel r_t = clip(a_t + (x_{t-1} − a_{t-1}), −1, 1) on the
  arm dims, claw untouched, r_0 = a_0; HDF5 keeps the clean label and `action_exec`; LeRobot `action` = r. Cause
  (disclosed): σ-0.1 per-step noise on a delta-on-hidden-target action makes the expert's clean labels carry a deadbeat
  correction the policy cannot see; fixed by labelling with the expert's setpoint step.
- 2026-10-02 (orchestrator): 0C attempt-1 mutant m-c (relabel applied to the claw) is a no-op because the claw is never
  noised; the runner's m-c' (arm correction shifted onto the claw) tests the same property and is flagged — accepted as
  an instrument finding (no gate effect).
- 2026-10-02: **Stage 1 attempt 1 FAIL (test pick 57.7 %, bar 90) -> kill fired as written: Stage 1 branch stopped;
  Stages 2, 3A, 3B, 4 and pi0.5 do not run.** Relabel worked as a fix of the label defect (loss 0.46 -> 0.012, +19.7
  points paired on validation); remaining failure = grasp precision (cube gripped rotated 25-45 deg / off-centre, slips
  in the lift; large 46.6 vs small 68.6 %).
- 2026-10-02/03: **Owner redesigns after the Stage 1 kill** (relayed by consultant 113): Stage 1 redesign (224 px /
  N 10, FAIL 68.3 %), its 2k-demo fix (FAIL 78.5 %), redesign #2 = cube spawn yaw +-15 deg (V15 PASS 96.8 %, c707676).
  Stages 2-4 inherit yaw +-15 deg (disclosed). Stage 2 (Stage 1 init) FAIL 22.8 %, no instruction grounding (c8c5b15).
- 2026-10-03: **Owner decision (owner's word "go", relayed by consultant 113): the 113 PERUN cap is raised from 100 to
  130 H200-h.** pi0.5 stays dropped; stack stays in Stage 4 (Stage 4 re-planned against 130). Stage 2 redesign =
  option (a), init `lerobot/smolvla_base`, bars unchanged, one diagnosed fix then park. s4r palette: nominal table
  neutral grey, randomizer low-saturation tones only, held-out = disjoint darker/lighter grey + low light, with a
  colour-separation / contrast unit test; Stage 4 nominal table differs from Stages 1-3 (disclosed, or re-render if
  cheap; decided in the Stage 4 re-plan). No hard stop.
- 2026-10-04 ~01:30: **Stage 2 redesign (a) (smolvla_base init) stopped by early abort (d)** (c050 1.3 % <= 5 %;
  s_75 / s_100 4.3 / 6.7 %; true = shuffled 0/100). **Consultant 113 decision under the owner's 24 h autonomy
  directive:** Stage 2 redesign #2 = the brief's 2a -> 2b split with a grounding-oriented recipe: init the Stage 1 V15
  checkpoint, trainable = vision + text layers + action expert (both LeRobot flags false, NO refreeze; memo option (b)),
  2a data seed 32200, tests 2a 11540 / val 21640, 2b 11550 / 11560 / val 21650; optional 4xH200 DDP (same global batch
  and schedule), 2x steps only if wall <= 6 h, decided before launch. Stage 4's language-free parts (s4r data, no-language
  baseline x3) start in parallel. Budget priority: Stage 2 -> 3A/3B -> baseline -> SmolVLA Stage 4 seeds (3 -> 2 if
  needed). Gate bars unchanged; disclosed in REPORT and README.
- 2026-10-04 ~09:31 → 2026-10-24 20:48 UTC: the session died after 2c0df0d; jobs 98302 / 98305 / 98309 were read on
  2026-10-24 (orch 002, task `002-window-s2a1.md`, autonomy window to 2026-10-25 10:00 UTC), nothing re-run.
- 2026-10-24 (orch 002): **2a attempt 1 FAIL** (test 11541 pick 31.5 %, wrong-cube 33.8 %; df10928) → **Stage 2 stopped
  for good** as written (d33e371 / task 002): no 2b, 3A, 3B or Stage 4 SmolVLA seeds. **Baseline stopped** (overfit
  check 0 / 100 vs 80 %; fa18efc, erratum 2489d48): "the no-language baseline is not a useful reference in 113".
- 2026-10-24 (orch 002): `--epochs 11 --max-steps 34440` used for 2a attempt 1 so that `--max-steps` binds (≈ 3,441
  steps / epoch on the paired set); `a1post (iii)` `pass` is then false by its full-epoch clause only (0 / 34,440 steps
  differ) — stated in `hpc/steps/s2a1_post.txt` before the run. `s1_report.py` gained the `s2a1` run (tooling only).
- 2026-10-25 post-kill grounding diagnosis (003), reported only, ≤ 2.0 H200-h — authority: consultant 113, under owner autonomy window 2026-10-24T20:48Z–2026-10-25T10:00Z, reversible by owner
- 2026-10-25 inference-guidance probe (004), validation only, reported only, ≤ 1.5 H200-h — authority: consultant 113, under owner autonomy window 2026-10-24T20:48Z–2026-10-25T10:00Z, reversible by owner
- 2026-10-25 005 early abort (b) read on loss_total (L_true + λ·hinge) instead of L_true, same bar and step; L_true ratio reported — authority: consultant 113 (owner-approved 005), instrument definition for a converged init, reversible by owner
- 2026-10-25 standing owner autonomy: consultant 113 approves and launches tasks and reversible design rulings inside brief 001 and the 130 H200-h cap; supersedes 001's "design changes wait for the robotics chat" — authority: owner ("confirmed: standing autonomy, approve everything inside the brief", relayed by consultant 113), reversible by owner
- 2026-10-25 Stage 2R one diagnosed fix = more steps (+34,440 from 2R 034440, same loss), (b) replaced by c050 regression / grounding guards for a continuation run, test 11543 — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-25 113 PERUN cap 130 -> 160 (180 on consultant projection) — authority: owner ('raise to 160 - even to 180 if you think it will get good results'), relayed by consultant 113
- 2026-10-26 2b amended: paired s2 data + contrastive loss (as 2R/2M) instead of plain flow on unpaired s2; bars unchanged — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-26 window cut-off 09:15 lifted for 2b; standing autonomy continues past 10:00 CET — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-26 3A / 3B amended: paired data + contrastive loss + VT, init = Stage 2 (2b) checkpoint; bars and tests unchanged — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-26 2b one fix = full 34,440-step schedule from the 2M pick; guards moved to c075 relative to the init's own 21650 score (−5) + slam > 10 %; c050 kept for b2 only — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-26 113 PERUN cap 160 -> 180 taken up for task 009 (3A ∥ 3B one size per scene, 1×H200 per stage, line 17.2 / cap 20 -> 113 ≈ 174 / 180; nothing launches above 180) — authority: owner ('raise to 160 - even to 180 if you think it will get good results', 2026-10-25) on the consultant's written projection (`_todo/113_vla_tabletop/009-stage3-single-size.md` § Budget), reversible by owner
- 2026-10-27 009 I2 read on the init's pick stream (≤ 0.10) instead of the union; union I2 reported — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-27 3B stopped at c075 by its guards; no fix (budget reserved for 3A); stack dropped from the final — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-27 3A one diagnosed fix = +17,754 steps from the 3A pick (half schedule, budget-bound), guard place < 70 % at the first save, test 11581 + retention 11544 — authority: consultant 113, standing owner autonomy, reversible by owner
- 2026-10-27 113 PERUN cap 180 -> 200 — authority: owner ('up the limit to 200h')
- 2026-10-27 3A fix stopped by its guard at 008877 (place 64.0 % < 70 %); 3A closed for good — authority: the registered guard (consultant 113 ruling)
- 2026-10-27 final evaluation of the passed / best checkpoints on the brief's splits, reported only (no Stage 4 training: 3A kill); no stack fix (41 % vs 70 % bar, 3A fix showed a restart hurts) — authority: consultant 113, standing owner autonomy, reversible by owner

## Facts verified (2026-10-24)
- PERUN reachable from the laptop (`ssh <cluster-host>`), queue empty at session start; `/runs` and `/data`
  intact after 20 days.
- Paired-set t = 0 frames through the trainer's torchcodec path: wrist differs within groups by median 1,584 px (raw
  renders: median 4.5 px) — per-episode h264 coding; not gated.

## Facts verified (2026-10-25, task 003)
- Instructions are 4-16 tokens with the checkpoint's own preprocessor (newline step + tokenizer, max_length 48, padding
  max_length): no truncation anywhere (results/g3_diag/prompt_len_*.json). t2a1_VT ran at lr 1e-4 (argv `lr: None`).
- SmolVLA 0.6.1 prefix = [64 x 2 image tokens, 48 language tokens (padded), 1 state]; prefill runs every layer as VLM
  self-attention; odd layers are cross-attention for the expert (its own k_proj / v_proj on the cached post-RoPE VLM
  K / V). Feature hooks proven exact on laptop (results/g3_diag/hook_check_s1a1U.json).
- Grounding diagnosis (GATES 003 RESULT): R3 data / loss for 034440 and S1 V15 - colour decodable from language tokens
  and expert-read K / V (CV 1.000), actions ignore it (instruction / noise effect 0.10 / 0.08); start noise picks the
  direction. The policy venv has no sklearn (probe = torch L-BFGS logistic regression in scripts/g3_probe.py).

## Facts verified (2026-10-25, task 004)
- Guided sampler (`vla_policy/guidance.py`, server `--guidance-w` + op `guide`, eval `--guidance-w`): w = 1 is the
  plain policy bit for bit on the same batch (laptop proof on s1a1_U); closed loop on 034440 reproduces s_100 (28.7 vs
  28.0 %). Cost: server 5.9 s / query at B 128 with K + 1 ≈ 3 branches (plain 2.0 s); ~0.2 H200-h per 300-ep run.
- Inference-time guidance on 034440 (validation 21640): no effect (best w = 2: +1.0 point); step 2 ratio 0.10 → 0.51
  at w = 8, so the instruction's pull on the chunk scales with w but stays below the start-noise effect.
- Europe left DST on 2026-10-25 03:00 CEST: PERUN job logs print +01:00 from then on; `ops/run_job.ps1` now measures
  durations in UTC (the smoke row logged −53.59 min before the fix; corrected to 6.40).

## Facts verified (2026-10-25, task 005 — Stage 2R)
- Instruction-contrastive hinge (`vla_policy/contrastive.py`, `train_smolvla --contrastive-lambda / --contrastive-margin`):
  λ = 0 is the plain trainer bit for bit; both branches share x_t / t / ε (asserted every step); the vision tower can be
  shared (fp32-equal to separate passes; bf16 within rounding). Cost 1.385× the plain step (smoke 131405).
- m by rule = 4.2103 (0.5 × mean squared first-chunk difference between paired members, normalized, 50 × 5 entries;
  per dim x / y / z / yaw / claw 14.2 / 14.6 / 1.1 / 12.3 / 0.0); cross-checked through the trainer's own reduction.
- At the language-ignoring fit the hinge gradient wrt v' is exactly 0 (unit test), yet in training the gap opened within
  ~2,000 steps (0.9 → 26.5) and the hinge was idle (< 0.3 % active) from ~12k steps.
- Stage 2R: test 11542 pick 67.7 %, wrong-cube 3.5 %; fix (+34,440 steps from 2R 034440) test 11543 pick 77.6 %,
  wrong-cube 3.9 %; probe named cube 93.6 → 95.5 % of grasps. Stage 2 closed for good (task 005 kill).
- The trainer stamps `GIT_HEAD` into train_summary at the end of a run: never push the repo to PERUN while a training job
  runs (both 005 runs carry the launch sha).
- `000_infra/bin/l2-wait.ps1` with an ssh -PollCmd: use `-o BatchMode=yes -o LogLevel=ERROR`, make "not there yet" exit 1
  (e.g. `test -f $f || exit 1`; grep on a missing file exits 2 = error), and avoid patterns with spaces in nested quotes.

## Facts verified (2026-10-26, task 006 — Stage 2M / 2b amended)
- Stage 2M (colours, 2–4 cubes): 6,000 paired demos (seed 32800 groups 0–3299) + 68,880 steps at lr 5e-5 of the
  contrastive loss from the 2R fix checkpoint → test 11544 pick 96.7 %, wrong-cube 0.1 % (PASS); named cube 99.5 % of grasps.
- 2b amended (sizes, 2–5 cubes, paired s2 seed 32900, m 4.2989): the 2M checkpoint on s2 validation picks 69.0 % with
  wrong-cube 18.3 % (size confusion) and slam 0.7 %; contrastive fine-tuning removed most size confusion (test wrong-cube
  4.4 %) but induced off-target table slams mid-training (17 % at c050) that only half-recovered (test slam 9.4 %); test
  11551 pick 68.1 % (FAIL). Stage 2 = colours PASS (2M), sizes FAIL.
- A re-run with the same seed / init / data reproduces the earlier run step for step (the 2b fix matched 132815 at step
  100 and at c050 within noise); `--resume` is tested for indices / lr exactly and loss to 1e-4, not bit-exact.
- Paired s3a layouts always have 3 members (2,000 / 2,000 groups) → whole-group kept counts are multiples of 3
  (`export_dataset.py --kept-stream` per-stream override; `s3_union_check.py --kept-task`).
- PERUN `repo` is a symlink to `repos/<sha>`; jobs run in the versioned dir they started with.
- `job.sbatch` direct command lines with `--bars "a>=b,c<d"` keep their quoting (bars recorded in the eval JSON).

## Facts verified (2026-10-26, task 008 — 2b sizes round 2)
- 3× paired s2 data (12,000) + 68,880 steps at lr 5e-5 from the 2b fix pick: test 11552 pick 79.4 % (68.1 before), slam
  9.1 % (9.4 before), wrong-cube 2.6 %; by cubes 91 / 84 / 72 / 70 %. The 2M lever raised precision but not the slams
  (wild reaches into the table, even with 2 cubes). Sizes closed for good (owner-approved consequence).
- A step of a running seq job reported the newly pushed sha (c100 vloss: cdbcbfb, job submitted on a0ba0dd): treat
  "jobs keep their tree" as unproven for later steps — push only with no PERUN job running when the code path matters.
- `l2-wait.ps1`: a `$` in -PollCmd is expanded locally by PowerShell; use `-Remote` (000_infra task 010) or a `$`-free
  poll (`squeue -h --me -o %i | grep -qx N && exit 1; exit 0`).
- `eval_policy.py --paired-targets` accepts s2a / s2 / s3a / s3b / s3a1 / s3b1 (d35b36a; it refused s2 before).

## Facts verified (2026-10-27, task 009 — Stage 3 place / stack, one cube size per scene)
- From the 2M colours pick (068880), paired + contrastive, 35,508 steps at lr 5e-5 on 1×H200 per stage (0.602 s/step,
  6.08 H200-h for the full schedule): 3A test place 72.7 % (bar 80), wrong pad 0.2 %; pick retention 90.2 % (2M 96.7,
  −6.5 points paired). 3B stack 40.7 % at c075 (stopped by its guards). Grounding is complete (3A probe: named cube
  289 / 289 grasps, named pad 246 / 246 releases); the failures are motor (grasp centring in clutter; releases off-centre,
  not released or not cleared within T 550; 3B wild reaches and drops ~50 %). The one 3A fix (+17,754 steps from the 3A pick, lr 5e-5 restarted, cosine) was stopped by its guard at 8,877: place validation 75.0 → 64.0 %, pick 88.7 → 83.3 % on the same seeds — continuing from a converged checkpoint with the lr restarted pulled it off its best point. 3A closed for good.
- I2 (mean L_true steps 1–50 ≤ 0.10) cannot hold on a union with a new task from an init trained without it (union I2
  4.47 / 2.99); read on the init's own stream (pick 0.0117) with a base-init mutant (0.3755) — consultant ruling 2026-10-27.
- `s1_tools a1post` `check_iii.pass` also demands n = steps/epoch × epochs; `--max-steps` that stops short of an epoch
  boundary makes it false with 0 differing steps (008 and 009 both) — read I4 as steps_differing.
- `paired_check` honours the export's `--kept-stream STAGE:SEED=N` override (`stream_kept`, 6595627).
- During training, the trainer stamps GIT_HEAD at the end: no repo push while a run trains; stage step files under
  `/runs/steps_scratch/` and pass `../../runs/steps_scratch/<file>` to seq.sbatch (`repo` → `repos/<sha>` symlink).
- s3a1 / s3b1 validation-to-test: 3A val c100 75.0 % → test 72.7 %; pick val 88.7 % → test 90.2 %.

## Facts verified (2026-10-27, task 010 — final evaluation, reported only)
- Fresh 1,000-episode streams, the models' own test flags: (a) seen reproduces each test within ~2.4 points (S1 95.3,
  2M 96.0, 3A place 71.5, 3A pick 87.8 %).
- s4r randomized visuals (grey table, `--visual train_rand --table-palette s4r`) break every 113 policy: 1.5–18.7 %
  (none was trained on randomized visuals).
- 2M colour grounding is tied to the cube identities seen as targets: on heldout_comp (green-large / yellow-small
  targets, one size per scene) 17.7 %, and every wrong grasp goes to one cube (yellow → green-small 330 / 330, green →
  yellow-large 313 / 313).
- After place training, the 3A checkpoint picks with "pick up" (87.8 %) but with the pick paraphrases grips the named
  cube and sets it down on a pad (12.2 %, drop 574 / 1,000; frames eps 0, 7).
- `heldout_comp` is undefined for the one-size samplers s3a1 / s3b1 (`vla_common/tasks.py:197`).
- Server latency on H200: p50 ≈ 2.03 s per batched query of 128 envs.
- Replay of the 2M (c) wrong grasps reproduces 10 / 10; 3A place (a) replays are the least reproducible (3 / 9).

## Facts verified (2026-10-28, task 011: demo video, figures, public push)
- `eval_policy.py --record` (viewer only, new runs): a fixed 960x720 viewer camera per env (aimed once at t = 0 at
  the on-table cubes; never fed to the policy) + the policy's own side / wrist frames + the evaluator's per-tick signals
  and the target / destination / fingertip projected into viewer pixels (`vla_tabletop/demo_rec.py`; projection
  checked by drawing it on frames and by unit tests). Isaac Lab 3 camera data are warp proxies: use `.torch`; quats xyzw.
- Demo seeds 51000–51099 (declared in hpc/JOBS.md before any job); 16 eps per stream at B 16: ~2 min per stream
  on one H200 incl. start-up; 13 streams = 0.40 H200-h. 113 = 179.35 / 200.
- Same-scene grounding (demo group 1, large red / yellow / blue): plain 2a-attempt-1 model told red closes beside it,
  told blue lifts the red cube, told yellow succeeds; the 2M model succeeds on all three.
- Demo sample rates (16 eps, viewer only, not measurements): S1 16/16, 2M 16/16, sizes 14/16, place 12/16, stack 6/16,
  2M randomized visuals 5/16, 2M held-out 2/16, 3A pick paraphrase 2/16; paired groups 0–5 (15 eps): plain 5/15, 2M 14/15.
- Card text is generated from the evaluator constants + result JSONs (`scripts/demo_facts.py` -> `media/cards.json`,
  checked by `tests/test_demo_facts.py`); `scripts/demo_gates.py` checks G1–G4.
- Public push: history holds the PERUN project id, cluster hostnames / node names and a Windows home path (no
  secrets); `main` is never rewritten; `scripts/make_public_tree.py --commit` builds a redacted single-commit orphan
  branch `public` (no checkout) which is pushed as the remote's main.
