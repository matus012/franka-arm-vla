# 113_vla_tabletop — Project Brief

Written by: robotics chat, 1 Oct 2026.
Pipeline: chat (this brief) → consultant CLI (review, may rename the folder, launches and manages the orchestrator, audits every gate, final summary) → orchestrator CLI (build, train, evaluate, report).
Standalone repo. Copies tools from 111 and 112 (never imports them). No trained model from 111 or 112 is used. The pretrained VLA weights come from Hugging Face.
Scope: **VLA only.** No VLM planner.

---

## 1. Purpose
- Learn how a vision-language-action model works by fine-tuning one.
- Input: camera images + typed instruction + the robot's own joint readings. Output: arm actions.
- Language must change behavior: several cubes in different colors and sizes, colored pads, stacking.
- Portfolio result: SmolVLA vs π0.5 vs a no-language baseline, on seen instructions, held-out combinations, reworded instructions and randomized visuals.

## 2. How this project is run (lessons from 111 and 112, mandatory)
- **Staged:** one change per stage. Every stage passes its gate before anything builds on it. Parallel branches are allowed only from a passed checkpoint.
- **Gates written first:** gates, early-abort signals, fallbacks and checkpoint-pick rules go into GATES.md and are committed before each run. Gates never move.
- **No blind retries.** On a failed gate, use that stage's pre-written fallback. If none is left, stop that branch and report.
- **Standing owner approval (as in 112):** diagnose, one fix, new attempt. Gates, thresholds and budgets never move. Design changes stop and go to the robotics chat.
- **Check close-up frames at every gate,** not just metrics.
- **Consultant audits** every gate against JSON results, git timestamps (gate committed before the run) and frames.
- **Smart test length** (all intermediate gates):
  - Run 300 test episodes.
  - If the result is more than 5 points from the bar, decide.
  - Otherwise extend to 1,000.
  - Final evaluations always use 1,000.
- **Carried-over lessons:**
  - Measure the expert + student-actuator ceiling early.
  - Budget training in epochs, not samples.
  - Randomization must contain the nominal scene, with held-out settings disjoint (unit-tested).
  - Stateful evaluators share the expert's blind spots.
  - Offline accuracy doesn't predict closed-loop success.
  - On PERUN, generate data in place and sync only checkpoints.
  - If DAgger is ever used (fallback only): first prove the expert is a function of the current state, on student-driven data. Otherwise use a relabeler.

## 3. World
- **Robot and table:** Franka Panda and table from 111/112. Physics settings from 111 (stiffer contacts, 4 substeps, finger armature).
- **Cubes:** 4 colors (red, green, blue, yellow) × 2 sizes.
  - Small = the 111 cube size.
  - Large = ~1.5× that, and it must fit the open gripper with margin.
  - [consultant] Numbers: the 111 cube is DexCube × 0.8 = 4.8 cm edge (111 context.md; its measured rest height 2.07 cm is relative to the env origin, so measure the real edge in 0A.1 and record it). The Franka opens 8.0 cm total (0.04 m per finger). 1.5× = 7.2 cm leaves 4 mm per side, and at the expert's 5° yaw tolerance the cube's footprint is 7.8 cm, so it does not fit with margin. Rule: large edge ≤ 6.0 cm (≈ 1.25×), i.e. edge·(cos 10° + sin 10°) ≤ 7.0 cm, giving ≥ 1 cm total clearance at a 10° yaw error. Same density gives about 0.42 kg (111 tested up to 0.5 kg at friction 1.0).
  - Same density for both sizes. Document the final sizes.
- **Pads:** flat colored squares on the table, in the same 4 colors, large enough for a large cube.
- **Physics:** nominal mass and friction only. No weight/friction randomization in 113 (disclosed).
- **Placement:** objects randomly placed, all in reach, minimum spacing, no overlaps.
- **Cameras:** side + wrist (from 112). A third fixed camera is allowed only if the 0A camera check fails with two.
- **Visual randomization:** the 112 randomizer with its tested split, used in Stage 4 and its data.
  - [consultant] The 112 randomizer tints the cube's hue across the full colour wheel. In 113 colour is the instruction, so a random hue would make "the red cube" wrong. For 113:
    - cube and pad hues stay at their named colours, with only small brightness and saturation jitter;
    - a unit test proves every jittered cube and pad colour is still nearest its own named colour, in both the train and held-out modes;
    - hue randomization applies to the table only;
    - light direction and camera-pose jitter stay as in 112;
    - the held-out band (table hue, low brightness, low light elevation) stays disjoint from train, and train contains the nominal scene, as in 112.

## 4. Tasks and instructions
- **Pick:** "pick up the small blue cube". Success: the named cube held ≥ 10 cm above the table at episode end.
- **Place:** "put the red cube on the green pad". Success: the named cube resting on the named pad, released, gripper clear.
- **Stack:** "stack the red cube on the blue cube". Success: the top cube resting on the named bottom cube, released, and stable for 1 s.
- **Instruction generator:**
  - Several templates per task.
  - Every instruction matches exactly one cube (and one destination). If two cubes share a color, the size must be stated.
- **Held-out sets** (never seen in training; disjointness unit-tested):
  - Compositions: 2 of the 8 color×size cubes are never pick targets in training (they still appear as distractors), plus a few held-out cube–pad and cube–cube pairs.
    - [consultant] "Never pick targets" means never the manipulated cube in any training task: not picked, not placed, and not the top cube of a stack. They may still be the bottom cube of a stack or a distractor. Neither held-out cube is the Stage 1 single cube.
  - Paraphrases: a separate set of reworded templates.

## 5. Expert (demo generator)
- Extend the 111 scripted controller (DLS IK phase machine, true positions, 95.6 % in 111) to pick among distractors, place on a pad, and stack.
- Release = open the claw above the destination, then retreat.
- [consultant] The 111 controller and phase machine hard-code the cube size (`CUBE_HALF` and `CUBE_REST_Z` = 0.0207 in `phases.py` and `metrics.py`). Replace them with per-cube values. Stack heights come from the bottom cube's size.
- Keep only successful demos (filtered by the success detectors).
- Inject small action noise during demos to widen the data (document σ).
- The expert is privileged and only used to generate data. Not used for DAgger unless a fallback calls for it (see §2).

## 6. Policies

### Main: SmolVLA
- `lerobot/smolvla_base` (~450M, Apache-2.0). Used in all stages.
- Vision encoder frozen vs unfrozen: decided in Stage 1 by a parallel run.
  - Rule: higher success wins. If within 3 points, pick frozen.
  - Why test it: a public SO-101 project found the frozen default couldn't adapt to its scenes.

### Comparison: π0.5
- `lerobot/pi05_base` (~4B; [consultant] 3.6B per the Hub). Final comparison only, 1 seed, PERUN only.
- LeRobot installed from GitHub main for π0.5, pinned to a commit.
  - [consultant] Not needed: π0.5 is in the LeRobot 0.6.1 release (`lerobot[pi]`). Pin `lerobot==0.6.1` for both SmolVLA and π0.5. Use a pinned main commit only if a documented fix is missing from 0.6.1.
  - [consultant] π0.5 needs the gated `google/paligemma-3b-pt-224` tokenizer. The owner accepts its licence on Hugging Face and provides a read-only HF token (laptop and PERUN). Without it, π0.5 is blocked; SmolVLA is not affected.
  - [consultant] π0.5 defaults to QUANTILE normalization (needs q01/q99 in the dataset stats). Either recompute the stats or set MEAN_STD, and record the choice in GATES.md.
  - [consultant] The Hub tags the weights `gemma` while the LeRobot docs say Apache-2.0. Treat them as Gemma terms (never publish), as already stated below.
- Weights are a Gemma derivative: never publish fine-tuned π0.5 weights. Code and results only.

### Baseline: no language
- The 112-style small CNN network, trained fresh on the same demos.
- Gets the task as perfect labels (task type, cube color, cube size, destination) instead of words. Rewording therefore can't hurt it.
- It measures what the pretrained VLA adds beyond perfect parsing. Report it that way.

### All policies
- **Inputs:** side + wrist images, joint positions/velocities, gripper width, and the instruction (text, or labels for the baseline).
- **Banned inputs:** object poses in any form, contact forces, privileged phase signals. A test enforces this.
- **Outputs:** the same action as the 112 student (Cartesian arm deltas incl. yaw through the same IK, hand held vertical, plus a binary claw), predicted in chunks.
- **Timing:** control rate as in 112. The camera rate and chunk execution length come from 0A.
  - [consultant] In 112 the control rate is 50 Hz (`decimation=2`, `dt=0.01`). 16.7 Hz is the 112 camera rate: a frame every 3 control steps, 1-frame latency.
  - [consultant] The dataset frame rate (one row per control step at 50 Hz, or one per camera frame) changes storage and training cost about 3×. Decide it in 0A.3/0B together with the chunk length, prove it with the overfit test, and write it into GATES.md before 0C.
- [consultant] **How the inputs map onto SmolVLA** (`smolvla_base` config, checked):
  - cameras: `camera1`/`camera2` via `rename_map`, with no third placeholder unless 0A adds a camera. Images are resized to 512 with padding inside the model, so any 0A size works.
  - state: joint pos 9 + vel 9 + gripper width 1 = 19 dims, padded to 32.
  - action: 5 dims, padded to 32.
  - chunks: `chunk_size` 50, `n_action_steps` = N from 0A.3.
  - normalization: MEAN_STD for state and action, identity for images. The binary claw is thresholded at 0.5 after un-normalizing.
  - instructions: at most 48 tokens.
  - frozen-vs-unfrozen = `freeze_vision_encoder` (default true; `train_expert_only` is also true by default, so record both flags).
  - π0.5: 224 px, the same 32-dim padding.
  - none of these needs a design change.
- [consultant] The baseline's 112 network is single-step, has 0.8 M parameters and took `prev_action` + `target` as inputs. For 113:
  - drop `prev_action` (copycat lesson) and `target`;
  - add the task labels;
  - make the head output an N-step chunk so it executes like the VLAs.

## 7. Software architecture
- **Two venvs:**
  - Sim venv: Python 3.12, torch 2.10.0 CUDA 13 pinned, Isaac Lab 3.0.0b2.post1 lightweight mode.
  - Policy venv: LeRobot + SmolVLA (+ π0.5), its own pinned versions.
  - They talk over localhost through a batched policy server, one request per camera frame for all envs.
- The same setup runs on PERUN inside the container(s), extended from 112's Apptainer image.
- [consultant] The policy venv needs Python ≥ 3.12. LeRobot 0.6.1 allows torch ≥ 2.7, < 2.12 and needs transformers ≥ 5.4, < 5.6. Pin torch 2.10.0+cu130 in the policy venv too (same as the sim venv; still separate venvs). Disable wandb in every LeRobot command (the `[training]` extra installs it).
- Demos are stored in LeRobot dataset format.
  - [consultant] Format v3.0 stores parquet + MP4 per camera (ffmpeg). On native Windows, torchcodec needs shared FFmpeg ≤ 7 DLLs and isn't verified with torch 2.10. Use the pyav decoding backend locally; PERUN (Linux) uses torchcodec. Don't use the image-only (PNG) mode for bulk data, it's about 35× larger. The 0B round-trip test runs on both platforms.
- **Hugging Face Hub pushes and Weights & Biases logging: OFF.** Downloads from the Hub are fine. Local TensorBoard only.

## 8. Stages and gates

### 0A — Scene + expert (in parallel with 0B)
- **0A.1 Scene:** cubes, pads and cameras load. A scripted grasp holds each cube size in ≥ 95 % of 300 trials.
- **0A.2 Expert:** 300 episodes per task type, nominal visuals.
  - Pass: pick ≥ 95 %, place ≥ 90 %, stack ≥ 85 %.
  - Fallback: up to 2 documented expert fixes per task.
  - If stack still fails: drop stack from the project and report it. If pick or place fail: stop.
  - Record each task's expert success as its ceiling.
- **0A.3 Chunk timing:** the expert observes only every N control steps (stale positions in between), with one camera-frame latency.
  - Candidates N = 5, 10, 20.
  - Pass: within 3 points of the unchunked expert on every task. Pick the largest N that passes.
  - [consultant] Emulate chunked execution honestly: every N steps the expert reads its whole observation (objects and its own arm), one camera frame late, then executes its next N actions open loop. IK still tracks at the 50 Hz control rate.
  - [consultant] The brief has no fallback if every candidate fails (112: a 10 Hz stale teacher already lost 4.5 points). Fallback: test N = 3 (112's proven 16.7 Hz rate). If that also fails, stop and report. This adds a smaller candidate; the 3-point bar is unchanged.
- **0A.4 Cameras:**
  - Image size candidates: 128, 224, 256.
  - Pass: every cube (incl. small ones) has ≥ 20 segmentation pixels in ≥ 90 % of side frames before the grasp; the wrist camera sees the target cube during approach and grasp; pads are visible in the side camera.
  - Pick the smallest passing size. Contact sheets per task.

### 0B — Pipeline (in parallel with 0A, built on the single-cube scene)
- Export demos to LeRobot format, with a round-trip test and decoded-episode video check.
- Sim ↔ policy-server bridge: untrained SmolVLA runs batched in the loop. Measure latency and throughput.
- **Overfit test:** SmolVLA fine-tuned on 10 demos from fixed initial states reproduces them (≥ 80 % success on those exact states). Catches action-unit, normalization and frame bugs before any real run.
- **Tests (must pass):**
  - banned inputs
  - frame/action alignment
  - instruction uniqueness (exactly one match)
  - held-out disjointness (compositions and paraphrases)
  - the visual randomization split
- Update the PERUN container with the policy venv. The first cluster job verifies the bridge on an H200.
- [consultant] **Measure training throughput** (GPU-seconds per sample on one H200, SmolVLA frozen and unfrozen, at the chosen frame rate and image size). Before 0C, write into GATES.md a per-stage GPU-h plan summing to ≤ 100 (π0.5 ≤ 30). Each stage's step count is set in epochs from that plan. Fit the demo counts to it, as §8 0C already allows.
- [consultant] The laptop (8 GB) is for sim, the expert, 0A, the bridge and inference smoke tests. LeRobot's guide puts SmolVLA fine-tuning at 10–16 GB at batch 8, so real fine-tunes (including the overfit test, unless it fits at a tiny batch) run on PERUN.

### 0C — Bulk demos (one PERUN run, after 0A and 0B pass)
- Guidance counts (adjust to throughput, document):
  - Stage 1: 500 single-cube picks
  - Stage 2: 2,000 picks among 2–5 cubes
  - Stage 3A: 2,000 places
  - Stage 3B: 2,000 stacks
  - Stage 4: a visually randomized copy of the mix (≥ 2,000)
- Only successful demos are kept. Report the expert success rate per batch.

### 1 — One cube
- "Pick up the cube", ~500 demos. SmolVLA frozen and unfrozen trained in parallel.
- Pass: pick ≥ 90 %, slam < 5 %, jam < 2 %. Choose the encoder setting by the §6 rule.
- Fallback: 1 diagnosed fix. Then stop.

### 2 — Which cube (colors + sizes, 2–5 cubes)
- Fine-tune from Stage 1.
- Pass (seen combinations): pick ≥ 85 %, wrong-cube rate < 5 %.
- Report held-out combinations (not gated).
- Fallback: split into 2a (colors only) then 2b (+ sizes), each gated the same way. Then stop.
- **π0.5 launches when Stage 2 passes:** fine-tune on all task data (nominal + randomized), 1 seed, sub-cap 30 GPU-h.

### 3A — Place (in parallel with 3B, from the Stage 2 checkpoint)
- Train on place + pick data (pick data kept to prevent forgetting).
- Pass: place ≥ 80 %, wrong-pad < 5 %, and pick success no more than 5 points below Stage 2.
- Fallback: 1 diagnosed fix. Then stop.

### 3B — Stack (in parallel with 3A, from the Stage 2 checkpoint)
- Train on stack + pick data.
- Pass: stack ≥ 70 %, wrong-cube < 5 %, pick no more than 5 points below Stage 2.
- Fallback: 1 diagnosed fix. If it still fails, the final drops stack and reports it as a limitation. Not a project stop.

### 4 — Final, all together
- Train SmolVLA on all task data (nominal + randomized visuals) from the Stage 2 checkpoint, 3 seeds in parallel.
- Train the baseline on the same data, 3 seeds.
- Bring in π0.5 (1 seed).
- **Evaluate every policy, 1,000 episodes per split:**
  - (a) seen instructions, nominal visuals
  - (b) seen instructions, randomized visuals
  - (c) held-out combinations
  - (d) held-out paraphrases (baseline: n/a)
- Break every split down by task type.
- **Final success criteria** (SmolVLA, 3-seed mean, split a): pick ≥ 85 %, place ≥ 75 %, stack ≥ 60 % (if stack is kept), wrong-object < 5 %, slam < 5 %, jam < 2 %.
- Splits (b)–(d) and the model comparison are reported honestly, not gated.
- If the criteria are not met within budget: ship anyway with honest numbers and a clear "why".

## 9. Evaluation
- **Metrics:** success per task, wrong-object rate, release failure (dropped off the pad/stack, or not released), stack stability, slam, jam, early close, drop, inference latency.
- Reuse the 111/112 detectors and evaluators. Phase-based metrics must work on student runs.
- Fixed test seeds, used once. Checkpoint picks on separate validation seeds. Rules written in GATES.md first.

## 10. Compute, parallelism, budget
- **Agents:** up to 10 at once (Claude Max). Stay inside its usage limits. Never buy extra usage.
- **PERUN:** parallel GPU jobs allowed; GPU-hours counted per job.
- **Budgets (hard stops):**
  - laptop ≤ 2 h per run and ≤ 20 h total
  - PERUN ≤ 100 H200 GPU-hours, including the π0.5 sub-cap of 30
- **Drop order if the budget gets tight:** π0.5 first, then stack.
- **Wall-time target:** 30–40 h.
- [consultant] **Reality check of the estimates.** Numbers come from LeRobot's hardware guide, measured on 4×H100 and scaled per sample. They are not yet measured on our data; 0B replaces them.
  - **SmolVLA:**
    - about 0.04 GPU-s per sample unfrozen and roughly a third of that frozen;
    - the documented recipe (20k steps × batch 64 = 1.28 M samples) ≈ 4–5 H200-h frozen, ≈ 12–14 unfrozen.
  - **π0.5:**
    - about 0.3 GPU-s per sample;
    - the 30 GPU-h sub-cap buys about 350k samples (~5k steps at batch 64), against the documented 30k steps. At that cap π0.5 will be under-trained relative to SmolVLA. Disclose it; the cap doesn't move.
    - Use π0.5's documented lighter variant (vision frozen, expert only) only if Stage 2 data show it fits the cap better, recorded in GATES.md first.
  - **Data size:** the demo counts are about 10k episodes. At 50 Hz with 5–15 s episodes that is several million frames, so one epoch over all data costs a large share of the budget. Hence the frame-rate decision (§6) and the throughput plan (0B).
  - **Verdict:** the 100 GPU-h budget is tight but workable with frozen-encoder runs and epoch-sized plans. If Stage 1 picks the unfrozen encoder, the per-stage plan must be redone, and the §10 drop order applies.
- [consultant] **PERUN jobs:** 112 used `gpu_short`, limit 2 h per job. Training runs longer than 2 h must checkpoint and resume across chained jobs, or use a longer partition if PERUN has one; check it in 0B. The π0.5 run is about 15 chained jobs serially on 1 GPU (wall time ~30 h) unless it uses multiple GPUs per job. Plan it so it doesn't push the project past 40 h.

## 11. Deliverables
- **Repo:** git, local only. README covering the story, results tables, GIFs, disclosures, and credit to 111/112.
- **Demo:** one command takes an instruction and a seed and writes a video: both camera views + close-up scene + the instruction overlaid.
- **Videos:**
  - success / near-miss / failure per task
  - held-out combination examples
  - wrong-object failures
  - SmolVLA vs π0.5 vs baseline on the same seeds
  - randomized visuals
- **GATES.md, REPORT.md,** and at the end `113_SUMMARY_FOR_MASTER.md` in the 112 format (TL;DR, results, stage table, fixes, lessons, disclosures, open items).
- **Tests:** everything listed in 0B, plus detector tests, the bridge smoke test, and eval and demo smoke tests.
- **Maintain:** context.md, status.txt, `<project>_plain.md`.
- **Disclosures:**
  - privileged scripted expert
  - arm through IK with the hand vertical
  - sim only
  - nominal physics only
  - the merged Stage 2 (if not split)
  - π0.5 has 1 seed
  - the baseline gets perfect labels
  - attempts per gate

## 12. Rules
- Full autonomy. No permission asks. Batch questions into REPORT.md.
- Never record video during training. Replay checkpoints afterwards.
- Seeds, TensorBoard, explicit device handling, type hints, fail fast. Commit at every gate.
- **Licenses:** Isaac Lab (BSD-3), Newton (Apache 2.0), LeRobot and SmolVLA (Apache-2.0), π0.5 weights (Gemma terms; never published). Don't copy NVIDIA assets into the repo.
- Never force-push, never make anything public (repo, HF dataset or model), never spend money.
