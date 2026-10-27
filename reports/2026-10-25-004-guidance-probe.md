# 2026-10-25 — 004 inference-time instruction-guidance probe

project: 113_vla_tabletop
session: 2026-10-25 00:48 – ~02:25 UTC (L2 orchestrator, headless)
task: `_todo/113_vla_tabletop/004-guidance-probe.md` (diagnosis on validation seeds only, not an attempt; reported only)

## Result

**Reading: "no effect".** At inference time, steering checkpoint 034440 toward the named cube does not make it pick
the named cube.

The steering ran at every denoising step: v = v_other + w (v_true − v_other), where v_other is the mean over the
instructions naming each other cube on the table. Measured on validation 21640, episodes 0–299 (the 2a validation
protocol):

| w | pick | wrong-cube | 2 / 3 / 4 cubes | slam / jam | step 2 instr / noise → ratio |
|---|---|---|---|---|---|
| 1 (plain, via the guided path) | **28.7 %** (86) | 38.3 % | 44.8 / 30.0 / 12.5 % | 0 / 0 | 2.76 / 26.9 → 0.103 |
| 2 (best) | 29.7 % (89) | 34.7 % | 44.8 / 33.0 / 12.5 % | 0 / 0 | 4.31 / 26.9 → 0.160 |
| 4 | 29.0 % (87) | 34.3 % | 46.9 / 30.0 / 11.5 % | 0 / 0.3 % | 7.61 / 26.8 → 0.284 |
| 8 | 26.7 % (80) | 33.7 % | 43.8 / 28.0 / 9.6 % | 0 / 0 | 13.26 / 25.9 → 0.512 |

- **Instrument check passes.** w = 1 reproduces the s_100 validation number: 28.7 % vs 28.0 %, within ±3.
- **Step 2 also reproduces.** At w = 1 it gives 003's ratio of 0.103.
- **No w reaches +10 points.** The best gain is +1.0, at w = 2. So the pre-registered reading is "no effect".
- **Cost: 1.081 H200-h.** This is 0.081 over the 1.0 line and inside the 1.5 cap. The 113 total is now 74.96 / 130.

## Why

Guidance does what it is supposed to do to the chunk:
- The instruction's pull on the first chunk grows roughly linearly with w: 2.8 → 13.3, a 4.8× increase at w = 8.

It does not change behaviour:
- Even at w = 8, a different start noise still moves the chunk twice as much as a different instruction (ratio 0.51
  against 1.5 for "uses language").
- So the arm's choice of cube is still made by the noise and the scene.
- Wrong-cube falls only 4–5 points, success by cube count keeps the chance × skill pattern, and at w = 8 success
  starts to fall (26.7 %): pushing harder distorts the motion before it fixes the choice.
- The frames look like unguided 2a: clean grasps of the wrong colour, the claw closing between two cubes, and a drop.

In plain terms, the colour signal the expert reads (003) is a small, mostly uninformative direction in its velocity
field, and amplifying it 8× at inference cannot stand in for training that makes the instruction matter.

Caveat: there is no empty-prompt branch, because the model was never trained with dropped language. Real
classifier-free guidance needs that training. This probe tests only whether the signal in the trained checkpoint can
be amplified.

## Next

- The owner decision stands: close 113, or write a redesign brief.
- What 004 adds against 003's options:
  - "Guidance alone at inference" is ruled out for 034440.
  - A guidance redesign needs language dropout during training, so that a real unconditional branch exists.
  - Alternatively, an instruction-contrastive or target-aware loss would make the chunk depend on the words (C0
    memo step 3).
- Both options need a new training run (~7 H200-h for a 2a-size run; 55.04 H200-h left).
- Pushback on the task, for the record: the "recovers" bar of pick ≥ 60 % was close to this checkpoint's ceiling.
  Success + wrong-cube at w = 1 is 67 %, so "recovers" needed ~90 % of wrong grasps converted. Lead F1 stated this in
  GATES before the run; the reading did not depend on it.

## Needs you

- Decision: close 113, or a redesign brief (language dropout + guidance, or a contrastive / target-aware loss).
- Infra, info only (blockers row): per the task, files were written via Bash and Python (the Write/Edit guard was not
  re-tested). Bash heredocs with long markdown content also fail in this harness ("unexpected EOF"); scratch +
  `cp` works.

## Intent

Task 004 was approved by the consultant under the owner's autonomy window. It measured whether the instruction signal
already in checkpoint 034440 can be amplified at inference into correct cube choice, on validation seeds only. It was
pre-registered in GATES (8c9ad87) before any job, with the readings written in verbatim. There was no training, no
test seed, no gate and no bar change.

## What happened

1. **Pre-registration (8c9ad87):** the GATES entry and the context.md deviation-log line.
2. **Code (42b02b8):**
   - `vla_common/guidance.py`: the other instructions, built by substituting the colour word, with the template
     unchanged.
   - `vla_policy/guidance.py`: expands the batch to the true row plus K other rows per env, and patches LeRobot's
     `denoise_step` with v_true + (w−1)(v_true − v_other). All rows of an env share x_t.
   - `vla_policy/server.py`: adds `--guidance-w` and the op `guide`. Reset forgets an env's others, and act refuses
     any env that has none.
   - Also changed: `vla_tabletop/bridge.py`, and the `scripts/eval_policy.py --guidance-w` and
     `scripts/g3_probe.py divonly --guidance-w` flags.
   - Readings: `scripts/g4_report.py`.
   - Tests: `tests/test_guidance.py` (written by an impl subagent) and `tests/test_g4_report.py`.
   - `tests/test_banned_inputs.py` (key whitelist) was narrowed explicitly: `msg["others"]` may appear only in
     `guide()`, and `guide()` must refuse when the server runs without guidance. This is the privileged-information
     channel; it is disclosed in GATES.
3. **Laptop proof (`scripts/g4_check.py`, s1a1_U, H 128):**
   - w = 1 equals plain on the same expanded batch exactly (0.0).
   - Against plain on the B rows alone the difference is 2e-4.
   - With the true instruction as its own "other" at w = 8, the difference is 2e-5.
   - w = 2 / 8 move the chunk.
4. **Laptop closed-loop smoke** (8 episodes, B 8, w 2): the op guide round trip works and the tokens are as sent.
5. **Lead review (fresh context): ACCEPT with findings, none blocking.**
   - Applied before the job (5f9c5dd): F1 a ceiling line, F4 a config-equality assert, F5 per-episode others, F7 a
     reference-integrator test.
   - F2, F3, F6 and F8 were noted in GATES.
6. **PERUN smoke 130745 (0.103 H200-h, not counted):** 5.94 s per query. The projection was 0.96 H200-h for the job
   and 1.14 for all of 004. That went over my own 1.1 guard if read as covering the whole task. Re-plan (GATES launch
   line, 32af7b2): `--time 01:10:00`, so 004 ≤ 1.35 ≤ cap 1.5 by construction.
7. **Job 130747 (0.890 H200-h):** w = 1, 2, 4 and 8, each followed by guided step 2. `results/s2a1_guidance_v21640.json`.
8. **Frames 130797 (0.088 H200-h):** a replay of w = 2's round 0. I looked at episodes 0, 3, 6, 28 and 57
   (`results/frames/s2a1_guid_w2_s21640/`). Episode 2 differs on replay, at knife-edge as recorded since 0A.2.
9. **GATES RESULT, docs, ledger rows.** The DST change at 03:00 CEST broke `ops/run_job.ps1` durations: the smoke row
   was logged at −53.59 min. Durations are now measured in UTC, and the row is corrected to 6.40.

## Evidence

Raw outputs are in `_todo/113_vla_tabletop/evidence/2026-10-25/004/`: job logs 130745 and 130747, the laptop proof and
smoke logs, the g4_report output, and pytest in both venvs.

```
$ grep -E 'EVAL_SUMMARY|G3_EXTRACT' results/s2a1_guidance/g4_probe_130747.out   (abridged)
EVAL_SUMMARY={"n": 300, "success": {"pick": 0.2867} ...}   # w 1
G3_EXTRACT={"guidance_w": 1.0, "div": {"full_chunk": {"instruction_effect": 2.764, "noise_effect": 26.89, "ratio": 0.1028 ...
EVAL_SUMMARY={"n": 300, "success": {"pick": 0.2967} ...}   # w 2
EVAL_SUMMARY={"n": 300, "success": {"pick": 0.29} ...}     # w 4
EVAL_SUMMARY={"n": 300, "success": {"pick": 0.2667} ...}   # w 8
G3_EXTRACT={"guidance_w": 8.0, "div": {"full_chunk": {"instruction_effect": 13.263, "noise_effect": 25.885, "ratio": 0.5124 ...
=== g4_probe on <node> NVIDIA H200 start 2026-10-25T02:09:15+01:00 repo 32af7b2 ... done 03:02:40+01:00
$ python scripts/g4_report.py ...
G4_REPORT={"sanity_w1": {"pick": 0.2867, "ref": 0.28, "pass": true}, "best_w": 2.0, "reading": "no effect", ...}
$ G4_CHECK (laptop, s1a1_U): a_w1_vs_plain_same_batch_maxabs 0.0, b 2.0e-4, c 2.0e-5, pass true
$ pytest (guidance, g4_report, banned_inputs, protocol, server_noise, eval_policy, g3_diag): 229 passed (policy venv), 229 passed (sim venv)
```

The full suite was not re-run. The changed modules (server, bridge, eval_policy, g3_probe) are covered by the suites
listed above.

skills: none
