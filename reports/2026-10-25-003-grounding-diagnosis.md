# 2026-10-25 — 003 grounding diagnosis

project: 113_vla_tabletop
session: 2026-10-24 23:55 – 2026-10-25 ~00:55 UTC (L2 orchestrator, headless)
task: `_todo/113_vla_tabletop/003-grounding-diagnosis.md` (diagnosis, not an attempt; reported only)

## Result

**Verdict R3 (data / loss) for both checkpoints:** the 2a attempt 1 policy (034440) and the S1 V15 init (041328).
- The colour word is perfectly linearly decodable from the language tokens and from the K/V the action expert reads,
  at every probed layer (CV accuracy 1.000 against a 0.36–0.43 shuffled-instruction control).
- After 2a training it also reaches the image tokens: layer 16 gives 0.92, and 0.93 on identical-scene pairs.
- Yet the first action chunk ignores it. The instruction effect is 0.103 × (034440) and 0.084 × (041328) the
  start-noise effect, against a bar of ≤ 1.5 × for "ignores".
- Cost: 0.120 H200-h of the 1.3 line; 113 total is 73.88 / 130.

| checkpoint | lang (L4–16) true / shuf | kv expert-read (L4–16) | img L4 / L8 / L12 / L16 (shuf ~0.4) | kv_img L16 | step 2 instr / noise → ratio | verdict |
|---|---|---|---|---|---|---|
| 2a a1 034440 | 1.000 / 0.38–0.41 → decodable | 1.000 / 0.42–0.43 → decodable | 0.48 / 0.68 / 0.85 / **0.92** → decodable | 0.75 (partial) | 2.76 / 26.9 → **0.103** (ignores) | **R3** |
| S1 V15 041328 | 1.000 / 0.36–0.38 → decodable | 1.000 / 0.39–0.40 → decodable | 0.55 / 0.66 / 0.84 / 0.89 → partial | 0.66 (partial) | 1.59 / 18.9 → **0.084** (ignores) | **R3** |

Step 0: the longest prompt is 16 tokens, against `tokenizer_max_length` 48 (1,686 prompts, none truncated). The t2a1_VT lr was 1e-4. Neither is an instrument finding.

## Why

The instruction does reach the expert's inputs, so R1 (text path) and R2 (fusion) are ruled out for the features
the expert reads. Even so, with the scene held fixed and only the words changed, the expert outputs nearly the same
chunk. Which way the arm goes is decided by the flow-matching start noise. For 034440, a second noise draw swings the
direction of the arm's planned xy motion by more than 45° in 31 % of layouts; a different instruction does so in 1 %.
That matches the random-cube signature of the 2a failure.

The training data does carry the signal: on paired training layouts, the demonstrations differ by target 16× more than
the policy's chunks do (43.8 vs 2.76). So the training objective never forced the policy to use the words it can see.
That is the R3 reading, and it points at a change in the training objective or data, not in the text encoder.

## Next

- Owner decision (as before): close 113, or a redesign brief. If you go for a redesign, R3 points at the loss / data
  family from the C0 memo (`evidence/2026-10-24/c0-language-grounding.md` steps 2–3):
  - language dropout + guidance at inference;
  - or an instruction-contrastive / target-aware loss.

  R3 argues against "unfreeze more of the VLM" alone, because the information is already in the expert's inputs.
- Caveat to weigh (lead review F2, written into GATES before the probe was read): with "decodable at ≥ 1 of 4
  layers", R1 and R2 were hard to reach by construction. The lang features include the colour token's own residual
  stream, and kv is a linear map that includes the language positions. **The R3 call rests on step 2.** Step 2's
  ceiling shows the metric can register a language effect.
- Pushback on the task: the GATES R2 rule uses kv only, while the task says "image tokens / the KV". Using img
  instead would not change the verdict: 034440 img is decodable, and V15 img is partial but its kv is decodable.

## Needs you

- Decision: close 113, or a redesign brief (R3 → loss / data family). PERUN left: 56.12 H200-h.
- Infra (info, blockers row): this background session's harness blocks Write/Edit in the main checkout until a
  worktree is entered, but `orch` runs without a worktree. Files were written via Bash / scratch + `cp`. Fix: set
  `"worktree": {"bgIsolation": "none"}` for L2 sessions, or change the orch skill.

## Intent

Task 003: diagnose where grounding is lost (readings R1 text / R2 mixing / R3 data-loss), using a linear probe on
the frozen policy's features plus an action-divergence test. Pre-registered in GATES (18970e1), with no training, no
fix, no test seed and no gate change.

## What happened

- **Pre-registration (18970e1).** GATES entry plus the context.md deviation-log line, committed before any job:
  readings verbatim, sample plan, feature definitions, probe, step 2, verdict rule.
- **Step 0, laptop CPU (`scripts/g3_prompt_len.py`).** Run with each checkpoint's own saved preprocessor; outputs
  `results/g3_diag/prompt_len_*.json`.
- **Code:**
  - `scripts/g3_dump_t0.py` (sim; the first query's t=0 inputs, as `ChunkExecutor`);
  - `scripts/g3_probe.py` (extract / divonly / probe; logistic regression in torch, because sklearn is not in the
    venv);
  - `tests/test_g3_diag.py` (15 tests, green in both venvs).
- **Laptop smokes, not counted:**
  - the extractor on `checkpoints/s1a1_U` with synthetic dumps;
  - the dump on 4 envs, with frames looked at.
- **PERUN 130693 `g3_diag` (0.094 H200-h).** Dumps: S_p 1,001 episodes / 405 groups, S_u 999. Features and step 2
  for both checkpoints. Shas match the records (07e87cc6…, c3dcef76…).
- **Lead review (fresh context, doctrine lens): ACCEPT, with F1 blocking for the probe.** F1: the verdict rule was
  applied first-match, but GATES says "any partial → mixed". Fixed and clarified in GATES (a249552) before any probe
  number was read; the verdict is unchanged. F2 (R1/R2 hard to reach) is disclosed. F3 (no ceiling) led to the data
  ceiling. F4 (no hook proof) led to the hook check, exact equality. F5 added an assert.
- **Supplementary step 2 view.** Pre-stated in 0179115, reported only. 130742 (0.026 H200-h) re-ran step 2
  bit-identically and saved the chunks (`results/g3_sup/view_*.json`). Per-dim ratios are 0.06–0.29, xyz-only
  0.10 / 0.25, raw 0.11 / 0.13: yaw scaling is not what drives the reading.
- **Ceiling** (pre-stated bcb2ca0). 130743 (CPU): the data's instruction effect is 43.8, in the same metric.
- **Probe** (laptop CPU): `results/g3_diag/probe_{034440,041328}.json`.
- **Errata in GATES:** clock times were estimates, so the commit times are the record (a6fc4cd); one launch line
  carried local time (f1e405c).

## Evidence

Raw outputs: `_todo/113_vla_tabletop/evidence/2026-10-25/` (job logs 130693 / 130742 / 130743, probe logs, hook check,
pytest in both venvs, laptop smoke logs, dumped frame sheet).

```
$ grep G3_ results/g3_diag/g3_diag_130693.out          (abridged)
G3_DUMP={"stream": "paired", "seed": 21645, "n": 1001, "groups": 405, ...}
G3_DUMP={"stream": "unpaired", "seed": 21640, "n": 999, "groups": 999, ...}
G3_EXTRACT={".../034440/...", "div": {"full_chunk": {"instruction_effect": 2.764, "noise_effect": 26.891, "ratio": 0.1028, "ignores_language": true}, ...}}
G3_EXTRACT={".../041328/...", "div": {"full_chunk": {"instruction_effect": 1.593, "noise_effect": 18.939, "ratio": 0.0841, "ignores_language": true}, ...}}
=== g3_diag on <node> NVIDIA H200 start 2026-10-25T02:22:18+02:00 repo 3d1f570 ... done 02:27:58+02:00
$ python scripts/g3_probe.py probe ... (034440)
lang L4 true 1.000 shuffled 0.383 -> decodable ... kv L16 true 1.000 shuffled 0.423 -> decodable
img L16 true 0.921 shuffled 0.427 -> decodable ; kv_img L16 0.754 / 0.421 -> partial
G3_PROBE family lang/img/kv decodable, kv_img partial -> verdict R3 data/loss
$ python scripts/g3_hook_check.py (laptop, s1a1_U)
G3_HOOKS={"calls_per_prefill": {"4": 1, "8": 1, "12": 1, "16": 1}, "final_norm_of_L16_capture_maxabs": 0.0, "kv_L*_K_half_vs_kproj_hook_maxabs": 0.0, "pass": true}
$ G3_CEIL data_instruction_effect full_chunk 43.84, first_10 23.27
$ pytest tests/test_g3_diag.py   -> 15 passed (policy venv), 15 passed (sim venv)
```

Full suite not re-run: no existing module was changed. Only new scripts and a new test file were added.

skills: none (see below)
