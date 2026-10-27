# Stage 2 grounding options — memo for the owner (lead, 2026-10-03; no runs)

**Situation.** At 50 % of training the Stage 2 policy picks the named cube in 12 % of validation episodes. Its behaviour
does not depend on which cube the instruction names: the paired closed-loop run gives 14 / 100 with the true instruction
and 11 / 100 with a shuffled one (z 1.13). The instruction does reach the model. Four instrument checks are clean; see
GATES "Stage 2 diagnosis", 90bfa0f. Offline, the text matters only weakly: about 9 % of the approach-phase loss. Stage 2
runs to its written end anyway (c100 pick, test 11520 / 11530 once). This memo is for the case where it fails. The
entry's only fallback, 2a / 2b, does **not** start automatically.

**Likely cause (hypothesis, not yet tested).** Every SmolVLA weight Stage 2 starts from was trained for 41k steps on
single-cube scenes, with instructions that never named a colour or size ("pick up the cube" templates). The action
expert learned that the text carries no information. The Stage 1 V15 parent shows no swap effect at all on s2 data
(+0.0006, CI includes 0). Ten s2 epochs have only begun to undo that.

**Rates used** (measured at H 224 / N 10, PERUN H200): training 0.00674 GPU-s / sample; one Stage 2 run at E2 = 10
(2.2 M samples) = 4.2 GPU-h and ≈ 4.2 h wall. Checkpoint pick ≈ 1.0, tests + frames ≈ 0.35, s2-size data set at 224
≈ 0.3 (+ ≈ 1.5 h of CPU export / checks).

| rank | option | what changes | GPU-h | wall | same test protocol? | expected grounding gain |
|---|---|---|---|---|---|---|
| 1 | **(a) init from `smolvla_base`** | Stage 2 init = the base model (the U arm otherwise unchanged); same data, recipe and steps. | ≈ 5.6 | ≈ 6.5 h | yes — the same s2 set, validation 21620, test 11520 + k / 11530 + k | **high (prior).** It removes the learned "ignore the text" prior and starts from a model pretrained on language-conditioned tasks. Cheapest change, one variable. Risk: it loses Stage 1's grasp skill. Stage 2's s2 data contains the same grasps, and Stage 1 learned them from 2,000 single-cube demos. |
| 2 | **(d) more steps / data, current setup** | E2 10 → 20 (or 4,000 demos). | ≈ 10 (E2 20) | ≈ 10.5 h | yes | **depends on c100.** If c100 shows the swap gap growing (check 4 is repeated on c100), more training is the cheapest reliable path. If the gap is still about 0, low. |
| 3 | **(c) grounding curriculum** | Phase 1: 2-cube scenes, distinct colours (new sampler + 2,000 demos). Phase 2: the s2 set. | ≈ 0.4 data + 2 × 5.2 ≈ 10.8 | ≈ 12–13 h | phase 2 yes (11520 + k); phase 1 needs its own validation | moderate. It makes the text-to-target signal maximal early, but it still starts from the text-blind init unless combined with (a). |
| 4 | **(b) train the text layers too** | Both LeRobot flags false **without** the refreeze: + text model 205 M (+ connector 12 M) trainable, ≈ 400 M in total. | ≈ 7.5 (≈ 1.5× per sample, estimate) | ≈ 7.5 h | yes | low to moderate. The pretrained text encoder already distinguishes colours. The weak link is more likely the action expert's use of the text, which (a) addresses directly. It also changes Stage 1's encoder-arm definition (a design change), with a higher overfit risk. |
| 5 | **(e) the written 2a → 2b split** | 2a = colours only (2–4 cubes, one size), from the Stage 1 checkpoint; then 2b on s2. | ≈ 11.5 | ≈ 13 h | no: 2a has its own streams (11540, 21640); 2b uses 11550 / 11560 | **low.** It keeps the text-blind init, and colour is exactly what the policy is not grounding. The kill fires if 2a fails, which ends 3A / 3B / 4. |

**Recommendation.** (a), optionally combined with (d)'s step count if c100 shows the gap still growing. It is one
dated owner revision of the Stage 2 entry (like the Stage 1 redesigns), with the same data and the same test protocol at
attempt k = 1 (11521 / 11531). Keep (e) as written but do not fire it on this failure mode.

**Budget (PERUN 113, H200-h).**

| item | H200-h |
|---|---|
| used through Stage 2 (estimate at its RESULT) | ≈ 45.5 |
| left of 100 | ≈ 54.5 |
| one Stage 2 retry, option (a) | ≈ 5.6 |
| 3A ∥ 3B (re-plan line / cap; data already made) | 12.6 / 14.0 |
| left for Stage 4 after that | ≈ 36 |
| Stage 4 as written at 224 / N 10 (SmolVLA × 3 + baseline × 3 + s4r data) | ≈ 50 |

Stage 4 does not fit with stack. The §10 drop order (stack next, ≈ −22) brings it to ≈ 28, which fits. The owner's s4r
palette decision is still pending. π0.5 stays dropped.

**What 3A / 3B / Stage 4 still need after a Stage 2 pass:**
- 3A / 3B: the union norm table against the Stage 2 checkpoint, then training (4.35 GPU-h each) and the two-skill
  checkpoint pick + tests. Data, tooling and entries (86640ee) are ready.
- Stage 4: the owner's s4r palette decision, its own re-plan entry, and the stack drop (or more budget).
