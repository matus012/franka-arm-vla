# 005 fix candidate: cross-instruction flow target (prepared, NOT registered, nothing launched)

Requested by consultant 113 (2026-10-25) while Stage 2R training job 131419 runs. This is CPU work only, with no
GATES entry. It may be chosen as the attempt's one diagnosed fix only after the verdict, and only if the evidence
points to it.

## Why
Stage 2R's hinge, at the language-ignoring fit (v' ≈ v ≈ u), has gradient 2 (v − u)(J' − J). That is first-order in
the fit residual, so the gap can stay at zero. The smoke showed exactly that: over 150 steps the gap was ≈ 0, the
hinge ≈ m, and L_true matched the plain run. Training job 131419 at step 100 shows the same: gap 0.0004, active
1.000. `test_iii_…` makes it exact: at v' = u_true the hinge's gradient with respect to v' is **0.0**.

## The term
SmolVLA's convention: x_t = t ε + (1 − t) a and u = ε − a. Take the same x_t, built from a_true, and the other
member's instruction ℓ'. If actions are deterministic given (o, ℓ'), the exact velocity is u' = (x_t − a_other) / t.

  L_cross = mean over rows of [ mean over the 50 × 5 entries of ( t · v(x_t, t | o, ℓ') − (x_t − a_other) )² ]

This is t²-weighted, i.e. t² ‖v' − u'‖², which keeps it stable at small t. At the symmetric point v' = u_true its
gradient with respect to v' is 2 t (a_other − a_true) / n. That is O(1) in the demos' difference, not in the
residual (test (iii)). It adds λ' · L_cross to the Stage 2R objective.

## The constraint that matters
**a_other is the right target for observation o only where the members' observations are identical: frame 0** (the
shared t = 0 state; data gate P1 / P2). After frame 0 the members' arms have moved differently, so the other member's
chunk at frame k is not the answer for this episode's frame-k observation. The code therefore applies the term to
frame-0 rows only (`frame_index == 0`).

With the registered sampler that gives a weak signal. WindowSampler draws one row per (episode, camera frame) window,
and there are 116 windows per episode. So at most 64 / 116 ≈ 0.55 rows per global batch are frame 0, and fewer if
frame 0's window holds several control rows. As wired, λ' acts on about one row every 2–5 steps.

A fix that relies on this term would need its own frame-0 rows in every batch: K extra frame-0 rows of train
episodes, added alongside the registered 64. That is a sampler / batch change, so it is a design decision for the
consultant's fix entry, not something to slip in. Two other possible extensions need evidence I do not have:
using early frames where the observations differ little, or an approximate a_other at matched progress.

## Code (uncommitted at the time of writing; committed with this note)
- `vla_policy/contrastive.py`:
  - `cross_terms(v_other, x_t, t, a_other, A, rows)`;
  - `flow_pair` now also returns v_true / v_other, captured by a forward hook on `action_out_proj` (the forward's
    v_t), with no extra pass;
  - `pick_other_eps` returns the member episode that ℓ' names, from the same rng draw as `pick_other`.
- `vla_policy/train_smolvla.py`:
  - `--cross-lambda λ'` (requires `--contrastive-lambda`);
  - `Contrast(cross_lambda, first_chunk)`: every table episode's first chunk, normalized as the preprocessor does;
  - in DDP the term is a mean over the global batch's frame-0 rows (the count is all-reduced);
  - `cross` / `cross_rows` are logged, and `loss_total` includes λ' · cross;
  - run_key carries λ', so a resume with another value is refused.
  - **With `--cross-lambda` absent the trainer's path is unchanged** (the hinge path still calls `pick_other`; the
    test_contrastive mutant still bites). The running job 131419 uses repo dd5b044, which has none of this.
- `tests/test_cross_target.py` (policy venv, CPU; 7 tests, plus test_contrastive / test_s2r_tools, 26 green):
  - (i) λ' = 0 (with λ = 0) is bit-equal to the plain trainer: one step (loss, gradients, RNG) and a whole run
    (losses, weights). λ' > 0 alone changes the gradients.
  - (ii) with x_t built from a_other, u' = u_other and the term is 0 at v' = u_other.
  - (iii) at v' = v = u_true the term's gradient equals 2 t (a_other − a_true) / n, nonzero, where the hinge's
    gradient is exactly 0.
  - Also tested: the flag guard, the frame-0 masking, and that `pick_other_eps` names a same-layout member.
- Not tested: a DDP-equivalence test for the cross term. The existing DDP test covers only the hinge.

## What the diagnosis should carry if early abort (b) fires (per the consultant)
The logged gap curve, the hinge-active fraction and L_true (steps.jsonl → `s2r_tools.py steps`, per-1,000-step
windows), the (b) ratios on loss_total and L_true, and, if c050 ran, its s_50 and the step 2 ratio. From those the
fix can be argued: no gap at all points to this term; a gap that opens and then stalls below m points to λ or m.
