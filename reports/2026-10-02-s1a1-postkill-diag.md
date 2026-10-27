# Stage 1 attempt 1: post-kill diagnosis - not a gate attempt, no decision

Question (robotics chat): does closed-loop frequency limit grasp precision? Checkpoint: Stage 1 attempt 1 U, 100 % checkpoint
(`/runs/s1a1_unfrozen/checkpoints/020664`, sha c804bae2...). Validation seed 21600, episodes 0-299, B 128 fixed layout, T_pick 350,
paired on the same 300 episodes. Only N (server `--n-action-steps` and client query period; `info` printed n_action_steps 20/10/5 and
the client's frame-index check passed) and `--sampling` differ. No test seeds, no training, GATES.md untouched.
Jobs 96068 (4 variants) + 96071 (frames): 0.396 H200-h. Script `scripts/s1a1_pk_report.py`, data `results/s1a1_postkill_diag.json`.

| variant | success | slam | jam | drop | early close | large | small | vs N20 (paired) |
|---|---|---|---|---|---|---|---|---|
| N=20 default (baseline) | 56.7 % | 0 | 0.3 % | 37.3 % | 75.0 % | 39.2 % | 74.8 % | - |
| N=10 | 58.0 % | 0 | 0.7 % | 31.3 % | 69.7 % | 45.1 % | 71.4 % | +1.3 pts (29 gained / 25 lost, z 0.5) |
| N=5 | 58.7 % | 0 | 0.3 % | 19.7 % | 60.3 % | 43.8 % | 74.1 % | +2.0 pts (41 / 35, z 0.7) |
| N=20 zero-noise | 65.0 % | 0 | 0.3 % | 31.0 % | 67.0 % | 54.2 % | 76.2 % | +8.3 pts (54 / 29, z 2.7) |

(Baseline re-measure 56.7 vs 57.3 % in the pick run: batch-composition solver non-determinism, 2 episodes.) Large n=153, small n=147.
Drop and early close are overlapping flags, not exclusive classes. Wall: N20 218 s, N10 322 s, N5 535 s (N5 is 2.5x the cost).
Grasp-yaw error: the evaluator does not log it (no yaw key in the episode records), so it is not reported.

## Reading
- Query frequency is not the limiter. Quadrupling it (N 20 -> 5) moves success +2.0 points with 41 gained / 35 lost episodes
  (z 0.7, consistent with zero); N 10 is the same. The 0A.3 "upper bound" caveat does not change this, since shorter chunks
  are strictly closer to the expert's closed loop. Large cubes gain a few points (39 -> 44-45 %) but stay far below small; N 5 even
  loses on small cubes. Drop falls with N (37 -> 20 %) while no-flag failures rise (0 -> 8.7 %), i.e. failures change class, not count.
- Sampling noise matters more than rate: zero-noise N20 +8.3 points (z 2.7), mostly on large cubes (39 -> 54 %), reproducing the
  D1 fix-diagnosis finding (default 35.0 -> zero 48.7 %). Still 25 points under the bar.

## What I looked at (results/frames/s1a1_pk/, N20 zero-noise, round 0, all 9 replays reproduced the recorded outcome)
Close-up | side | wrist sheets at grasp confirmation and at t=340: ep 0 (success), 1, 3, 6, 8, 9, 13, 26 (failures, 7 of 8 large
cubes except ep 1) and 117 (failure with no flag). Seen in the wrist views: in every failure the cube is
grasped rotated about 30-45 degrees (diamond in the wrist image) and/or clamped on a corner or edge off the pad centre, and often
tilted; at t=340 the cube lies on the table (drop) or, in ep 8, is still pinched between the fingertips without having
been lifted off properly. The success (ep 0) shows a cube square to the fingers, centred. This matches the stated failure mode and
points at grasp alignment (yaw / lateral approach error of the policy before closing) rather than reaction time: the claw closes at
the point where the policy is already misaligned, at any query rate.

## Caveats
Validation, 300 episodes (CI about +-5.6 points); one checkpoint; a single N-sweep per sampling mode (N<20 with zero noise not run);
observations only, no causal test of the yaw explanation (no yaw log). Not a gate attempt, no decision, no test seeds consumed.
