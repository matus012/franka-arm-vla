# Expert close-timeout diagnostic (2026-10-01, not a gate)

**Finding (0B.3):** in 7 of the 10 overfit demos (Stage 1, seed 31600, eps 0-9) the expert lifts exactly 50 steps after
closing (CLOSE_MAX timeout), a clock the policy cannot see. Reproduced here: 7 / 10 (states 0, 2, 3, 4, 7, 8, 9).

**Root cause:** the contact half of the gate is fine (both target pads > 1 N on a median 92 % of CLOSE steps, PAD_STEPS
run reaches 45). The stillness half never holds: it requires EACH finger joint |v| < 5 mm/s for 5 steps, but with the
effort-limited (6.1 N) claw the cube + both fingers keep shuttling sideways between the finger limits (the drift seen in
0A.1). Each finger moves at ~3-5 cm/s (still on a median 4 % of CLOSE steps, longest run 1 step), while the gripper
WIDTH barely changes (width rate < 5 mm/s on 86 % of CLOSE steps). So the gate only fires when a grasp happens to start
without shuttle; otherwise the 50-step timeout lifts.

**Timeout fraction, committed expert ("settle"), validation seeds, 300 episodes each:** Stage 1 (21700) 57.0 % (small
60.1 / large 53.7); pick s2 (21200) 56.3 % (60.6 / 51.7); place s3a (21210) 58.3 % (67.7 / 48.3); stack s3b (21220)
57.0 % (64.5 / 50.6). Overfit states (31600, n = 10): 70 % (small 7 / 8, large 0 / 2).

**Candidate (flag `Expert(lift_trigger="width")`, OFF by default; committed behaviour and every passed gate unchanged):**
same gate, but the stillness test uses the gripper-width rate |v_left + v_right| < 5 mm/s (the policy sees both finger
velocities in its 19-dim state) instead of each finger's speed. Validation, same seeds / episodes, ON vs OFF:
timeout 0 % vs 56-58 % on every task; success Stage 1 100 / 100 %, pick 100 / 99.7 %, place 100 / 100 %, stack 98.7 /
96.7 % (drops 5 vs 10); overfit states 10 / 10 gate-triggered, 10 / 10 success. Close phase shortens from 50 to a median
10 steps.

**Not changed:** whether to switch the expert (and redo 0A.2 / 0A.3 with it) is the owner's decision. Data:
`results/expert_close_diag.json`, per-episode logs `results/expert_close_diag/*.json`, script
`scripts/diag_expert_close.py`. GPU 11.6 min.
