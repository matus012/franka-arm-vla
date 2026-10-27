# Stage 1 one-fix decision criteria (pre-registered 2026-10-02 ~06:00, before any diagnostic number)

Context: Stage 1 attempt 0 FAIL (GATES 6e460ae): U chosen, test pick 35.9 % (bar 90). Diagnosis: sigma-0.1 demo noise
moves the hidden commanded target; the next clean label cancels it (slope -0.96..-0.99) -> ~0.1 white label jitter;
loss plateau 0.46. Consultant guidance (2026-10-02): sigma = 0 re-collection is a design change (owner); fixes within
standing approval are chosen on VALIDATION evidence only (seed 21600 / s1 VAL episodes; no test seeds).

- **D1 inference-only:** existing chosen U checkpoint, validation seed 21600, 300 episodes, paired with default
  sampling: (i) zero flow-matching initial noise, (ii) mean of K = 8 sampled chunks per query (claw thresholded after
  averaging). Fix (a) qualifies if a variant reaches >= 80 % success with slam < 5 % and jam < 2 %; both qualify ->
  higher; tie -> zero noise.
- **D2 relabel (only if D1 does not qualify):** label redefined relative to observable state, keeping the sigma-0.1
  executed states. Qualifies only if ALL hold: |slope| on the previous executed noise < 0.1 and white step-to-step std at
  the sigma-0 level (0.014-0.045); open-loop executability >= 95 % expert success on >= 100 validation episodes; a short
  validation-only U fit (<= 0.5 GPU-h) on relabelled s1 TRAIN gets clearly below the 0.46 plateau on s1 VAL. If chosen,
  every 0C set is relabelled and re-gated (0C items 1-5) before Stage 2.
- **Neither qualifies:** Stage 1 is parked for the owner (blockers.md + REPORT.md); no attempt is spent overnight.
- **Attempt 1** (whichever fix): one change, test seed 11501, U arm, everything else as attempt 0; entry committed in
  GATES.md before the run. The U re-plan entry (GPU-h plan clause) is committed before any Stage 2 run.
