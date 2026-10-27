# Demo videos (2026-10-02)

Videos are in `videos/raw/` (gitignored; each < 1 MB, with a `.json` of detector flags next to it); stills (start / mid / end frame, 2/3 scale)
are in `videos/stills/`. Layout: close-up scene | side | wrist (delayed 128 px frames the policy sees). Run from the repo root through
`ops/run_job.ps1` (PowerShell, `& .\ops\run_job.ps1 -Job <j> -LogFile logs\jobs\<j>.log -PyArgs ('<args>')`).

Policy: Stage 1 attempt-1 picked checkpoint (`results/s1a1.json`, unfrozen arm, 11501), `checkpoints/s1a1_U/pretrained_model`
(copied from PERUN `runs/s1a1_unfrozen/checkpoints/020664/pretrained_model`, model.safetensors sha256 `c804bae2...f88d8` = the recorded one;
gitignored). It scores 57.7 % pick on the test set, so these clips show a policy that failed its gate (kill fired, GATES 123b253).
`scripts/demo.py` got a `--stage s1` flag (single-cube scene, "pick up the cube", episode 0 of the seed's stream as in eval_policy) for this.

## Stage 1 policy, "pick up the cube" (SmolVLA, N=20, nominal visuals, validation seeds, fp32 on the laptop)

Seeds tried in order, all recorded: 21600 success, 21601 success, 21602 success, 21603 grasped-then-dropped, 21604 success,
21605 grasped-then-dropped; 21606 was started and killed by me (laptop budget), no result. First of each type is kept.
No seed with "never grasped" was seen in 6 tries, so the "failure" clip is a second grasp-then-drop case (see caveat).

Command (seed S): `scripts\demo.py --stage s1 --instruction "pick up the cube" --seed S --policy smolvla --ckpt checkpoints\s1a1_U\pretrained_model --out videos\raw\s1_pick_S.mp4`

| video (`videos/raw/`) | seed | outcome | what is seen (stills `videos/stills/<stem>_{start,mid,end}.png`) |
|---|---|---|---|
| s1_pick_21600.mp4 | 21600 | SUCCESS (completion step 160) | Small green cube; gripper descends, closes on it at ~3.5 s and holds it lifted off the table to the end (step 349). |
| s1_pick_21603.mp4 | 21603 | NEAR-MISS: grasp confirmed, then `drop`, `early_close`, not success | Small red cube next to the blue pad; gripper reaches it and closes (grasp confirmed at ~2.4 s) but the cube slips out and lies on the table under/beside the gripper from ~3.6 s; the gripper hovers there until the end without re-grasping. |
| s1_pick_21605.mp4 | 21605 | FAILURE: grasp confirmed, then `drop`, `early_close`, not success | Small green cube away from the pads; gripper comes down on it (~2.4 s), the cube ends up on the table, the gripper rises and stays open above the empty spot, cube lies untouched to the end. |

Caveat: 21603 and 21605 have identical detector flags (grasp_confirmed, drop, early_close); I told them apart only by eye (21603: cube
ends next to the gripper; 21605: gripper leaves the cube). The "lifted < 10 cm" criterion was not measured. The overlay text
"grasp confirmed" is sticky (it stays after the cube is dropped).

## Expert references (scripted expert 0A.2, chunked N=10; the policy stages that would learn these did not run)

Command: `scripts\demo.py --instruction "<instr>" --seed S --policy expert --out videos\raw\<stem>.mp4`

| video | seed | instruction | outcome | what is seen |
|---|---|---|---|---|
| expert_pick_21700.mp4 | 21700 | "pick up the red cube" | SUCCESS | Five cubes (red, yellow, green x2, ...) and four pads; the hand picks the red cube out of the distractors and lifts it (phase LIFT) at the end. |
| expert_place_21701.mp4 | 21701 | "put the blue cube on the yellow pad" | SUCCESS | Blue cube carried across (LOWER at 5.5 s) and set down on the yellow pad; DONE at the end, hand retreated, blue cube on the yellow pad. |
| expert_stack_21702.mp4 | 21702 | "stack the blue cube on the small green cube" | SUCCESS | Blue cube lifted, lowered onto the small green cube beside the yellow/red items; at the end the blue cube sits on the green one and the hand has released. |

Laptop GPU this task (incl. 2 RAM-aborted attempts, 1 killed run): ~28 min, laptop total 7.09 -> 7.55 h of 20 h (`ops/gpu_log.csv`).
Note: sim + fp32 SmolVLA server needs more free RAM than the 1 GiB `run_job` guard allows when ~4.5 GB is free; I closed the browser to run.
