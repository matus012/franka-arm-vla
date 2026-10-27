# 113 figure set (final results)

`fig1.png` what it does, `fig2.png` how it was built and tested, `fig3.png` proof (a grounding, b stage bars, c transfer),
`cards.png` six concept cards. 2x PNGs of 1600 px wide pages. Every image is a real 113 frame or plot; items drawn as
schematics are labelled "illustrative" / "schematic" in the figure (flow-matching noise curves, the 50-cell chunk strip).

## Regenerate (from `figures/src/`, repo root = `..\..`)
```
..\..\.venv\Scripts\python assets.py        # frames + matplotlib charts -> assets/   (numpy, matplotlib, pillow, imageio)
..\..\.venv\Scripts\python build_html.py    # -> fig1.html fig2.html fig3.html cards.html
node shoot.js fig1 fig2 fig3 cards          # Playwright chromium, 2x -> ../fig1.png ...
```
One-time: `npm i playwright && npx playwright install chromium` inside `figures/src` (local `node_modules`, git-ignored).
Font: Inter if installed, else Segoe UI (Inter is not installed on lpc; the PNGs use Segoe UI / matplotlib falls back likewise).

## Where each number comes from
| number | source |
|---|---|
| Stage bars in 3b: one cube 96.8 [95.5, 97.7] bar 90; plain colours 31.5 [28.7, 34.4] wrong cube 33.8; 2R 67.7 [64.7, 70.5] wrong 3.5 and fix 77.6 [74.9, 80.1] wrong 3.9; 2M 96.7 [95.4, 97.6] wrong 0.1; sizes round 2 79.4 [76.8, 81.8] wrong 2.6; place 72.7 [69.9, 75.4] wrong pad 0.2; stack validation 40.7 [35.3, 46.3] | `README.md` Results tables, checked against `GATES.md` RESULT lines (2a a1 l.4315; 2R l.5057, 5178; stack l.5863). Bars: S1 90, colours/sizes 85, place 80, stack 70 (`README.md` tables). Typed into `STAGES` in `assets.py` (not read from a json: only the 010 evaluation has one). |
| 3c and the 96.0 % / [94.6, 97.0] in Fig 1; 64.3 % wrong cube | `results/final_eval.json` (read by `assets.py`; `reports/2026-10-27-final.md` table agrees) |
| Baseline note (0 / 100, not usable) | `README.md` Disclosures / Stage 4 row; `GATES.md` baseline overfit check; shown as text only, no bar |
| 3a plain row: wrong cube 33.8 %, shuffled 21 vs 19 of 100 | `README.md` "The short version" (2a attempt 1) |
| Probe: colour CV 1.000 vs shuffled 0.36-0.43; instruction / noise 0.103 | `reports/2026-10-25-003-grounding-diagnosis.md` (2a attempt 1 checkpoint row) |
| Hinge: lambda = 1, m = 4.21 | `GATES.md` Stage 2R entry (l.~4840), `reports/2026-10-25-005-stage2r.md`; loss curves in card 4 / Fig 2 T4 = `results/s2m/steps_noindex.jsonl` (final colour model, 68,880 steps; other-instruction loss ~25 vs margin 4.21, hinge active on 0.035 % of the last 10k steps, printed by `assets.py`) |
| Expert ceilings 100 / 100 / 97 %, 6,000 paired demos, 224 px, N = 10 (50 predicted, 10 run, 50 Hz, camera 16.7 Hz, 1 frame latency), pick = held >= 10 cm | `README.md`; `vla_common/timing.py` (PERIOD 3, LATENCY 1, N 10) |
| Policy inputs: side + wrist RGB, 19-dim state, text | `README.md` "What the robot sees and does"; `vla_tabletop/scene.py` (cameras side + wrist) |

## Image sources
- Fig 1 frames, Fig 3a, cards 3: `videos/final_010/02_2m_grounding_layout14_{red,blue,yellow}_ep3{1,2,3}.mp4` (close-up | side | wrist triptych, split at x 0:320 / 324:564 / 568:808). Fig 1 uses the blue run at frames 0, 38, 76, 116 (= 0, 2.3, 4.6, 7.0 s at 16.7 fps).
- Fig 2 expert thumbnails: `videos/raw/expert_{pick,stack}_*.mp4`; dataset thumbnails: `results/frames/0c_s2a_pair_224_y15/0c_s2a_pair_rl224_y15.png` (decoded from the LeRobot dataset); T5: last frame of the blue run.

## Update after coordinator review
- Fig 1 step 4 states the evaluator's pick rule (end of the 7 s episode, named cube held by both fingers, lifted >= 10 cm above its rest height; `vla_tabletop/detectors.py` pick_dz = 0.100, T 350 steps at 50 Hz per the coordinator, not re-read by me).
- Fig 2 T2 shows the expert's ceiling after its one fix (pick 100, place 100, stack 97.3 %: `README.md`, `status.txt` 2026-10-02 00:56, the expert that generated the data) and names the first version (99.8 / 100 / 98.0, `status.txt` 2026-10-01 17:24).
- Fig 3a is now rendered from `results/demo/grounding/traj.json` when it exists (`panel_a()` in `build_html.py`; schema as sent by the coordinator; heading text comes from `scene_note`; viewer image labelled as not the policy input; ✓/✗ + reason per run, paths drawn in the target colour). Tested only on a fake file kept outside the repo (`TRAJ=<file>` env override). Without the file, the colour-model frames from final_010 and a marked "Pending" box for the plain model are shown.
- 2026-10-27 late: real `traj.json` landed; Fig 3a is now built from it (Pending box gone), caption states one scene illustrates the mechanism, rates are the 1,000-episode tests (plain 31.5 %, final 96.7 %).
