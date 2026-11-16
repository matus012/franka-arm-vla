# 2026-10-28 — 012: demo playable on GitHub Pages + 113 storage freed on PERUN and lpc

project: 113_vla_tabletop
session: l2 113 (orch 012), 2026-10-28 14:24 – 15:50 (CET)
task: _todo/113_vla_tabletop/012-pages-cleanup.md (done)
approval used (owner's exact words, 2026-10-28, in the task file): "A + full cleanup, keep the 6 final models on the laptop."

## Result

**The demo now plays in the browser, the 6 final models are on the laptop (verified), and 113's storage is freed:
916 GB on PERUN and 13 GB on lpc.** All 4 gates PASS. No force-push, no visibility change, no branch deleted, no other
project touched.

- **Pages:** https://matus012.github.io/franka-arm-vla/ shows a video player for the full 4:52 demo (original file,
  12 MB, already fast-start), the 4 figures, the results table and a link back to the repo.
  - Served from a new `gh-pages` branch (7708348) that holds only `index.html`, `113_demo.mp4`, the 4 PNGs and `.nojekyll`.
  - GitHub enabled Pages on that branch by itself when the branch was pushed, so the API call returned "already enabled".
- **README:** "▶ Watch the full demo (4:52)" sits under the hero GIF. On public `main` it is a new commit 2a261f6 on
  top of 573c596: a plain fast-forward push, and the rescan was clean.
- **Models:** `models_final/<name>/` holds weights + config only (6 files each, 9.1 GB in total), with a
  `models_final/README.md` that says which model is which, its result, and how to load and evaluate it. The folder is git-ignored.
- **PERUN:** the whole `/mnt/project/<project-id>/113_vla_tabletop` folder is gone. The project dir went from 1,121 GB to 205 GB.
- **lpc:** the following are deleted: `outputs/`, `checkpoints/`, both venvs (`.venv/`, `policy/.venv/`), `.pytest_cache/`, and 587 untracked
  frame / dump files under `results/`. The project folder went from 23 GB to 9.9 GB, of which 9.1 GB is `models_final/`.

### The 6 final models

| name | run dir / step | result (judged once) | result JSON | ckpt_sha = sha256(model.safetensors) |
|---|---|---|---|---|
| s1_single_cube | runs/s1y15_U/checkpoints/041328 | pick 96.8 %, PASS | results/s1y_U_test_s11504.json | c3dcef76f53e00e5954469970c25c38980a396723f7bbbf2b85919e6cc6688f5 |
| s2a_plain | runs/s2a1_VT/checkpoints/034440 | pick 31.5 %, FAIL | results/s2a1_test_s11541.json | 07e87cc6f5b2533ab4b931e356e7cf944e6b53d49bbf2cc0109ab416290cb3ff |
| s2m_colour | runs/s2m_VT/checkpoints/068880 | pick 96.7 %, PASS | results/s2m_test_s11544.json | e4741cf66cbbb57686df5a2ae7948e46a3a35ac65bc8c79bb48a598a32ca0f64 |
| s2b_sizes | runs/s2bp2_VT/checkpoints/068880 | pick 79.4 %, FAIL | results/s2bp2_test_s11552.json | ee011bf902f9a1d956ee4d6a47dcc2e3f2cfee681a9a79ea53b458a18c55cc12 |
| s3a_place | runs/s3a1p_VT/checkpoints/035508 | place 72.7 %, FAIL | results/s3a1p_test_place_s11580.json | f6adef8890138871f025994b9aeaa5714e67440c9601a272e001a998bc562f84 |
| s3b_stack | runs/s3b1p_VT/checkpoints/026631 | stack 40.7 % (val, stopped), FAIL | results/s3b1p/c075/s3b1p_VT_val_stack_c075_s21695.json | 8f3ee7711987616a2bc0bee8e27228650de887cf7c1e931b5fe03283c1c2edde |

**How each checkpoint was chosen:** each is the `ckpt` field of the result JSON that the README, REPORT and demo cards
cite. The demo's d11 render steps used the same paths.

## Gates

| gate | check | result |
|---|---|---|
| G1 | 6 models on lpc; sha256 match; load smoke | **PASS.** 36 / 36 files match the PERUN sha256, and all 6 `ckpt_sha` match their result JSONs (`verify_sha.txt`). The load smoke on the RTX 4060 (policy venv, strict fp32) returned a finite (2, 10, 5) chunk with a 0/1 claw for all 6 (`load_smoke.txt`). It ran before the venvs were deleted. |
| G2 | Pages 200 + plays; README link live; rescan clean; no force | **PASS.** The page, the mp4 (`video/mp4`, Accept-Ranges, a range request returns 206) and the 4 PNGs all return 200 (`pages_http.txt`). In the browser the video played: t = 150 → 160 s, 1280×720, no error (`pages_playing_t160.jpg`). The figures decode at 3,200 px (`pages_figures.jpg`). At 375 px the page has no horizontal scroll (`pages_mobile.jpg`). The raw public README has the link on line 15 (`readme_link.txt`). Both builds scanned clean (`make_public_tree.txt`). The push was a plain fast-forward 573c596..2a261f6 plus the new gh-pages branch (`git_push.txt`). |
| G3 | PERUN folder gone; usage before / after; others untouched | **PASS.** The folder is gone. Project dir: 113 was 916 GB of it before, and 205 GB remain after. 112 5.9 G, 114 138 G, lockon 11 G and omot 51 G are the same before and after. `squeue --me` was empty both before and after (`perun_du_before.txt`, `perun_du_after.txt`). |
| G4 | lpc tracked files intact; size before / after | **PASS.** 0 tracked files deleted or modified, and the non-`??` lines of `git status` are identical before and after (`lpc_git_status_{before,after}.txt`). Size: 23 GB → 9.9 GB (`lpc_du_{before,after}.txt`). |

### PERUN sizes deleted (`du -sh`, before)

| item | size |
|---|---|
| data | 546 G |
| runs | 331 G |
| cache | 23 G |
| repos | 14 G |
| vla_tabletop.sif | 4.0 G |
| repo.tgz | 110 M |
| build | 356 K |
| build.log | 108 K |
| repo (empty) | 0 |
| def + sif.sha256 | 8 K |
| **113 total** | **916 G** |

### lpc sizes deleted (`du -sh`, before)

| item | size |
|---|---|
| `outputs/` | 5.6 G |
| `.venv/` | 4.3 G |
| `policy/.venv/` | 3.5 G |
| `checkpoints/` (s1a1_U) | 1.2 G |
| `results/` frames + dumps (571 mp4, 9 npz, 6 pt, 1 png) | ~0.95 G |
| `.pytest_cache/` | 76 K |

The disk's free space rose less than these sizes add up to. uv venvs hard-link into the shared uv cache, so deleting a
venv frees only its own files, and `du` counts a hard-linked file only once.

## Why (decisions)

- **Pages from a branch with no code, not from `main /docs`.** The task asked for this, and it keeps the page from
  serving any code. The builder exports the page files as committed at HEAD, adds `.nojekyll`, and runs the same
  identifier scan as the public tree. A new test checks that the site holds exactly those files and that the page
  references every one of them.
- **The public README goes out as a snapshot commit on top of the public `main`.** It is not a new orphan. The new
  `make_public_tree.py --parent` builds it, so the public history stays linear and the push is a plain fast-forward.
- **Left in `results/` on purpose** (outside "raw untracked frames and dumps"; listed in `lpc_results_kept_logs.txt`
  and `lpc_results_kept_json.txt`):
  - 24 untracked training-step logs and job logs (jsonl / log / out / txt, ~100 MB);
  - 179 untracked eval / dev JSONs (~8 MB).
- **Large untracked dirs not in the approval, left as they are:**
  - `data/` (102 MB, ignored);
  - `logs/smoke/` (106 MB, ignored).
  - Smaller ones: `datasets/` (21 MB) and `figures/src/node_modules/` (19 MB).

## Next

- Nothing scheduled. 113 is closed.
- Re-running anything needs:
  - `uv sync` in the root and in `policy/`;
  - the models from `models_final/`;
  - datasets rebuilt from the scripts (0C steps), since the PERUN data are gone.

## Needs you

- None blocking.
- Optional: run `000_infra/settings/sync.ps1` for the 011 skill, and commit the `000_infra/hpc_usage.md` rows for 011 / 012 (uncommitted, because 000_infra has other changes).

## Evidence

`_todo/113_vla_tabletop/evidence/2026-10-28/012/`:
- the copy: `copy.log`;
- the hashes and load smoke: `perun_sha256.txt`, `verify_sha.txt`, `load_smoke.txt`;
- the publish: `make_public_tree.txt`, `git_push.txt`, `ls_remote_before.txt`, `pages_enable.json`, `pages_config.json`, `pages_http.txt`, `readme_link.txt`, `pages_*.jpg`;
- PERUN usage: `perun_du_{before,after}.txt`;
- laptop cleanup: `lpc_du_{before,after}.txt`, `lpc_git_status_{before,after}.txt`, `lpc_results_deleted.txt`, `lpc_results_kept_{logs,json}.txt`.

Commits: 2a8249c (page source, builder `--pages` / `--parent`, README link, `models_final/` ignored), 1a7e081
(`scripts/load_final_model.py`), plus the end-ritual commit.

skills: none
