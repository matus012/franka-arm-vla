# CONTRACT — 113 interfaces (v1, frozen 2026-10-01 before 0A/0B are built)

The brief (`113_project.md`) is the plan; this file only fixes the names and shapes that parallel work meets at.
A change to anything here is a commit to this file first (`contract: <change>`), announced in `status.txt`.

## Packages and venvs
| path | venv | may import |
|---|---|---|
| `lift_rl/` | sim | frozen copy of 111 (via 112). **Never edited.** |
| `vision_lift/` | sim | frozen copy of 112. **Never edited.** 113 code copies from it or imports its helpers (cameras, frames, envs). |
| `vla_common/` | both | stdlib + numpy + h5py only (task catalog, instructions, held-out splits, protocol, episode schema, timing). |
| `vla_tabletop/` | sim (`.venv`, Python 3.12, Isaac Lab 3.0.0b2.post1, torch 2.10.0+cu130) | isaaclab, torch, `lift_rl`, `vision_lift`, `vla_common`. Never lerobot. |
| `vla_policy/` | policy (`policy/.venv`, Python 3.12, `lerobot[smolvla,pi]==0.6.1`, torch 2.10.0+cu130) | lerobot, torch, `vla_common`. Never isaaclab. |

Laptop GPU runs: `powershell -File ops\run_job.ps1 -Job <name> -LogFile logs\jobs\<name>.log -PyArgs "<script + args>" [-Venv policy]`
(GPU lock = one job at a time across all agents, 2 h kill, 20 h total). Short (< 1 min) smoke checks may run directly.

## World catalog (`vla_common/catalog.py`)
- `COLORS = ("red", "green", "blue", "yellow")`, `SIZES = ("small", "large")`.
- `CubeId = (color, size)`; 8 cubes. Prim / scene names: `cube_<color>_<size>`; pads: `pad_<color>`.
- Nominal RGB per colour (linear 0–1) is defined once here and used by the scene, the randomizer and its tests.
- Edge lengths: small = the measured 111 cube edge (0A.1 records it), large ≤ 6.0 cm (brief §3 [consultant] rule).
  Stored as `EDGE_M = {"small": ..., "large": ...}`; same density.
- Every env always holds all 8 cubes and 4 pads; objects not used in an episode are parked off the table, outside
  both camera views and outside the reach box (a `parked` flag per object is part of the episode state).

## Tasks and instructions (`vla_common/tasks.py`)
- `TaskSpec(task: "pick"|"place"|"stack", target: CubeId, dest: str|CubeId|None, cubes: list[CubeId], pads: list[str])`
  — `dest` = pad colour (place), bottom CubeId (stack), None (pick). `cubes` / `pads` = objects on the table.
- `instruction(spec, rng, split="train"|"paraphrase") -> str`: template text; size is stated whenever another cube
  on the table shares the target's (or bottom cube's) colour; templates of the two splits are disjoint.
- `labels(spec) -> np.ndarray[int64] (4,)`: (task, target colour, target size, destination) for the baseline;
  destination index = pad colour 0–3, bottom cube 4–11, none 12.
- `HELD_OUT_TARGETS` (2 cubes, never the manipulated cube in training; may be a stack bottom or a distractor; neither
  is the Stage 1 cube), `HELD_OUT_PLACE_PAIRS`, `HELD_OUT_STACK_PAIRS`; `sample_spec(stage, split, rng)`.
- Matching: `matches(spec, instruction) -> list[CubeId]` (a parser used only by the uniqueness test) must return
  exactly one cube (and one destination).

## Sim observation and action (policy-facing)
- **state** `(19,) float32` = `joint_pos_rel` (9: 7 arm + 2 finger, relative to default) + `joint_vel_rel` (9) +
  `gripper_width` (1, m) — the 112 `StudentObsCfg` minus `target` and `prev_action`. Whitelist enforced by
  `tests/test_banned_inputs.py`; nothing else crosses the bridge except images and the instruction / labels.
- **images** `side`, `wrist`: `(H, H, 3) uint8` RGB, H from 0A.4 (128 | 224 | 256). Camera code from
  `vision_lift/cameras.py`; a frame is captured every `PERIOD` control steps and seen `LATENCY` frames late
  (`vision_lift/frames.py` semantics; 113 values from 0A.3, stored in `vla_common/timing.py`).
- **action** `(5,) float32` = 112 student action: dx, dy, dz, dyaw in [-1, 1] (1 cm / 3° per step, same IK, hand
  vertical) + claw in {0., 1.} (1 = close; executed as > 0.5). Control 50 Hz (dt 0.01, decimation 2).
- Labels are stored **as executed** (after clipping).

## Episode files (sim → exporter): `vla_common/episode.py`
HDF5, one file per chunk of episodes, group `ep_<k>` per kept episode:
| dataset | shape / dtype | meaning |
|---|---|---|
| `side_rgb`, `wrist_rgb` | (F, H, H, 3) uint8 | frames captured at camera ticks (step 0, PERIOD, 2·PERIOD, …) |
| `state` | (T, 19) float32 | state at control step t, before the action |
| `action` | (T, 5) float32 | executed action at step t |
| `frame_idx` | (T,) int32 | index into the frames the policy sees at step t (delayed) |

Attributes: `instruction`, `split`, `task`, `spec_json`, `labels`, `seed`, `env_id`, `success`, `metrics_json`,
`period`, `latency`, `image_size`, `visual_mode`, `action_noise_sigma`, `expert_version`. File attrs: `git_sha`,
`created`, `stage`. Only successful episodes are kept for training; batch success rates go into a sidecar JSON.

## Bridge (`vla_common/protocol.py`, server `vla_policy/server.py`, client `vla_tabletop/bridge.py`)
- `multiprocessing.connection` on `127.0.0.1:<port>` with an authkey; messages are dicts of str/int/float/list and
  numpy arrays encoded as `{"__nd__": dtype, "shape": [...], "data": bytes}` (no pickled numpy across venvs).
- Requests: `{"op": "info"}`, `{"op": "reset", "env_ids": int32 (B,)}`, `{"op": "act", "env_ids": int32 (B,),
  "side": uint8 (B,H,H,3), "wrist": uint8 (B,H,H,3), "state": float32 (B,19), "instruction": list[str] (B)}`
  (+ `"labels": int64 (B,4)` for the baseline). One `act` per query step for all envs that need a new chunk.
- Response to `act`: `{"actions": float32 (B, n_action_steps, 5), "server_s": float}`; the client executes the
  chunk open loop, one action per control step, claw thresholded at 0.5.
- Server CLI: `--policy smolvla|pi05|baseline --ckpt <dir|hub id> --n-action-steps N --device cuda`.

## Dataset (exporter `vla_policy/export.py`)
LeRobot dataset format v3.0, local only (`datasets/<name>/`, never pushed). Features: `observation.images.side`,
`observation.images.wrist` (video), `observation.state` (19), `action` (5), `task` (instruction string); extra
per-episode columns for split / labels. Row rate (one per control step or one per camera frame) is decided in
0A.3/0B, proved by the overfit test and written into GATES.md before 0C. Decoding: pyav on Windows, torchcodec on PERUN.

## Results
Every gate writes `results/<gate>.json` (numbers, seeds, git sha, wall time) and a contact sheet
`results/frames/<gate>/*.png`; both get copied to `_todo/113_vla_tabletop/evidence/<date>/` by the orchestrator.

## v2 amendments (2026-10-01, from the lead's contract review; supersede v1 where they differ)
- **a) Labels vs executed actions.** Episode files store `action` = the clean expert action (clipped to the action
  box, i.e. executable) = the training label, and `action_exec` (T, 5) float32 = the action actually executed
  (clean + demo noise sigma, clipped). Without noise they are equal. Exporter writes `action` (label) to the dataset.
- **b) Stateless server.** `reset` carries `"seeds": int64 (B,)`; the server keeps no LeRobot action queue (use
  `predict_action_chunk` or equivalent) and uses one noise generator per env seeded from `reset`, so results do not
  depend on batch size or batch order.
- **c) `info` response:** `{"policy", "n_action_steps", "chunk_size", "image_size", "state_dim": 19, "action_dim": 5,
  "period", "latency", "ckpt_sha"}`; the client fails fast on any mismatch with `vla_common/timing.py` / its config.
- **d) Single source of truth for timing:** `vla_common/timing.py` holds PERIOD, LATENCY, N; 113 code always passes
  them explicitly to `vision_lift/frames.py` functions (never relies on that module's defaults). Test asserts it.
- **e)** `vla_tabletop/` never uses `lift_rl` `CUBE_REST_Z` / `CUBE_HALF` (test greps for it).
- **f)** Episode attrs add `episode_index`, `T`, `n_action_steps`, `detector_version`.
- **g) Camera mapping fixed:** camera1 = side, camera2 = wrist, for SmolVLA and pi0.5.
- **h)** Instruction test also checks <= 48 tokens with the SmolVLA tokenizer (policy venv).
- **i) Interpretation (deviation log):** one `act` per query step (every N control steps) for all envs, not one per
  camera frame — with chunked execution the policy only looks at the newest frame at query steps.
- **j) Parked objects** are placed off the table on the ground, at rest, spaced apart, outside both camera frusta
  and the reach box; the scene test asserts they did not move by > 1 mm over an episode.
- **k) Pads in every scene.** Pads are on the table in pick and stack episodes too (as distractors), not only in place
  episodes, so the scene alone does not reveal the task (lead finding 1).
- **l) 2026-10-02 (Stage 1 one fix, GATES "Stage 0C attempt 1"):** the LeRobot training column `action` = r, the
  expert's setpoint step: r_0 = a_0, r_t = clip(a_t + (x_{t-1} - a_{t-1}), -1, 1) on the arm dims, claw = a. HDF5 keeps
  the clean label `action` (a) and `action_exec` (x). Supersedes "dataset `action` = the clean label" for the training
  column only.
