# Third-party code, models and assets

This repository's own code is MIT (`LICENSE`, (c) 2026 Matus Filo). Nothing below is redistributed except where a
row says "copied"; all dependencies are installed from the pinned lock files (`uv.lock`, `policy/uv.lock`).

| component | licence | how it is used here |
|---|---|---|
| Isaac Lab 3.0.0b2.post1, Newton 1.2.1, MuJoCo-Warp 3.8.1, NVIDIA Warp 1.13.0 | BSD-3-Clause / Apache-2.0 | pip dependencies (physics, Newton Warp renderer) |
| Isaac Lab source, copied in part: `lift_rl/env_cfg.py` (stock `Isaac-Lift-Cube-Franka-v0` cfg), `lift_rl/physics.py` (stack-task Newton preset), `lift_rl/guarded_env.py` (`ManagerBasedRLEnv.step`), `lift_rl/agents.py` (PPO runner cfg values), `scripts/templates_112/cli_args.py` | BSD-3-Clause, Copyright (c) 2022-2026, The Isaac Lab Project Developers | copied and modified; notice kept in the file headers and in `LICENSE_111`; these files stay BSD-3-Clause |
| NVIDIA Isaac Sim / Omniverse runtime and assets, incl. the Franka Panda and table USD assets | NVIDIA Omniverse / Isaac Sim licence terms (not open source) | fetched by Isaac Lab at run time from NVIDIA's asset server; **not copied into or redistributed by this repository**; users must accept NVIDIA's terms themselves |
| PyTorch 2.10.0 / torchvision 0.25.0 | BSD-3-Clause | pip dependencies |
| LeRobot 0.6.1 | Apache-2.0 | pip dependency (policy venv): datasets, SmolVLA, pi0.5 loaders |
| `lerobot/smolvla_base` (SmolVLA) | Apache-2.0 (model card, https://huggingface.co/lerobot/smolvla_base) | pretrained weights downloaded from the Hub, fine-tuned locally / on PERUN; fine-tuned weights are **not released** |
| SmolVLM2-500M-Video-Instruct backbone (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) | Apache-2.0 (model card, https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | backbone of SmolVLA, downloaded via LeRobot; not redistributed |
| `lerobot/pi05_base`, `google/paligemma-3b-pt-224` tokenizer | Gemma terms (treated as such) | downloaded only (baseline comparison); fine-tuned weights are never published |
| ffmpeg (via `imageio-ffmpeg` / `imageio[ffmpeg]`; PyAV for dataset decoding) | wrapper BSD-2-Clause; ffmpeg binary LGPL/GPL depending on build | used to encode videos; the binary is **not redistributed** (videos in `media/`, `videos/`, `results/frames/` are our own renders) |
| 111_lift_rl (`lift_rl/`, `LICENSE_111`) | MIT (owner's own), with the Isaac Lab-derived files above under BSD-3-Clause | frozen copy (via 112), never modified |
| 112_vision_lift (`vision_lift/`, `scripts/templates_112`, `tests/templates_112`, `hpc/templates_112`) | MIT (owner's own); `scripts/templates_112/cli_args.py` is BSD-3-Clause (Isaac Lab) | frozen copy / templates, never modified |
| h5py, SciPy, imageio, matplotlib, TensorBoard, pytest | BSD / MIT-style | pip dependencies |

No dataset, checkpoint, or NVIDIA asset is part of this repository; `LICENSE` is MIT because no dependency's licence
requires otherwise (all linked/imported components are permissive; the Gemma-terms models are not redistributed).
