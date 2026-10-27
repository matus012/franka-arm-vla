# Policy venv (`policy/.venv`) — exact versions (uv.lock is authoritative)

Built 2026-10-01 on the laptop (native Windows 11) with `uv lock` + `uv sync` from `policy/`
(dot-source `scripts\uv_env.ps1` first). Separate from the sim venv (`.venv`); they talk only over the bridge.

| package | version | note |
|---|---|---|
| Python | 3.12.13 | uv-managed CPython |
| lerobot | 0.6.1 | extras `smolvla,pi,training` (training pulls wandb; every LeRobot command runs with `--wandb.enable=false`) |
| torch | 2.10.0+cu130 | pytorch-cu130 index (explicit), same pin as the sim venv; `torch.cuda.is_available()` = True |
| torchvision | 0.25.0+cu130 | pytorch-cu130 index |
| transformers | 5.5.4 | LeRobot 0.6.1 needs >= 5.4, < 5.6 |
| accelerate | 1.15.0 | |
| datasets | 4.8.5 | |
| huggingface-hub | 1.33.0 | downloads only; no pushes |
| tokenizers | 0.22.2 | |
| safetensors | 0.8.0 | |
| av (PyAV) | 15.1.0 | video encode (libsvtav1) + decode backend on Windows |
| torchcodec | 0.10.0, Linux only | `override-dependencies` marker `sys_platform == 'linux'`: not installed on Windows (cannot load without shared FFmpeg DLLs; LeRobot falls back to pyav anyway); 0.10.x because 0.11 needs torch 2.11 (same pin as `hpc/policy/pyproject.toml`) |
| h5py | 3.16.0 | contract episode files |
| numpy | 2.2.6 | |
| wandb | 0.27.2 | installed by `lerobot[training]`, never enabled |

Checked constraints (LeRobot 0.6.1 METADATA): torch >= 2.7, < 2.12; torchvision >= 0.22, < 0.27;
transformers >= 5.4, < 5.6 (via the `smolvla` / `pi` extras); av >= 15, < 16.

Verify:
`policy\.venv\Scripts\python.exe -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"`
-> `0.6.1 2.10.0+cu130 True`.

Hub assets used (downloads, cached under `%USERPROFILE%\.cache\huggingface`): `lerobot/smolvla_base`
(revision d9f33c94a60fb382c90dea2164c96845bd955e28, model.safetensors sha256
7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb), `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
(revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467; VLM config + tokenizer).
