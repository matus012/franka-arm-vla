# Container build test (WSL, 2026-09-30)

Environment: WSL2 Ubuntu-24.04 (`wsl -l -v`: Ubuntu-24.04 + docker-desktop, both were Stopped). No docker inside the distro
(Docker Desktop engine was not running, WSL integration off) -> built with **Apptainer directly from `vision_lift.def`**
(Bootstrap: docker, `nvidia/cuda:13.0.1-runtime-ubuntu24.04`). `sudo` needs a password, so everything ran as `wsl -u root`.
The `Dockerfile` is the equivalent recipe but was NOT built (no docker daemon).

## Commands
```
wsl -d Ubuntu-24.04 -u root -- bash -c "apt-get install -y software-properties-common squashfs-tools uidmap fuse2fs fuse-overlayfs;
   add-apt-repository -y ppa:apptainer/ppa; apt-get update; apt-get install -y apptainer"   # apptainer 1.5.4
mkdir /root/vl; cp hpc/{pyproject.toml,uv.lock,vision_lift.def} /root/vl/; cd /root/vl
apptainer build vision_lift.sif vision_lift.def          # rootful (root in WSL), build dir on WSL ext4
apptainer exec --containall vision_lift.sif python -c "import torch, isaaclab, newton, warp, h5py, scipy, pxr ..."
```
## Result: PASS (imports); GPU checks not possible in WSL
- Build wall time **855 s (~14 min)**, image **4.67 GB** (`vision_lift.sif`, left at `/root/vl/` inside WSL, not in the repo). Python 3.12.14 (uv-managed, /opt/python).
- `%test` inside the build printed: `torch 2.10.0+cu130 warp 1.13.0 newton 1.2.1`.
- Run-time check: `torch 2.10.0+cu130 cuda_avail False cu 13.0 warp 1.13.0 newton 1.2.1 h5py 3.16.0 scipy 1.16.2 pxr ok`.
- `import isaaclab_newton.renderers, isaaclab_tasks, isaaclab.sensors, rsl_rl; from isaaclab_tasks.utils import launch_simulation` -> `render+tasks import ok`.
- `from pxr import Usd, UsdGeom` works (`Usd 0.25.11`).
- **CUDA is not visible to torch inside the container in WSL** (`--nv` and a manual `/usr/lib/wsl/lib` + `/dev/dxg` bind both gave `cuda_avail False`); expected, not debugged. No sim/render was run in WSL. The real check is `slurm/render_check.sbatch` on an H200 node (there `--nv` uses the host driver).
- The base image is `runtime` (no compilers). Warp compiles kernels with its bundled NVRTC (pip `nvidia-cuda-nvrtc`), so no nvcc should be needed; unverified until a GPU run.

## Import-chain library notes
- Installed (apt): `libgl1 libglu1-mesa libx11-6 libxext6 libxrender1 libsm6 libxt6 libgomp1 ca-certificates curl git`. The set was a generous guess; the whole chain imports with it, not bisected for a minimal set. No EGL/Vulkan/X server packages were installed and none were needed for imports.
- **`import usdex.core` aborts on Linux** (`terminate called after ... error_already_set`), after the warning "usd-core 25.11 is installed alongside usd-exchange, but provides incompatible OpenUSD modules". Windows only warns. Nothing in isaaclab / isaaclab_newton / isaaclab_tasks imports `usdex` (grepped), and the task/render import chain works, so it is harmless unless our code imports `usdex`. It comes from the pins (usd-core + usd-exchange together, both required by the brief); not changed.
