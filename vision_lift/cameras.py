"""Student cameras (brief section 0.2): a fixed side camera and a wrist camera on panda_hand, Newton Warp renderer.

- Side camera: prim per env, posed once from an eye/look-at pair in the env frame (`aim_side`).
- Wrist camera: prim under Robot/panda_hand with a fixed mount offset (ROS convention: +z forward, +x right, +y down),
  `update_latest_camera_pose=True`. Verified (tests/test_cameras.py) to follow the hand in lightweight mode.
- Instance segmentation (`instance_segmentation_fast`, raw int32) returns Newton shape indices; the cube's visual
  shape id per env is looked up once from the Newton model's shape labels (`cube_shape_ids`).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

APERTURE = 20.955  # mm, horizontal aperture (Isaac Lab default)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def tilt_quat_xyzw(theta_deg: float) -> tuple[float, float, float, float]:
    """ROS-convention camera rotation in the hand frame: optical axis = hand +z tilted by theta toward hand -x, image
    right = hand +y. Returns (x, y, z, w)."""
    from scipy.spatial.transform import Rotation

    s, c = math.sin(math.radians(theta_deg)), math.cos(math.radians(theta_deg))
    R = np.array([[0.0, -c, -s], [1.0, 0.0, 0.0], [0.0, -s, c]])  # columns: cam x, y, z in hand coordinates
    return tuple(float(v) for v in Rotation.from_matrix(R).as_quat())


# Default mount / pose (fixed by GATES.md Stage 0.2 before its run; changes are counted fallbacks)
SIDE_EYE = (1.15, 0.45, 0.55)
SIDE_LOOK = (0.50, 0.00, 0.18)
SIDE_FOCAL = 20.0
WRIST_POS = (0.065, 0.0, 0.03)
WRIST_TILT_DEG = 30.0
WRIST_FOCAL = 12.0
RES = 128
DATA_TYPES = ("rgb", "depth", "instance_segmentation_fast")


def _spawn(focal: float):
    return sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=400.0, horizontal_aperture=APERTURE,
                                      clipping_range=(0.01, 10.0))


def side_cam_cfg(res: int = RES, focal: float = SIDE_FOCAL, data_types=DATA_TYPES, name: str = "SideCam") -> CameraCfg:
    return CameraCfg(prim_path="{ENV_REGEX_NS}/" + name, update_latest_camera_pose=True, data_types=list(data_types),
                     spawn=_spawn(focal), width=res, height=res,
                     renderer_cfg=NewtonWarpRendererCfg(colorize_instance_segmentation=False))


def wrist_cam_cfg(res: int = RES, pos=WRIST_POS, tilt_deg: float = WRIST_TILT_DEG, focal: float = WRIST_FOCAL,
                  data_types=DATA_TYPES, name: str = "WristCam") -> CameraCfg:
    return CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand/" + name, update_latest_camera_pose=True,
                     offset=CameraCfg.OffsetCfg(pos=tuple(pos), rot=tilt_quat_xyzw(tilt_deg), convention="ros"),
                     data_types=list(data_types), spawn=_spawn(focal), width=res, height=res,
                     renderer_cfg=NewtonWarpRendererCfg(colorize_instance_segmentation=False))


def add_student_cameras(cfg, res: int = RES, data_types=DATA_TYPES) -> None:
    cfg.scene.side_cam = side_cam_cfg(res, data_types=data_types)
    cfg.scene.wrist_cam = wrist_cam_cfg(res, data_types=data_types)


def aim_side(u, eye=SIDE_EYE, look=SIDE_LOOK, name: str = "side_cam", env_ids=None) -> None:
    o = u.scene.env_origins if env_ids is None else u.scene.env_origins[env_ids]
    e = o + torch.as_tensor(eye, dtype=torch.float32, device=u.device)
    k = o + torch.as_tensor(look, dtype=torch.float32, device=u.device)
    u.scene[name].set_world_poses_from_view(e, k, env_ids=env_ids)


def cube_shape_ids(u) -> torch.Tensor:
    """(N,) Newton shape index of each env's cube visual mesh (as written by instance_segmentation_fast)."""
    from isaaclab_newton.physics import NewtonManager

    labels = NewtonManager.get_model().shape_label
    ids = []
    for i in range(u.num_envs):
        key = f"/World/envs/env_{i}/Object/visuals"
        hit = [k for k, lab in enumerate(labels) if lab.startswith(key)]
        if len(hit) != 1:
            raise RuntimeError(f"env {i}: expected one cube visual shape, found {hit}")
        ids.append(hit[0])
    return torch.tensor(ids, device=u.device)


def cube_pixels(u, name: str, cube_ids: torch.Tensor) -> torch.Tensor:
    """(N,) number of pixels showing the env's own cube in camera `name`."""
    seg = _t(u.scene[name].data.output["instance_segmentation_fast"])[..., 0]
    return (seg == cube_ids[:, None, None]).sum(dim=(1, 2))


def rgb(u, name: str) -> torch.Tensor:
    return _t(u.scene[name].data.output["rgb"])[..., :3]


def depth(u, name: str) -> torch.Tensor:
    return _t(u.scene[name].data.output["depth"])[..., 0]


def student_views(u, height: int) -> np.ndarray:
    """(N, height, height//2... ) side (top) and wrist (bottom) RGB views stacked, upscaled (nearest) to `height`."""
    s, w = rgb(u, "side_cam"), rgb(u, "wrist_cam")
    both = torch.cat([s, w], dim=1)  # (N, 2H, W, 3)
    k = max(1, height // both.shape[1])
    both = both.repeat_interleave(k, 1).repeat_interleave(k, 2)
    out = torch.zeros(both.shape[0], height, both.shape[2], 3, dtype=torch.uint8, device=both.device)
    out[:, : both.shape[1]] = both.to(torch.uint8)
    return out.cpu().numpy()
