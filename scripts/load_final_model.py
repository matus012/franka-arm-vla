"""Load one of the final 113 models (models_final/<name>/, weights + config) and run one forward on a dummy batch.

Run with the policy venv (policy/pyproject.toml; `uv sync` inside policy/ recreates it):
    policy\\.venv\\Scripts\\python scripts\\load_final_model.py models_final\\s2m_colour [--device cuda]
Prints the weights sha256 (= `ckpt_sha` in the result JSONs), the served input size and the action chunk shape.
The models expect the 113 observation: side + wrist RGB (H x H x 3 uint8, H = 224), the 19-dim state
(vla_common/protocol.py) and the instruction text; they return N = 10 actions of 5 dims (arm 4 + claw 0/1).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_policy.server import SmolVLAServer  # noqa: E402


def smoke(ckpt: Path, device: str, instruction: str) -> dict:
    t0 = time.time()
    srv = SmolVLAServer(str(ckpt), n_action_steps=10, device=device)
    H, B = srv.image_size, 2
    rng = np.random.default_rng(0)
    ids = np.arange(B, dtype=np.int32)
    srv.reset({"env_ids": ids, "seeds": np.array([1, 2], np.int64)})
    msg = {"env_ids": ids, "side": rng.integers(0, 255, (B, H, H, 3), dtype=np.uint8),
           "wrist": rng.integers(0, 255, (B, H, H, 3), dtype=np.uint8),
           "state": rng.normal(size=(B, 19)).astype(np.float32), "instruction": [instruction] * B}
    a = srv.act(msg)
    ok = a.shape == (B, 10, 5) and bool(np.isfinite(a).all()) and set(np.unique(a[..., 4])) <= {0.0, 1.0}
    return {"ckpt": ckpt.name, "ckpt_sha": srv.ckpt_sha, "image_size": H, "device": srv.info()["device"],
            "action_shape": list(a.shape), "finite_and_binary_claw": ok, "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--instruction", default="pick up the red cube")
    a = ap.parse_args()
    bad = 0
    for c in a.ckpt:
        r = smoke(c, a.device, a.instruction)
        print(json.dumps(r), flush=True)
        bad += not r["finite_and_binary_claw"]
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
