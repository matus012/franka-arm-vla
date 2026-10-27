"""Stage 0.5: training throughput (samples/s through StepDataset + DataLoader + bf16 step), RGB and RGB-D, and a
random-init student checkpoint for the student-driven eval throughput run."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from vision_lift.dataset import StepDataset, list_files  # noqa: E402
from vision_lift.student import StudentPolicy  # noqa: E402
from vision_lift.train import bc_loss, proprio_stats, save_student  # noqa: E402

if __name__ == "__main__":
    files = list_files(ROOT / "datasets" / "s05")
    dev = torch.device("cuda:0")
    res = {"files": len(files)}
    for depth in (False, True):
        for workers in (4, 8):
            ds = StepDataset(files, depth=depth)
            m = StudentPolicy(depth=depth)
            m.set_norm(*proprio_stats(ds))
            m.to(dev)
            opt = torch.optim.AdamW(m.parameters(), 3e-4)
            dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=workers, drop_last=True,
                                             persistent_workers=True, pin_memory=True, prefetch_factor=4)
            n, t0 = 0, None
            for i, b in enumerate(dl):
                if i == 10:
                    torch.cuda.synchronize()
                    t0 = time.time()
                loss, _ = bc_loss(m, b, dev, augment=True)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if i >= 10:
                    n += 256
                if i == 110:
                    break
            torch.cuda.synchronize()
            res[f"{'rgbd' if depth else 'rgb'}_w{workers}_samples_per_s"] = round(n / (time.time() - t0), 1)
            print(res, flush=True)
            del dl
        save_student(ROOT / "logs" / "bench" / f"random_{'rgbd' if depth else 'rgb'}.pt", m.cpu(), {"note": "untrained, throughput only"})
    res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6)
    out = ROOT / "results" / "s05_throughput" / "train_bench.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
