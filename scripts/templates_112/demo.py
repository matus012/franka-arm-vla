"""One-command demo (brief section 8): replays a student checkpoint (after training) and writes to videos/:
  - student_view_<outcome>.mp4   close-up with the phase overlay | the delayed side + wrist frames the student saw,
                                 for one success, one near miss (< 10 cm) and one failure (when they occur)
  - grid_3x3.mp4 / grid_3x3.gif  9 episodes
  - baseline_vs_student.mp4      the modular baseline (left) vs the student (right), same seed / episode
  - visual_rand_off_vs_on.mp4    the student on nominal visuals (left) vs randomized train visuals (right)
  - INDEX.md                     file, checkpoint, seed, visuals, outcome per file
Usage: python scripts/demo.py --ckpt checkpoints/.../epoch_xxx.pt [--world P6] [--seed 12345] [--num_envs 32]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--world", default="P6")
p.add_argument("--seed", type=int, default=12345)
p.add_argument("--num_envs", type=int, default=32, help="episodes replayed per condition (singles need a failure)")
p.add_argument("--visual_seed", type=int, default=12347)
p.add_argument("--out", default="videos")
a = p.parse_args()

import imageio.v2 as iio  # noqa: E402
import numpy as np  # noqa: E402

out = ROOT / a.out
raw = out / "raw"


def worker(tag: str, *extra: str) -> dict:
    d = raw / tag
    if not (d / "episodes.json").exists():
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "demo_worker.py"), "--world", a.world, "--seed",
                        str(a.seed), "--out", str(d.relative_to(ROOT)), *extra], cwd=ROOT, check=True)
    return json.loads((d / "episodes.json").read_text())


def side_by_side(left: Path, right: Path, dst: Path) -> None:
    ra, rb = iio.get_reader(str(left)), iio.get_reader(str(right))
    w = iio.get_writer(str(dst), fps=25, codec="libx264", quality=7, macro_block_size=8)
    for fa, fb in zip(ra, rb):
        w.append_data(np.concatenate([fa[::2, ::2], np.full((fa.shape[0] // 2, 6, 3), 255, np.uint8), fb[::2, ::2]], 1))
    w.close()
    ra.close()
    rb.close()


def grid(files: list[Path], dst: Path, gif: Path) -> None:
    readers = [iio.get_reader(str(f)) for f in files]
    w = iio.get_writer(str(dst), fps=25, codec="libx264", quality=7, macro_block_size=8)
    frames_gif = []
    for k, frs in enumerate(zip(*readers)):
        small = [f[::3, ::3] for f in frs]
        rows = [np.concatenate(small[r * 3:(r + 1) * 3], 1) for r in range(3)]
        g = np.concatenate(rows, 0)
        w.append_data(g)
        if k % 5 == 0:
            frames_gif.append(g[::2, ::2])
    w.close()
    for r in readers:
        r.close()
    iio.mimsave(str(gif), frames_gif, duration=0.2, loop=0)


ck = ["--ckpt", a.ckpt]
nom = worker("student_nominal", "--policy", "student", *ck, "--num_envs", str(a.num_envs))
vis = worker("student_train_visuals", "--policy", "student", *ck, "--num_envs", "9", "--visual", "train",
             "--visual_seed", str(a.visual_seed))
base = worker("baseline_nominal", "--policy", "baseline", "--num_envs", "9")
index = ["# Demo videos (112_vision_lift)", "", f"Student checkpoint: `{a.ckpt}`; world {a.world}; seed {a.seed}. "
         "All replayed after training. Left panel: close-up with the phase overlay (111 phase machine on the true state; "
         "target cube green when the cube is within 3 cm). Right panel: the delayed side + wrist frames the student saw. "
         "First 2 s at 0.25x (grasp in slow motion), then 1x.", "", "| file | what | visuals | outcome |", "|---|---|---|---|"]
eps = nom["episodes"]
picks = {"success": next((e for e in eps if e["success"]), None),
         "near_miss": next((e for e in eps if not e["success"] and e["dist_m"] < 0.10), None),
         "failure": next((e for e in eps if not e["success"] and e["dist_m"] >= 0.10), None)}
for cls, e in picks.items():
    if e is None:
        index.append(f"| — | {cls}: none among the {len(eps)} replayed episodes | nominal | — |")
        continue
    dst = out / f"student_view_{cls}.mp4"
    shutil.copyfile(raw / "student_nominal" / e["file"], dst)
    index.append(f"| `{dst.name}` | student, env {e['env']} | nominal | success={e['success']}, final distance "
                 f"{e['dist_m'] * 100:.1f} cm |")
grid([raw / "student_nominal" / f"env{i}.mp4" for i in range(9)], out / "grid_3x3.mp4", out / "grid_3x3.gif")
index.append(f"| `grid_3x3.mp4` / `.gif` | student, envs 0-8 | nominal | {sum(e['success'] for e in eps[:9])}/9 success |")
side_by_side(raw / "baseline_nominal" / "env0.mp4", raw / "student_nominal" / "env0.mp4", out / "baseline_vs_student.mp4")
index.append(f"| `baseline_vs_student.mp4` | left: modular baseline, right: student, env 0 | nominal | baseline "
             f"{base['episodes'][0]['success']}, student {eps[0]['success']} |")
side_by_side(raw / "student_nominal" / "env0.mp4", raw / "student_train_visuals" / "env0.mp4",
             out / "visual_rand_off_vs_on.mp4")
index.append(f"| `visual_rand_off_vs_on.mp4` | left: nominal visuals, right: randomized (train) visuals, env 0 | both | "
             f"{eps[0]['success']} / {vis['episodes'][0]['success']} |")
(out / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
print("\n".join(index))
