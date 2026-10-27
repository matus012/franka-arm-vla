"""Task 011 machine gates G1-G4 on the composed demo (media/) against the render JSONs (--rec) and the sources.

  python scripts/demo_gates.py [--rec results/demo/d11]   -> prints one line per gate, exit 1 on any failure
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
MEDIA = ROOT / "media"


def probe(path: Path) -> dict:
    """Duration / streams of an mp4 from ffmpeg's own header read (imageio-ffmpeg's binary; no ffprobe needed)."""
    import imageio_ffmpeg

    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"],
                       capture_output=True, text=True)
    err = r.stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    v = re.search(r"Stream #\S+: Video: (\w+).*?, (\d+)x(\d+).*?, ([\d.]+) fps", err)
    frames = re.findall(r"frame=\s*(\d+)", err)
    return {"rc": r.returncode, "duration_s": int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None,
            "codec": v[1] if v else None, "wh": (int(v[2]), int(v[3])) if v else None, "fps": float(v[4]) if v else None,
            "decoded_frames": int(frames[-1]) if frames else 0, "size_mb": path.stat().st_size / 2 ** 20}


def g1() -> tuple[bool, str]:
    p = probe(MEDIA / "113_demo.mp4")
    ok = (p["rc"] == 0 and p["codec"] == "h264" and p["wh"] in ((1280, 720), (1920, 1080)) and abs(p["fps"] - 30) < 0.01
          and 180 <= p["duration_s"] <= 300 and p["size_mb"] <= 50 and p["decoded_frames"] >= 0.99 * p["duration_s"] * 30)
    return ok, f"G1 {'PASS' if ok else 'FAIL'}: {p}"


def g2_g4() -> tuple[bool, str]:
    from demo_facts import check

    errs = check(json.loads((MEDIA / "cards.json").read_text()))
    return not errs, f"G2/G4 {'PASS' if not errs else 'FAIL'}: cards.json = fresh build from evaluator constants + result " \
                     f"JSONs, every number in the audited write-ups{'' if not errs else ' -- ' + '; '.join(errs)}"


def g3(rec: Path) -> tuple[bool, str]:
    from demo_compose import RUNS, reason

    errs, n = [], 0
    for gj in sorted((MEDIA / "grids").glob("*.json")):
        if gj.stem == "grounding":
            continue
        g = json.loads(gj.read_text())
        ev = json.loads((rec / f"{g['stream']}.json").read_text())
        assert RUNS.get(gj.stem) == g["stream"], gj
        recs = sorted(ev["episodes"], key=lambda r: r["index"])[:16]
        if [e["index"] for e in g["episodes"]] != list(range(16)):
            errs.append(f"{gj.stem}: not episodes 0-15 in order")
        for e, r in zip(g["episodes"], recs):
            n += 1
            if e["success"] != bool(r["success"]) or e["shown_verdict"] != reason(r) or e["instruction"] != r["instruction"]:
                errs.append(f"{gj.stem} ep{r['index']}: grid verdict differs from the evaluator record")
        verdict_pass = gj.stem in ("s1", "2m")
        first_fail = next((r["index"] for r in recs if not r["success"]), 0)
        if g["full_size_episode"] != (0 if verdict_pass else first_fail):
            errs.append(f"{gj.stem}: full-size episode {g['full_size_episode']} is not the declared one")
        if ev["config"]["seed"] not in range(51000, 51100):
            errs.append(f"{gj.stem}: seed outside the declared demo range")
    return not errs, f"G3 {'PASS' if not errs else 'FAIL'}: {n} grid tiles = evaluator records, declared streams / episodes" \
                     f"{'' if not errs else ' -- ' + '; '.join(errs)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rec", type=Path, default=ROOT / "results" / "demo" / "d11")
    a = p.parse_args()
    res = [g1(), g2_g4(), g3(a.rec)]
    print("\n".join(m for _, m in res))
    sys.exit(0 if all(ok for ok, _ in res) else 1)


if __name__ == "__main__":
    main()
