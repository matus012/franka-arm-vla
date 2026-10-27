"""GATES 0B.4 mutants: each listed test must kill its listed mutant (each run once). The source file is mutated in
place, the test file run with the right venv's python, and the file restored byte-exactly (sha256 + `git diff --quiet`
checked after every mutant). Never commit while this runs.

  python scripts/s0b4_mutants.py --only cpu   -> results/0b4_mutants_cpu.txt   (all mutants whose test needs no GPU)
  python scripts/s0b4_mutants.py --only sim   -> results/0b4_mutants_sim.txt   (the cube-pose mutant: test_sim_113, GPU;
                                                 run through ops/run_job.ps1)
  python scripts/s0b4_mutants.py --merge      -> results/0b4_mutants.txt
"""

from __future__ import annotations

import datetime
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = {"sim": ROOT / ".venv" / "Scripts" / "python.exe", "policy": ROOT / "policy" / ".venv" / "Scripts" / "python.exe"}
_CUBE_HUE = ("hsv_to_rgb(*(lambda x: ((x[..., 0] + 0.05 * (torch.rand(x.shape[:2], generator=gen) * 2 - 1)) % 1.0, "
             "x[..., 1], x[..., 2]))(rgb_to_hsv(_jitter_objects(cube_base, n, gen, nom))))")
# name -> (group, source file, [(old, new)], test file, venv)
MUTANTS = {
    "1 banned inputs: a cube-pose term appended to the state": (
        "sim", "vla_tabletop/scene.py",
        [("@configclass\nclass ObservationsCfg:",
          "def _mutant_cube_pos(env):\n    return _t(env.scene[CUBE_NAMES[0]].data.root_pos_w) - env.scene.env_origins\n"
          "\n\n@configclass\nclass ObservationsCfg:"),
         ("        gripper_width = ObsTerm(func=VE.student_gripper_width)  # 1: finger width [m]\n",
          "        gripper_width = ObsTerm(func=VE.student_gripper_width)  # 1: finger width [m]\n"
          "        cube_pos = ObsTerm(func=_mutant_cube_pos)  # MUTANT\n")],
        "tests/test_sim_113.py", "sim"),
    "2a alignment: latency 0 (runtime frame buffer)": (
        "cpu", "vla_tabletop/bridge.py",
        [("self.frames = DelayedFrames(period, latency)", "self.frames = DelayedFrames(period, 0)")],
        "tests/test_alignment.py", "policy"),
    "2b alignment: export off by one row (row t gets label t+1)": (
        "cpu", "vla_policy/export.py",
        [('"observation.state": a["state"][t], "action": a["action"][t],',
          '"observation.state": a["state"][t], "action": a["action"][min(t + 1, a["action"].shape[0] - 1)],')],
        "tests/test_alignment.py", "policy"),
    "3 instruction uniqueness: size never stated": (
        "cpu", "vla_common/tasks.py",
        [("say_size = rng.random() < P_SIZE_UNFORCED or collide", "say_size = False")],
        "tests/test_instructions.py", "sim"),
    "4 held-out disjointness: one held-out place pair admitted to the train sampler": (
        "cpu", "vla_common/tasks.py",
        [("if (t, p) not in HELD_OUT_PLACE_PAIRS:", "if (t, p) not in HELD_OUT_PLACE_PAIRS[1:]:")],
        "tests/test_heldout.py", "sim"),
    "5a visual randomization: cube hue jitter enabled (+-0.05 hue on every cube)": (
        "cpu", "vla_tabletop/visual_rand.py",
        [('"cube_rgb": _jitter_objects(cube_base, n, gen, nom),', f'"cube_rgb": {_CUBE_HUE},')],
        "tests/test_visual_rand.py", "sim"),
    "5b visual randomization: held-out band overlapping train (hue 0.30-0.47 vs train 0.00-0.35)": (
        "cpu", "vla_tabletop/visual_rand.py",
        [('"heldout": {"hues": ((0.38, 0.47),),', '"heldout": {"hues": ((0.30, 0.47),),')],
        "tests/test_visual_rand.py", "sim"),
}


def pytest(test: str, venv: str) -> tuple[int, list[str], str]:
    r = subprocess.run([str(PY[venv]), "-m", "pytest", test, "-q", "-p", "no:cacheprovider", "-x"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1:] or [""]
    return r.returncode, sorted(set(re.findall(r"FAILED ([\w/\\.]+::[\w\[\]\-.,]+)", r.stdout))), tail[0]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    if "--merge" in sys.argv:
        parts = [(ROOT / "results" / f"0b4_mutants_{g}.txt").read_text(encoding="utf-8") for g in ("cpu", "sim")]
        ok = all("ALL_KILLED=True" in p for p in parts)
        (ROOT / "results" / "0b4_mutants.txt").write_text("\n".join(parts) + f"\nGATE_0B4_MUTANTS_ALL_KILLED={ok}\n",
                                                          encoding="utf-8")
        print(f"ALL_KILLED={ok}")
        return
    group = sys.argv[sys.argv.index("--only") + 1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = [f"# 0B.4 mutants, group {group}, {datetime.datetime.now().isoformat(timespec='seconds')}, git HEAD {head}",
             "# per mutant: unmutated test result, mutant result + failing tests, restore check (sha256 + git diff)"]
    all_killed = True
    for name, (g, rel, reps, test, venv) in MUTANTS.items():
        if g != group:
            continue
        src = ROOT / rel
        orig = src.read_bytes()
        h0 = sha(src)
        code0, failed0, tail0 = pytest(test, venv)
        text = orig.decode("utf-8")
        for old, new in reps:
            if "\r\n" in text:  # CRLF working copy
                old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
            assert text.count(old) == 1, f"{name}: pattern not found exactly once: {old!r}"
            text = text.replace(old, new)
        try:
            src.write_bytes(text.encode("utf-8"))
            code, failed, tail = pytest(test, venv)
        finally:
            src.write_bytes(orig)
        clean = subprocess.run(["git", "diff", "--quiet", "--", rel], cwd=ROOT).returncode == 0
        restored = sha(src) == h0 and clean
        killed = code0 == 0 and code != 0
        all_killed &= killed and restored
        lines += [f"{name}", f"  file {rel}; test {test} ({venv} venv)",
                  f"  unmutated: {'GREEN' if code0 == 0 else 'RED ' + str(failed0)} ({tail0})",
                  f"  mutant: {'KILLED' if killed else 'SURVIVED'} by {failed or tail} ({tail})",
                  f"  restore: sha256 {'identical' if sha(src) == h0 else 'DIFFERENT'}, git diff "
                  f"{'clean' if clean else 'DIRTY'} -> {'OK' if restored else 'FAILED'}"]
        print("\n".join(lines[-5:]), flush=True)
    lines.append(f"ALL_KILLED={all_killed}")
    out = ROOT / "results" / f"0b4_mutants_{group}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1])


if __name__ == "__main__":
    main()
