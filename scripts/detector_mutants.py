"""GATES.md mutant checks, CPU only (no simulator). The source file is restored afterwards.
  --set detectors : "Instrument proof first": tests/test_detectors.py vs the pre-registered detector mutants m1-m5
                    -> results/0a_detector_mutants.txt
  --set expert    : 0A.3 faithfulness: tests/test_chunked_expert.py vs the zero-latency-snapshot and
                    closed-loop-between-queries mutants -> results/0a3_expert_mutants.txt

Usage: python scripts/detector_mutants.py --set detectors
"""

from __future__ import annotations

import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = {"detectors": ("vla_tabletop/detectors.py", "tests/test_detectors.py", "0a_detector_mutants.txt"),
        "expert": ("vla_tabletop/expert.py", "tests/test_chunked_expert.py", "0a3_expert_mutants.txt")}
EXPERT_MUTANTS = {
    "zero-latency snapshot (killed by ii)": [
        ("fs = frame_step(t, self.period, self.latency)", "fs = frame_step(t, self.period, 0)")],
    "closed loop between queries (killed by i)": [("query = t % self.n_query == 0", "query = True")],
    "commanded target read fresh at t instead of the frame step (killed by iv)": [
        ("cmd = self.snaps[fs]  #", "cmd = snap_fn()  #")],
}
MUTANTS = {
    "m1 one rest height for both sizes (large judged with the small cube's rest z)": [
        ("e, zr = geo.edge.to(s.cube_pos.device), geo.z_rest.to(s.cube_pos.device)",
         "e, zr = geo.edge.to(s.cube_pos.device), geo.z_rest.to(s.cube_pos.device)[:1].expand(8)"),
        ("zr_t = self.geo.z_rest.to(z_t.device)[tg]", "zr_t = self.geo.z_rest.to(z_t.device)[0].expand_as(z_t)")],
    "m2 released checks only the left pad": [
        ("self.released = (s.f_l <= C) & (s.f_r <= C) & (s.f_p <= C) & (s.f_a <= C)", "self.released = (s.f_l <= C)")],
    "m3 stack stability window 1 step instead of 50": [("k_stack: int = 50", "k_stack: int = 1")],
    "m4 slam read from the cube-table force": [
        ("self.slam |= s.f_ht > th.slam_n", "self.slam |= s.f_ct.amax(-1) > th.slam_n")],
    "m5 pad-region test in the world frame instead of the pad frame": [
        ("local = to_frame(c_pos[:, :, None, :], pad_pos[:, None, :, :], quat_to_mat(pad_quat)[:, None])",
         "local = c_pos[:, :, None, :] - pad_pos[:, None, :, :]")],
}


def pytest(test: str) -> tuple[int, list[str]]:
    r = subprocess.run([sys.executable, "-m", "pytest", test, "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode, sorted(set(re.findall(r"FAILED [\w/]+\.py::(\w+)", r.stdout)))


def main() -> None:
    which = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "detectors"
    src_rel, test, out_name = SETS[which]
    SRC = ROOT / src_rel
    mutants = MUTANTS if which == "detectors" else EXPERT_MUTANTS
    orig = SRC.read_text(encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", src_rel, test], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = [f"# {which} mutants (GATES.md), {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"# git HEAD {sha}{' (UNCOMMITTED CHANGES: ' + dirty + ')' if dirty else ''}; {test} against {src_rel}, one mutation each"]
    code, failed = pytest(test)
    lines.append(f"unmutated: {'GREEN' if code == 0 else 'RED ' + str(failed)}")
    all_killed = code == 0
    try:
        for name, reps in mutants.items():
            src = orig
            for old, new in reps:
                assert src.count(old) == 1, f"{name}: pattern not found exactly once: {old}"
                src = src.replace(old, new)
            SRC.write_text(src, encoding="utf-8", newline="\n")
            code, failed = pytest(test)
            killed = code != 0
            all_killed &= killed
            lines.append(f"{name}: {'KILLED' if killed else 'SURVIVED'} by {failed}")
    finally:
        SRC.write_text(orig, encoding="utf-8", newline="\n")
    lines.append(f"ALL_KILLED={all_killed}")
    out = ROOT / "results" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
