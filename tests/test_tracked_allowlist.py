"""Public-repo guard: every tracked file is on the allowlist (scripts/make_public_tree.py), none > 50 MB, no dataset /
checkpoint / array dump, and every tracked raw job log is hostname-clean once the public-export redaction is applied
(the pushed tree is that redacted export; the identifiers themselves are scrubbed by make_public_tree, not here)."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_public_tree", ROOT / "scripts" / "make_public_tree.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="git/.git absent")


def _tracked_now() -> list[tuple[str, int]]:
    """Working-tree view of `git ls-files` (index), sizes from disk; files removed from disk are skipped."""
    names = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [(n, (ROOT / n).stat().st_size) for n in names.split("\0") if n and (ROOT / n).is_file()]


def test_tracked_files_are_allowlisted() -> None:
    bad = M.violations(_tracked_now())
    assert not bad, "\n".join(bad[:50])


def test_raw_logs_clean_after_redaction() -> None:
    bad = []
    for path, size in _tracked_now():
        if Path(path).suffix.lower() in M.RAW_LOG_EXT or path.startswith("logs/"):
            text, _ = M.redact((ROOT / path).read_text(encoding="utf-8", errors="surrogateescape"))
            hits = {k: v for k, v in M.scan_text(text, path).items() if k not in ("email", "secret_assign")}
            if hits:
                bad.append(f"{path}: {list(hits)}")
    assert not bad, "\n".join(bad[:50])


def test_guard_catches_bad_entries() -> None:
    entries = [("big.json", M.MAX_BYTES + 1), ("results/a.npz", 10), ("data/x.json", 1), ("secret.pt", 1),
               ("stray.docx", 1), ("results/ok.json", 10), ("scripts/ok.py", 10), ("README.md", 5)]
    flagged = {v.split(":")[0] for v in M.violations(entries)}
    assert flagged == {"big.json", "results/a.npz", "data/x.json", "secret.pt", "stray.docx"}
