"""Build the public snapshot of this repo: allowlisted tracked files of HEAD, identifiers redacted, re-scanned.

    python scripts/make_public_tree.py --out DIR            # export + redact + scan; exit 1 if anything remains
    python scripts/make_public_tree.py --out DIR --commit   # also write a single-commit orphan branch `public`
    python scripts/make_public_tree.py --scan-history       # report identifier hits over every blob of every commit

`main` is never rewritten. `--commit` uses a temporary index and `--work-tree=DIR`, so nothing is checked out here.
Redaction table (exact, ordered; numbers and results are never touched):
    <project-id> -> <project-id>        <host>.<cluster-domain> / <cluster-domain> -> <cluster-host> / <cluster-domain>
    login0N -> <cluster-login>           gpuNN / cnNN node names -> <node>
    <home>, C:\\\\Users\\\\matus, <home> -> <home>   (the following path separators are kept)
"""

from __future__ import annotations

import argparse
import collections
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- allowlist (also enforced by tests/test_tracked_allowlist.py) ----
ALLOWED_TOP_DIRS = {"docs", "figures", "hpc", "lift_rl", "logs", "media", "ops", "policy", "reports", "results",
                    "scripts", "tests", "videos", "vision_lift", "vla_common", "vla_policy", "vla_tabletop"}
ALLOWED_TOP_FILES = {".gitattributes", ".gitignore", "113_project.md", "113_vla_tabletop_plain.md", "GATES.md",
                     "LICENSE", "LICENSE_111", "README.md", "REPORT.md", "THIRD_PARTY.md", "conftest.py",
                     "context.md", "pyproject.toml", "status.txt", "uv.lock"}
ALLOWED_EXT = {".py", ".js", ".html", ".md", ".txt", ".json", ".jsonl", ".png", ".mp4", ".gif", ".csv", ".toml", ".lock", ".sbatch",
               ".sh", ".ps1", ".def", ".out", ".err", ".log", ".rc"}
ALLOWED_NAMES = {"DONE", ".gitignore", ".gitattributes", "LICENSE", "LICENSE_111"}
FORBIDDEN_EXT = {".npz", ".npy", ".pt", ".pth", ".safetensors", ".h5", ".hdf5", ".parquet", ".onnx", ".ckpt", ".bin",
                 ".pkl", ".pickle", ".sif", ".bundle", ".zip", ".tar", ".gz", ".7z"}
FORBIDDEN_DIRS = {"datasets", "data", "checkpoints", "outputs", ".venv", "__pycache__", "wandb"}
MAX_BYTES = 50 * 1024 * 1024
BINARY_EXT = {".png", ".mp4", ".gif"}
RAW_LOG_EXT = {".out", ".err", ".log", ".rc"}

# ---- redaction table: (name, regex, replacement), applied in order ----
REDACTIONS: list[tuple[str, re.Pattern[str], str]] = [
    ("project_id", re.compile(r"<project-id>"), "<project-id>"),
    ("cluster_host", re.compile(r"\b[A-Za-z0-9-]+\.perun\.tuke\.sk\b"), "<cluster-host>"),
    ("cluster_domain", re.compile(r"\bperun\.tuke\.sk\b"), "<cluster-domain>"),
    ("login_node", re.compile(r"\blogin0\d\b"), "<cluster-login>"),
    ("node", re.compile(r"\b(?:gpu|cn)\d{2}\b"), "<node>"),
    ("win_home", re.compile(r"(?i)\b[a-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)matus\b|/(?:mnt/)?[a-z]/Users/matus\b"), "<home>"),
]

# ---- identifier scan (patterns that must have zero hits in the export) ----
OWNER_EMAILS = {"matus.filo25@gmail.com", "noreply@anthropic.com"}
SCAN: dict[str, re.Pattern[str]] = {
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    "wandb_key": re.compile(r"(?i)wandb_api_key\W{0,4}[0-9a-f]{40}|\bwandb_v1_\w{20,}"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "secret_assign": re.compile(r"(?i)\b(?:token|api[_-]?key|secret|passwd|password)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    "project_id": re.compile(r"perun\d{8}"),
    "cluster_host": re.compile(r"perun\.tuke\.sk"),
    "login_node": re.compile(r"\blogin0\d\b"),
    "node": re.compile(r"\b(?:gpu|cn)\d{2}\b"),
    "private_ip": re.compile(r"(?<![\w.\-])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
                             r"|192\.168\.\d{1,3}\.\d{1,3}|158\.197\.\d{1,3}\.\d{1,3})(?![\w.])"),
    "win_home": re.compile(r"(?i)\b[a-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)matus\b|/(?:mnt/)?[a-z]/Users/matus\b"),
    "home_dir": re.compile(r"(?<![\w.])/(?:home|users)/[a-z][a-z0-9_]+(?=/)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
}
# Account-specific patterns (cluster user name, ...) stay out of the repo: one regex per line in the untracked
# ops/private_scan_patterns.txt (gitignored), scanned as "account_<n>".
_PRIV = ROOT / "ops" / "private_scan_patterns.txt"
if _PRIV.exists():
    for _n, _line in enumerate(x.strip() for x in _PRIV.read_text(encoding="utf-8").splitlines()):
        if _line and not _line.startswith("#"):
            SCAN[f"account_{_n}"] = re.compile(_line)
SCAN_SKIP = {("private_ip", ".lock")} | {("email", e) for e in BINARY_EXT}  # lockfile versions / random bytes in images


def git(*args: str, cwd: Path = ROOT, env: dict | None = None, text: bool = True) -> subprocess.CompletedProcess:
    import os
    e = {**os.environ, **(env or {})}
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=text, check=True, env=e)


def tracked(rev: str = "HEAD") -> list[tuple[str, int]]:
    """[(path, size)] of every file in `rev` (git ls-tree)."""
    out = git("ls-tree", "-r", "-z", "--long", rev).stdout
    res = []
    for rec in out.split("\0"):
        if rec:
            meta, path = rec.split("\t", 1)
            res.append((path, int(meta.split()[3])))
    return res


def violations(entries: list[tuple[str, int]]) -> list[str]:
    """Allowlist violations for [(path, size)]; empty list = pass."""
    bad = []
    for path, size in entries:
        p = Path(path)
        parts = p.parts
        ext = p.suffix.lower()
        if len(parts) == 1 and path not in ALLOWED_TOP_FILES:
            bad.append(f"{path}: top-level file not allowlisted")
        elif len(parts) > 1 and parts[0] not in ALLOWED_TOP_DIRS:
            bad.append(f"{path}: top-level dir not allowlisted")
        if ext in FORBIDDEN_EXT:
            bad.append(f"{path}: forbidden type {ext}")
        elif ext not in ALLOWED_EXT and p.name not in ALLOWED_NAMES:
            bad.append(f"{path}: extension {ext or '(none)'} not allowlisted")
        if FORBIDDEN_DIRS & set(parts[:-1]):
            bad.append(f"{path}: forbidden directory")
        if size > MAX_BYTES:
            bad.append(f"{path}: {size} bytes > 50 MB")
    return bad


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, rx, repl in REDACTIONS:
        text, n = rx.subn(repl, text)
        if n:
            counts[name] = n
    return text, counts


def scan_text(text: str, path: str = "") -> dict[str, list[str]]:
    """{pattern: [matched strings]} (emails of the owner / anthropic noreply are ignored)."""
    ext = Path(path).suffix.lower()
    hits: dict[str, list[str]] = {}
    for name, rx in SCAN.items():
        if (name, ext) in SCAN_SKIP:
            continue
        found = [m.group(0) for m in rx.finditer(text)]
        if name == "email":
            found = [f for f in found if f.lower() not in OWNER_EMAILS]
        if found:
            hits[name] = found
    return hits


def export(out: Path, rev: str = "HEAD") -> dict[str, int]:
    entries = tracked(rev)
    bad = violations(entries)
    if bad:
        raise SystemExit("allowlist violations:\n  " + "\n  ".join(bad))
    if out.exists():
        shutil.rmtree(out)
    totals: collections.Counter[str] = collections.Counter()
    for path, _ in entries:
        data = git("cat-file", "blob", f"{rev}:{path}", text=False).stdout
        if Path(path).suffix.lower() not in BINARY_EXT:
            s, c = redact(data.decode("utf-8", errors="surrogateescape"))
            totals.update(c)
            data = s.encode("utf-8", errors="surrogateescape")
        dest = out / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return dict(totals)


def scan_dir(out: Path) -> dict[str, dict[str, int]]:
    """{pattern: {path: n}} over every file of the export (binaries scanned as latin-1)."""
    res: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for f in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = f.relative_to(out).as_posix()
        for name, found in scan_text(f.read_bytes().decode("latin-1"), rel).items():
            res[name][rel] = len(found)
    return dict(res)


def scan_history() -> None:
    """Every blob of every commit: per-pattern hit count, files, distinct values (top 6); marks 'in HEAD tree'."""
    head = {git("rev-parse", f"HEAD:{p}").stdout.strip(): p for p, _ in tracked()}
    objs = git("rev-list", "--all", "--objects").stdout.splitlines()
    paths = {l.split(" ", 1)[0]: (l.split(" ", 1) + [""])[1] for l in objs}
    batch = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input="\n".join(paths).encode() + b"\n",
                           capture_output=True, check=True).stdout
    cnt: collections.Counter[str] = collections.Counter()
    files: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    vals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    in_head: collections.Counter[str] = collections.Counter()
    pos, nblob = 0, 0
    while pos < len(batch):
        eol = batch.index(b"\n", pos)
        oid, typ, size = batch[pos:eol].decode().split()
        body = batch[eol + 1: eol + 1 + int(size)]
        pos = eol + 1 + int(size) + 1
        if typ != "blob":
            continue
        nblob += 1
        path = paths[oid]
        for name, found in scan_text(body.decode("latin-1"), path).items():
            cnt[name] += len(found)
            files[name][path] += len(found)
            vals[name].update(found)
            if oid in head:
                in_head[name] += len(found)
    print(f"blobs {nblob}, distinct paths {len(set(paths.values()))}")
    for name in SCAN:
        print(f"{name}: hits {cnt[name]}, files {len(files[name])}, hits in HEAD blobs {in_head[name]}")
        if cnt[name]:
            print("   files:", files[name].most_common(6))
            print("   values:", vals[name].most_common(6))


def commit_tree(out: Path, branch: str, msg: str, parent: str | None = None) -> str:
    """Commit `out` as branch `branch` (orphan, or on top of `parent`), without touching this working tree."""
    import tempfile
    idx = Path(tempfile.mkdtemp()) / "index"
    env = {"GIT_INDEX_FILE": str(idx)}
    git("-c", "core.autocrlf=false", f"--work-tree={out}", "add", "-A", "--force", env=env)
    tree = git("write-tree", env=env).stdout.strip()
    c = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", msg).stdout.strip()
    git("update-ref", f"refs/heads/{branch}", c)
    shutil.rmtree(idx.parent, ignore_errors=True)
    return c


def commit_orphan(out: Path, parent: str | None = None) -> str:
    """Branch `public` from `out`: a single orphan commit, or (with `parent`) the next snapshot on top of it."""
    sha = git("rev-parse", "--short", "HEAD").stdout.strip()
    msg = f"franka-arm-vla: public snapshot of 113_vla_tabletop ({sha})\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
    return commit_tree(out, "public", msg, parent)


# GitHub Pages site (branch gh-pages): the page, the demo video and the figures only, no code
PAGES = {"index.html": "docs/pages/index.html", "113_demo.mp4": "media/113_demo.mp4", "fig1.png": "figures/fig1.png",
         "fig2.png": "figures/fig2.png", "fig3.png": "figures/fig3.png", "cards.png": "figures/cards.png"}


def build_pages(out: Path) -> dict[str, dict[str, int]]:
    """Copy the PAGES files (as committed at HEAD) into `out` plus .nojekyll; return scan hits (empty = clean)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for dst, src in PAGES.items():
        (out / dst).write_bytes(git("show", f"HEAD:{src}", text=False).stdout)
    (out / ".nojekyll").write_bytes(b"")
    return scan_dir(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--scan-history", action="store_true")
    ap.add_argument("--parent", help="commit `public` on top of this rev (the current public main) instead of orphan")
    ap.add_argument("--pages", action="store_true", help="build the gh-pages site into --out (with --commit: branch gh-pages)")
    a = ap.parse_args()
    if a.scan_history:
        scan_history()
        return 0
    if a.pages:
        if a.out is None:
            ap.error("--out required")
        left = build_pages(a.out)
        for name, f in left.items():
            print(f"REMAINS {name}: {sum(f.values())} in {len(f)} files, e.g. {list(f.items())[:5]}")
        if left:
            return 1
        print(f"pages clean: {a.out} ({', '.join(sorted(p.name for p in a.out.iterdir()))})")
        if a.commit:
            sha = git("rev-parse", "--short", "HEAD").stdout.strip()
            msg = f"franka-arm-vla: demo page (from 113_vla_tabletop {sha})\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
            print("gh-pages ->", commit_tree(a.out, "gh-pages", msg))
        return 0
    if a.out is None:
        ap.error("--out required")
    totals = export(a.out)
    print("redactions applied:", totals)
    left = scan_dir(a.out)
    for name, f in left.items():
        print(f"REMAINS {name}: {sum(f.values())} in {len(f)} files, e.g. {list(f.items())[:5]}")
    if left:
        return 1
    print(f"export clean: {a.out}")
    if a.commit:
        print("public ->", commit_orphan(a.out, a.parent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
