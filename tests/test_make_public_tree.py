"""Redaction table of scripts/make_public_tree.py on synthetic strings."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_public_tree", ROOT / "scripts" / "make_public_tree.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def test_redaction_table() -> None:
    src = ("/project/<project-id>/113_vla_tabletop on <cluster-host> via <cluster-login>; job 95678 on <node> and <node>, "
           "<home>\\A_MAIN\\ws, JSON C:\\\\Users\\\\matus\\\\A_MAIN, <home>/x, <cluster-domain>")
    out, counts = M.redact(src)
    assert out == ("/project/<project-id>/113_vla_tabletop on <cluster-host> via <cluster-login>; job 95678 on <node> and <node>, "
                   "<home>\\A_MAIN\\ws, JSON <home>\\\\A_MAIN, <home>/x, <cluster-domain>")
    assert counts["node"] == 2 and counts["project_id"] == 1
    assert not M.scan_text(out)


def test_numbers_and_results_untouched() -> None:
    s = '{"success": 0.275, "job": 98204, "seed": 21620, "gpu_h": 0.241, "sha": "gpu0123abc", "gpu1": 3, "ncn3": 1}'
    assert M.redact(s)[0] == s


def test_scan_flags_leaks_and_ignores_owner_email() -> None:
    hits = M.scan_text("hf_" + "a" * 36 + " <project-id> <node> " + "10.1.2." + "3 a" + "@b.org " + "matus.filo25@gmail.com noreply@anthropic.com")
    assert set(hits) == {"hf_token", "project_id", "node", "private_ip", "email"}
    assert hits["email"] == ["a" + "@b.org"]
    assert "private_ip" not in M.scan_text('version = "10.4.0.' + '35"', "uv.lock")


def test_pages_site_holds_only_the_page_video_and_figures(tmp_path: Path) -> None:
    left = M.build_pages(tmp_path / "site")
    names = sorted(p.name for p in (tmp_path / "site").iterdir())
    assert names == sorted([*M.PAGES, ".nojekyll"]), names
    assert not left, left
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    for name in M.PAGES:
        assert name == "index.html" or f'"{name}"' in html, name  # every asset is referenced by the page
