from __future__ import annotations

from pathlib import Path

from mka.scan import run_scan

from conftest import make_config


def test_scan_lists_flagged_rows(fixture_corpus: Path, capsys: object) -> None:
    code = run_scan(make_config(fixture_corpus))
    out = capsys.readouterr().out
    assert code == 0
    assert "production-only" in out
    assert "does not search neighbors" in out
    assert "spec_legacy" in out
    assert "2021-03" in out
    assert "spec_current" not in out


def test_scan_missing_corpus_dir(tmp_path: Path, capsys: object) -> None:
    missing = tmp_path / "nope"
    code = run_scan(make_config(missing))
    err = capsys.readouterr().err
    assert code == 1
    assert "corpus directory not found" in err


def test_scan_missing_manifest(tmp_path: Path, capsys: object) -> None:
    code = run_scan(make_config(tmp_path))
    err = capsys.readouterr().err
    assert code == 1
    assert "manifest not found" in err


def test_scan_bad_manifest(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    code = run_scan(make_config(tmp_path))
    err = capsys.readouterr().err
    assert code == 1
    assert "cannot read manifest" in err
