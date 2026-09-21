from __future__ import annotations

import json
from pathlib import Path

import pytest

from mka.ingest import prepare_chunks, run_ingest

from conftest import make_config

DOC = "# Title\n\n**Revision:** 2025-01\n\nBody.\n"


def _write_corpus(root: Path, documents: list[dict], files: dict[str, str]) -> Path:
    (root / "docs").mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"documents": documents}),
        encoding="utf-8",
    )
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _row(doc_id: str, **overrides: object) -> dict:
    fields = {
        "doc_id": doc_id,
        "path": f"docs/{doc_id}.md",
        "title": doc_id,
        "doc_type": "spec",
        "audience": "all",
        "model": "MD-7000",
        "last_updated": "2025-01-10",
        "version": "2025-01",
        "flagged_outdated": False,
    }
    fields.update(overrides)
    return fields


def test_prepare_fixture_corpus(fixture_corpus: Path) -> None:
    chunks, errors = prepare_chunks(make_config(fixture_corpus))
    assert errors == []
    assert [chunk.id for chunk in chunks] == ["spec_current::0", "spec_legacy::0"]


@pytest.fixture
def skip_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mka.ingest._write_index", lambda cfg, chunks: None)


def test_run_ingest_prints_count(
    fixture_corpus: Path, skip_index: None, capsys: object
) -> None:
    assert run_ingest(make_config(fixture_corpus)) == 0
    assert "chunks: 2" in capsys.readouterr().out


def test_run_ingest_writes_index(fixture_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_write(cfg, chunks) -> None:
        del cfg
        seen.append(len(chunks))

    monkeypatch.setattr("mka.ingest._write_index", fake_write)
    assert run_ingest(make_config(fixture_corpus)) == 0
    assert seen == [2]


def test_invalid_audience_skips_row(tmp_path: Path, skip_index: None) -> None:
    root = _write_corpus(
        tmp_path,
        [
            _row("good"),
            _row("bad", audience="everyone"),
        ],
        {"docs/good.md": DOC, "docs/bad.md": DOC},
    )
    chunks, errors = prepare_chunks(make_config(root))
    assert [chunk.id for chunk in chunks] == ["good::0"]
    assert any("invalid audience" in err and "bad" in err for err in errors)
    assert run_ingest(make_config(root)) == 1


def test_invalid_version_skips_row(tmp_path: Path) -> None:
    root = _write_corpus(
        tmp_path,
        [_row("bad", version="2025")],
        {"docs/bad.md": DOC},
    )
    chunks, errors = prepare_chunks(make_config(root))
    assert chunks == []
    assert any("invalid version" in err for err in errors)


def test_missing_file_fails(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path, [_row("gone")], {})
    chunks, errors = prepare_chunks(make_config(root))
    assert chunks == []
    assert any("missing file" in err for err in errors)


def test_orphan_doc_fails(tmp_path: Path, skip_index: None) -> None:
    root = _write_corpus(
        tmp_path,
        [_row("kept")],
        {"docs/kept.md": DOC, "docs/extra.md": DOC},
    )
    chunks, errors = prepare_chunks(make_config(root))
    assert [chunk.id for chunk in chunks] == ["kept::0"]
    assert any("orphan" in err and "docs/extra.md" in err for err in errors)
    assert run_ingest(make_config(root)) == 1


def test_staple_fails_row(tmp_path: Path) -> None:
    stapled = "# One\n\n**Revision:** 2025-01\n\n# Two\n\nBody.\n"
    root = _write_corpus(tmp_path, [_row("glued")], {"docs/glued.md": stapled})
    chunks, errors = prepare_chunks(make_config(root))
    assert chunks == []
    assert any("stapled" in err for err in errors)


def test_missing_corpus_dir(tmp_path: Path, capsys: object) -> None:
    missing = tmp_path / "nope"
    assert run_ingest(make_config(missing)) == 1
    captured = capsys.readouterr()
    assert "corpus directory not found" in captured.err
    assert "chunks: 0" in captured.out
