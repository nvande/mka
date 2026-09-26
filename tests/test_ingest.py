from __future__ import annotations

import json
from pathlib import Path

import pytest

from mka.ingest import prepare_chunks, run_ingest
from mka.safety import WarningExcerpt

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
    out = capsys.readouterr().out
    assert "chunks: 2" in out
    assert "--- stats ---" not in out


def test_run_ingest_stats_reports_when_no_api_calls(
    fixture_corpus: Path, skip_index: None, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert run_ingest(make_config(fixture_corpus), stats=True) == 0
    out = capsys.readouterr().out
    assert "chunks: 2" in out
    assert out.index("chunks: 2") < out.index("--- stats ---")
    assert "workflow: ingest" in out
    assert "token_cost_usd: 0.00000000" in out
    assert "pinecone_calls: 0" in out
    assert "latency_ms:" in out


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


def test_run_ingest_caches_warnings_on_every_chunk_of_a_file(
    fixture_corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    seen: list[str] = []

    def fake_classify(chat, *, doc_id, title, path, text):
        del chat, title, path, text
        seen.append(doc_id)
        return [WarningExcerpt("Never exceed 2,100 psi.", "technician")]

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("mka.ingest.classify_source_warnings", fake_classify)
    monkeypatch.setattr("mka.ingest.make_chat", lambda cfg: object())
    written: list[list] = []
    monkeypatch.setattr("mka.ingest._write_index", lambda cfg, chunks: written.append(chunks))

    assert run_ingest(make_config(fixture_corpus)) == 0
    # One classify per source file, not per chunk.
    assert sorted(seen) == ["spec_current", "spec_legacy"]
    for chunk in written[0]:
        assert chunk.metadata["warnings_cached"] is True
        assert chunk.metadata["warning_excerpts"] == ["Never exceed 2,100 psi."]
        assert chunk.metadata["warning_audiences"] == ["technician"]
    assert "warnings: cached on 2/2 chunks" in capsys.readouterr().out


def test_run_ingest_leaves_the_cache_empty_when_the_classifier_fails(
    fixture_corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "mka.ingest.classify_source_warnings", lambda chat, **kwargs: None
    )
    monkeypatch.setattr("mka.ingest.make_chat", lambda cfg: object())
    written: list[list] = []
    monkeypatch.setattr("mka.ingest._write_index", lambda cfg, chunks: written.append(chunks))

    # Still upserts. Check 5 on ask is the gate for these chunks.
    assert run_ingest(make_config(fixture_corpus)) == 0
    assert all(not chunk.metadata["warnings_cached"] for chunk in written[0])
    assert "warnings: cached on 0/2 chunks" in capsys.readouterr().out


def test_run_ingest_skips_warning_cache_without_key(
    fixture_corpus: Path, skip_index: None, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert run_ingest(make_config(fixture_corpus)) == 0
    assert "warnings:" not in capsys.readouterr().out


def test_missing_corpus_dir(tmp_path: Path, capsys: object) -> None:
    missing = tmp_path / "nope"
    assert run_ingest(make_config(missing)) == 1
    captured = capsys.readouterr()
    assert "corpus directory not found" in captured.err
    assert "chunks: 0" in captured.out
