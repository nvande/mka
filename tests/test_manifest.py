from __future__ import annotations

import json
from pathlib import Path

import pytest

from mka.types import load_manifest


def test_load_manifest_reads_closed_fields(fixture_corpus: Path) -> None:
    rows = load_manifest(fixture_corpus)
    by_id = {row.doc_id: row for row in rows}
    assert set(by_id) == {"spec_current", "spec_legacy"}
    assert by_id["spec_legacy"].flagged_outdated is True
    assert by_id["spec_current"].flagged_outdated is False
    assert by_id["spec_current"].version == "2025-01"
    assert not hasattr(by_id["spec_current"], "notes")


def test_load_manifest_missing_documents_key(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError):
        load_manifest(tmp_path)


def test_load_manifest_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_manifest(tmp_path)
