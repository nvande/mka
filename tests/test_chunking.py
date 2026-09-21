from __future__ import annotations

import pytest

from mka.chunking import ChunkError, chunk_document
from mka.types import ManifestRow

QA_DOC = """# FAQ: Cold Storage

**Audience:** All internal users
**Revision:** 2025-01

## Q: What's your recommended configuration for a -10°F freezer dock?
**A:** The standard recommendation is an MD-9000 air-powered dock leveler.

## Q: Why air-powered instead of hydraulic in cold storage?
**A:** Hydraulic fluid viscosity increases at low temperatures.

## Q: Do we need special finishes for food facilities?
**A:** Stainless steel lip option (+$1,200) is recommended for wash-down.

## Q: How do we handle condensation and frost?
**A:** Thermal separation is the primary mitigation.
"""

TABLE_FAQ = """# FAQ: Warranty Coverage

**Audience:** All internal users
**Revision:** 2025-01

## Dock levelers

| Component | Warranty period |
|---|---|
| Platform and frame (structural) | 5 years |
"""

STAPLED_H1 = """# First title

**Revision:** 2025-01

# Second title

Body.
"""

STAPLED_REVISION = """# Title

**Revision:** 2025-01

Revision: 2024-01

Body.
"""


def _row(**overrides: object) -> ManifestRow:
    fields: dict = {
        "doc_id": "faq_cold_storage",
        "path": "docs/faq_cold_storage.md",
        "title": "FAQ: Cold Storage",
        "doc_type": "faq",
        "audience": "all",
        "model": "multi",
        "last_updated": "2025-01-15",
        "version": "2025-01",
        "flagged_outdated": False,
    }
    fields.update(overrides)
    return ManifestRow(**fields)


def test_qa_shape_drops_preamble_and_numbers_questions() -> None:
    chunks = chunk_document(_row(), QA_DOC)
    assert [chunk.id for chunk in chunks] == [
        "faq_cold_storage::q0",
        "faq_cold_storage::q1",
        "faq_cold_storage::q2",
        "faq_cold_storage::q3",
    ]
    assert all(chunk.text.startswith("## Q:") for chunk in chunks)
    assert not any(chunk.text.startswith("# FAQ") for chunk in chunks)
    assert chunks[2].metadata["title"] == "FAQ: Cold Storage"
    assert chunks[2].metadata["parent_title"] == "FAQ: Cold Storage"


def test_qa_pricing_only_on_stainless_lip_question() -> None:
    chunks = chunk_document(_row(), QA_DOC)
    flags = {chunk.id: chunk.metadata["contains_pricing"] for chunk in chunks}
    assert flags == {
        "faq_cold_storage::q0": False,
        "faq_cold_storage::q1": False,
        "faq_cold_storage::q2": True,
        "faq_cold_storage::q3": False,
    }


def test_table_faq_is_one_whole_file_chunk() -> None:
    row = _row(doc_id="faq_warranty", path="docs/faq_warranty.md", title="FAQ: Warranty")
    chunks = chunk_document(row, TABLE_FAQ)
    assert len(chunks) == 1
    assert chunks[0].id == "faq_warranty::0"
    assert chunks[0].text.startswith("# FAQ: Warranty Coverage")
    assert "## Dock levelers" in chunks[0].text
    assert chunks[0].metadata["contains_pricing"] is False


def test_staple_extra_h1_fails() -> None:
    with pytest.raises(ChunkError, match="H1"):
        chunk_document(_row(), STAPLED_H1)


def test_staple_extra_revision_fails() -> None:
    with pytest.raises(ChunkError, match="Revision"):
        chunk_document(_row(), STAPLED_REVISION)


def test_oversize_without_headings_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import mka.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_EMBED_TOKENS", 1)
    with pytest.raises(ChunkError, match="exceeds"):
        chunk_document(_row(doc_id="spec_x"), "# Title\n\n**Revision:** 2025-01\n\nword word word\n")


def test_oversize_splits_on_existing_h2(monkeypatch: pytest.MonkeyPatch) -> None:
    import mka.chunking as chunking

    def token_len(text: str) -> int:
        return 100 if "## One" in text and "## Two" in text else 5

    monkeypatch.setattr(chunking, "_token_len", token_len)
    monkeypatch.setattr(chunking, "MAX_EMBED_TOKENS", 50)
    text = "# Title\n\n**Revision:** 2025-01\n\n## One\nshort\n\n## Two\nshort\n"
    chunks = chunk_document(_row(doc_id="spec_x"), text)
    assert [chunk.id for chunk in chunks] == ["spec_x::0", "spec_x::1", "spec_x::2"]
    assert any(chunk.text.startswith("## One") for chunk in chunks)
    assert any(chunk.text.startswith("## Two") for chunk in chunks)
