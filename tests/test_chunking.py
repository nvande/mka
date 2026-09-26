from __future__ import annotations

from pathlib import Path

import pytest

from mka.chunking import (
    MAX_METADATA_BYTES,
    ChunkError,
    _Piece,
    _fit_size,
    attach_warnings,
    chunk_document,
)
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


def test_metadata_starts_with_an_empty_warning_cache() -> None:
    meta = chunk_document(_row(), QA_DOC)[0].metadata
    assert meta["warnings_cached"] is False
    assert meta["contains_warning"] is False
    assert meta["warning_text"] == ""
    assert meta["warning_excerpts"] == []
    assert meta["warning_audiences"] == []


def test_attach_warnings_copies_the_file_pass_onto_a_chunk() -> None:
    chunk = chunk_document(_row(), QA_DOC)[0]
    rows = [("Never exceed 2,100 psi.", "all"), ("Quote 8 weeks.", "sales")]
    assert attach_warnings(chunk, rows) is True
    assert chunk.metadata["warnings_cached"] is True
    assert chunk.metadata["contains_warning"] is True
    assert chunk.metadata["warning_excerpts"] == [
        "Never exceed 2,100 psi.",
        "Quote 8 weeks.",
    ]
    assert chunk.metadata["warning_audiences"] == ["all", "sales"]
    # The raw join keeps every excerpt a verbatim substring for the receipt.
    for text, _ in rows:
        assert text in chunk.metadata["warning_text"]


def test_attach_warnings_marks_a_file_with_no_hazards_as_cached() -> None:
    chunk = chunk_document(_row(), QA_DOC)[0]
    assert attach_warnings(chunk, []) is True
    assert chunk.metadata["warnings_cached"] is True
    assert chunk.metadata["contains_warning"] is False


def test_attach_warnings_leaves_the_record_alone_when_it_will_not_fit() -> None:
    chunk = chunk_document(_row(), QA_DOC)[0]
    before = dict(chunk.metadata)
    assert attach_warnings(chunk, [("x" * MAX_METADATA_BYTES, "all")]) is False
    assert chunk.metadata == before
    assert chunk.metadata["warnings_cached"] is False


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

    monkeypatch.setattr(chunking, "token_len", token_len)
    monkeypatch.setattr(chunking, "MAX_EMBED_TOKENS", 50)
    text = "# Title\n\n**Revision:** 2025-01\n\n## One\nshort\n\n## Two\nshort\n"
    chunks = chunk_document(_row(doc_id="spec_x"), text)
    assert [chunk.id for chunk in chunks] == ["spec_x::0", "spec_x::1", "spec_x::2"]
    assert any(chunk.text.startswith("## One") for chunk in chunks)
    assert any(chunk.text.startswith("## Two") for chunk in chunks)


LIP_CONTROL = """# Service Procedure: MD-7000 Lip Control Troubleshooting

**Equipment:** Meridian MD-7000 Hydraulic Dock Leveler
**Revision:** 2024-08

## Symptom: Lip will not extend
Check lip cylinder wiring harness at junction J3.

## Symptom: Lip will not retract
Inspect the lip return spring tension.

## Symptom: Lip falls short of trailer (incomplete extension)
Verify dock sensor plate alignment.

## Symptom: Lip extends then immediately retracts
Most commonly caused by false trigger on the lip limit switch.

## Related parts
- Lip cylinder seal kit: MD7-LCS-12
"""

DOCKGUARD = """# Service Procedure: DockGuard Vehicle Restraint Fault Diagnostics

**Equipment:** DockGuard Vehicle Restraint (all revisions)
**Revision:** 2024-10

## Fault code reference

### E01 — No trailer detected
Control panel does not register a trailer at the dock.

### E02 — Hook will not extend
Restraint hook fails to move toward engagement position.

### E03 — Hook will not retract
Hook stays engaged after release command.

### E04 — Light tree communication fault
Inside control panel cannot communicate with outside red/green light tree.

### E05 — Emergency stop engaged
E-stop button has been pressed at the dock.
"""

PM_CHECKLIST = """# Annual Preventive Maintenance Checklist

**Frequency:** Annually

## Purpose
Annual PM is required to maintain warranty coverage.

## Dock levelers (all models)
- Inspect all hinge points
### Hydraulic models (MD-7000)
- Check hydraulic fluid level

## Industrial doors
- Inspect lift cables

## Vehicle restraints (DockGuard)
- Verify hook engagement

## Electrical (all equipment)
- Torque-check all line-side connections

## Documentation
All items above must be documented in the equipment maintenance log.
"""

SINGLE_PROCEDURE = """# Service Procedure: MD-7000 Hydraulic Pressure Reset

**Equipment:** Meridian MD-7000 Hydraulic Dock Leveler
**Revision:** 2025-01

## Symptoms indicating pressure reset may be required
- Leveler fails to lower under load

## Required tools
- 9/16" open-end wrench

## Procedure
### Step 1 — Isolate power
Engage the dock leveler disconnect.

## Safety limits
- Never exceed 2,100 psi.

## Documentation
Log date, technician name, and resulting pressure.
"""

SPEC_WITH_TOPICS = """# Product spec

**Revision:** 2025-01

## Dock levelers
Capacity table.

## Industrial doors
Speed table.

## Vehicle restraints
Hook table.
"""


def _service_row(doc_id: str, **overrides: object) -> ManifestRow:
    return _row(
        doc_id=doc_id,
        path=f"docs/{doc_id}.md",
        title=doc_id,
        doc_type="service",
        audience="technician",
        **overrides,
    )


def test_service_symptoms_staple_preamble_and_related_parts() -> None:
    chunks = chunk_document(_service_row("service_md7000_lip_control"), LIP_CONTROL)
    assert [chunk.id for chunk in chunks] == [
        "service_md7000_lip_control::s0",
        "service_md7000_lip_control::s1",
        "service_md7000_lip_control::s2",
        "service_md7000_lip_control::s3",
    ]
    assert all("**Equipment:**" in chunk.text for chunk in chunks)
    assert all("## Related parts" in chunk.text for chunk in chunks)
    assert all("MD7-LCS-12" in chunk.text for chunk in chunks)
    assert all(chunk.metadata["parent_title"] == "service_md7000_lip_control" for chunk in chunks)
    assert chunks[0].text.count("## Symptom:") == 1
    assert "Lip will not extend" in chunks[0].text
    assert "Lip will not retract" not in chunks[0].text
    assert "Lip will not retract" in chunks[1].text
    assert "Lip will not extend" not in chunks[1].text


def test_service_faults_keep_preamble_and_isolate_codes() -> None:
    chunks = chunk_document(_service_row("service_dockguard_diagnostics"), DOCKGUARD)
    assert [chunk.id for chunk in chunks] == [
        "service_dockguard_diagnostics::s0",
        "service_dockguard_diagnostics::s1",
        "service_dockguard_diagnostics::s2",
        "service_dockguard_diagnostics::s3",
        "service_dockguard_diagnostics::s4",
    ]
    assert all("**Equipment:**" in chunk.text for chunk in chunks)
    assert all("## Fault code reference" in chunk.text for chunk in chunks)
    assert "### E01" in chunks[0].text
    assert "Hook will not extend" not in chunks[0].text
    assert "### E02" in chunks[1].text
    assert "No trailer detected" not in chunks[1].text


def test_service_pm_topics_share_purpose_and_documentation() -> None:
    chunks = chunk_document(_service_row("service_annual_pm_checklist"), PM_CHECKLIST)
    assert [chunk.id for chunk in chunks] == [
        "service_annual_pm_checklist::s0",
        "service_annual_pm_checklist::s1",
        "service_annual_pm_checklist::s2",
        "service_annual_pm_checklist::s3",
    ]
    assert all("## Purpose" in chunk.text for chunk in chunks)
    assert all("## Documentation" in chunk.text for chunk in chunks)
    assert "Dock levelers" in chunks[0].text
    assert "Hydraulic models" in chunks[0].text
    assert "Industrial doors" in chunks[1].text
    assert "Hydraulic models" not in chunks[1].text
    assert "Vehicle restraints" in chunks[2].text
    assert "Electrical" in chunks[3].text


def test_single_procedure_service_file_stays_one_chunk() -> None:
    chunks = chunk_document(_service_row("service_md7000_hydraulic_reset"), SINGLE_PROCEDURE)
    assert [chunk.id for chunk in chunks] == ["service_md7000_hydraulic_reset::0"]
    assert "## Procedure" in chunks[0].text
    assert "## Safety limits" in chunks[0].text


def test_spec_with_several_h2_stays_whole_file() -> None:
    chunks = chunk_document(_row(doc_id="spec_multi", doc_type="spec"), SPEC_WITH_TOPICS)
    assert [chunk.id for chunk in chunks] == ["spec_multi::0"]
    assert "## Dock levelers" in chunks[0].text
    assert "## Vehicle restraints" in chunks[0].text


def test_one_symptom_stays_whole_file() -> None:
    text = """# Service

**Revision:** 2024-08

## Symptom: Lip will not extend
Check the harness.

## Related parts
- MD7-LCS-12
"""
    chunks = chunk_document(_service_row("service_one_symptom"), text)
    assert [chunk.id for chunk in chunks] == ["service_one_symptom::0"]


def test_oversize_service_issue_restaples_preamble_and_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    import mka.chunking as chunking

    def token_len(text: str) -> int:
        return 100 if "## One" in text and "## Two" in text else 5

    monkeypatch.setattr(chunking, "token_len", token_len)
    monkeypatch.setattr(chunking, "MAX_EMBED_TOKENS", 50)
    row = _service_row("service_big")
    piece = _Piece(
        "service_big::s0",
        "**Equipment:** MD-7000\n## One\nshort\n## Two\nshort\n## Related parts\nkits\n",
        "**Equipment:** MD-7000\n",
        "## Related parts\nkits\n",
    )
    chunks = _fit_size(row, piece)
    assert [chunk.id for chunk in chunks] == ["service_big::s0::0", "service_big::s0::1"]
    assert all("**Equipment:**" in chunk.text for chunk in chunks)
    assert all("## Related parts" in chunk.text for chunk in chunks)
    assert "## Two" not in chunks[0].text
    assert "## One" not in chunks[1].text


def test_corpus_service_sanity() -> None:
    docs = Path(__file__).resolve().parents[1] / "corpus" / "docs"
    cases = [
        ("service_md7000_lip_control.md", "service_md7000_lip_control", 4, "::s"),
        ("service_dockguard_diagnostics.md", "service_dockguard_diagnostics", 5, "::s"),
        ("service_annual_pm_checklist.md", "service_annual_pm_checklist", 4, "::s"),
        ("service_md7000_hydraulic_reset.md", "service_md7000_hydraulic_reset", 1, "::0"),
        ("service_rapidroll_photoeye.md", "service_rapidroll_photoeye", 1, "::0"),
        ("service_thermaguard_spring.md", "service_thermaguard_spring", 1, "::0"),
    ]
    if not all((docs / name).is_file() for name, *_ in cases):
        pytest.skip("corpus not present")
    lip = chunk_document(
        _service_row("service_md7000_lip_control"),
        (docs / "service_md7000_lip_control.md").read_text(encoding="utf-8"),
    )
    assert all("**Equipment:**" in chunk.text for chunk in lip)
    assert all("## Related parts" in chunk.text for chunk in lip)
    dock = chunk_document(
        _service_row("service_dockguard_diagnostics"),
        (docs / "service_dockguard_diagnostics.md").read_text(encoding="utf-8"),
    )
    assert "Hook will not extend" not in dock[0].text
    pm = chunk_document(
        _service_row("service_annual_pm_checklist"),
        (docs / "service_annual_pm_checklist.md").read_text(encoding="utf-8"),
    )
    assert "Hydraulic models" in pm[0].text
    assert "Hydraulic models" not in pm[1].text
    for name, doc_id, count, suffix in cases:
        chunks = chunk_document(
            _service_row(doc_id),
            (docs / name).read_text(encoding="utf-8"),
        )
        assert len(chunks) == count, name
        assert all(suffix in chunk.id for chunk in chunks), name
