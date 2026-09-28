from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mka.chunking import MAX_METADATA_BYTES, chunk_document
from mka.safety import (
    WarningExcerpt,
    assign_warnings,
    attach_warnings,
    cached_warnings,
    excerpt_visible,
    extract_warnings,
    print_safety_staple,
    resolve_warnings,
)
from mka.types import Hit, ManifestRow

CORPUS_DOCS = Path(__file__).resolve().parents[1] / "corpus" / "docs"

SERVICE_DOC = """# Hydraulic Reset

**Classification:** Technician only — do not distribute to operators
**Revision:** 2025-01

> ⚠ **DANGER — STORED PRESSURE**
> Bleed the circuit before opening any fitting.

## Symptom: Platform drifts

Check the relief valve.

## Symptom: Pump runs hot

Check fluid level.

## Safety limits

- **Never** exceed 2,100 psi.
- If pressure remains erratic, stop and escalate.

## Contact

Call service engineering.
"""

SPEC_DOC = """# MD-7000 Specification

## Safety features

- Velocity fuse
- Lip lock

## Safety

Meets ANSI MH30.1.
"""


def _hit(**over: object) -> Hit:
    data: dict = {
        "id": "svc::0",
        "score": 0.9,
        "text": "Body.",
        "title": "Hydraulic Reset",
        "path": "docs/svc.md",
        "doc_id": "svc",
        "model": "MD-7000",
        "doc_type": "service",
        "flagged_outdated": False,
        "contains_warning": False,
        "warning_text": "",
    }
    data.update(over)
    return Hit(**data)


def _row(**over: object) -> ManifestRow:
    fields: dict = {
        "doc_id": "svc",
        "path": "docs/svc.md",
        "title": "Hydraulic Reset",
        "doc_type": "service",
        "audience": "technician",
        "model": "MD-7000",
        "last_updated": "2025-01-10",
        "version": "2025-01",
        "flagged_outdated": False,
    }
    fields.update(over)
    return ManifestRow(**fields)


# extract_warnings


def test_extract_finds_every_marked_note_in_order() -> None:
    found = extract_warnings(SERVICE_DOC)
    assert found == [
        WarningExcerpt(
            "**Classification:** Technician only — do not distribute to operators",
            "technician",
        ),
        WarningExcerpt(
            "> ⚠ **DANGER — STORED PRESSURE**\n> Bleed the circuit before opening any fitting.",
            "all",
        ),
        WarningExcerpt(
            "- **Never** exceed 2,100 psi.\n- If pressure remains erratic, stop and escalate.",
            "all",
        ),
    ]


def test_extract_returns_verbatim_substrings() -> None:
    for item in extract_warnings(SERVICE_DOC):
        assert item.text in SERVICE_DOC


def test_extract_ignores_spec_catalog_lists() -> None:
    assert extract_warnings(SPEC_DOC) == []


def test_extract_ignores_ordinary_prose_and_plain_blockquotes() -> None:
    text = (
        "# FAQ\n\n## Q: Should we grease the track?\n**A:** Do not grease the track;"
        " keep it dry. Never is a strong word.\n\n> A customer said it was fine.\n"
    )
    assert extract_warnings(text) == []


def test_extract_skips_a_bare_audience_line_without_an_instruction() -> None:
    assert extract_warnings("# Doc\n\n**Audience:** Sales team only\n\nBody.\n") == []
    assert extract_warnings("# Doc\n\n**Classification:** Technician only\n\nBody.\n") == []


def test_extract_role_directed_section_carries_its_audience() -> None:
    text = "# Compliance\n\n## Coverage\n\nUL 325.\n\n## Notes for field sales\n\nDo not promise CE.\n"
    assert extract_warnings(text) == [WarningExcerpt("Do not promise CE.", "sales")]


def test_extract_section_stops_at_a_rule_or_same_level_heading() -> None:
    text = "## Critical rules\n\n- Never adjust with the door open.\n\n---\n\nFooter.\n"
    assert extract_warnings(text) == [WarningExcerpt("- Never adjust with the door open.", "all")]
    text = "## What NOT to do\n\n- Never stand on the lip.\n\n### Why\n\nIt tips.\n\n## Next\n\nMore.\n"
    assert extract_warnings(text) == [
        WarningExcerpt("- Never stand on the lip.\n\n### Why\n\nIt tips.", "all")
    ]


def test_extract_over_the_real_corpus_matches_the_marked_hazards() -> None:
    by_file = {
        path.name: extract_warnings(path.read_text(encoding="utf-8"))
        for path in sorted(CORPUS_DOCS.glob("*.md"))
    }
    hydraulic = [item.text for item in by_file["service_md7000_hydraulic_reset.md"]]
    assert any("**Never** exceed 2,100 psi" in text for text in hydraulic)
    spring = [item.text for item in by_file["service_thermaguard_spring.md"]]
    assert any(text.startswith("> ⚠ **DANGER") for text in spring)
    assert any("**Never** adjust springs with the door open" in text for text in spring)
    quickstart = [item.text for item in by_file["operator_quickstart.md"]]
    assert any("**Never** stand on the lip" in text for text in quickstart)
    assert by_file["compliance_certifications.md"][0].audience == "sales"
    # Maintenance advice inside FAQ answers is not a hazard note.
    for name, found in by_file.items():
        if name.startswith("faq_") or name.startswith("spec_"):
            assert found == [], name


# assign_warnings and attach_warnings


def test_assign_puts_a_note_inside_one_chunk_on_that_chunk_only() -> None:
    chunks = chunk_document(_row(), SERVICE_DOC)
    assert len(chunks) == 2
    local = WarningExcerpt("> ⚠ **CAUTION**\n> Pump housing is hot.", "all")
    # Put the note inside the second symptom's body only.
    chunks[1] = replace(chunks[1], text=chunks[1].text.replace("Check fluid level.", local.text))
    assert assign_warnings([local], chunks) == 2
    assert chunks[0].metadata["warning_excerpts"] == []
    assert chunks[1].metadata["warning_excerpts"] == [local.text]


def test_assign_puts_a_note_in_no_chunk_on_every_chunk() -> None:
    chunks = chunk_document(_row(), SERVICE_DOC)
    dropped = WarningExcerpt("Only in the header, which the splitter removed.", "sales")
    assert assign_warnings([dropped], chunks) == 2
    for chunk in chunks:
        assert chunk.metadata["warning_excerpts"] == [dropped.text]
        assert chunk.metadata["warning_audiences"] == ["sales"]
        assert chunk.metadata["warnings_cached"] is True


def test_assign_real_service_doc_puts_shared_notes_on_every_piece() -> None:
    chunks = chunk_document(_row(), SERVICE_DOC)
    assign_warnings(extract_warnings(SERVICE_DOC), chunks)
    for chunk in chunks:
        texts = chunk.metadata["warning_excerpts"]
        assert any(text.startswith("> ⚠ **DANGER") for text in texts)
        assert any("**Never** exceed 2,100 psi" in text for text in texts)
        assert any(text.startswith("**Classification:**") for text in texts)


def test_attach_warnings_copies_rows_onto_a_chunk() -> None:
    chunk = chunk_document(_row(), SERVICE_DOC)[0]
    rows = [("Never exceed 2,100 psi.", "all"), ("Quote 8 weeks.", "sales")]
    assert attach_warnings(chunk, rows) is True
    assert chunk.metadata["warnings_cached"] is True
    assert chunk.metadata["contains_warning"] is True
    assert chunk.metadata["warning_excerpts"] == ["Never exceed 2,100 psi.", "Quote 8 weeks."]
    assert chunk.metadata["warning_audiences"] == ["all", "sales"]
    for text, _ in rows:
        assert text in chunk.metadata["warning_text"]


def test_attach_warnings_marks_a_clean_file_as_cached() -> None:
    chunk = chunk_document(_row(), SERVICE_DOC)[0]
    assert attach_warnings(chunk, []) is True
    assert chunk.metadata["warnings_cached"] is True
    assert chunk.metadata["contains_warning"] is False


def test_attach_warnings_leaves_the_record_alone_when_it_will_not_fit() -> None:
    chunk = chunk_document(_row(), SERVICE_DOC)[0]
    before = dict(chunk.metadata)
    assert attach_warnings(chunk, [("x" * MAX_METADATA_BYTES, "all")]) is False
    assert chunk.metadata == before


# cached_warnings and resolve_warnings


def test_cached_warnings_reads_the_ingest_cache() -> None:
    hit = _hit(
        text="Body.",
        warnings_cached=True,
        warning_text="Header note.",
        warning_excerpts=("Header note.",),
        warning_audiences=("sales",),
    )
    assert cached_warnings([hit]) == [(hit, [WarningExcerpt("Header note.", "sales")])]


def test_cached_warnings_is_none_when_any_record_is_uncached_or_drifted() -> None:
    fresh = _hit(warnings_cached=True, warning_excerpts=(), warning_audiences=())
    uncached = _hit(id="svc::1")
    drifted = _hit(
        id="svc::2",
        warnings_cached=True,
        warning_excerpts=("Not in the record any more.",),
        warning_audiences=("all",),
    )
    assert cached_warnings([fresh]) == [(fresh, [])]
    assert cached_warnings([fresh, uncached]) is None
    assert cached_warnings([fresh, drifted]) is None


def test_resolve_warnings_prefers_the_cache_and_extracts_on_a_miss() -> None:
    cached = _hit(
        text="## Safety limits\n\n- Never exceed 2,100 psi.\n",
        warnings_cached=True,
        warning_excerpts=("- Never exceed 2,100 psi.",),
        warning_audiences=("all",),
    )
    assert resolve_warnings([cached]) == [(cached, [WarningExcerpt("- Never exceed 2,100 psi.", "all")])]
    miss = _hit(text="## Safety limits\n\n- Never exceed 2,100 psi.\n")
    assert resolve_warnings([miss]) == [(miss, [WarningExcerpt("- Never exceed 2,100 psi.", "all")])]


# printing


def test_excerpt_visible_by_audience() -> None:
    assert excerpt_visible("all", "sales")
    assert excerpt_visible("sales", "sales")
    assert not excerpt_visible("sales", "technician")
    assert not excerpt_visible("technician", "sales")


def test_print_safety_staple_dedupes_and_filters_by_role(capsys) -> None:
    first = _hit(id="svc::0")
    second = _hit(id="svc::1")
    shared = WarningExcerpt("> ⚠ DANGER\n> Shared preamble.", "all")
    sales_only = WarningExcerpt("Do not promise CE.", "sales")
    print_safety_staple([(first, [shared, sales_only]), (second, [shared])], "technician")
    out = capsys.readouterr().out
    assert out.count("⚠ WARNING — Hydraulic Reset (docs/svc.md).") == 1
    assert out.count("Shared preamble.") == 1
    assert "Do not promise CE." not in out


def test_print_safety_staple_prints_nothing_when_no_note_is_visible(capsys) -> None:
    hit = _hit()
    print_safety_staple([(hit, [WarningExcerpt("Sales note.", "sales")])], "technician")
    print_safety_staple([(hit, [])], "sales")
    assert capsys.readouterr().out == ""
