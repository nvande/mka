from __future__ import annotations

import re
from pathlib import Path

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


def test_questions_keep_the_title_and_stay_isolated() -> None:
    chunks = chunk_document(_row(), QA_DOC)
    assert [chunk.id for chunk in chunks] == [
        "faq_cold_storage::1",
        "faq_cold_storage::2",
        "faq_cold_storage::3",
        "faq_cold_storage::4",
    ]
    assert all(chunk.text.startswith("# FAQ: Cold Storage") for chunk in chunks)
    assert all("**Revision:** 2025-01" in chunk.text for chunk in chunks)
    assert all(chunk.text.count("## Q:") == 1 for chunk in chunks)
    assert "Why air-powered" not in chunks[0].text
    assert "Why air-powered" in chunks[1].text
    assert "recommended configuration" not in chunks[1].text
    assert chunks[2].metadata["title"] == "FAQ: Cold Storage"
    assert chunks[2].metadata["parent_title"] == "FAQ: Cold Storage"


def test_table_faq_is_one_whole_file_chunk() -> None:
    row = _row(doc_id="faq_warranty", path="docs/faq_warranty.md", title="FAQ: Warranty")
    chunks = chunk_document(row, TABLE_FAQ)
    assert len(chunks) == 1
    assert chunks[0].id == "faq_warranty::0"
    assert chunks[0].text.startswith("# FAQ: Warranty Coverage")
    assert "## Dock levelers" in chunks[0].text


def test_staple_extra_h1_fails() -> None:
    with pytest.raises(ChunkError, match="H1"):
        chunk_document(_row(), STAPLED_H1)


def test_oversize_without_headings_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import mka.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_EMBED_TOKENS", 1)
    with pytest.raises(ChunkError, match="exceeds"):
        chunk_document(_row(doc_id="spec_x"), "# Title\n\n**Revision:** 2025-01\n\nword word word\n")


def test_sibling_sections_keep_the_title_and_front_matter() -> None:
    text = "# Title\n\n**Revision:** 2025-01\n\n## One\nshort\n\n## Two\nshort\n"
    chunks = chunk_document(_row(doc_id="spec_x"), text)
    assert [chunk.id for chunk in chunks] == ["spec_x::1", "spec_x::2"]
    assert all(chunk.text.startswith("# Title") for chunk in chunks)
    assert all("**Revision:** 2025-01" in chunk.text for chunk in chunks)
    assert "## One" in chunks[0].text and "## Two" not in chunks[0].text
    assert "## Two" in chunks[1].text and "## One" not in chunks[1].text


def test_fenced_headings_do_not_split_the_outline() -> None:
    text = """# Title

**Revision:** 2025-01

## One

```
# Second title
## Fake
```

alpha

## Two
beta
"""
    chunks = chunk_document(_row(doc_id="fenced"), text)
    assert [chunk.id for chunk in chunks] == ["fenced::1", "fenced::2"]
    assert "## Fake" in chunks[0].text
    assert "beta" not in chunks[0].text
    assert "## One" not in chunks[1].text
    assert chunks[1].text.startswith("# Title")
    assert "**Revision:**" in chunks[1].text


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

### Air-powered models (MD-9000)
- Inspect the air bag

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

### Step 2 — Locate the relief valve
The relief valve is on the power unit.

### Step 6 — Tighten locknut
Torque the locknut.

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


def test_sibling_sections_pull_down_the_title_not_each_other() -> None:
    chunks = chunk_document(_service_row("service_md7000_lip_control"), LIP_CONTROL)
    assert [chunk.id for chunk in chunks] == [
        "service_md7000_lip_control::1",
        "service_md7000_lip_control::2",
        "service_md7000_lip_control::3",
        "service_md7000_lip_control::4",
        "service_md7000_lip_control::5",
    ]
    assert all(chunk.text.startswith("# Service Procedure: MD-7000 Lip Control") for chunk in chunks)
    assert all("**Equipment:**" in chunk.text for chunk in chunks)
    assert all(chunk.metadata["parent_title"] == "service_md7000_lip_control" for chunk in chunks)
    assert chunks[0].text.count("## Symptom:") == 1
    assert "Lip will not extend" in chunks[0].text
    assert "Lip will not retract" not in chunks[0].text
    assert "## Related parts" not in chunks[0].text
    assert "Lip will not retract" in chunks[1].text
    assert "Lip will not extend" not in chunks[1].text
    assert "## Related parts" in chunks[-1].text
    assert "MD7-LCS-12" in chunks[-1].text
    assert "Lip will not extend" not in chunks[-1].text


def test_nested_sections_keep_the_parent_heading() -> None:
    chunks = chunk_document(_service_row("service_dockguard_diagnostics"), DOCKGUARD)
    assert [chunk.id for chunk in chunks] == [
        "service_dockguard_diagnostics::1",
        "service_dockguard_diagnostics::2",
        "service_dockguard_diagnostics::3",
        "service_dockguard_diagnostics::4",
        "service_dockguard_diagnostics::5",
    ]
    assert all("**Equipment:**" in chunk.text for chunk in chunks)
    assert all("## Fault code reference" in chunk.text for chunk in chunks)
    assert "### E01" in chunks[0].text
    assert "Hook will not extend" not in chunks[0].text
    assert "### E02" in chunks[1].text
    assert "No trailer detected" not in chunks[1].text


def test_parent_prose_stays_its_own_chunk_and_children_keep_the_heading() -> None:
    chunks = chunk_document(_service_row("service_annual_pm_checklist"), PM_CHECKLIST)
    assert len(chunks) == 8
    assert all("**Frequency:**" in chunk.text for chunk in chunks)
    purpose = next(chunk for chunk in chunks if "## Purpose" in chunk.text)
    assert "Hydraulic models" not in purpose.text
    assert "## Documentation" not in purpose.text
    levelers = next(chunk for chunk in chunks if "Inspect all hinge points" in chunk.text)
    assert "## Dock levelers (all models)" in levelers.text
    assert "Hydraulic models" not in levelers.text
    hydraulic = next(chunk for chunk in chunks if "### Hydraulic models" in chunk.text)
    assert "## Dock levelers (all models)" in hydraulic.text
    assert "Inspect all hinge points" not in hydraulic.text
    assert "Air-powered" not in hydraulic.text
    doors = next(chunk for chunk in chunks if "Inspect lift cables" in chunk.text)
    assert "## Industrial doors" in doors.text
    assert "## Documentation" in chunks[-1].text
    assert "warranty coverage" not in chunks[-1].text


def test_numbered_steps_stay_with_their_parent_section() -> None:
    chunks = chunk_document(_service_row("service_md7000_hydraulic_reset"), SINGLE_PROCEDURE)
    procedure = next(chunk for chunk in chunks if "### Step 1" in chunk.text)
    assert "### Step 6" in procedure.text
    assert "## Safety limits" not in procedure.text
    assert "## Required tools" not in procedure.text
    safety = next(chunk for chunk in chunks if "## Safety limits" in chunk.text)
    assert safety.text.startswith("# Service Procedure: MD-7000 Hydraulic Pressure Reset")
    assert "**Revision:**" in safety.text
    assert "Step 1" not in safety.text
    assert [chunk.id for chunk in chunks] == [
        "service_md7000_hydraulic_reset::1",
        "service_md7000_hydraulic_reset::2",
        "service_md7000_hydraulic_reset::3",
        "service_md7000_hydraulic_reset::4",
        "service_md7000_hydraulic_reset::5",
    ]


def test_outline_split_ignores_doc_type() -> None:
    spec = chunk_document(_row(doc_id="spec_multi", doc_type="spec"), SPEC_WITH_TOPICS)
    service = chunk_document(_row(doc_id="spec_multi", doc_type="service"), SPEC_WITH_TOPICS)
    assert [chunk.id for chunk in spec] == ["spec_multi::1", "spec_multi::2", "spec_multi::3"]
    assert [chunk.text for chunk in spec] == [chunk.text for chunk in service]
    assert "## Dock levelers" in spec[0].text
    assert "## Industrial doors" not in spec[0].text
    assert "## Vehicle restraints" in spec[2].text
    assert all(chunk.text.startswith("# Product spec") for chunk in spec)
    assert all("**Revision:**" in chunk.text for chunk in spec)


def test_one_section_stays_whole_file() -> None:
    text = """# Service

**Revision:** 2024-08

## Symptom: Lip will not extend
Check the harness.
"""
    chunks = chunk_document(_service_row("service_one_symptom"), text)
    assert [chunk.id for chunk in chunks] == ["service_one_symptom::0"]
    assert "## Symptom: Lip will not extend" in chunks[0].text
    assert "**Revision:**" in chunks[0].text


def test_oversize_sequence_splits_under_the_parent_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    import mka.chunking as chunking

    def token_len(text: str) -> int:
        return 100 if "Step 1" in text and "Step 2" in text else 5

    monkeypatch.setattr(chunking, "token_len", token_len)
    monkeypatch.setattr(chunking, "MAX_EMBED_TOKENS", 50)
    text = """# Title

**Revision:** 2025-01

## Procedure

### Step 1 — A
short

### Step 2 — B
short
"""
    chunks = chunk_document(_row(doc_id="spec_x"), text)
    assert [chunk.id for chunk in chunks] == ["spec_x::1", "spec_x::2"]
    assert all(chunk.text.startswith("# Title") for chunk in chunks)
    assert all("**Revision:**" in chunk.text for chunk in chunks)
    assert all("## Procedure" in chunk.text for chunk in chunks)
    assert "Step 2" not in chunks[0].text
    assert "Step 1" not in chunks[1].text


def test_corpus_service_sanity() -> None:
    docs = Path(__file__).resolve().parents[1] / "corpus" / "docs"
    cases = [
        ("service_md7000_lip_control.md", "service_md7000_lip_control", 5),
        ("service_dockguard_diagnostics.md", "service_dockguard_diagnostics", 5),
        ("service_annual_pm_checklist.md", "service_annual_pm_checklist", 10),
        ("service_md7000_hydraulic_reset.md", "service_md7000_hydraulic_reset", 5),
        ("service_rapidroll_photoeye.md", "service_rapidroll_photoeye", 3),
        ("service_thermaguard_spring.md", "service_thermaguard_spring", 3),
    ]
    if not all((docs / name).is_file() for name, *_ in cases):
        pytest.skip("corpus not present")
    lip = chunk_document(
        _service_row("service_md7000_lip_control"),
        (docs / "service_md7000_lip_control.md").read_text(encoding="utf-8"),
    )
    assert all("**Equipment:**" in chunk.text for chunk in lip)
    assert all(chunk.text.startswith("# Service Procedure: MD-7000 Lip Control") for chunk in lip)
    assert "## Related parts" in lip[-1].text
    assert "## Related parts" not in lip[0].text
    dock = chunk_document(
        _service_row("service_dockguard_diagnostics"),
        (docs / "service_dockguard_diagnostics.md").read_text(encoding="utf-8"),
    )
    assert all("## Fault code reference" in chunk.text for chunk in dock)
    assert "Hook will not extend" not in dock[0].text
    pm = chunk_document(
        _service_row("service_annual_pm_checklist"),
        (docs / "service_annual_pm_checklist.md").read_text(encoding="utf-8"),
    )
    levelers = next(chunk for chunk in pm if "Inspect all hinge points" in chunk.text)
    hydraulic = next(chunk for chunk in pm if "### Hydraulic models" in chunk.text)
    air = next(chunk for chunk in pm if "### Air-powered models" in chunk.text)
    assert "Hydraulic models" not in levelers.text
    assert "## Dock levelers (all models)" in hydraulic.text
    assert "Air-powered" not in hydraulic.text
    assert "## Dock levelers (all models)" in air.text
    assert all("**Frequency:**" in chunk.text for chunk in pm)
    for name, doc_id, count in cases:
        chunks = chunk_document(
            _service_row(doc_id),
            (docs / name).read_text(encoding="utf-8"),
        )
        assert len(chunks) == count, name
        assert all(re.fullmatch(rf"{doc_id}::[1-9]\d*", chunk.id) for chunk in chunks), name
