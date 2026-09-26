from __future__ import annotations

import json

import pytest

from mka.safety import (
    WarningExcerpt,
    cached_warnings,
    classify_source_warnings,
    classify_warnings,
    excerpt_in_source,
    infer_audience,
    is_catalog_spec,
    print_safety_staple,
    resolve_warnings,
)
from mka.types import Hit


class FakeChat:
    def __init__(self, reply: str | BaseException) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, *, system: str, user: str, json_object: bool = False) -> str:
        self.calls.append((system, user, json_object))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def make_hit(**over: object) -> Hit:
    data: dict = {
        "id": "svc::0",
        "score": 0.9,
        "text": "Reset at 1,800 psi. Never exceed 2,100 psi.",
        "title": "Hydraulic reset",
        "path": "docs/service_md7000_hydraulic_reset.md",
        "doc_id": "service_md7000_hydraulic_reset",
        "model": "MD-7000",
        "doc_type": "service",
        "flagged_outdated": False,
        "contains_warning": True,
        "warning_text": "DANGER: lockout/tagout before work.",
    }
    data.update(over)
    return Hit(**data)


def test_excerpt_in_source_verbatim_keep_paraphrase_drop() -> None:
    hit = make_hit()
    assert excerpt_in_source("Never exceed 2,100 psi.", hit)
    assert excerpt_in_source("DANGER: lockout/tagout before work.", hit)
    assert not excerpt_in_source("do not go over 2100", hit)
    assert not excerpt_in_source("", hit)


def test_classify_warnings_keeps_verbatim() -> None:
    hit = make_hit()
    chat = FakeChat(
        json.dumps(
            {
                "warnings": [
                    {"id": hit.id, "excerpts": ["Never exceed 2,100 psi."]}
                ]
            }
        )
    )
    out = classify_warnings(chat, [hit])
    assert out == [(hit, [WarningExcerpt("Never exceed 2,100 psi.", "all")])]
    assert chat.calls[0][2] is True


def test_classify_warnings_fail_closed_bad_json() -> None:
    hit = make_hit()
    assert classify_warnings(FakeChat("not json"), [hit]) is None
    assert classify_warnings(FakeChat(""), [hit]) is None


def test_classify_warnings_raises_on_a_failed_call() -> None:
    # None is "no warnings found" to the caller. A failed call must not look
    # like a clean pass, so it raises and ask reports it as an error.
    with pytest.raises(RuntimeError, match="boom"):
        classify_warnings(FakeChat(RuntimeError("boom")), [make_hit()])


def test_classify_warnings_fail_closed_invented_id() -> None:
    hit = make_hit()
    chat = FakeChat(json.dumps({"warnings": [{"id": "nope", "excerpts": []}]}))
    assert classify_warnings(chat, [hit]) is None


def test_classify_warnings_fail_closed_when_all_excerpts_fail_receipt() -> None:
    hit = make_hit()
    chat = FakeChat(
        json.dumps({"warnings": [{"id": hit.id, "excerpts": ["paraphrased danger"]}]})
    )
    assert classify_warnings(chat, [hit]) is None


def test_classify_warnings_zero_excerpts_is_clean_pass() -> None:
    hit = make_hit(text="Photo-eye range is 40 ft.", warning_text="")
    chat = FakeChat(json.dumps({"warnings": [{"id": hit.id, "excerpts": []}]}))
    assert classify_warnings(chat, [hit]) == [(hit, [])]


def test_resolve_warnings_reads_the_cache_without_calling_the_model() -> None:
    hit = make_hit(
        warnings_cached=True,
        warning_excerpts=("Never exceed 2,100 psi.",),
        warning_audiences=("technician",),
    )
    chat = FakeChat(RuntimeError("the model must not run"))
    assert resolve_warnings(chat, [hit], "psi?") == [
        (hit, [WarningExcerpt("Never exceed 2,100 psi.", "technician")])
    ]
    assert chat.calls == []


def test_cached_warnings_keeps_the_audience_resolved_at_ingest() -> None:
    # The chunk lost the "For sales" heading in the split. infer_audience
    # would fall back to "all"; the stored value is the one ingest resolved
    # against the whole file.
    hit = make_hit(
        text="Quote lead time as 8 weeks.",
        warning_text="Quote lead time as 8 weeks.",
        warnings_cached=True,
        warning_excerpts=("Quote lead time as 8 weeks.",),
        warning_audiences=("sales",),
    )
    assert infer_audience("Quote lead time as 8 weeks.", hit) == "all"
    assert cached_warnings([hit]) == [
        (hit, [WarningExcerpt("Quote lead time as 8 weeks.", "sales")])
    ]


def test_cached_warnings_zero_excerpts_is_a_clean_pass() -> None:
    hit = make_hit(text="Photo-eye range is 40 ft.", warning_text="", warnings_cached=True)
    assert cached_warnings([hit]) == [(hit, [])]


def test_cached_warnings_falls_back_when_one_chunk_was_never_classified() -> None:
    cached = make_hit(id="a", warnings_cached=True)
    uncached = make_hit(id="b")
    assert cached_warnings([cached, uncached]) is None


def test_cached_warnings_falls_back_when_the_cache_drifted_from_the_record() -> None:
    # The excerpt is no longer verbatim in this record, so the cache is not
    # trusted and the live pass runs instead.
    drifted = make_hit(
        warnings_cached=True,
        warning_excerpts=("Never exceed 1,000 psi.",),
        warning_audiences=("all",),
    )
    assert cached_warnings([drifted]) is None
    mismatched = make_hit(
        warnings_cached=True,
        warning_excerpts=("Never exceed 2,100 psi.",),
        warning_audiences=(),
    )
    assert cached_warnings([mismatched]) is None


def test_resolve_warnings_runs_the_model_on_a_cache_miss() -> None:
    hit = make_hit()
    chat = FakeChat(
        json.dumps({"warnings": [{"id": hit.id, "excerpts": ["Never exceed 2,100 psi."]}]})
    )
    assert resolve_warnings(chat, [hit], "psi?") == [
        (hit, [WarningExcerpt("Never exceed 2,100 psi.", "all")])
    ]
    assert len(chat.calls) == 1


SOURCE_FILE = """# Hydraulic reset

**Revision:** 2025-01

## Safety limits
Never exceed 2,100 psi.

## Notes for field sales
Quote lead time as 8 weeks.
"""


def test_classify_source_warnings_resolves_audience_from_the_file() -> None:
    chat = FakeChat(
        json.dumps(
            {
                "warnings": [
                    {
                        "id": "svc",
                        "excerpts": [
                            "Never exceed 2,100 psi.",
                            "Quote lead time as 8 weeks.",
                        ],
                    }
                ]
            }
        )
    )
    out = classify_source_warnings(
        chat, doc_id="svc", title="Hydraulic reset", path="docs/svc.md", text=SOURCE_FILE
    )
    assert out == [
        WarningExcerpt("Never exceed 2,100 psi.", "all"),
        WarningExcerpt("Quote lead time as 8 weeks.", "sales"),
    ]


def test_classify_source_warnings_no_rows_is_a_clean_pass() -> None:
    chat = FakeChat(json.dumps({"warnings": []}))
    out = classify_source_warnings(
        chat, doc_id="svc", title="Hydraulic reset", path="docs/svc.md", text=SOURCE_FILE
    )
    assert out == []


def test_classify_source_warnings_returns_none_on_an_unusable_reply() -> None:
    for reply in ("not json", ""):
        assert (
            classify_source_warnings(
                FakeChat(reply),
                doc_id="svc",
                title="Hydraulic reset",
                path="docs/svc.md",
                text=SOURCE_FILE,
            )
            is None
        )


def test_classify_source_warnings_raises_on_a_failed_call() -> None:
    # Ingest catches this per file and leaves that file's cache empty, so
    # ask still gates it live rather than shipping an unwarned answer.
    with pytest.raises(RuntimeError, match="boom"):
        classify_source_warnings(
            FakeChat(RuntimeError("boom")),
            doc_id="svc",
            title="Hydraulic reset",
            path="docs/svc.md",
            text=SOURCE_FILE,
        )


SPEC_MD7000 = """## Hydraulic system
- Velocity fuse engages automatically on hydraulic line failure

## Safety features
- Full-width integrated toe guards per ANSI MH30.1
- Automatic velocity fuse (hydraulic)
- Maintenance strut with lockout provision
- Night lock / cross-traffic float position

## Operating environment
- Temperature: -20°F to 120°F
"""


def test_is_catalog_spec_drops_safety_features_keeps_limits() -> None:
    spec = make_hit(text=SPEC_MD7000, warning_text="")
    assert is_catalog_spec("Full-width integrated toe guards per ANSI MH30.1", spec)
    assert is_catalog_spec(
        "## Safety features\n- Full-width integrated toe guards per ANSI MH30.1",
        spec,
    )
    limits = make_hit(
        text="## Safety limits\nNever exceed 2,100 psi.",
        warning_text="",
    )
    assert not is_catalog_spec("Never exceed 2,100 psi.", limits)
    door = make_hit(
        text='## Safety\n- Through-beam photo-eye at 6" above floor\n',
        warning_text="",
    )
    assert is_catalog_spec('Through-beam photo-eye at 6" above floor', door)


def test_classify_warnings_drops_safety_features_without_fail_closed() -> None:
    hit = make_hit(
        text=SPEC_MD7000,
        warning_text="",
        title="MD-7000 Specification",
        path="docs/spec_md7000.md",
        doc_id="spec_md7000",
        doc_type="spec",
    )
    bullets = (
        "- Full-width integrated toe guards per ANSI MH30.1\n"
        "- Automatic velocity fuse (hydraulic)\n"
        "- Maintenance strut with lockout provision\n"
        "- Night lock / cross-traffic float position"
    )
    chat = FakeChat(
        json.dumps({"warnings": [{"id": hit.id, "excerpts": [bullets]}]})
    )
    assert classify_warnings(chat, [hit], "MD-7000 capacity?") == [(hit, [])]


def test_classify_warnings_keeps_hazard_when_mixed_with_features() -> None:
    text = SPEC_MD7000 + "\n## Safety limits\nNever exceed 2,100 psi.\n"
    hit = make_hit(text=text, warning_text="")
    chat = FakeChat(
        json.dumps(
            {
                "warnings": [
                    {
                        "id": hit.id,
                        "excerpts": [
                            "Full-width integrated toe guards per ANSI MH30.1",
                            "Never exceed 2,100 psi.",
                        ],
                    }
                ]
            }
        )
    )
    out = classify_warnings(chat, [hit], "capacity?")
    assert out == [(hit, [WarningExcerpt("Never exceed 2,100 psi.", "all")])]


def test_infer_audience_defaults_all_unless_note_names_a_role() -> None:
    unmarked = make_hit()
    assert infer_audience("Never exceed 2,100 psi.", unmarked) == "all"
    sales = make_hit(
        text=(
            "## Notes for field sales\n\n"
            "Do not represent equipment as meeting standards not listed."
        )
    )
    assert (
        infer_audience(
            "Do not represent equipment as meeting standards not listed.",
            sales,
        )
        == "sales"
    )
    explicit = make_hit(text="For sales: route certification gaps to engineering.")
    assert infer_audience("For sales: route certification gaps to engineering.", explicit) == "sales"
    tech = make_hit(text="## Technician only\n\nLockout before you open the hood.")
    assert infer_audience("Lockout before you open the hood.", tech) == "technician"


def test_classify_warnings_attaches_sales_audience_from_heading() -> None:
    note = "Do not represent equipment as meeting standards not listed."
    hit = make_hit(
        id="comp::0",
        text=f"## Notes for field sales\n\n{note}",
        warning_text="",
        contains_warning=False,
        title="Compliance",
        path="docs/compliance_certifications.md",
        doc_id="compliance_certifications",
        doc_type="reference",
    )
    chat = FakeChat(
        json.dumps({"warnings": [{"id": hit.id, "excerpts": [note]}]})
    )
    out = classify_warnings(chat, [hit])
    assert out == [(hit, [WarningExcerpt(note, "sales")])]


def test_classify_warnings_accepts_object_excerpts() -> None:
    hit = make_hit()
    chat = FakeChat(
        json.dumps(
            {
                "warnings": [
                    {
                        "id": hit.id,
                        "excerpts": [
                            {"text": "Never exceed 2,100 psi.", "audience": "sales"}
                        ],
                    }
                ]
            }
        )
    )
    out = classify_warnings(chat, [hit])
    assert out == [(hit, [WarningExcerpt("Never exceed 2,100 psi.", "all")])]


def test_print_safety_staple_dedupes_by_doc_id(capsys) -> None:
    excerpt = "Never exceed 2,100 psi."
    a = make_hit(id="svc::0")
    b = make_hit(id="svc::1")
    print_safety_staple(
        [
            (a, [WarningExcerpt(excerpt, "all")]),
            (b, [WarningExcerpt(excerpt, "all")]),
        ],
        "sales",
    )
    out = capsys.readouterr().out
    assert out.count("⚠ WARNING") == 1
    assert out.count(excerpt) == 1


def test_print_safety_staple_hides_sales_notes_from_technicians(capsys) -> None:
    shared = WarningExcerpt("Never exceed 2,100 psi.", "all")
    sales_only = WarningExcerpt("Do not represent equipment as CE marked.", "sales")
    hit = make_hit()
    print_safety_staple([(hit, [shared, sales_only])], "technician")
    tech_out = capsys.readouterr().out
    assert "Never exceed 2,100 psi." in tech_out
    assert "Do not represent equipment as CE marked." not in tech_out
    print_safety_staple([(hit, [shared, sales_only])], "sales")
    sales_out = capsys.readouterr().out
    assert "Never exceed 2,100 psi." in sales_out
    assert "Do not represent equipment as CE marked." in sales_out


def test_print_safety_staple_skips_header_when_role_filters_all(capsys) -> None:
    hit = make_hit()
    print_safety_staple(
        [(hit, [WarningExcerpt("Route certification gaps to engineering.", "sales")])],
        "technician",
    )
    assert capsys.readouterr().out == ""
