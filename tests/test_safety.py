from __future__ import annotations

import json

from mka.safety import classify_warnings, excerpt_in_source, print_safety_staple
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
    assert out == [(hit, ["Never exceed 2,100 psi."])]
    assert chat.calls[0][2] is True


def test_classify_warnings_fail_closed_bad_json() -> None:
    hit = make_hit()
    assert classify_warnings(FakeChat("not json"), [hit]) is None
    assert classify_warnings(FakeChat(""), [hit]) is None
    assert classify_warnings(FakeChat(RuntimeError("boom")), [hit]) is None


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


def test_print_safety_staple_dedupes_by_doc_id(capsys) -> None:
    excerpt = "Never exceed 2,100 psi."
    a = make_hit(id="svc::0")
    b = make_hit(id="svc::1")
    print_safety_staple([(a, [excerpt]), (b, [excerpt])])
    out = capsys.readouterr().out
    assert out.count("⚠ SAFETY") == 1
    assert out.count(excerpt) == 1
