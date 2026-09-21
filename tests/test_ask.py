from __future__ import annotations

import json

from mka.ask import (
    BANNER,
    DECIDE_SYS,
    GEN_SYS,
    REDIRECT,
    REFUSE,
    REFUSE_SYS,
    SCOPE_SYS,
    TRIVIAL,
    TRIVIAL_TOO_LONG,
    classify_scope,
    explain_refuse,
    format_refuse,
    outdated_siblings,
    pinecone_filter,
    receipt_ok,
    retrieve,
    run_ask,
    trivial_reject,
)
from mka.safety import WARN_SYS
from mka.types import Hit, ManifestRow

from conftest import make_config


class FakeChat:
    def __init__(self, reply: str | BaseException) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, *, system: str, user: str, json_object: bool = False) -> str:
        self.calls.append((system, user, json_object))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


class ScriptedChat:
    def __init__(self, replies: dict[str, str | BaseException]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, *, system: str, user: str, json_object: bool = False) -> str:
        self.calls.append((system, user, json_object))
        reply = self.replies[system]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeEmbeddings:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2]
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [self.vector]


def make_hit(**over: object) -> Hit:
    data: dict = {
        "id": "spec_md7000::0",
        "score": 0.9,
        "text": "Capacity is 35,000 lbs.",
        "title": "MD-7000 Specification",
        "path": "docs/spec_md7000.md",
        "doc_id": "spec_md7000",
        "model": "MD-7000",
        "doc_type": "spec",
        "contains_warning": False,
        "warning_text": "",
    }
    data.update(over)
    return Hit(**data)


def make_row(**over: object) -> ManifestRow:
    data: dict = {
        "doc_id": "spec_md7000",
        "path": "docs/spec_md7000.md",
        "title": "MD-7000 Specification",
        "doc_type": "spec",
        "audience": "all",
        "model": "MD-7000",
        "last_updated": "2024-01",
        "version": "2024-01",
        "flagged_outdated": False,
    }
    data.update(over)
    return ManifestRow(**data)


def gen_json(answer: str, *ids: str) -> str:
    return json.dumps({"answer": answer, "citation_ids": list(ids)})


def warn_json(*rows: dict) -> str:
    return json.dumps({"warnings": list(rows)})


def refuse_json(reason: str, detail: str = "") -> str:
    return json.dumps({"reason": reason, "detail": detail})


def test_trivial_empty_and_whitespace() -> None:
    assert trivial_reject("", 4000) == TRIVIAL
    assert trivial_reject("   \n", 4000) == TRIVIAL


def test_trivial_too_long() -> None:
    assert trivial_reject("x" * 5, 4) == TRIVIAL_TOO_LONG


def test_trivial_ok() -> None:
    assert trivial_reject("What is MD-9000 capacity?", 4000) is None


def test_scope_allow_only_exact_token() -> None:
    chat = FakeChat("ALLOW")
    assert classify_scope(chat, "MD-9000 capacity?") == "ALLOW"


def test_scope_deny_extra_prose_and_punctuation() -> None:
    assert classify_scope(FakeChat("ALLOW extra"), "q") == "DENY"
    assert classify_scope(FakeChat("ALLOW."), "q") == "DENY"
    assert classify_scope(FakeChat("DENY"), "q") == "DENY"
    assert classify_scope(FakeChat(""), "q") == "DENY"
    assert classify_scope(FakeChat(RuntimeError("boom")), "q") == "DENY"


def test_pinecone_filter_roles() -> None:
    assert pinecone_filter("sales") == {
        "$and": [
            {"flagged_outdated": {"$eq": False}},
            {"doc_type": {"$ne": "service"}},
        ]
    }
    assert pinecone_filter("technician") == {
        "$and": [
            {"flagged_outdated": {"$eq": False}},
            {"doc_type": {"$ne": "pricing"}},
            {"contains_pricing": {"$eq": False}},
        ]
    }


def test_receipt_ok() -> None:
    hits = [make_hit(id="a"), make_hit(id="b")]
    assert receipt_ok(["a"], hits)
    assert receipt_ok(["b", "a"], hits)
    assert not receipt_ok([], hits)
    assert not receipt_ok(["a", "nope"], hits)
    assert not receipt_ok(["made-up"], hits)


def test_outdated_sibling_md7000_spec_only() -> None:
    live = make_hit()
    legacy = make_row(
        doc_id="spec_md7000_legacy",
        path="docs/spec_md7000_legacy.md",
        title="MD-7000 Specification (legacy)",
        version="2019-06",
        flagged_outdated=True,
    )
    current = make_row()
    multi = make_row(
        doc_id="spec_multi_legacy",
        path="docs/spec_multi_legacy.md",
        title="Multi spec (legacy)",
        model="multi",
        flagged_outdated=True,
    )
    faq_legacy = make_row(
        doc_id="faq_md7000_legacy",
        path="docs/faq_md7000_legacy.md",
        title="FAQ (legacy)",
        doc_type="faq",
        flagged_outdated=True,
    )
    siblings = outdated_siblings([live], [legacy, current, multi, faq_legacy])
    assert siblings == [legacy]


def test_run_ask_trivial_prints_banner_no_llm(tmp_path, capsys) -> None:
    chat = FakeChat("ALLOW")
    code = run_ask(make_config(tmp_path), "sales", "  ", chat=chat)
    out = capsys.readouterr().out
    assert code == 0
    assert chat.calls == []
    assert BANNER in out
    assert TRIVIAL in out


def test_run_ask_redirect_on_deny(tmp_path, capsys) -> None:
    chat = FakeChat("DENY")
    code = run_ask(make_config(tmp_path), "sales", "Write me a poem", chat=chat)
    out = capsys.readouterr().out
    assert code == 0
    assert BANNER in out
    assert REDIRECT in out
    assert chat.calls[0][0] == SCOPE_SYS
    assert chat.calls[0][1] == "Write me a poem"
    assert "sales" not in chat.calls[0][0].lower()


def test_run_ask_empty_usable_refuses_without_links(tmp_path, capsys) -> None:
    chat = FakeChat("ALLOW")
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-9000 capacity?",
        chat=chat,
        retriever=lambda query, role: [],
    )
    captured = capsys.readouterr()
    assert code == 0
    assert BANNER in captured.out
    assert REFUSE in captured.out
    assert "Related:" not in captured.out
    assert captured.err == ""
    assert chat.calls == [(SCOPE_SYS, "MD-9000 capacity?", False)]


def test_run_ask_below_floor_refuses_without_links(tmp_path, capsys) -> None:
    hit = make_hit(score=0.19)
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=FakeChat("ALLOW"),
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert REFUSE in out
    assert "Related:" not in out


def test_run_ask_technician_price_belt_drops_hits(tmp_path, capsys) -> None:
    priced = make_hit(id="price::0", text="List price $10,550", score=0.99)
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "What does the MD-9000 cost?",
        chat=FakeChat("ALLOW"),
        retriever=lambda query, role: [priced],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert REFUSE in out
    assert "Related:" not in out
    assert "$10,550" not in out


def test_format_refuse_templates() -> None:
    assert (
        format_refuse("MISSING_INFO", "dock leveler prices")
        == f"{REFUSE} Missing information: dock leveler prices"
    )
    assert format_refuse("AMBIGUOUS", "which dock leveler model") == (
        "The question was too ambiguous to answer. "
        "Please clarify which dock leveler model in your question and ask again."
    )
    assert format_refuse("UNKNOWN", "anything") == REFUSE
    assert format_refuse("MISSING_INFO", "") == REFUSE
    assert format_refuse("NOPE", "dock leveler prices") == REFUSE
    assert format_refuse("MISSING_INFO", "x" * 121) == REFUSE


def test_explain_refuse_fail_closed() -> None:
    hit = make_hit()
    assert explain_refuse(FakeChat("not json"), "cheapest?", [hit]) == REFUSE
    assert explain_refuse(FakeChat(RuntimeError("boom")), "cheapest?", [hit]) == REFUSE
    assert (
        explain_refuse(FakeChat(refuse_json("MISSING_INFO", "")), "cheapest?", [hit])
        == REFUSE
    )


def test_run_ask_decide_refuse_prints_missing_info(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "REFUSE",
            REFUSE_SYS: refuse_json("MISSING_INFO", "a CE mark number"),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "CE number?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"{REFUSE} Missing information: a CE mark number" in out
    assert "Related:" in out
    assert "- MD-7000 Specification (docs/spec_md7000.md)" in out
    assert "Sources:" not in out
    assert [call[0] for call in chat.calls] == [SCOPE_SYS, DECIDE_SYS, REFUSE_SYS]
    assert chat.calls[2][2] is True
    payload = json.loads(chat.calls[2][1])
    assert payload["query"] == "CE number?"
    assert "role" not in payload


def test_run_ask_ambiguous_capacity_without_model(tmp_path, capsys) -> None:
    # Same asked slot, three live specs, three different numbers. Not missing.
    hits = [
        make_hit(
            id="spec_md5000::0",
            title="MD-5000 Specification",
            path="docs/spec_md5000.md",
            doc_id="spec_md5000",
            model="MD-5000",
            text="Rated lifting capacity: 30,000 lbs",
        ),
        make_hit(
            id="spec_md7000::0",
            title="MD-7000 Specification",
            path="docs/spec_md7000.md",
            doc_id="spec_md7000",
            model="MD-7000",
            text="Rated lifting capacity: 35,000 lbs",
        ),
        make_hit(
            id="spec_md9000::0",
            title="MD-9000 Specification",
            path="docs/spec_md9000.md",
            doc_id="spec_md9000",
            model="MD-9000",
            text="Rated lifting capacity: 40,000 lbs",
        ),
    ]
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "REFUSE",
            REFUSE_SYS: refuse_json("AMBIGUOUS", "which dock leveler model"),
        }
    )
    query = "What is the rated lifting capacity of the dock leveler?"
    code = run_ask(
        make_config(tmp_path),
        "sales",
        query,
        chat=chat,
        retriever=lambda q, role: hits,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (
        "The question was too ambiguous to answer. "
        "Please clarify which dock leveler model in your question and ask again."
    ) in out
    assert "Related:" in out
    assert "- MD-5000 Specification (docs/spec_md5000.md)" in out
    assert "- MD-7000 Specification (docs/spec_md7000.md)" in out
    assert "- MD-9000 Specification (docs/spec_md9000.md)" in out
    payload = json.loads(chat.calls[2][1])
    assert payload["query"] == query
    assert [chunk["id"] for chunk in payload["chunks"]] == [
        "spec_md5000::0",
        "spec_md7000::0",
        "spec_md9000::0",
    ]


def test_run_ask_ambiguous_md7000_price_without_config(tmp_path, capsys) -> None:
    # Q5 is answerable only with size and voltage; this query names neither.
    hit = make_hit(
        id="pricing_md7000_2026::0",
        title="MD-7000 Pricing — 2026",
        path="docs/pricing_md7000_2026.md",
        doc_id="pricing_md7000_2026",
        model="MD-7000",
        doc_type="pricing",
        text=(
            "MD-7000, 6 ft × 8 ft, 230V 3-phase: $8,450. "
            "MD-7000, 7 ft × 8 ft, 230V 3-phase: $9,100. "
            "MD-7000, 7 ft × 10 ft, 230V 3-phase: $10,200. "
            "Voltage upgrade to 460V: +$350."
        ),
    )
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "REFUSE",
            REFUSE_SYS: refuse_json(
                "AMBIGUOUS", "which MD-7000 platform size and voltage"
            ),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "What is the list price for an MD-7000?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert (
        "The question was too ambiguous to answer. "
        "Please clarify which MD-7000 platform size and voltage "
        "in your question and ask again."
    ) in out
    assert "Related:" in out
    assert "$8,450" not in out.split("Related:")[0]


def test_run_ask_decide_refuse_unknown_keeps_canned_line(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "REFUSE",
            REFUSE_SYS: refuse_json("UNKNOWN", ""),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "CE number?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert REFUSE in out
    assert "Missing information:" not in out
    assert "ambiguous" not in out.lower()
    assert "Related:" in out


def test_run_ask_bad_citation_hides_answer(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("should not print", "invented-id"),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "should not print" not in out
    assert REFUSE in out
    assert "Related:" in out


def test_run_ask_check5_fail_closed_hides_answer(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("35,000 lbs.", hit.id),
            WARN_SYS: "not-json",
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "35,000 lbs." not in out
    assert REFUSE in out
    assert "Related:" in out


def test_run_ask_happy_path_prints_answer_staple_sources_sibling(
    tmp_path, capsys
) -> None:
    excerpt = "Never exceed 2,100 psi."
    hit = make_hit(text=f"Reset at 1,800 psi. {excerpt}", contains_warning=True)
    legacy = make_row(
        doc_id="spec_md7000_legacy",
        path="docs/spec_md7000_legacy.md",
        title="MD-7000 Specification (legacy)",
        version="2019-06",
        flagged_outdated=True,
    )
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("35,000 lbs.", hit.id),
            WARN_SYS: warn_json({"id": hit.id, "excerpts": [excerpt]}),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [hit],
        rows=[legacy, make_row()],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "35,000 lbs." in out
    assert REFUSE not in out
    assert "⚠ SAFETY — required warnings from MD-7000 Specification (docs/spec_md7000.md)." in out
    assert excerpt in out
    assert f"- MD-7000 Specification (docs/spec_md7000.md, {hit.id})" in out
    assert (
        "Outdated revision (not used): MD-7000 Specification (legacy) — 2019-06 "
        "(spec_md7000_legacy, docs/spec_md7000_legacy.md)"
    ) in out
    systems = [call[0] for call in chat.calls]
    assert systems == [SCOPE_SYS, DECIDE_SYS, GEN_SYS, WARN_SYS]
    assert chat.calls[2][2] is True
    assert chat.calls[3][2] is True
    gen_payload = json.loads(chat.calls[2][1])
    assert gen_payload["query"] == "MD-7000 capacity?"
    assert "role" not in gen_payload


def test_run_ask_clean_warnings_skip_staple(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("35,000 lbs.", hit.id),
            WARN_SYS: warn_json({"id": hit.id, "excerpts": []}),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [hit],
        rows=[],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "35,000 lbs." in out
    assert "SAFETY" not in out
    assert "Sources:" in out


def test_retrieve_uses_role_filter_and_price_belt(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    hits = [
        make_hit(id="ok::0", text="Capacity is 40,000 lbs."),
        make_hit(id="priced::0", text="List price $10,550"),
    ]

    def fake_query(cfg, vector, filter, top_k):
        captured["vector"] = vector
        captured["filter"] = filter
        captured["top_k"] = top_k
        return hits

    monkeypatch.setattr("mka.ask.store.query", fake_query)
    embeddings = FakeEmbeddings([0.3, 0.4])
    cfg = make_config(tmp_path)
    out = retrieve(cfg, embeddings, "capacity?", "technician")
    assert embeddings.texts == ["capacity?"]
    assert captured["vector"] == [0.3, 0.4]
    assert captured["filter"] == pinecone_filter("technician")
    assert captured["top_k"] == cfg.top_k
    assert [hit.id for hit in out] == ["ok::0"]
