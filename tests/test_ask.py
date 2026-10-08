from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mka.ask import (
    BANNER,
    DECIDE_SYS,
    GEN_SYS,
    PRICING_DENIED,
    REDIRECT,
    REFUSE,
    SUBJECTIVE,
    REFUSE_SYS,
    SCOPE_CHOICES,
    SCOPE_INSTRUCTIONS,
    TRIVIAL,
    TRIVIAL_TOO_LONG,
    classify_scope,
    answer_supported,
    complete_citations,
    explain_refuse,
    related_sub_chunks,
    note_is_grounded,
    format_refuse,
    outdated_note,
    pinecone_filter,
    retrieve,
    run_ask,
    superseded_hits,
    trivial_reject,
)
from mka.llm import OpenAIChat
from mka.products import catalog_hint
from mka.types import Hit

from conftest import make_config


class FakeChat:
    def __init__(
        self,
        reply: str | BaseException,
        support_reply: str | BaseException = "Supported",
        note_reply: str | BaseException = "Ungrounded",
    ) -> None:
        self.reply = reply
        self.support_reply = support_reply
        self.note_reply = note_reply
        self.calls: list[tuple[str, str, bool]] = []
        self.scope: list[str] = []
        self.scope_instructions: list[str] = []
        self.support: list[str] = []
        self.support_instructions: list[str] = []
        self.notes: list[str] = []
        self.note_instructions: list[str] = []

    def classify(self, text: str, *, instructions: str, choices: list[dict]) -> str:
        if _is_support(choices):
            self.support.append(text)
            self.support_instructions.append(instructions)
            if isinstance(self.support_reply, BaseException):
                raise self.support_reply
            return self.support_reply
        if _is_note(choices):
            self.notes.append(text)
            self.note_instructions.append(instructions)
            if isinstance(self.note_reply, BaseException):
                raise self.note_reply
            return self.note_reply
        self.scope.append(text)
        self.scope_instructions.append(instructions)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply

    def complete(self, *, system: str, user: str, json_object: bool = False) -> str:
        self.calls.append((system, user, json_object))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


class ScriptedChat:
    def __init__(
        self,
        replies: dict[str, str | BaseException],
        scope_label: str = "ALLOW OTHER",
        support_label: str | BaseException = "Supported",
        note_label: str | BaseException = "Ungrounded",
    ) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str, bool]] = []
        self.scope: list[str] = []
        self.scope_label = scope_label
        self.support: list[str] = []
        self.support_label = support_label
        self.notes: list[str] = []
        self.note_label = note_label

    def classify(self, text: str, *, instructions: str, choices: list[dict]) -> str:
        del instructions
        if _is_support(choices):
            self.support.append(text)
            if isinstance(self.support_label, BaseException):
                raise self.support_label
            return self.support_label
        if _is_note(choices):
            self.notes.append(text)
            if isinstance(self.note_label, BaseException):
                raise self.note_label
            return self.note_label
        self.scope.append(text)
        return self.scope_label

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
        return [self.vector for _ in texts]


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
        "flagged_outdated": False,
        "contains_warning": False,
        "warning_text": "",
    }
    data.update(over)
    return Hit(**data)


def _is_support(choices: list[dict]) -> bool:
    return {choice.get("value") for choice in choices} >= {"Supported", "Unsupported"}


def _is_note(choices: list[dict]) -> bool:
    return {choice.get("value") for choice in choices} >= {"Grounded", "Ungrounded"}


def systems(chat) -> list[str]:
    return [call[0] for call in chat.calls]


def only_call(chat, system: str) -> tuple[str, str, bool]:
    matches = [call for call in chat.calls if call[0] == system]
    assert len(matches) == 1, f"expected one {system[:30]!r} call, got {len(matches)}"
    return matches[0]


def assert_decided_and_drafted_together(chat, *, after: str | None) -> None:
    # Generate runs on a worker thread alongside decide, so its position is
    # unspecified. The main-thread sequence is fixed and each call runs once.
    # Scope is a decision, not one of these chat completions.
    seen = systems(chat)
    main_thread = [system for system in seen if system != GEN_SYS]
    assert main_thread == [DECIDE_SYS] + ([after] if after else [])
    assert seen.count(GEN_SYS) == 1
    assert chat.scope


def gen_json(answer: str, *ids: str, outdated_note: str | None = None) -> str:
    return json.dumps(
        {"answer": answer, "citation_ids": list(ids), "outdated_note": outdated_note}
    )


LEGACY = dict(
    id="spec_md7000_legacy::0",
    title="MD-7000 Specification (SUPERSEDED)",
    path="docs/spec_md7000_legacy.md",
    doc_id="spec_md7000_legacy",
    text="Revision: 2021-03. Rated lifting capacity: 30,000 lbs. Motor: 1.5 HP.",
    flagged_outdated=True,
)


def refuse_json(reason: str, detail: str = "") -> str:
    return json.dumps({"reason": reason, "detail": detail})


def test_trivial_empty_and_whitespace() -> None:
    assert trivial_reject("", 4000) == TRIVIAL
    assert trivial_reject("   \n", 4000) == TRIVIAL


def test_trivial_too_long() -> None:
    assert trivial_reject("x" * 5, 4) == TRIVIAL_TOO_LONG


def test_trivial_ok() -> None:
    assert trivial_reject("What is MD-9000 capacity?", 4000) is None


def test_scope_allow_only_exact_label() -> None:
    chat = FakeChat("ALLOW OTHER")
    assert classify_scope(chat, "MD-9000 capacity?") == "ALLOW OTHER"
    assert "Catalog terms" in chat.scope[0]
    assert classify_scope(FakeChat("ALLOW PRICE"), "q") == "ALLOW PRICE"
    assert classify_scope(FakeChat("allow other"), "q") == "ALLOW OTHER"
    assert classify_scope(FakeChat("allow price"), "q") == "ALLOW PRICE"


def test_scope_leveler_use_case_sends_catalog_hint() -> None:
    chat = FakeChat("ALLOW OTHER")
    query = "What is the best leveler for a blast freezer?"
    assert classify_scope(chat, query) == "ALLOW OTHER"
    assert chat.scope[0] == catalog_hint(query)
    assert "leveler → dock leveler" in chat.scope[0]


def test_scope_deny_extra_prose_and_punctuation() -> None:
    assert classify_scope(FakeChat("ALLOW OTHER extra"), "q") == "DENY"
    assert classify_scope(FakeChat("ALLOW OTHER."), "q") == "DENY"
    assert classify_scope(FakeChat("ALLOW PRICE."), "q") == "DENY"
    assert classify_scope(FakeChat("ALLOW"), "q") == "DENY"
    assert classify_scope(FakeChat("DENY"), "q") == "DENY"
    assert classify_scope(FakeChat(""), "q") == "DENY"


def test_scope_raises_rather_than_denying_on_a_failed_call() -> None:
    # DENY here would print the out-of-scope redirect for an API outage and
    # send the user off to reword a question that was fine.
    with pytest.raises(RuntimeError, match="boom"):
        classify_scope(FakeChat(RuntimeError("boom")), "q")


def test_scope_subjective_exact_token() -> None:
    assert classify_scope(FakeChat("SUBJECTIVE"), "q") == "SUBJECTIVE"
    assert classify_scope(FakeChat("subjective"), "q") == "SUBJECTIVE"
    assert classify_scope(FakeChat("SUBJECTIVE."), "q") == "DENY"
    assert classify_scope(FakeChat("SUBJECTIVE extra"), "q") == "DENY"


def test_pinecone_filter_roles() -> None:
    assert pinecone_filter("sales") == {"doc_type": {"$ne": "service"}}
    assert pinecone_filter("technician") == {"doc_type": {"$ne": "pricing"}}


def test_scope_instructions_allow_broad_product_asks() -> None:
    assert "Do not require a model number" in SCOPE_INSTRUCTIONS
    assert "every product we document" in SCOPE_INSTRUCTIONS
    assert "catalog" in SCOPE_INSTRUCTIONS
    assert {choice["value"] for choice in SCOPE_CHOICES} == {
        "ALLOW OTHER",
        "ALLOW PRICE",
        "SUBJECTIVE",
        "DENY",
    }
    assert "ALLOW PRICE when the user is asking about pricing" in SCOPE_INSTRUCTIONS
    assert "does not ask about pricing" in SCOPE_INSTRUCTIONS
    assert "These words refer to our products" in SCOPE_INSTRUCTIONS


def test_decide_and_generate_treat_conflict_as_answer() -> None:
    assert "conflicting evidence" in DECIDE_SYS
    assert "report the disagreement rather than select a value" in DECIDE_SYS
    assert "absence of a statement as a definite negative" in DECIDE_SYS
    assert "flagged_outdated" in GEN_SYS
    assert "Do not use information from chunks where flagged_outdated is true" in GEN_SYS
    assert '"outdated_note": string|null' in GEN_SYS
    assert "do not select one value" in GEN_SYS
    assert "Cite every chunk that directly supports a fact" in GEN_SYS
    assert "disagree about the same fact" in REFUSE_SYS
    assert "air-powered" not in GEN_SYS
    assert "hydraulic" not in GEN_SYS.lower()


def test_answer_supported_sends_the_answer_and_cited_chunks() -> None:
    cited = make_hit(id="spec_md7000::1", text="Capacity is 35,000 lbs.")
    other = make_hit(id="faq::1", text="The lip is 16 in.")
    chat = FakeChat("DENY")
    assert answer_supported(chat, "35,000 lbs.", [cited.id, "invented"], [cited, other])
    sent = chat.support[0]
    assert "Answer:\n35,000 lbs." in sent
    assert "Capacity is 35,000 lbs." in sent
    assert cited.id in sent
    assert "The lip is 16 in." not in sent
    assert "invented" not in sent
    assert "every factual claim" in chat.support_instructions[0]
    assert answer_supported(FakeChat("DENY", "supported"), "35,000 lbs.", [cited.id], [cited])
    assert not answer_supported(FakeChat("DENY", "Unsupported"), "40,000 lbs.", [cited.id], [cited])
    assert not answer_supported(FakeChat("DENY", "Supported."), "35,000 lbs.", [cited.id], [cited])
    assert not answer_supported(FakeChat("DENY", "Supported extra"), "35,000 lbs.", [cited.id], [cited])
    assert not answer_supported(FakeChat("DENY", ""), "35,000 lbs.", [], [cited])
    with pytest.raises(RuntimeError, match="boom"):
        answer_supported(FakeChat("DENY", RuntimeError("boom")), "35,000 lbs.", [cited.id], [cited])


def test_run_ask_trivial_prints_banner_no_llm(tmp_path, capsys) -> None:
    chat = FakeChat("ALLOW OTHER")
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
    assert "--- stats ---" not in out
    assert chat.calls == []
    assert chat.scope == ["Write me a poem"]
    assert "sales" not in chat.scope_instructions[0].lower()
    assert "technician" not in chat.scope_instructions[0].lower()


def test_run_ask_missing_index_explains_ingest(tmp_path, capsys, monkeypatch) -> None:
    class Missing:
        def query(self, **kwargs):
            raise RuntimeError("[404 NOT_FOUND] Resource mka-poc not found")

    monkeypatch.setattr("mka.store._index", lambda cfg: Missing())
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=FakeChat("ALLOW OTHER"),
        embeddings=FakeEmbeddings(),
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "Pinecone index 'mka-poc' does not exist" in captured.err
    assert "mka ingest" in captured.err
    assert "[404" not in captured.err


def test_run_ask_reports_a_failed_gate_as_an_error_not_a_redirect(tmp_path, capsys) -> None:
    # The scope gate blowing up used to print the out-of-scope redirect and
    # exit 0, which is indistinguishable from the assistant declining.
    chat = FakeChat(RuntimeError("Connection error."))
    code = run_ask(make_config(tmp_path), "sales", "MD-7000 price?", chat=chat)
    captured = capsys.readouterr()
    assert code == 1
    assert REDIRECT not in captured.out
    assert "error: Connection error." in captured.err


def test_run_ask_stats_prices_reported_tokens(tmp_path, capsys) -> None:
    class Decisions:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-6-luna"
            assert kwargs["questions"][0]["type"] == "choice"
            return SimpleNamespace(
                answers=[SimpleNamespace(type="choice", choice="DENY")],
                usage=SimpleNamespace(input_tokens=1000),
            )

    client = SimpleNamespace(decisions=Decisions())
    chat = OpenAIChat(client, "gpt-5.4-nano")
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "Tell me a joke",
        chat=chat,
        stats=True,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert REDIRECT in out
    assert out.index(REDIRECT) < out.index("--- stats ---")
    assert "workflow: ask" in out
    assert "chat_calls: 0 model=- prompt_tokens=0 completion_tokens=0" in out
    assert "decision_calls: 1 model=gpt-6-luna prompt_tokens=1000" in out
    # 1000 * $0.10 / 1M, input only
    assert "token_cost_usd: 0.00010000" in out
    assert "pinecone_calls: 0" in out
    assert "latency_ms:" in out


def test_run_ask_subjective_stops_before_retrieve(tmp_path, capsys) -> None:
    query = "What Meridian dock leveler will make me the coolest with my friends on linkedin?"
    called = {"retrieve": False}

    def retriever(q: str, role: str) -> list:
        del q, role
        called["retrieve"] = True
        return [make_hit()]

    chat = FakeChat("SUBJECTIVE")
    code = run_ask(
        make_config(tmp_path),
        "technician",
        query,
        chat=chat,
        retriever=retriever,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert BANNER in out
    assert SUBJECTIVE in out
    assert REDIRECT not in out
    assert REFUSE not in out
    assert "Related:" not in out
    assert called["retrieve"] is False
    assert chat.calls == []
    assert chat.scope == [catalog_hint(query)]
    assert "dock leveler → dock leveler" in chat.scope[0]
    assert "sales" not in chat.scope_instructions[0].lower()
    assert "technician" not in chat.scope_instructions[0].lower()


def test_run_ask_empty_usable_refuses_without_links(tmp_path, capsys) -> None:
    chat = FakeChat("ALLOW OTHER")
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
    assert chat.calls == []
    assert chat.scope == [catalog_hint("MD-9000 capacity?")]
    assert "md-9000 → dock leveler" in chat.scope[0]


def test_run_ask_below_floor_refuses_without_links(tmp_path, capsys) -> None:
    hit = make_hit(score=0.19)
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=FakeChat("ALLOW OTHER"),
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert REFUSE in out
    assert "Related:" not in out


def test_run_ask_technician_allow_price_is_denied_before_retrieval(tmp_path, capsys) -> None:
    called = {"retrieve": False}

    def retriever(query, role):
        called["retrieve"] = True
        return [make_hit()]

    chat = FakeChat("ALLOW PRICE")
    for query in (
        "What is the list price for a 7 ft by 10 ft MD-7000 with 460V?",
        "How much for an MD-9000?",
    ):
        code = run_ask(make_config(tmp_path), "technician", query, chat=chat, retriever=retriever)
        out = capsys.readouterr().out
        assert code == 0
        assert PRICING_DENIED in out
        assert REFUSE not in out
        assert REDIRECT not in out
    assert called["retrieve"] is False
    assert len(chat.scope) == 2
    assert chat.calls == []

    # The words in the question do not decide this. ALLOW OTHER proceeds.
    other = FakeChat("ALLOW OTHER")
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "MD-7000 pricing",
        chat=other,
        retriever=retriever,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert PRICING_DENIED not in out
    assert called["retrieve"] is True

    # Out of scope is still a redirect, including a poem that mentions a price.
    denied = FakeChat("DENY")
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "Write a poem about the MD-7000 price",
        chat=denied,
        retriever=retriever,
    )
    out = capsys.readouterr().out
    assert REDIRECT in out
    assert PRICING_DENIED not in out


def test_run_ask_technician_price_belt_drops_pricing_documents(tmp_path, capsys) -> None:
    # A question with no price word can still pull a pricing document. The belt
    # after retrieval uses the document type.
    priced = make_hit(id="price::0", doc_type="pricing", text="Stainless lip option: $1,200", score=0.99)
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "Which lip option is recommended for wash-down zones?",
        chat=FakeChat("ALLOW OTHER"),
        retriever=lambda query, role: [priced],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert REFUSE in out
    assert "$1,200" not in out


def test_run_ask_sales_price_ask_is_not_denied(tmp_path, capsys) -> None:
    hit = make_hit(id="pricing::0", doc_type="pricing", text="7 ft × 10 ft, 230V: $10,200. 460V: +$350.")
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("$10,550.", hit.id),
        },
        scope_label="ALLOW PRICE",
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "What is the list price for a 7 ft by 10 ft MD-7000 with 460V?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "$10,550." in out
    assert PRICING_DENIED not in out


def test_superseded_hits_match_model_and_doc_type() -> None:
    current = make_hit()
    legacy = make_hit(**LEGACY)
    legacy_pricing = make_hit(**{**LEGACY, "id": "old_price::0", "doc_type": "pricing"})
    other_model = make_hit(**{**LEGACY, "id": "old_9000::0", "model": "MD-9000"})
    usable = [current, legacy, legacy_pricing, other_model]
    assert superseded_hits(usable, [current]) == [legacy]
    # Nothing current cited means nothing is superseded.
    assert superseded_hits(usable, [legacy]) == []


def test_outdated_note_rules() -> None:
    legacy = make_hit(**LEGACY)
    # No superseded hit: the model's note is dropped, whatever it says.
    assert outdated_note("The old spec lists 30,000 lbs.", []) is None
    # An accepted sentence is used as given.
    assert (
        outdated_note("The superseded 2021-03 revision lists 30,000 lbs.", [legacy])
        == "The superseded 2021-03 revision lists 30,000 lbs."
    )
    # No accepted sentence: canned line. Whether a sentence is acceptable is a decision.
    canned = outdated_note(None, [legacy])
    assert canned.startswith("Note: a superseded revision is also on file")
    assert "MD-7000 Specification (SUPERSEDED) (docs/spec_md7000_legacy.md)" in canned
    assert outdated_note("   ", [legacy]) == canned


def test_note_is_grounded_fail_closed() -> None:
    legacy = make_hit(**LEGACY)
    chat = FakeChat("DENY", note_reply="Grounded")
    assert note_is_grounded(chat, "The old spec lists 30,000 lbs.", [legacy])
    sent = chat.notes[0]
    assert "Note:\nThe old spec lists 30,000 lbs." in sent
    assert "30,000 lbs" in sent
    assert legacy.title in sent
    assert "every factual claim" in chat.note_instructions[0]
    assert not note_is_grounded(
        FakeChat("DENY", note_reply="Ungrounded"), "lists 25,000 lbs.", [legacy]
    )
    assert note_is_grounded(
        FakeChat("DENY", note_reply="grounded"), "lists 30,000 lbs.", [legacy]
    )
    assert not note_is_grounded(
        FakeChat("DENY", note_reply="Grounded."), "lists 30,000 lbs.", [legacy]
    )
    assert not note_is_grounded(
        FakeChat("DENY", note_reply="Grounded extra"), "lists 30,000 lbs.", [legacy]
    )
    with pytest.raises(RuntimeError, match="boom"):
        note_is_grounded(
            FakeChat("DENY", note_reply=RuntimeError("boom")),
            "lists 30,000 lbs.",
            [legacy],
        )


def test_run_ask_superseded_revision_always_gets_a_note_and_source(tmp_path, capsys) -> None:
    current = make_hit()
    legacy = make_hit(**LEGACY)
    # The model cites only the current doc and gives no note. Python adds both.
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("35,000 lbs.", current.id),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [current, legacy],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert chat.notes == []
    assert "35,000 lbs." in out
    assert "Note: a superseded revision is also on file" in out
    assert f"- MD-7000 Specification (SUPERSEDED) (docs/spec_md7000_legacy.md, {legacy.id})" in out
    assert out.index("35,000 lbs.") < out.index("Note:") < out.index("Sources:")


def test_naming_the_hydraulic_alternative_does_not_note_a_conflict(tmp_path, capsys) -> None:
    current = make_hit(text="Model MD-7000. Rated capacity 35,000 lbs. Motor 2 HP.")
    legacy = make_hit(
        **{
            **LEGACY,
            "text": "Model MD-7000. Revision 2021-03. Rated lifting capacity: 30,000 lbs.",
        }
    )
    faq = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="The standard recommendation is an MD-9000 air-powered dock leveler.",
    )
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json(
                "Specify the MD-9000. Hydraulic fluid viscosity increases significantly "
                "at low temperatures, slowing the MD-7000 response and stressing seals.",
                faq.id,
            ),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "What equipment should I specify and why?",
        chat=chat,
        retriever=lambda query, role: [faq, current, legacy],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "viscosity" in out
    assert "Note:" not in out
    assert legacy.id not in out


def test_run_ask_superseded_note_from_model_when_grounded(tmp_path, capsys) -> None:
    current = make_hit()
    legacy = make_hit(**LEGACY)
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json(
                "35,000 lbs.",
                current.id,
                legacy.id,
                outdated_note="The superseded 2021-03 revision lists 30,000 lbs.",
            ),
        },
        note_label="Grounded",
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [current, legacy],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "The superseded 2021-03 revision lists 30,000 lbs." in out
    assert "Note: a superseded revision" not in out
    # Cited once, not duplicated by the superseded append.
    assert out.count(legacy.id) == 1
    assert "30,000 lbs" in chat.notes[0]
    assert legacy.text in chat.notes[0]


def test_run_ask_ungrounded_outdated_note_uses_the_canned_line(tmp_path, capsys) -> None:
    current = make_hit()
    legacy = make_hit(**LEGACY)
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json(
                "35,000 lbs.",
                current.id,
                outdated_note="The old revision lists 25,000 lbs.",
            ),
        },
        note_label="Ungrounded",
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [current, legacy],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "35,000 lbs." in out
    assert "25,000" not in out
    assert "Note: a superseded revision is also on file" in out
    assert legacy.text in chat.notes[0]


def test_format_refuse_templates() -> None:
    assert (
        format_refuse("MISSING_INFO", "dock leveler prices")
        == f"{REFUSE} Missing information: dock leveler prices"
    )
    assert format_refuse("AMBIGUOUS", "which dock leveler model") == (
        "The question was too ambiguous to answer. "
        "Please clarify which dock leveler model in your question and ask again."
    )
    assert format_refuse("UNKNOWN", "European market certifications") == (
        f"{REFUSE} Related topic: European market certifications"
    )
    assert format_refuse("UNKNOWN", "") == REFUSE
    assert format_refuse("MISSING_INFO", "") == REFUSE
    assert format_refuse("NOPE", "dock leveler prices") == REFUSE
    assert format_refuse("MISSING_INFO", "x" * 121) == REFUSE


def test_explain_refuse_fail_closed() -> None:
    hit = make_hit()
    assert explain_refuse(FakeChat("not json"), "cheapest?", [hit]) == REFUSE
    with pytest.raises(RuntimeError, match="boom"):
        explain_refuse(FakeChat(RuntimeError("boom")), "cheapest?", [hit])
    assert (
        explain_refuse(FakeChat(refuse_json("MISSING_INFO", "")), "cheapest?", [hit])
        == REFUSE
    )


def test_run_ask_decide_refuse_prints_missing_info(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
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
    # The draft was started alongside decide and discarded on REFUSE.
    assert_decided_and_drafted_together(chat, after=REFUSE_SYS)
    refuse_call = only_call(chat, REFUSE_SYS)
    assert refuse_call[2] is True
    payload = json.loads(refuse_call[1])
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
    payload = json.loads(only_call(chat, REFUSE_SYS)[1])
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
    assert "Related topic:" not in out
    assert "ambiguous" not in out.lower()
    assert "Related:" in out


def test_run_ask_decide_refuse_unknown_prints_related_topic(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            DECIDE_SYS: "REFUSE",
            REFUSE_SYS: refuse_json("UNKNOWN", "European market certifications"),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "Give me the European market certifications of every Meridian leveler",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"{REFUSE} Related topic: European market certifications" in out
    assert "Related:" in out


def test_run_ask_unsupported_hides_answer(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("should not print", hit.id),
        },
        support_label="Unsupported",
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
    assert "should not print" in chat.support[0]
    assert hit.text in chat.support[0]


def test_run_ask_support_failure_is_an_error_not_a_refuse(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("should not print", hit.id),
        },
        support_label=RuntimeError("Connection error."),
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "should not print" not in captured.out
    assert REFUSE not in captured.out
    assert "error: Connection error." in captured.err


def test_run_ask_happy_path_prints_answer_staple_sources(
    tmp_path, capsys
) -> None:
    # No ingest cache on this record, so the staple comes from extracting
    # the chunk's own marked section. No model call is involved.
    excerpt = "- **Never** exceed 2,100 psi."
    hit = make_hit(text=f"Reset at 1,800 psi.\n\n## Safety limits\n\n{excerpt}\n")
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("35,000 lbs.", hit.id),
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
    assert "35,000 lbs." in out
    assert REFUSE not in out
    assert "⚠ WARNING — MD-7000 Specification (docs/spec_md7000.md)." in out
    assert excerpt in out
    assert f"- MD-7000 Specification (docs/spec_md7000.md, {hit.id})" in out
    assert "Outdated revision" not in out
    assert_decided_and_drafted_together(chat, after=None)
    gen_call = only_call(chat, GEN_SYS)
    assert gen_call[2] is True
    gen_payload = json.loads(gen_call[1])
    assert gen_payload["query"] == "MD-7000 capacity?"
    assert "role" not in gen_payload
    assert gen_payload["chunks"][0]["flagged_outdated"] is False
    assert chat.support
    assert "35,000 lbs." in chat.support[0]
    assert hit.text in chat.support[0]


def test_run_ask_stale_cache_falls_back_to_the_chunk_text(tmp_path, capsys) -> None:
    # The cached excerpt no longer appears in the record: the cache drifted.
    # Ask ignores it and extracts from the chunk instead of trusting it.
    hit = make_hit(
        text="Reset at 1,800 psi.\n\n## Safety limits\n\n- Never exceed 2,100 psi.\n",
        warnings_cached=True,
        warning_excerpts=("Stale text that is not in the chunk.",),
        warning_audiences=("all",),
    )
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("1,800 psi.", hit.id),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "Reset pressure?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Stale text" not in out
    assert "- Never exceed 2,100 psi." in out


def test_run_ask_staples_from_the_ingest_cache_without_a_warn_call(
    tmp_path, capsys
) -> None:
    excerpt = "Never exceed 2,100 psi."
    hit = make_hit(
        text=f"Reset at 1,800 psi. {excerpt}",
        contains_warning=True,
        warnings_cached=True,
        warning_excerpts=(excerpt,),
        warning_audiences=("all",),
    )
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("35,000 lbs.", hit.id),
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
    assert "35,000 lbs." in out
    assert excerpt in out
    assert_decided_and_drafted_together(chat, after=None)


def test_run_ask_refuse_discards_a_finished_draft(tmp_path, capsys) -> None:
    # Generate returned a perfectly valid answer. Decide said REFUSE. The
    # answer must not print, and its citations must not appear.
    hit = make_hit()
    chat = ScriptedChat(
        {
            DECIDE_SYS: "REFUSE",
            GEN_SYS: gen_json("35,000 lbs. Trust me.", hit.id),
            REFUSE_SYS: refuse_json("MISSING_INFO", "the asked value"),
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
    assert "Trust me" not in out
    assert "Sources:" not in out
    assert f"{REFUSE} Missing information: the asked value" in out
    assert_decided_and_drafted_together(chat, after=REFUSE_SYS)


def test_run_ask_refuse_ignores_a_draft_that_errored(tmp_path, capsys) -> None:
    # A transport error in the speculative draft is irrelevant on REFUSE.
    hit = make_hit()
    chat = ScriptedChat(
        {
            DECIDE_SYS: "REFUSE",
            GEN_SYS: RuntimeError("draft connection dropped"),
            REFUSE_SYS: refuse_json("UNKNOWN", ""),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    captured = capsys.readouterr()
    assert code == 0
    assert REFUSE in captured.out
    assert captured.err == ""


def test_run_ask_answer_surfaces_a_draft_that_errored(tmp_path, capsys) -> None:
    # Decide said ANSWER but the draft call failed: that is an outage, exit 1.
    hit = make_hit()
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: RuntimeError("draft connection dropped"),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    captured = capsys.readouterr()
    assert code == 1
    assert REFUSE not in captured.out
    assert "error: draft connection dropped" in captured.err


def test_run_ask_clean_warnings_skip_staple(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("35,000 lbs.", hit.id),
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
    assert "35,000 lbs." in out
    assert "⚠ WARNING" not in out
    assert "Sources:" in out


def test_run_ask_hides_sales_note_from_technician(tmp_path, capsys) -> None:
    note = "Do not represent equipment as meeting standards not listed."
    hit = make_hit(
        text=f"## Notes for field sales\n\n{note}",
        contains_warning=False,
        warning_text="",
    )
    chat = ScriptedChat(
        {
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("No CE mark as stock.", hit.id),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "European certifications?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No CE mark as stock." in out
    assert note not in out
    assert "⚠ WARNING" not in out


def test_retrieve_uses_role_filter_and_price_belt(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    hits = [
        make_hit(id="ok::0", text="Capacity is 40,000 lbs."),
        make_hit(id="dollar::0", text="Stainless lip option: $1,200"),
        make_hit(id="priced::0", text="List price $10,550", doc_type="pricing"),
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
    assert [hit.id for hit in out] == ["ok::0", "dollar::0"]


def test_complete_citations_adds_retrieved_docs_that_name_the_same_model() -> None:
    faq = make_hit(
        id="faq::q0",
        doc_id="faq_selection_guide",
        text="The MD-5000 is not rated for true freezer environments. Specify the MD-9000.",
    )
    cold = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        text="The standard recommendation is an MD-9000 air-powered dock leveler.",
    )
    ids = complete_citations([faq.id], "Specify the MD-9000 air-powered model.", [faq, cold])
    assert ids == [faq.id, cold.id]


def test_complete_citations_skips_docs_already_cited() -> None:
    a = make_hit(id="a::0", doc_id="spec_md9000", text="MD-9000 capacity 40,000")
    b = make_hit(id="a::1", doc_id="spec_md9000", text="MD-9000 temperature -40")
    assert complete_citations([a.id], "The MD-9000 is rated to -40°F.", [a, b]) == [a.id]


def test_related_sub_chunks_pulls_all_models_with_hydraulic_or_air_powered(
    tmp_path, monkeypatch
) -> None:
    all_models = make_hit(
        id="service_annual_pm_checklist::s0",
        doc_id="service_annual_pm_checklist",
        doc_type="service",
        text="## Dock levelers (all models)\nInspect all hinge points.",
        score=0.2,
    )
    hydraulic = make_hit(
        id="service_annual_pm_checklist::s1",
        doc_id="service_annual_pm_checklist",
        doc_type="service",
        text="### Hydraulic models\nCheck hydraulic fluid level.",
        score=0.8,
    )
    air = make_hit(
        id="service_annual_pm_checklist::s2",
        doc_id="service_annual_pm_checklist",
        doc_type="service",
        text="### Air-powered models\nInspect the air bag.",
        score=0.7,
    )
    doors = make_hit(
        id="service_annual_pm_checklist::s3",
        doc_id="service_annual_pm_checklist",
        doc_type="service",
        text="## Industrial doors\nClean and align photo-eyes.",
        score=0.1,
    )
    monkeypatch.setattr(
        "mka.ask.store.query",
        lambda cfg, vector, filter, top_k: [all_models, hydraulic, air, doors],
    )
    out = related_sub_chunks(make_config(tmp_path), "technician", [0.1], [hydraulic])
    assert [hit.id for hit in out] == [hydraulic.id, all_models.id, air.id]


def test_related_sub_chunks_pulls_same_family_from_the_document(tmp_path, monkeypatch) -> None:
    recommendation = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="## Q: Which leveler?\n**A:** The MD-9000 air-powered dock leveler.",
        score=0.74,
    )
    hydraulic = make_hit(
        id="faq_cold_storage::q1",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="## Q: Why not hydraulic?\n**A:** The MD-7000 hydraulic leveler slows down in the cold.",
        score=0.0,
    )
    door = make_hit(
        id="faq_cold_storage::q4",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="## Q: What about track heating?\n**A:** Heated tracks are an option on the ThermaGuard 600.",
        score=0.0,
    )
    calls: list[dict] = []

    def fake_query(cfg, vector, filter, top_k):
        del cfg, vector
        calls.append(filter)
        assert top_k == 64
        return [recommendation, door, hydraulic]

    monkeypatch.setattr("mka.ask.store.query", fake_query)
    out = related_sub_chunks(make_config(tmp_path), "technician", [0.1], [recommendation])
    assert len(calls) == 1
    assert calls[0]["$and"][-1] == {"doc_id": {"$in": ["faq_cold_storage"]}}
    assert [hit.id for hit in out] == [recommendation.id, hydraulic.id]
    assert out[1].score == recommendation.score


def test_related_sub_chunks_loads_every_chunked_file_in_one_query(tmp_path, monkeypatch) -> None:
    faq = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="Specify the MD-9000 dock leveler.",
        score=0.7,
    )
    service = make_hit(
        id="service_md7000_lip_control::s0",
        doc_id="service_md7000_lip_control",
        doc_type="service",
        text="Check the MD-7000 dock leveler lip.",
        score=0.6,
    )
    faq_rel = make_hit(
        id="faq_cold_storage::q1",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="The MD-7000 hydraulic leveler slows down in the cold.",
    )
    service_rel = make_hit(
        id="service_md7000_lip_control::s1",
        doc_id="service_md7000_lip_control",
        doc_type="service",
        text="The hydraulic leveler lip stays out.",
    )
    calls: list[dict] = []

    def fake_query(cfg, vector, filter, top_k):
        del cfg, vector, top_k
        calls.append(filter)
        return [faq_rel, service_rel]

    monkeypatch.setattr("mka.ask.store.query", fake_query)
    out = related_sub_chunks(make_config(tmp_path), "technician", [0.1], [faq, service])
    assert len(calls) == 1
    assert calls[0]["$and"][-1] == {
        "doc_id": {"$in": ["faq_cold_storage", "service_md7000_lip_control"]}
    }
    assert [hit.id for hit in out] == [faq.id, faq_rel.id, service.id, service_rel.id]


def test_related_sub_chunks_does_not_cascade_from_a_relative(tmp_path, monkeypatch) -> None:
    seed = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="Specify the MD-9000 dock leveler.",
        score=0.74,
    )
    # Shares the seed's family, and also names a door. That door family must
    # not pull the track-heating chunk.
    both = make_hit(
        id="faq_cold_storage::q1",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="The MD-9000 pairs with a ThermaGuard 600 insulated door.",
        score=0.4,
    )
    door = make_hit(
        id="faq_cold_storage::q4",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="Heated tracks are an option on the ThermaGuard 600.",
        score=0.3,
    )
    monkeypatch.setattr(
        "mka.ask.store.query",
        lambda cfg, vector, filter, top_k: [seed, both, door],
    )
    out = related_sub_chunks(make_config(tmp_path), "sales", [0.1], [seed])
    assert [hit.id for hit in out] == [seed.id, both.id]


def test_related_sub_chunks_skips_a_chunk_with_no_catalog_family(tmp_path, monkeypatch) -> None:
    def fake_query(cfg, vector, filter, top_k):
        del cfg, vector, filter, top_k
        raise AssertionError("a chunk with no catalog family should not load the file")

    monkeypatch.setattr("mka.ask.store.query", fake_query)
    hit = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="## Q: How often?\n**A:** Quarterly.",
    )
    out = related_sub_chunks(make_config(tmp_path), "sales", [0.1], [hit])
    assert out == [hit]


def test_related_sub_chunks_skips_an_unchunked_file(tmp_path, monkeypatch) -> None:
    def fake_query(cfg, vector, filter, top_k):
        del cfg, vector, filter, top_k
        raise AssertionError("a whole file is not a split procedure")

    monkeypatch.setattr("mka.ask.store.query", fake_query)
    hit = make_hit(
        id="service_md7000_hydraulic_reset::0",
        doc_id="service_md7000_hydraulic_reset",
        doc_type="service",
        text="Set the MD-7000 relief valve to 1,800 psi.",
    )
    out = related_sub_chunks(make_config(tmp_path), "technician", [0.1], [hit])
    assert out == [hit]


def test_related_sub_chunks_keeps_a_dollar_amount_and_drops_a_pricing_document(
    tmp_path, monkeypatch
) -> None:
    recommendation = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="Specify the MD-9000 dock leveler.",
        score=0.7,
    )
    dollar = make_hit(
        id="faq_cold_storage::q1",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="The MD-9000 stainless lip is +$1,200.",
    )
    sheet = make_hit(
        id="pricing_md7000_2026::1",
        doc_id="pricing_md7000_2026",
        doc_type="pricing",
        text="The MD-9000 list is $12,000.",
    )
    monkeypatch.setattr(
        "mka.ask.store.query",
        lambda cfg, vector, filter, top_k: [dollar, sheet],
    )
    out = related_sub_chunks(make_config(tmp_path), "technician", [0.1], [recommendation])
    assert [hit.id for hit in out] == [recommendation.id, dollar.id]


def test_related_sub_chunks_pulls_service_chunks_in_the_same_family(tmp_path, monkeypatch) -> None:
    lip = make_hit(
        id="service_md7000_lip_control::s0",
        doc_id="service_md7000_lip_control",
        doc_type="service",
        text="## Symptom: Lip will not extend\nCheck the MD-7000 dock leveler.",
        score=0.8,
    )
    retract = make_hit(
        id="service_md7000_lip_control::s1",
        doc_id="service_md7000_lip_control",
        doc_type="service",
        text="## Symptom: Lip will not retract\nThe hydraulic leveler lip stays out.",
        score=0.0,
    )
    door = make_hit(
        id="service_md7000_lip_control::s2",
        doc_id="service_md7000_lip_control",
        doc_type="service",
        text="## Symptom: Door will not open\nAlign the RapidRoll photo-eye.",
        score=0.0,
    )
    calls: list[int] = []

    def fake_query(cfg, vector, filter, top_k):
        del cfg, vector, filter
        calls.append(top_k)
        return [door, retract, lip]

    monkeypatch.setattr("mka.ask.store.query", fake_query)
    out = related_sub_chunks(make_config(tmp_path), "technician", [0.1], [lip])
    assert calls == [64]
    assert [hit.id for hit in out] == [lip.id, retract.id]
    assert out[1].score == lip.score


def test_retrieve_follow_up_pulls_missing_model_spec(tmp_path, monkeypatch) -> None:
    faq = make_hit(
        id="faq::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="Pair the MD-9000 with a ThermaGuard 600 and a RapidRoll 400.",
        score=0.7,
    )
    spec = make_hit(
        id="spec_thermaguard600::0",
        doc_id="spec_thermaguard600",
        doc_type="spec",
        title="ThermaGuard 600 Insulated Sectional Door",
        text="ThermaGuard 600 insulated door for cold storage.",
        score=0.6,
    )
    calls: list[int] = []

    def fake_query(cfg, vector, filter, top_k):
        calls.append(top_k)
        if len(calls) == 1:
            return [faq]
        return [spec]

    monkeypatch.setattr("mka.ask.store.query", fake_query)
    out = retrieve(make_config(tmp_path), FakeEmbeddings(), "freezer dock?", "sales")
    assert [hit.id for hit in out] == [faq.id, spec.id]
    assert calls[0] == make_config(tmp_path).top_k
    assert calls[1] == 4

