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
    SCOPE_SYS,
    TRIVIAL,
    TRIVIAL_TOO_LONG,
    cap_per_doc,
    classify_scope,
    complete_citations,
    explain_refuse,
    follow_up_reasons,
    include_documented_cause,
    format_refuse,
    outdated_note,
    pinecone_filter,
    receipt_ok,
    retrieve,
    run_ask,
    strip_follow_ups,
    strip_outdated_claims,
    superseded_hits,
    trivial_reject,
)
from mka.llm import OpenAIChat
from mka.products import catalog_hint
from mka.types import Hit

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


def systems(chat) -> list[str]:
    return [call[0] for call in chat.calls]


def only_call(chat, system: str) -> tuple[str, str, bool]:
    matches = [call for call in chat.calls if call[0] == system]
    assert len(matches) == 1, f"expected one {system[:30]!r} call, got {len(matches)}"
    return matches[0]


def assert_decided_and_drafted_together(chat, *, before: str, after: str | None) -> None:
    # Generate runs on a worker thread alongside decide, so its position is
    # unspecified. The main-thread sequence is fixed and each call runs once.
    seen = systems(chat)
    main_thread = [system for system in seen if system != GEN_SYS]
    assert main_thread == [before, DECIDE_SYS] + ([after] if after else [])
    assert seen.count(GEN_SYS) == 1


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


def test_scope_allow_only_exact_token() -> None:
    chat = FakeChat("ALLOW")
    assert classify_scope(chat, "MD-9000 capacity?") == "ALLOW"
    assert "Catalog terms" in chat.calls[0][1]


def test_scope_leveler_use_case_sends_catalog_hint() -> None:
    chat = FakeChat("ALLOW")
    query = "What is the best leveler for a blast freezer?"
    assert classify_scope(chat, query) == "ALLOW"
    assert chat.calls[0][1] == catalog_hint(query)
    assert "leveler → dock leveler" in chat.calls[0][1]


def test_scope_deny_extra_prose_and_punctuation() -> None:
    assert classify_scope(FakeChat("ALLOW extra"), "q") == "DENY"
    assert classify_scope(FakeChat("ALLOW."), "q") == "DENY"
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
    assert pinecone_filter("technician") == {
        "$and": [
            {"doc_type": {"$ne": "pricing"}},
            {"contains_pricing": {"$eq": False}},
        ]
    }


def test_scope_sys_allows_broad_meridian_product_asks() -> None:
    assert "broad but still about our catalog" in SCOPE_SYS
    assert "every / all / each Meridian product" in SCOPE_SYS
    assert "Do not require a model number" in SCOPE_SYS


def test_decide_and_generate_treat_conflict_as_answer() -> None:
    assert "conflicting values" in DECIDE_SYS
    assert "Do not REFUSE only because two documents disagree" in DECIDE_SYS
    assert "Absence is not a negative answer" in DECIDE_SYS
    assert "flagged_outdated" in GEN_SYS
    assert "do not mention outdated or superseded documents in the answer" in GEN_SYS
    assert '"outdated_note": string|null' in GEN_SYS
    assert "do not pick a winner" in GEN_SYS
    assert "Cite every provided chunk that states a product you named" in GEN_SYS
    assert "disagree on the same fact" in REFUSE_SYS


def test_receipt_ok() -> None:
    hits = [make_hit(id="a"), make_hit(id="b")]
    assert receipt_ok(["a"], hits)
    assert receipt_ok(["b", "a"], hits)
    assert not receipt_ok([], hits)
    assert not receipt_ok(["a", "nope"], hits)
    assert not receipt_ok(["made-up"], hits)


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
    assert "--- stats ---" not in out
    assert chat.calls[0][0] == SCOPE_SYS
    assert chat.calls[0][1] == "Write me a poem"
    assert "sales" not in chat.calls[0][0].lower()


def test_run_ask_missing_index_explains_ingest(tmp_path, capsys, monkeypatch) -> None:
    class Missing:
        def query(self, **kwargs):
            raise RuntimeError("[404 NOT_FOUND] Resource mka-poc not found")

    monkeypatch.setattr("mka.store._index", lambda cfg: Missing())
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "MD-7000 capacity?",
        chat=FakeChat("ALLOW"),
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
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )

    class Completions:
        def create(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="DENY"))],
                usage=usage,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
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
    assert "chat_calls: 1 model=gpt-5.4-nano prompt_tokens=1000 completion_tokens=10" in out
    # 1000 * $0.20 / 1M + 10 * $1.25 / 1M
    assert "token_cost_usd: 0.00021250" in out
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
    assert chat.calls == [(SCOPE_SYS, catalog_hint(query), False)]
    assert "dock leveler → dock leveler" in chat.calls[0][1]
    assert "sales" not in chat.calls[0][0].lower()
    assert "technician" not in chat.calls[0][0].lower()


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
    assert chat.calls == [(SCOPE_SYS, catalog_hint("MD-9000 capacity?"), False)]
    assert "md-9000 → dock leveler" in chat.calls[0][1]


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


def test_run_ask_technician_price_ask_is_denied_before_retrieval(tmp_path, capsys) -> None:
    called = {"retrieve": False}

    def retriever(query, role):
        called["retrieve"] = True
        return [make_hit()]

    chat = FakeChat("ALLOW")
    for query in (
        "What is the list price for a 7 ft by 10 ft MD-7000 with 460V?",
        "What does the MD-9000 cost?",
        "Can you quote me a ThermaGuard 600?",
        "MD-7000 pricing",
    ):
        code = run_ask(make_config(tmp_path), "technician", query, chat=chat, retriever=retriever)
        out = capsys.readouterr().out
        assert code == 0
        assert PRICING_DENIED in out
        assert REFUSE not in out
    assert called["retrieve"] is False
    # Scope still runs first: a poem about prices is a redirect, not a price denial.
    assert all(call[0] == SCOPE_SYS for call in chat.calls)


def test_run_ask_technician_price_belt_still_drops_untagged_price_hits(tmp_path, capsys) -> None:
    # A question with no price word can still pull a priced chunk. The belt
    # after retrieval is the second cut.
    priced = make_hit(id="price::0", text="Stainless lip option: $1,200", score=0.99)
    code = run_ask(
        make_config(tmp_path),
        "technician",
        "Which lip option is recommended for wash-down zones?",
        chat=FakeChat("ALLOW"),
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
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json("$10,550.", hit.id),
        }
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
    # Grounded numbers: the model's sentence is used.
    assert (
        outdated_note("The superseded 2021-03 revision lists 30,000 lbs.", [legacy])
        == "The superseded 2021-03 revision lists 30,000 lbs."
    )
    # A number the superseded text does not contain: canned line instead.
    canned = outdated_note("The old revision lists 25,000 lbs.", [legacy])
    assert canned.startswith("Note: a superseded revision is also on file")
    assert "MD-7000 Specification (SUPERSEDED) (docs/spec_md7000_legacy.md)" in canned
    assert "25,000" not in canned
    # No sentence from the model: canned line.
    assert outdated_note(None, [legacy]) == canned
    assert outdated_note("   ", [legacy]) == canned


def test_strip_outdated_claims() -> None:
    answer = (
        "The MD-5000 is not rated for blast freezers. Specify the MD-9000.\n\n"
        "Note: An outdated document lists a conflicting temperature rating for the MD-5000."
    )
    assert strip_outdated_claims(answer) == (
        "The MD-5000 is not rated for blast freezers. Specify the MD-9000."
    )
    assert strip_outdated_claims("Capacity is 35,000 lbs.") == "Capacity is 35,000 lbs."


def test_strip_follow_ups_drops_the_offer_to_continue() -> None:
    answer = (
        "Specify the MD-9000 with a ThermaGuard 600.\n\n"
        "If you want, tell me your dock pit and electrical/air constraints and "
        "I can map these to the installation requirements."
    )
    assert strip_follow_ups(answer) == "Specify the MD-9000 with a ThermaGuard 600."
    assert strip_follow_ups("The MD-9000 needs 80–100 psi compressed air.") == (
        "The MD-9000 needs 80–100 psi compressed air."
    )


def test_run_ask_superseded_revision_always_gets_a_note_and_source(tmp_path, capsys) -> None:
    current = make_hit()
    legacy = make_hit(**LEGACY)
    # The model cites only the current doc and gives no note. Python adds both.
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
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
            SCOPE_SYS: "ALLOW",
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
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json(
                "35,000 lbs.",
                current.id,
                legacy.id,
                outdated_note="The superseded 2021-03 revision lists 30,000 lbs.",
            ),
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
    assert "The superseded 2021-03 revision lists 30,000 lbs." in out
    assert "Note: a superseded revision" not in out
    # Cited once, not duplicated by the superseded append.
    assert out.count(legacy.id) == 1


def test_run_ask_invented_outdated_claim_is_stripped(tmp_path, capsys) -> None:
    hit = make_hit(
        id="faq_selection_guide::q4",
        title="FAQ: Which Dock Leveler Should I Choose?",
        doc_id="faq_selection_guide",
        doc_type="faq",
        model="multi",
        text="The MD-5000 has an operating temperature floor of -20°F.",
    )
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
            DECIDE_SYS: "ANSWER",
            GEN_SYS: gen_json(
                "No. Specify the MD-9000.\n\nNote: An outdated document lists a conflicting rating.",
                hit.id,
                outdated_note="An outdated document lists a conflicting temperature rating.",
            ),
        }
    )
    code = run_ask(
        make_config(tmp_path),
        "sales",
        "Can I install a mechanical dock leveler in a blast freezer?",
        chat=chat,
        retriever=lambda query, role: [hit],
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "No. Specify the MD-9000." in out
    assert "outdated" not in out.lower()


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
    # The draft was started alongside decide and discarded on REFUSE.
    assert_decided_and_drafted_together(chat, before=SCOPE_SYS, after=REFUSE_SYS)
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
    assert "Related topic:" not in out
    assert "ambiguous" not in out.lower()
    assert "Related:" in out


def test_run_ask_decide_refuse_unknown_prints_related_topic(tmp_path, capsys) -> None:
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
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


def test_run_ask_happy_path_prints_answer_staple_sources(
    tmp_path, capsys
) -> None:
    # No ingest cache on this record, so the staple comes from extracting
    # the chunk's own marked section. No model call is involved.
    excerpt = "- **Never** exceed 2,100 psi."
    hit = make_hit(text=f"Reset at 1,800 psi.\n\n## Safety limits\n\n{excerpt}\n")
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
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
    assert_decided_and_drafted_together(chat, before=SCOPE_SYS, after=None)
    gen_call = only_call(chat, GEN_SYS)
    assert gen_call[2] is True
    gen_payload = json.loads(gen_call[1])
    assert gen_payload["query"] == "MD-7000 capacity?"
    assert "role" not in gen_payload
    assert gen_payload["chunks"][0]["flagged_outdated"] is False


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
            SCOPE_SYS: "ALLOW",
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
            SCOPE_SYS: "ALLOW",
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
    assert_decided_and_drafted_together(chat, before=SCOPE_SYS, after=None)


def test_run_ask_refuse_discards_a_finished_draft(tmp_path, capsys) -> None:
    # Generate returned a perfectly valid answer. Decide said REFUSE. The
    # answer must not print, and its citations must not appear.
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
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
    assert_decided_and_drafted_together(chat, before=SCOPE_SYS, after=REFUSE_SYS)


def test_run_ask_refuse_ignores_a_draft_that_errored(tmp_path, capsys) -> None:
    # A transport error in the speculative draft is irrelevant on REFUSE.
    hit = make_hit()
    chat = ScriptedChat(
        {
            SCOPE_SYS: "ALLOW",
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
            SCOPE_SYS: "ALLOW",
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
            SCOPE_SYS: "ALLOW",
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
            SCOPE_SYS: "ALLOW",
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
    assert captured["top_k"] == cfg.retrieve_pool
    assert [hit.id for hit in out] == ["ok::0"]


def test_cap_per_doc_keeps_order_and_limit() -> None:
    hits = [
        make_hit(id="faq::q0", doc_id="faq"),
        make_hit(id="faq::q1", doc_id="faq"),
        make_hit(id="faq::q2", doc_id="faq"),
        make_hit(id="spec::0", doc_id="spec"),
    ]
    assert [hit.id for hit in cap_per_doc(hits, 2)] == ["faq::q0", "faq::q1", "spec::0"]
    assert [hit.id for hit in cap_per_doc(hits, 0)] == [hit.id for hit in hits]


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


def test_follow_up_reasons_keeps_the_adjacent_why_question(tmp_path, monkeypatch) -> None:
    recommendation = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text=(
            "## Q: What's your recommended configuration for a -10°F freezer dock?\n"
            "**A:** MD-9000."
        ),
        score=0.74,
    )
    why = make_hit(
        id="faq_cold_storage::q1",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text=(
            "## Q: Why air-powered instead of hydraulic in cold storage?\n"
            "**A:** Hydraulic fluid viscosity increases."
        ),
        score=0.0,
    )
    finishes = make_hit(
        id="faq_cold_storage::q2",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="## Q: Do we need special finishes?\n**A:** Stainless.",
        score=0.0,
    )

    def fake_fetch(cfg, ids):
        del cfg
        assert "faq_cold_storage::q1" in ids
        return [why, finishes]

    monkeypatch.setattr("mka.ask.store.fetch", fake_fetch)
    out = follow_up_reasons(
        make_config(tmp_path),
        "technician",
        "What equipment should I specify and why?",
        [recommendation],
    )
    assert [hit.id for hit in out] == [recommendation.id, why.id]
    assert out[1].score == recommendation.score


def test_follow_up_reasons_skips_a_plain_lookup(tmp_path, monkeypatch) -> None:
    def fake_fetch(cfg, ids):
        del cfg, ids
        raise AssertionError("plain lookup should not fetch neighbors")

    monkeypatch.setattr("mka.ask.store.fetch", fake_fetch)
    hit = make_hit(id="faq_cold_storage::q0", doc_id="faq_cold_storage", doc_type="faq")
    out = follow_up_reasons(
        make_config(tmp_path), "sales", "What is the rated lifting capacity?", [hit]
    )
    assert out == [hit]


def test_follow_up_reasons_hides_priced_why_from_technician(tmp_path, monkeypatch) -> None:
    recommendation = make_hit(
        id="faq_cold_storage::q0",
        doc_id="faq_cold_storage",
        doc_type="faq",
        score=0.7,
    )
    priced = make_hit(
        id="faq_cold_storage::q1",
        doc_id="faq_cold_storage",
        doc_type="faq",
        text="## Q: Why this option?\n**A:** Stainless lip +$1,200.",
    )
    monkeypatch.setattr("mka.ask.store.fetch", lambda cfg, ids: [priced])
    out = follow_up_reasons(
        make_config(tmp_path),
        "technician",
        "Which leveler should I specify and why?",
        [recommendation],
    )
    assert [hit.id for hit in out] == [recommendation.id]


WHY = (
    "## Q: Why air-powered instead of hydraulic in cold storage?\n"
    "**A:** Hydraulic fluid viscosity increases significantly at low temperatures, "
    "slowing the MD-7000's response and stressing seals. The MD-9000 uses compressed "
    "air, which performs consistently down to -40°F, and eliminates the risk of "
    "hydraulic fluid contamination in food zones.\n"
)


def test_include_documented_cause_adds_the_omitted_mechanism() -> None:
    why = make_hit(id="faq_cold_storage::q1", doc_id="faq_cold_storage", doc_type="faq", text=WHY)
    draft = (
        "Specify the MD-9000. It has no hydraulic fluid and operates down to -40°F, "
        "so it meets the temperature requirement."
    )
    answer, ids = include_documented_cause(
        "What equipment should I specify and why?",
        draft,
        [why],
    )
    assert "viscosity" in answer
    assert answer.startswith(draft)
    assert ids == [why.id]


def test_include_documented_cause_leaves_a_complete_answer() -> None:
    why = make_hit(id="faq_cold_storage::q1", doc_id="faq_cold_storage", doc_type="faq", text=WHY)
    draft = (
        "Specify the MD-9000 because hydraulic fluid viscosity increases significantly "
        "at low temperatures, slowing the MD-7000 response and stressing seals."
    )
    answer, ids = include_documented_cause(
        "What equipment should I specify and why?",
        draft,
        [why],
    )
    assert answer == draft
    assert ids == []


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
    assert calls[0] == make_config(tmp_path).retrieve_pool
    assert calls[1] == 4

