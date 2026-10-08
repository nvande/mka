from __future__ import annotations

from mka.usage import Call, Ledger, PineconeCall, render, token_cost_usd


def _chat(model: str = "gpt-5.4-nano", prompt: int | None = 1000, completion: int | None = 10) -> Call:
    return Call("chat", model, prompt, completion, latency_ms=5)


def test_chat_cost_uses_list_price() -> None:
    # 1000 * $0.20 / 1M + 10 * $1.25 / 1M
    assert token_cost_usd([_chat()]) == 0.0002125


def test_snapshot_model_uses_the_base_rate() -> None:
    assert token_cost_usd([_chat(model="gpt-5.4-nano-2026-03-17")]) == 0.0002125


def test_unknown_model_or_missing_usage_is_not_priced_as_zero() -> None:
    assert token_cost_usd([_chat(model="gpt-unknown")]) is None
    assert token_cost_usd([_chat(prompt=None, completion=None)]) is None
    text = render(Ledger("ask", calls=[_chat(prompt=None, completion=None)]))
    assert "token_cost_usd: unknown" in text
    assert "prompt_tokens=unreported" in text


def test_render_with_no_calls() -> None:
    text = render(Ledger("ask"))
    assert "workflow: ask" in text
    assert "latency_ms:" in text
    assert "chat_calls: 0" in text
    assert "embed_calls: 0" in text
    assert "decision_calls: 0" in text
    assert "token_cost_usd: 0.00000000" in text
    assert "pinecone_calls: 0" in text


def test_decision_cost_is_input_tokens_only() -> None:
    # 1000 * $0.10 / 1M. No output charge.
    call = Call("decision", "gpt-6-luna", 1000, 0, latency_ms=1)
    assert token_cost_usd([call]) == 0.0001


def test_render_lists_pinecone_calls_with_detail() -> None:
    ledger = Ledger(
        "ask",
        calls=[_chat(), Call("embed", "text-embedding-3-small", 16, 0, latency_ms=2)],
        pinecone=[PineconeCall("query", 40.0, {"top_k": 8, "matches": 8, "read_units": 1})],
    )
    text = render(ledger)
    assert "chat_calls: 1 model=gpt-5.4-nano prompt_tokens=1000 completion_tokens=10" in text
    assert "embed_calls: 1 model=text-embedding-3-small prompt_tokens=16" in text
    assert "pinecone_calls: 1" in text
    assert "  query: latency_ms=40 top_k=8 matches=8 read_units=1" in text
    # 0.0002125 + 16 * 0.02 / 1M
    assert "token_cost_usd: 0.00021282" in text
