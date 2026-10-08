from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mka import usage
from mka.llm import OpenAIChat, OpenAIEmbeddings, make_chat, make_embeddings

from conftest import make_config


class _FakeEmbedAPI:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        del model
        self.batch_sizes.append(len(input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[float(i)])
                for i in range(len(input))
            ]
        )


def test_unknown_provider_raises(tmp_path: Path) -> None:
    cfg = replace(make_config(tmp_path), llm_provider="not-a-provider")
    with pytest.raises(ValueError, match="unknown llm_provider"):
        make_embeddings(cfg)
    with pytest.raises(ValueError, match="unknown llm_provider"):
        make_chat(cfg)


def test_embeddings_batch_at_64() -> None:
    api = _FakeEmbedAPI()
    client = SimpleNamespace(embeddings=api)
    embedder = OpenAIEmbeddings(client, "text-embedding-3-small", batch_size=64)
    vectors = embedder.embed(["t"] * 65)
    assert len(vectors) == 65
    assert api.batch_sizes == [64, 1]


def test_embeddings_record_token_cost_per_batch(tmp_path: Path) -> None:
    class _UsageEmbedAPI:
        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            del model
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=i, embedding=[0.0]) for i in range(len(input))
                ],
                usage=SimpleNamespace(prompt_tokens=len(input), total_tokens=len(input)),
            )

    client = SimpleNamespace(embeddings=_UsageEmbedAPI())
    embedder = OpenAIEmbeddings(client, "text-embedding-3-small", batch_size=64)
    with usage.track("ingest") as ledger:
        vectors = embedder.embed(["t"] * 65)
    assert len(vectors) == 65
    assert [call.prompt_tokens for call in ledger.calls] == [64, 1]
    # 65 * $0.02 / 1M
    assert "token_cost_usd: 0.00000130" in usage.render(ledger)
    assert "embed_calls: 2" in usage.render(ledger)


def test_embeddings_empty() -> None:
    api = _FakeEmbedAPI()
    client = SimpleNamespace(embeddings=api)
    embedder = OpenAIEmbeddings(client, "text-embedding-3-small")
    assert embedder.embed([]) == []
    assert api.batch_sizes == []


def test_classify_sends_a_choice_question_and_reads_the_label() -> None:
    captured: dict = {}

    class _Decisions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                answers=[SimpleNamespace(type="choice", choice="ALLOW")],
                usage=SimpleNamespace(input_tokens=12),
            )

    client = SimpleNamespace(decisions=_Decisions())
    chat = OpenAIChat(client, "gpt-5.4-nano", "gpt-6-luna")
    with usage.track("ask") as ledger:
        assert (
            chat.classify("MD-9000?", instructions="pick one", choices=[{"value": "ALLOW"}])
            == "ALLOW"
        )
    assert captured["model"] == "gpt-6-luna"
    assert captured["input"] == "MD-9000?"
    assert captured["questions"] == [
        {
            "type": "choice",
            "name": "scope",
            "instructions": "pick one",
            "choices": [{"value": "ALLOW"}],
        }
    ]
    assert ledger.calls[0].kind == "decision"
    assert ledger.calls[0].prompt_tokens == 12


def test_classify_fail_closes_a_refusal_and_raises_on_an_api_error() -> None:
    class _Decisions:
        def create(self, **kwargs):
            del kwargs
            return SimpleNamespace(answers=[SimpleNamespace(type="refusal", choice=None)])

    chat = OpenAIChat(SimpleNamespace(decisions=_Decisions()), "gpt-5.4-nano")
    assert chat.classify("poem", instructions="pick one", choices=[]) == "DENY"

    class _Down:
        def create(self, **kwargs):
            del kwargs
            raise RuntimeError("down")

    chat = OpenAIChat(SimpleNamespace(decisions=_Down()), "gpt-5.4-nano")
    with pytest.raises(RuntimeError, match="down"):
        chat.classify("poem", instructions="pick one", choices=[])


def test_chat_json_object_sets_response_format() -> None:
    captured: dict = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    chat = OpenAIChat(client, "gpt-5.4-nano")
    assert chat.complete(system="s", user="u", json_object=True) == '{"ok": true}'
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0
    captured.clear()
    chat.complete(system="s", user="u")
    assert "response_format" not in captured
