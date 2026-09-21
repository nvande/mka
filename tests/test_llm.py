from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_embeddings_empty() -> None:
    api = _FakeEmbedAPI()
    client = SimpleNamespace(embeddings=api)
    embedder = OpenAIEmbeddings(client, "text-embedding-3-small")
    assert embedder.embed([]) == []
    assert api.batch_sizes == []


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
