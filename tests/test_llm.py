from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mka.llm import OpenAIEmbeddings, make_chat, make_embeddings

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
