from __future__ import annotations

from typing import Protocol

from mka.config import Config


class Chat(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


class Embeddings(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def make_chat(cfg: Config) -> Chat:
    del cfg
    raise NotImplementedError("LLM adapter is not implemented yet")


def make_embeddings(cfg: Config) -> Embeddings:
    del cfg
    raise NotImplementedError("embeddings adapter is not implemented yet")
