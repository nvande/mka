from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from mka.config import Config

EMBED_BATCH = 64


class Chat(Protocol):
    def complete(self, *, system: str, user: str, json_object: bool = False) -> str: ...


class Embeddings(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIChat:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def complete(self, *, system: str, user: str, json_object: bool = False) -> str:
        kwargs: dict = {}
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class OpenAIEmbeddings:
    def __init__(self, client: OpenAI, model: str, batch_size: int = EMBED_BATCH) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            out.extend([item.embedding for item in ordered])
        return out


def make_chat(cfg: Config) -> Chat:
    _require_openai(cfg)
    return OpenAIChat(OpenAI(), cfg.chat_model)


def make_embeddings(cfg: Config) -> Embeddings:
    _require_openai(cfg)
    return OpenAIEmbeddings(OpenAI(), cfg.embed_model)


def _require_openai(cfg: Config) -> None:
    if cfg.llm_provider != "openai":
        raise ValueError(f"unknown llm_provider: {cfg.llm_provider}")
