from __future__ import annotations

import time
from typing import Protocol

from openai import OpenAI

from mka import usage
from mka.config import Config

EMBED_BATCH = 64


class Chat(Protocol):
    def complete(self, *, system: str, user: str, json_object: bool = False) -> str: ...

    def classify(self, text: str, *, instructions: str, choices: list[dict]) -> str: ...


class Embeddings(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIChat:
    def __init__(self, client: OpenAI, model: str, decision_model: str = "gpt-6-luna") -> None:
        self._client = client
        self._model = model
        self._decision_model = decision_model

    def classify(self, text: str, *, instructions: str, choices: list[dict]) -> str:
        """Fixed-choice scope label. A refusal or unusable answer is DENY.

        An API error is left to raise. That is an outage, not a verdict.
        """
        started = time.perf_counter()
        response = None
        try:
            response = self._client.decisions.create(
                model=self._decision_model,
                input=text,
                questions=[
                    {
                        "type": "choice",
                        "name": "scope",
                        "instructions": instructions,
                        "choices": choices,
                    }
                ],
            )
            return _decision_choice(response)
        finally:
            usage.record_decision(self._decision_model, response, usage.elapsed_ms(started))

    def complete(self, *, system: str, user: str, json_object: bool = False) -> str:
        kwargs: dict = {}
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = None
        try:
            # Gates and the answer both run at temperature 0. A creative
            # sample is how a citation id or a one-token decision drifts.
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
        finally:
            usage.record_chat(self._model, response, usage.elapsed_ms(started))


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
            started = time.perf_counter()
            response = None
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                out.extend([item.embedding for item in ordered])
            finally:
                usage.record_embed(self._model, response, usage.elapsed_ms(started))
        return out


def make_chat(cfg: Config) -> Chat:
    _require_openai(cfg)
    return OpenAIChat(OpenAI(), cfg.chat_model, cfg.decision_model)


def _decision_choice(response: object) -> str:
    answers = getattr(response, "answers", None)
    if not answers:
        return "DENY"
    answer = answers[0]
    if getattr(answer, "type", None) != "choice":
        return "DENY"
    choice = getattr(answer, "choice", None)
    value = getattr(choice, "value", choice)
    return "" if value is None else str(value)


def make_embeddings(cfg: Config) -> Embeddings:
    _require_openai(cfg)
    return OpenAIEmbeddings(OpenAI(), cfg.embed_model)


def _require_openai(cfg: Config) -> None:
    if cfg.llm_provider != "openai":
        raise ValueError(f"unknown llm_provider: {cfg.llm_provider}")
