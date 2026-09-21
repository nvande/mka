from __future__ import annotations

from mka.config import Config
from mka.types import Chunk, Hit


def ensure_index(cfg: Config) -> None:
    del cfg
    raise NotImplementedError("store is not implemented yet")


def upsert(cfg: Config, vectors: list[Chunk], embeddings: list[list[float]]) -> None:
    del cfg, vectors, embeddings
    raise NotImplementedError("store is not implemented yet")


def query(
    cfg: Config,
    vector: list[float],
    filter: dict,
    top_k: int,
) -> list[Hit]:
    del cfg, vector, filter, top_k
    raise NotImplementedError("store is not implemented yet")
