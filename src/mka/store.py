from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from pinecone import Pinecone, ServerlessSpec

from mka import usage
from mka.config import Config
from mka.types import Chunk, Hit

UPSERT_BATCH = 100


@contextmanager
def _timed(op: str, **detail: Any) -> Iterator[dict]:
    """Time one Pinecone call and record it once, success or failure.

    Yields a dict for anything only known after the call returns.
    """
    started = time.perf_counter()
    try:
        yield detail
    except Exception as exc:
        detail["error"] = str(exc)
        raise
    finally:
        usage.record_pinecone(op, usage.elapsed_ms(started), **detail)


def ensure_index(cfg: Config) -> None:
    with _timed("ensure_index", index=cfg.pinecone_index) as detail:
        client = _client(cfg)
        detail["created"] = not client.has_index(cfg.pinecone_index)
        if detail["created"]:
            client.create_index(
                name=cfg.pinecone_index,
                dimension=cfg.embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cfg.pinecone_cloud, region=cfg.pinecone_region),
            )


def wipe_namespace(cfg: Config) -> None:
    with _timed("delete_namespace", namespace=cfg.pinecone_namespace):
        try:
            _index(cfg).delete(delete_all=True, namespace=cfg.pinecone_namespace)
        except Exception as exc:
            # Pinecone raises when the namespace does not exist yet. That is
            # an empty index, not a failed delete.
            if "namespace" not in str(exc).lower():
                raise


def delete_index(cfg: Config, *, wait: float = 60, poll: float = 1) -> bool:
    client = _client(cfg)
    if not client.has_index(cfg.pinecone_index):
        return False
    client.delete_index(cfg.pinecone_index)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if not client.has_index(cfg.pinecone_index):
            return True
        time.sleep(poll)
    raise RuntimeError(f"timed out waiting for index {cfg.pinecone_index} to delete")


def upsert(cfg: Config, vectors: list[Chunk], embeddings: list[list[float]]) -> None:
    if len(vectors) != len(embeddings):
        raise ValueError("embedding count does not match chunk count")
    index = _index(cfg)
    records = [
        {"id": chunk.id, "values": values, "metadata": chunk.metadata}
        for chunk, values in zip(vectors, embeddings, strict=True)
    ]
    for start in range(0, len(records), UPSERT_BATCH):
        batch = records[start : start + UPSERT_BATCH]
        with _timed("upsert", namespace=cfg.pinecone_namespace, vectors=len(batch)):
            index.upsert(vectors=batch, namespace=cfg.pinecone_namespace)


def query(cfg: Config, vector: list[float], filter: dict, top_k: int) -> list[Hit]:
    with _timed("query", namespace=cfg.pinecone_namespace, top_k=top_k) as detail:
        response = _index(cfg).query(
            vector=vector,
            filter=filter,
            top_k=top_k,
            include_metadata=True,
            namespace=cfg.pinecone_namespace,
        )
        hits = [_hit(match) for match in response.matches]
        detail["matches"] = len(hits)
        read_units = usage.lookup(usage.lookup(response, "usage"), "read_units")
        if read_units is not None:
            detail["read_units"] = read_units
        return hits


def _hit(match: object) -> Hit:
    meta = getattr(match, "metadata", None) or {}
    return Hit(
        id=match.id,
        score=float(match.score),
        text=str(meta.get("text", "")),
        title=str(meta.get("title", "")),
        path=str(meta.get("path", "")),
        doc_id=str(meta.get("doc_id", "")),
        model=str(meta.get("model", "")),
        doc_type=str(meta.get("doc_type", "")),
        flagged_outdated=bool(meta.get("flagged_outdated", False)),
        contains_warning=bool(meta.get("contains_warning", False)),
        warning_text=str(meta.get("warning_text", "")),
        warnings_cached=bool(meta.get("warnings_cached", False)),
        warning_excerpts=_str_tuple(meta.get("warning_excerpts")),
        warning_audiences=_str_tuple(meta.get("warning_audiences")),
    )


def _client(cfg: Config) -> Pinecone:
    return Pinecone(api_key=cfg.pinecone_api_key)


def _index(cfg: Config):
    return _client(cfg).Index(cfg.pinecone_index)


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)
