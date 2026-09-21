from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from mka.config import Config
from mka.types import Chunk, Hit

UPSERT_BATCH = 100


def ensure_index(cfg: Config) -> None:
    client = _client(cfg)
    if client.has_index(cfg.pinecone_index):
        return
    client.create_index(
        name=cfg.pinecone_index,
        dimension=cfg.embed_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=cfg.pinecone_cloud, region=cfg.pinecone_region),
    )


def wipe_namespace(cfg: Config) -> None:
    try:
        _index(cfg).delete(delete_all=True, namespace=cfg.pinecone_namespace)
    except Exception as exc:
        if "namespace" not in str(exc).lower():
            raise


def upsert(cfg: Config, vectors: list[Chunk], embeddings: list[list[float]]) -> None:
    if len(vectors) != len(embeddings):
        raise ValueError("embedding count does not match chunk count")
    index = _index(cfg)
    batch: list[dict] = []
    for chunk, values in zip(vectors, embeddings, strict=True):
        batch.append({"id": chunk.id, "values": values, "metadata": chunk.metadata})
        if len(batch) == UPSERT_BATCH:
            index.upsert(vectors=batch, namespace=cfg.pinecone_namespace)
            batch = []
    if batch:
        index.upsert(vectors=batch, namespace=cfg.pinecone_namespace)


def query(
    cfg: Config,
    vector: list[float],
    filter: dict,
    top_k: int,
) -> list[Hit]:
    response = _index(cfg).query(
        vector=vector,
        filter=filter,
        top_k=top_k,
        include_metadata=True,
        namespace=cfg.pinecone_namespace,
    )
    hits: list[Hit] = []
    for match in response.matches:
        meta = match.metadata or {}
        hits.append(
            Hit(
                id=match.id,
                score=float(match.score),
                text=str(meta.get("text", "")),
                title=str(meta.get("title", "")),
                path=str(meta.get("path", "")),
                doc_id=str(meta.get("doc_id", "")),
                model=str(meta.get("model", "")),
                doc_type=str(meta.get("doc_type", "")),
                contains_warning=bool(meta.get("contains_warning", False)),
                warning_text=str(meta.get("warning_text", "")),
            )
        )
    return hits


def _client(cfg: Config) -> Pinecone:
    return Pinecone(api_key=cfg.pinecone_api_key)


def _index(cfg: Config):
    return _client(cfg).Index(cfg.pinecone_index)
