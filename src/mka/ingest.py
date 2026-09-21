from __future__ import annotations

import json
import os
import re
import sys

from mka import store
from mka.chunking import ChunkError, chunk_document
from mka.config import Config
from mka.llm import make_embeddings
from mka.types import Chunk, ManifestRow, load_manifest

ALLOWED_AUDIENCE = frozenset({"sales", "technician", "all"})
VERSION_RE = re.compile(r"^\d{4}-\d{2}$")


def run_ingest(cfg: Config) -> int:
    chunks, errors = prepare_chunks(cfg)
    for message in errors:
        print(message, file=sys.stderr)
    if chunks:
        try:
            _write_index(cfg, chunks)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    print(f"chunks: {len(chunks)}")
    return 1 if errors else 0


def _write_index(cfg: Config, chunks: list[Chunk]) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to ingest")
    if not cfg.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to ingest")
    embeddings = make_embeddings(cfg)
    values = embeddings.embed([chunk.text for chunk in chunks])
    if len(values) != len(chunks):
        raise RuntimeError("embedding count does not match chunk count")
    store.ensure_index(cfg)
    store.wipe_namespace(cfg)
    store.upsert(cfg, chunks, values)


def prepare_chunks(cfg: Config) -> tuple[list[Chunk], list[str]]:
    if not cfg.corpus_dir.is_dir():
        return [], [f"error: corpus directory not found: {cfg.corpus_dir}"]
    manifest_path = cfg.corpus_dir / "manifest.json"
    if not manifest_path.is_file():
        return [], [f"error: manifest not found: {manifest_path}"]
    try:
        rows = load_manifest(cfg.corpus_dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [], [f"error: cannot read manifest: {exc}"]

    chunks: list[Chunk] = []
    errors: list[str] = []
    for row in rows:
        problem = _allowlist_error(row)
        if problem:
            errors.append(problem)
            continue
        body_path = cfg.corpus_dir / row.path
        if not body_path.is_file():
            errors.append(f"error: {row.doc_id}: missing file {row.path}")
            continue
        try:
            text = body_path.read_text(encoding="utf-8")
            chunks.extend(chunk_document(row, text))
        except (OSError, ChunkError) as exc:
            errors.append(f"error: {row.doc_id}: {exc}")

    listed = {row.path for row in rows}
    docs_dir = cfg.corpus_dir / "docs"
    if docs_dir.is_dir():
        for path in sorted(docs_dir.glob("*.md")):
            rel = path.relative_to(cfg.corpus_dir).as_posix()
            if rel not in listed:
                errors.append(f"error: orphan not in manifest: {rel}")
    return chunks, errors


def _allowlist_error(row: ManifestRow) -> str | None:
    if row.audience not in ALLOWED_AUDIENCE:
        return f"error: {row.doc_id}: invalid audience {row.audience!r}"
    if not VERSION_RE.fullmatch(row.version):
        return f"error: {row.doc_id}: invalid version {row.version!r}"
    return None
