"""The ingest command. Read the manifest, chunk, embed, then replace the namespace.

A bad manifest row fails only that file and never gets upserted. The rest of
the corpus still loads, so one broken document doesn't block a rebuild.
Wiping the namespace is safe here because ingest is the only writer and the
corpus is the entire index. If a second live source ever writes to this
index, it can't share this replace.
"""

from __future__ import annotations

import json
import os
import re
import sys

from mka import store, usage
from mka.chunking import ChunkError, chunk_document
from mka.config import Config
from mka.llm import make_embeddings
from mka.safety import assign_warnings, extract_warnings
from mka.spinner import Spinner
from mka.types import Chunk, ManifestRow, load_manifest

ALLOWED_AUDIENCE = frozenset({"sales", "technician", "all"})
VERSION_RE = re.compile(r"^\d{4}-\d{2}$")


def run_ingest(cfg: Config, *, stats: bool = False) -> int:
    return usage.reported("ingest", lambda: _run_ingest(cfg), stats=stats)


def _run_ingest(cfg: Config) -> int:
    chunks, errors = prepare_chunks(cfg)
    for message in errors:
        print(message, file=sys.stderr)
    if chunks:
        spin = Spinner()
        try:
            _cache_warnings(cfg, chunks)
            _write_index(cfg, chunks)
        except Exception as exc:
            spin.stop()
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            spin.stop()
    if errors:
        print(f"chunks: {len(chunks)}")
        return 1
    print(f"ingest complete: {len(chunks)} chunks")
    return 0


def _cache_warnings(cfg: Config, chunks: list[Chunk]) -> None:
    # Hazard notes are read off the whole source file, not the chunk, so a
    # note in a header the splitter dropped still reaches the chunks it
    # covers. assign_warnings applies the scope rule and writes the cache.
    cached = 0
    for group in _by_doc(chunks).values():
        meta = group[0].metadata
        try:
            text = (cfg.corpus_dir / str(meta["path"])).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"warning: {meta['doc_id']}: warning cache skipped: {exc}", file=sys.stderr)
            continue
        cached += assign_warnings(extract_warnings(text), group)
    print(f"warnings: cached on {cached}/{len(chunks)} chunks")


def _by_doc(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(str(chunk.metadata["doc_id"]), []).append(chunk)
    return groups


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
    # Full replace. A doc removed from the corpus must not keep answering.
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

    # A markdown file with no manifest row has no audience or version.
    # Fail it instead of embedding an unlabeled chunk.
    listed = {row.path for row in rows}
    docs_dir = cfg.corpus_dir / "docs"
    if docs_dir.is_dir():
        for path in sorted(docs_dir.glob("*.md")):
            rel = path.relative_to(cfg.corpus_dir).as_posix()
            if rel not in listed:
                errors.append(f"error: orphan not in manifest: {rel}")
    return chunks, errors


def _allowlist_error(row: ManifestRow) -> str | None:
    # Pinecone filters are exact. An unknown audience or a version that is
    # not YYYY-MM would slip past the role filter, so the row is rejected.
    if row.audience not in ALLOWED_AUDIENCE:
        return f"error: {row.doc_id}: invalid audience {row.audience!r}"
    if not VERSION_RE.fullmatch(row.version):
        return f"error: {row.doc_id}: invalid version {row.version!r}"
    return None
