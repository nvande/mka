from __future__ import annotations

import json
import re

import tiktoken

from mka.pricing import contains_pricing
from mka.types import Chunk, ManifestRow

MAX_EMBED_TOKENS = 8191
MAX_METADATA_BYTES = 40_000

_H1 = re.compile(r"^# ", re.M)
_REVISION = re.compile(r"(?i)^(\*\*Revision:\*\*|Revision:)", re.M)
_QA_MARK = re.compile(r"^## Q:", re.M)
_QA_SPLIT = re.compile(r"(?=^## Q:)", re.M)
_H2_SPLIT = re.compile(r"(?=^## )", re.M)

_encoding: tiktoken.Encoding | None = None


class ChunkError(Exception):
    """File cannot be chunked (staple or size)."""


def chunk_document(row: ManifestRow, text: str) -> list[Chunk]:
    _reject_staple(text)
    pieces = _split_shape(row, text)
    chunks: list[Chunk] = []
    for chunk_id, body in pieces:
        chunks.extend(_fit_size(row, chunk_id, body))
    return chunks


def _reject_staple(text: str) -> None:
    if len(_H1.findall(text)) > 1:
        raise ChunkError("stapled file: more than one H1")
    if len(_REVISION.findall(text)) > 1:
        raise ChunkError("stapled file: more than one Revision header")


def _split_shape(row: ManifestRow, text: str) -> list[tuple[str, str]]:
    if _QA_MARK.search(text):
        parts = [part for part in _QA_SPLIT.split(text) if part.startswith("## Q:")]
        if not parts:
            raise ChunkError(f"{row.doc_id}: Q&A shape but no questions")
        return [(f"{row.doc_id}::q{i}", part) for i, part in enumerate(parts)]
    return [(f"{row.doc_id}::0", text)]


def _fit_size(row: ManifestRow, chunk_id: str, body: str) -> list[Chunk]:
    chunk = _make_chunk(row, chunk_id, body)
    if not _over_limit(chunk):
        return [chunk]
    parts = [part for part in _H2_SPLIT.split(body) if part.strip()]
    if len(parts) <= 1:
        raise ChunkError(f"{chunk_id} exceeds embed or metadata size")
    fitted: list[Chunk] = []
    for i, part in enumerate(parts):
        part_id = f"{row.doc_id}::{i}"
        piece = _make_chunk(row, part_id, part)
        if _over_limit(piece):
            raise ChunkError(f"{part_id} exceeds embed or metadata size")
        fitted.append(piece)
    return fitted


def _make_chunk(row: ManifestRow, chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata=_metadata(row, text))


def _metadata(row: ManifestRow, text: str) -> dict:
    return {
        "doc_id": row.doc_id,
        "path": row.path,
        "title": row.title,
        "parent_title": row.title,
        "doc_type": row.doc_type,
        "audience": row.audience,
        "model": row.model,
        "version": row.version,
        "last_updated": row.last_updated,
        "flagged_outdated": row.flagged_outdated,
        "contains_pricing": contains_pricing(text),
        "contains_warning": False,
        "warning_text": "",
        "text": text,
    }


def _over_limit(chunk: Chunk) -> bool:
    if _token_len(chunk.text) > MAX_EMBED_TOKENS:
        return True
    packed = json.dumps(chunk.metadata, ensure_ascii=False).encode()
    return len(packed) > MAX_METADATA_BYTES


def _token_len(text: str) -> int:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return len(_encoding.encode(text))
