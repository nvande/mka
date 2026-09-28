"""Chunking on document shape, not on filename or a fixed token window.

Most files stay whole as one chunk, which keeps the original context together.
I only split the shapes that repeat: an FAQ becomes one chunk per ``## Q:``,
and a service doc becomes one chunk per issue with the shared preamble and
tail added back to each piece. A single procedure is never broken up. If a
piece still runs past the embedding or metadata limit, I split it on an
existing ``##`` or raise. Nothing is truncated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

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
_SYMPTOM = re.compile(r"^## Symptom:", re.M)
_FAULT = re.compile(r"^### E\d+", re.M)
_FAULT_SPLIT = re.compile(r"(?=^### E\d+)", re.M)
_PROCEDURE_H2 = re.compile(r"^## Procedure\b", re.M)

_SHARED_H2 = frozenset(
    {
        "purpose",
        "documentation",
        "related parts",
        "required tools",
        "procedure",
        "safety limits",
        "prerequisites",
        "critical rules",
        "fault code reference",
    }
)

_encoding: tiktoken.Encoding | None = None


class ChunkError(Exception):
    """File cannot be chunked (staple or size)."""


@dataclass(frozen=True)
class _Piece:
    chunk_id: str
    body: str
    preamble: str = ""
    tail: str = ""


def chunk_document(row: ManifestRow, text: str) -> list[Chunk]:
    _reject_staple(text)
    chunks: list[Chunk] = []
    for piece in _split_shape(row, text):
        chunks.extend(_fit_size(row, piece))
    return chunks


def _reject_staple(text: str) -> None:
    # Two documents pasted into one file must not become one vector.
    if len(_H1.findall(text)) > 1:
        raise ChunkError("stapled file: more than one H1")
    if len(_REVISION.findall(text)) > 1:
        raise ChunkError("stapled file: more than one Revision header")


def _split_shape(row: ManifestRow, text: str) -> list[_Piece]:
    if _QA_MARK.search(text):
        parts = [part for part in _QA_SPLIT.split(text) if part.startswith("## Q:")]
        if not parts:
            raise ChunkError(f"{row.doc_id}: Q&A shape but no questions")
        return [_Piece(f"{row.doc_id}::q{i}", part) for i, part in enumerate(parts)]
    for splitter in (_split_symptoms, _split_faults, _split_pm_topics):
        pieces = splitter(row, text)
        if pieces:
            return pieces
    return [_Piece(f"{row.doc_id}::0", text)]


def _split_symptoms(row: ManifestRow, text: str) -> list[_Piece] | None:
    if len(_SYMPTOM.findall(text)) < 2:
        return None
    lead, sections = _h2_sections(text)
    first = next(i for i, section in enumerate(sections) if _is_symptom(section))
    last = max(i for i, section in enumerate(sections) if _is_symptom(section))
    preamble = lead + "".join(sections[:first])
    tail = "".join(sections[last + 1 :])
    issues = [section for section in sections[first : last + 1] if _is_symptom(section)]
    return _service_pieces(row, preamble, issues, tail)


def _split_faults(row: ManifestRow, text: str) -> list[_Piece] | None:
    if len(_FAULT.findall(text)) < 2:
        return None
    preamble = ""
    faults: list[str] = []
    for part in _FAULT_SPLIT.split(text):
        if _FAULT.match(part):
            faults.append(part)
        else:
            preamble += part
    last = faults[-1]
    fault_body, trailing = _h2_sections(last)
    if trailing:
        faults[-1] = fault_body
        tail = "".join(trailing)
    else:
        tail = ""
    return _service_pieces(row, preamble, faults, tail)


def _split_pm_topics(row: ManifestRow, text: str) -> list[_Piece] | None:
    # doc_type is required here so a spec with several ## sections stays
    # one chunk. A "## Procedure" heading is one job, not a list of topics.
    if row.doc_type != "service":
        return None
    if _PROCEDURE_H2.search(text):
        return None
    lead, sections = _h2_sections(text)
    topic_indexes = [i for i, section in enumerate(sections) if not _is_shared_h2(section)]
    if len(topic_indexes) < 2:
        return None
    first, last = topic_indexes[0], topic_indexes[-1]
    preamble = lead + "".join(sections[:first])
    tail = "".join(sections[last + 1 :])
    issues: list[str] = []
    current = ""
    for section in sections[first : last + 1]:
        if _is_shared_h2(section):
            current += section
            continue
        if current:
            issues.append(current)
        current = section
    if current:
        issues.append(current)
    return _service_pieces(row, preamble, issues, tail)


def _service_pieces(row: ManifestRow, preamble: str, issues: list[str], tail: str) -> list[_Piece]:
    pieces: list[_Piece] = []
    for i, issue in enumerate(issues):
        pieces.append(
            _Piece(
                f"{row.doc_id}::s{i}",
                _join_staple(preamble, issue, tail),
                preamble,
                tail,
            )
        )
    return pieces


def _h2_sections(text: str) -> tuple[str, list[str]]:
    lead: list[str] = []
    sections: list[str] = []
    for part in _H2_SPLIT.split(text):
        if part.startswith("## "):
            sections.append(part)
        else:
            lead.append(part)
    return "".join(lead), sections


def _is_symptom(section: str) -> bool:
    return section.startswith("## Symptom:")


def _is_shared_h2(section: str) -> bool:
    # A single "## Symptom:" inside a procedure is context for that job.
    # The symptom splitter already ran, and it only splits when there are
    # two or more. Treating "symptom" as shared keeps the PM splitter from
    # cutting the procedure apart.
    name = _h2_name(section)
    if name in _SHARED_H2:
        return True
    return name.startswith("symptom")


def _h2_name(section: str) -> str:
    line = section.split("\n", 1)[0]
    if line.startswith("## "):
        return line[3:].strip().lower()
    return ""


def _join_staple(preamble: str, issue: str, tail: str) -> str:
    return "".join(part for part in (preamble, issue, tail) if part)


def _fit_size(row: ManifestRow, piece: _Piece) -> list[Chunk]:
    # Walls are the embedding model (~8191 tokens) and Pinecone's metadata
    # cap. This corpus fits. The split exists so a glued-together file fails
    # loudly instead of being truncated into a useless vector. Service
    # pieces keep the same preamble and tail, so a DANGER above the first
    # issue stays on every piece.
    chunk = _make_chunk(row, piece.chunk_id, piece.body)
    if not over_limit(chunk):
        return [chunk]
    if piece.preamble or piece.tail:
        middle = _unstaple(piece.body, piece.preamble, piece.tail)
        parts = [part for part in _H2_SPLIT.split(middle) if part.strip()]
        if len(parts) <= 1:
            raise ChunkError(f"{piece.chunk_id} exceeds embed or metadata size")
        fitted: list[Chunk] = []
        for i, part in enumerate(parts):
            part_id = f"{piece.chunk_id}::{i}"
            body = _join_staple(piece.preamble, part, piece.tail)
            extra = _make_chunk(row, part_id, body)
            if over_limit(extra):
                raise ChunkError(f"{part_id} exceeds embed or metadata size")
            fitted.append(extra)
        return fitted
    parts = [part for part in _H2_SPLIT.split(piece.body) if part.strip()]
    if len(parts) <= 1:
        raise ChunkError(f"{piece.chunk_id} exceeds embed or metadata size")
    fitted = []
    for i, part in enumerate(parts):
        part_id = f"{row.doc_id}::{i}"
        extra = _make_chunk(row, part_id, part)
        if over_limit(extra):
            raise ChunkError(f"{part_id} exceeds embed or metadata size")
        fitted.append(extra)
    return fitted


def _unstaple(body: str, preamble: str, tail: str) -> str:
    middle = body
    if preamble:
        if not middle.startswith(preamble):
            raise ChunkError("service chunk missing preamble")
        middle = middle[len(preamble) :]
    if tail:
        if not middle.endswith(tail):
            raise ChunkError("service chunk missing tail")
        middle = middle[: len(middle) - len(tail)]
    return middle


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
        # doc_type is not trusted. A FAQ row can still contain a price.
        "contains_pricing": contains_pricing(text),
        # Hazard-note cache. Empty at chunk time; safety.assign_warnings
        # fills these once ingest has read the whole source file.
        "contains_warning": False,
        "warning_text": "",
        "warnings_cached": False,
        "warning_excerpts": [],
        "warning_audiences": [],
        "text": text,
    }


def over_limit(chunk: Chunk) -> bool:
    if token_len(chunk.text) > MAX_EMBED_TOKENS:
        return True
    packed = json.dumps(chunk.metadata, ensure_ascii=False).encode()
    return len(packed) > MAX_METADATA_BYTES


def token_len(text: str) -> int:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return len(_encoding.encode(text))
