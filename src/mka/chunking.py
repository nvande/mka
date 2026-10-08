"""Split a markdown document on its heading outline.

A chunk is one section. The document title, the front matter under that
title, and every ancestor heading are copied onto the chunk, so a section
retrieved on its own still says which document and parent it came from.

Numbered steps stay with their parent heading. They are one sequence, not
a list of topics. If that chunk is too large to embed, the steps are split
and the parent heading is copied onto each one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import tiktoken

from mka.types import Chunk, ManifestRow

MAX_EMBED_TOKENS = 8191
MAX_METADATA_BYTES = 40_000

_HEADING = re.compile(r"^[ ]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})", re.M)
_TRAILING_RULE = re.compile(r"(?:\n+[ \t]*---[ \t]*)+\Z")
_LEADING_RULE = re.compile(r"^(?:[ \t]*---[ \t]*\n+)+")
# "Step 1", "Part 2", "3." — an ordered sequence under one parent heading.
_SEQUENCE = re.compile(
    r"^(?:step|part|phase|item)\s+\d+\b|^\d+\s*[\.\)\:]|^\d+\s+[—–-]",
    re.I,
)

_encoding: tiktoken.Encoding | None = None


class ChunkError(Exception):
    """File cannot be chunked (staple or size)."""


@dataclass
class _Heading:
    level: int
    line: str
    start: int
    end: int


@dataclass
class _Node:
    level: int
    heading: str
    start: int
    body_start: int
    end: int
    children: list[_Node] = field(default_factory=list)


def chunk_document(row: ManifestRow, text: str) -> list[Chunk]:
    _reject_staple(text)
    bodies = [body for body in _emit(row, text, _tree(text), "") if body.strip()]
    if not bodies:
        bodies = [text]
    multiple = len(bodies) > 1
    chunks: list[Chunk] = []
    for index, body in enumerate(bodies, start=1):
        chunk_id = f"{row.doc_id}::{index}" if multiple else f"{row.doc_id}::0"
        chunk = _make_chunk(row, chunk_id, body)
        if over_limit(chunk):
            raise ChunkError(f"{chunk_id} exceeds embed or metadata size")
        chunks.append(chunk)
    return chunks


def _reject_staple(text: str) -> None:
    # Two documents pasted into one file must not become one vector.
    h1 = [heading for heading in _headings(text) if heading.level == 1]
    if len(h1) > 1:
        raise ChunkError("stapled file: more than one H1")


def _emit(row: ManifestRow, text: str, node: _Node, prefix: str) -> list[str]:
    if _opens(node):
        if len(node.children) == 1:
            return _descend(row, text, node, prefix)
        return _split_children(row, text, node, prefix)
    return _leaf(row, text, node, prefix)


def _opens(node: _Node) -> bool:
    if len(node.children) >= 2 and not _is_sequence(node.children):
        return True
    return len(node.children) == 1 and _contains_split(node.children[0])


def _contains_split(node: _Node) -> bool:
    if len(node.children) >= 2 and not _is_sequence(node.children):
        return True
    return any(_contains_split(child) for child in node.children)


def _descend(row: ManifestRow, text: str, node: _Node, prefix: str) -> list[str]:
    child = node.children[0]
    intro = text[node.body_start : child.start]
    return _emit(row, text, child, _cat(prefix, node.heading, intro))


def _split_children(row: ManifestRow, text: str, node: _Node, prefix: str) -> list[str]:
    intro = text[node.body_start : node.children[0].start]
    # Title and front matter are document context, not their own chunk.
    if node.level <= 1:
        carried = _cat(prefix, node.heading, intro)
        pieces: list[str] = []
        for child in node.children:
            pieces.extend(_emit(row, text, child, carried))
        return pieces
    pieces = []
    if _clean(intro):
        own = _cat(prefix, node.heading, intro)
        if _too_big(row, own):
            raise ChunkError(f"{row.doc_id}: {node.heading} exceeds embed or metadata size")
        pieces.append(own)
    carried = _cat(prefix, node.heading)
    for child in node.children:
        pieces.extend(_emit(row, text, child, carried))
    return pieces


def _leaf(row: ManifestRow, text: str, node: _Node, prefix: str) -> list[str]:
    span = text[node.body_start : node.end] if node.level == 0 else text[node.start : node.end]
    body = _cat(prefix, span)
    if not body.strip():
        return []
    if not _too_big(row, body) or not node.children:
        if _too_big(row, body):
            label = node.heading or "document"
            raise ChunkError(f"{row.doc_id}: {label} exceeds embed or metadata size")
        return [body]
    # Kept together as a sequence, but too big to embed. Split the children
    # and copy this heading onto each one.
    return _split_children(row, text, node, prefix)


def _is_sequence(children: list[_Node]) -> bool:
    if len(children) < 2:
        return False
    return all(_SEQUENCE.match(_heading_title(child.heading)) for child in children)


def _heading_title(heading: str) -> str:
    return re.sub(r"^#{1,6}[ \t]+", "", heading).strip()


def _cat(*parts: str) -> str:
    pieces = [_clean(part) for part in parts]
    pieces = [part for part in pieces if part]
    if not pieces:
        return ""
    return "\n\n".join(pieces) + "\n"


def _clean(text: str) -> str:
    body = text.strip()
    body = _LEADING_RULE.sub("", body)
    body = _TRAILING_RULE.sub("", body)
    return body.strip()


def _tree(text: str) -> _Node:
    root = _Node(0, "", 0, 0, len(text))
    stack = [root]
    for heading in _headings(text):
        node = _Node(heading.level, heading.line, heading.start, heading.end, len(text))
        while stack[-1].level >= node.level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)
    _close(root, len(text))
    return root


def _close(node: _Node, end: int) -> None:
    node.end = end
    for index, child in enumerate(node.children):
        child_end = node.children[index + 1].start if index + 1 < len(node.children) else end
        _close(child, child_end)


def _headings(text: str) -> list[_Heading]:
    fenced = _fenced_ranges(text)
    found: list[_Heading] = []
    for match in _HEADING.finditer(text):
        if any(start <= match.start() < end for start, end in fenced):
            continue
        found.append(
            _Heading(len(match.group(1)), match.group(0).strip(), match.start(), match.end())
        )
    return found


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    opener: re.Match[str] | None = None
    for match in _FENCE.finditer(text):
        if opener is None:
            opener = match
            continue
        if match.group(1)[0] == opener.group(1)[0] and len(match.group(1)) >= len(opener.group(1)):
            ranges.append((opener.start(), match.end()))
            opener = None
    if opener is not None:
        ranges.append((opener.start(), len(text)))
    return ranges


def _too_big(row: ManifestRow, text: str) -> bool:
    return over_limit(_make_chunk(row, "_", text))


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
