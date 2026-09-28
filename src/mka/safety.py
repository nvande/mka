"""Find safety warnings in the documents and print the ones that apply under the answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mka.chunking import over_limit
from mka.types import Chunk, Hit

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)
_HAZARD_HEADING = re.compile(
    r"^(?:safety limits?|critical rules?|what not to do|danger|warning|caution|"
    r"hazards?|safety (?:rules|notes|precautions)|lockout)\b",
    re.I,
)
_SALES = re.compile(r"\b(?:notes?\s+for\s+(?:field\s+)?sales|for\s+sales|sales(?:\s+team)?\s+only)\b", re.I)
_TECH = re.compile(r"\b(?:for\s+technicians?|technicians?\s+only)\b", re.I)
_ADMONITION_FIRST_LINE = re.compile(r"^>.*\b(?:DANGER|WARNING|CAUTION)\b", re.M)
_RESTRICTION_LINE = re.compile(
    r"^\*\*(?:Audience|Classification):\*\*.*\b(?:do not (?:share|distribute)|without approval)\b.*$",
    re.I | re.M,
)
_RULE = re.compile(r"^---[ \t]*$", re.M)


@dataclass(frozen=True)
class WarningExcerpt:
    text: str
    audience: str


def extract_warnings(text: str) -> list[WarningExcerpt]:
    """Every marked hazard or handling note in a source text, verbatim, in order."""
    found: list[tuple[int, WarningExcerpt]] = []
    found += _sections(text)
    found += _admonitions(text)
    found += _restriction_lines(text)
    found.sort(key=lambda item: item[0])
    out: list[WarningExcerpt] = []
    seen: set[str] = set()
    for _start, excerpt in found:
        if excerpt.text not in seen:
            seen.add(excerpt.text)
            out.append(excerpt)
    return out


def _sections(text: str) -> list[tuple[int, WarningExcerpt]]:
    headings = list(_HEADING.finditer(text))
    out: list[tuple[int, WarningExcerpt]] = []
    for i, match in enumerate(headings):
        level, title = len(match.group(1)), match.group(2)
        audience = _audience(title)
        if not (_HAZARD_HEADING.match(title) or audience != "all"):
            continue
        start = match.end()
        end = len(text)
        for later in headings[i + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        rule = _RULE.search(text, start, end)
        if rule:
            end = rule.start()
        body = text[start:end].strip()
        if body:
            out.append((start, WarningExcerpt(body, audience)))
    return out


def _admonitions(text: str) -> list[tuple[int, WarningExcerpt]]:
    out: list[tuple[int, WarningExcerpt]] = []
    for match in _ADMONITION_FIRST_LINE.finditer(text):
        start = match.start()
        end = start
        for line in text[start:].splitlines(keepends=True):
            if not line.startswith(">"):
                break
            end += len(line)
        block = text[start:end].strip()
        out.append((start, WarningExcerpt(block, _audience(block))))
    return out


def _restriction_lines(text: str) -> list[tuple[int, WarningExcerpt]]:
    return [
        (match.start(), WarningExcerpt(match.group(0).strip(), _audience(match.group(0))))
        for match in _RESTRICTION_LINE.finditer(text)
    ]


def _audience(probe: str) -> str:
    if _SALES.search(probe):
        return "sales"
    if _TECH.search(probe):
        return "technician"
    return "all"


def assign_warnings(excerpts: list[WarningExcerpt], chunks: list[Chunk]) -> int:
    """Copy one file's notes onto its chunks by the scope rule. Returns chunks cached."""
    rows: dict[str, list[tuple[str, str]]] = {chunk.id: [] for chunk in chunks}
    for excerpt in excerpts:
        homes = [chunk for chunk in chunks if excerpt.text in chunk.text]
        for chunk in homes or chunks:
            rows[chunk.id].append((excerpt.text, excerpt.audience))
    return sum(attach_warnings(chunk, rows[chunk.id]) for chunk in chunks)


def attach_warnings(chunk: Chunk, rows: list[tuple[str, str]]) -> bool:
    """Store (excerpt, audience) rows on a record. False leaves it untouched.

    warning_text is the raw join so excerpt_in_source still sees a verbatim
    substring for a header note the chunk body does not contain.
    """
    saved = dict(chunk.metadata)
    chunk.metadata.update(
        {
            "contains_warning": bool(rows),
            "warning_text": "\n\n".join(text for text, _ in rows),
            "warnings_cached": True,
            "warning_excerpts": [text for text, _ in rows],
            "warning_audiences": [audience for _, audience in rows],
        }
    )
    if over_limit(chunk):
        chunk.metadata.clear()
        chunk.metadata.update(saved)
        return False
    return True


def excerpt_in_source(excerpt: str, hit: Hit) -> bool:
    blob = hit.text + "\n" + (hit.warning_text or "")
    return bool(excerpt) and excerpt in blob


def cached_warnings(cited: list[Hit]) -> list[tuple[Hit, list[WarningExcerpt]]] | None:
    """Notes from the ingest cache. None if any cited record was not cached
    or its cache no longer matches the record."""
    out: list[tuple[Hit, list[WarningExcerpt]]] = []
    for hit in cited:
        if not hit.warnings_cached:
            return None
        if len(hit.warning_excerpts) != len(hit.warning_audiences):
            return None
        rows: list[WarningExcerpt] = []
        for text, audience in zip(hit.warning_excerpts, hit.warning_audiences):
            if not excerpt_in_source(text, hit):
                return None
            rows.append(WarningExcerpt(text, audience))
        out.append((hit, rows))
    return out


def resolve_warnings(cited: list[Hit]) -> list[tuple[Hit, list[WarningExcerpt]]]:
    """The ingest cache when it covers every cited record, else extract from
    each record's own text. The fallback cannot see a header note the
    splitter dropped; re-running ingest restores it."""
    cached = cached_warnings(cited)
    if cached is not None:
        return cached
    return [(hit, extract_warnings(hit.text)) for hit in cited]


def excerpt_visible(audience: str, role: str) -> bool:
    return audience == "all" or audience == role


def print_safety_staple(
    warnings: list[tuple[Hit, list[WarningExcerpt]]], role: str
) -> None:
    printed: set[tuple[str, str]] = set()
    for hit, excerpts in warnings:
        unique: list[str] = []
        for item in excerpts:
            if not excerpt_visible(item.audience, role):
                continue
            key = (hit.doc_id, item.text)
            if key in printed:
                continue
            printed.add(key)
            unique.append(item.text)
        if not unique:
            continue
        print(f"⚠ WARNING — {hit.title} ({hit.path}).")
        print()
        for excerpt in unique:
            print(excerpt)
            print()
