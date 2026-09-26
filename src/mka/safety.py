"""Hazard notes on the chunks the answer actually cited.

The model proposes verbatim excerpts. Python keeps an excerpt only when it
is a contiguous substring of that chunk, then drops catalog "Safety
features" lists. Explicit "For sales" / "For technicians" wording sets the
audience; the model's audience label is not used. If every proposed excerpt
fails the substring check, the answer is refused. Zero real warnings is a
clean pass, and the answer prints with no staple.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from mka.llm import Chat
from mka.types import Hit

WARN_SYS = """Find hazard warnings in these chunks, not product specs.
Include only DANGER, WARNING, CAUTION, Critical rules, Safety limits,
What NOT to do, lockout/tagout, never-exceed limits, and Notes that are
themselves warnings (including role-directed notes such as For sales).
Exclude catalog lists: "## Safety features", "## Safety", toe guards,
velocity fuses, night locks, maintenance struts, photo-eyes listed as
equipment, and any other feature/spec bullets. Those are specs, not warnings.
Return excerpts: [] for a specification chunk.
Do not copy a feature list because the heading says Safety.
Do not skip a real hazard warning on a cited procedure.
Reply JSON only:
{"warnings":[{"id": chunk_id, "excerpts":[{"text":"verbatim from the chunk","audience":"all"}]}]}
audience is all (default), sales, or technician.
Use sales when the note or its heading says For sales, Notes for field sales, or sales only.
Use technician when the note or its heading says For technicians or technician only.
Otherwise audience is all.
Copy excerpts exactly. Do not summarize.
If a chunk has no hazard warnings, return excerpts: []."""

_SPEC_HEADING = re.compile(
    r"^(?:safety features|safety|features|standard features|safety equipment)$",
    re.I,
)
_WARNING_HEADING = re.compile(
    r"^(?:safety limits|critical rules|what not to do|danger|warning|caution)\b",
    re.I,
)

_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.M)
_SALES_NOTE = re.compile(
    r"\b(?:for\s+(?:field\s+)?sales|notes?\s+for\s+(?:field\s+)?sales|"
    r"field\s+sales|sales\s+only)\b",
    re.I,
)
_TECH_NOTE = re.compile(
    r"\b(?:for\s+technicians?|technician\s+only|techs?\s+only)\b",
    re.I,
)


@dataclass(frozen=True)
class WarningExcerpt:
    text: str
    audience: str


def excerpt_in_source(excerpt: str, hit: Hit) -> bool:
    # warning_text is empty at ingest. It is included so a cached excerpt
    # still has to be verbatim from this record, not a paraphrase.
    blob = hit.text + "\n" + (hit.warning_text or "")
    return bool(excerpt) and excerpt in blob


def heading_before(excerpt: str, text: str) -> str:
    idx = text.find(excerpt)
    if idx < 0:
        return ""
    headings = _HEADING.findall(text[:idx])
    return headings[-1].strip() if headings else ""


def infer_audience(excerpt: str, hit: Hit) -> str:
    # The heading or the note itself wins over whatever audience the model
    # returned. classify_warnings discards that label and calls this.
    heading = heading_before(excerpt, hit.text)
    if not heading and hit.warning_text:
        heading = heading_before(excerpt, hit.warning_text)
    probe = f"{heading}\n{excerpt}"
    if _SALES_NOTE.search(probe):
        return "sales"
    if _TECH_NOTE.search(probe):
        return "technician"
    return "all"


def excerpt_visible(audience: str, role: str) -> bool:
    return audience == "all" or audience == role


def is_catalog_spec(excerpt: str, hit: Hit) -> bool:
    # "## Safety" on a spec is a parts list (toe guards, photo-eyes).
    # "## Safety limits" on a procedure is a hazard. The heading decides.
    # A real warning heading short-circuits so a spec heading elsewhere
    # in the excerpt cannot hide it.
    heading = heading_before(excerpt, hit.text)
    if not heading and hit.warning_text:
        heading = heading_before(excerpt, hit.warning_text)
    inner = [part.strip() for part in _HEADING.findall(excerpt)]
    for candidate in (heading, *inner):
        if not candidate:
            continue
        if _WARNING_HEADING.match(candidate):
            return False
        if _SPEC_HEADING.match(candidate):
            return True
    return False


def classify_warnings(
    chat: Chat, cited: list[Hit], query: str = ""
) -> list[tuple[Hit, list[WarningExcerpt]]] | None:
    user = json.dumps(
        {
            "query": query,
            "chunks": [
                {
                    "id": hit.id,
                    "title": hit.title,
                    "path": hit.path,
                    "text": hit.text,
                    "warning_text": hit.warning_text,
                }
                for hit in cited
            ]
        },
        ensure_ascii=False,
    )
    raw = chat.complete(system=WARN_SYS, user=user, json_object=True)
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    rows = data.get("warnings", [])
    if not isinstance(rows, list):
        return None
    by_id = {hit.id: hit for hit in cited}
    out: list[tuple[Hit, list[WarningExcerpt]]] = []
    in_source = 0
    missed = 0
    for row in rows:
        if not isinstance(row, dict):
            return None
        hit = by_id.get(row.get("id"))
        if hit is None:
            return None
        accepted: list[WarningExcerpt] = []
        excerpts = row.get("excerpts") or []
        if not isinstance(excerpts, list):
            return None
        for item in excerpts:
            parsed = _parse_excerpt(item)
            if parsed is None:
                return None
            text, _audience = parsed
            if not excerpt_in_source(text, hit):
                missed += 1
                continue
            in_source += 1
            # Quoted from the chunk, but it is a feature list. Drop it.
            # An empty result after this drop is still a pass.
            if is_catalog_spec(text, hit):
                continue
            accepted.append(
                WarningExcerpt(text=text, audience=infer_audience(text, hit))
            )
        out.append((hit, accepted))
    # The model claimed warnings and none of them appear in the chunk.
    # That is not "no warnings." Refuse the answer.
    if missed and in_source == 0:
        return None
    return out


def classify_source_warnings(
    chat: Chat, *, doc_id: str, title: str, path: str, text: str
) -> list[WarningExcerpt] | None:
    """Ingest pass. Classify a whole source file once, before it is chunked.

    The file still carries the headings a later split can strip, so the
    audience resolved here is better grounded than one inferred from a lone
    chunk. Same prompt and same Python verification as the ask-time pass.
    None is a classifier error: leave the cache empty and let ask gate it.
    """
    whole = Hit(
        id=doc_id,
        score=1.0,
        text=text,
        title=title,
        path=path,
        doc_id=doc_id,
        model="",
        doc_type="",
        flagged_outdated=False,
        contains_warning=False,
        warning_text="",
    )
    out = classify_warnings(chat, [whole])
    if out is None:
        return None
    # No rows at all is a file with nothing to warn about, not a failure.
    return out[0][1] if out else []


def cached_warnings(cited: list[Hit]) -> list[tuple[Hit, list[WarningExcerpt]]] | None:
    """Rebuild Check 5 from the ingest cache, with no model call.

    None means the cache does not cover this set and the live pass has to
    run. Every excerpt is still checked against the record it rides on, so a
    vector that drifted from its cache falls back instead of printing a note
    nothing verified.
    """
    out: list[tuple[Hit, list[WarningExcerpt]]] = []
    for hit in cited:
        if not hit.warnings_cached:
            return None
        if len(hit.warning_excerpts) != len(hit.warning_audiences):
            return None
        accepted: list[WarningExcerpt] = []
        for text, audience in zip(hit.warning_excerpts, hit.warning_audiences):
            if not excerpt_in_source(text, hit):
                return None
            accepted.append(WarningExcerpt(text=text, audience=audience))
        out.append((hit, accepted))
    return out


def resolve_warnings(
    chat: Chat, cited: list[Hit], query: str = ""
) -> list[tuple[Hit, list[WarningExcerpt]]] | None:
    """Check 5. The ingest cache when it covers every cited chunk, else the model."""
    cached = cached_warnings(cited)
    if cached is not None:
        return cached
    return classify_warnings(chat, cited, query)


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
        print(
            f"⚠ WARNING — {hit.title} ({hit.path}). "
        )
        print()
        for excerpt in unique:
            print(excerpt)
            print()


def _parse_excerpt(item: object) -> tuple[str, str | None] | None:
    if isinstance(item, str):
        return item, None
    if not isinstance(item, dict):
        return None
    text = item.get("text")
    if text is None:
        text = item.get("excerpt")
    if not isinstance(text, str):
        return None
    audience = item.get("audience")
    if audience is not None and not isinstance(audience, str):
        return None
    return text, audience


