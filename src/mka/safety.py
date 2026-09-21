from __future__ import annotations

import json
from typing import Any

from mka.llm import Chat
from mka.types import Hit

WARN_SYS = """Find every safety note or general warning in these chunks.
Include DANGER, WARNING, CAUTION, Critical rules, Safety limits,
What NOT to do, and Notes that are themselves warnings.
Exclude product "Safety features" and sales notes that are not hazards.
Reply JSON only: {"warnings":[{"id": chunk_id, "excerpts":["verbatim from the chunk"]}]}
Copy excerpts exactly. Do not summarize. Do not decide they are irrelevant to the question.
If a chunk has no safety notes or warnings, return excerpts: []."""


def excerpt_in_source(excerpt: str, hit: Hit) -> bool:
    blob = hit.text + "\n" + (hit.warning_text or "")
    return bool(excerpt) and excerpt in blob


def classify_warnings(chat: Chat, cited: list[Hit]) -> list[tuple[Hit, list[str]]] | None:
    user = json.dumps(
        {
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
    try:
        raw = chat.complete(system=WARN_SYS, user=user, json_object=True)
    except Exception:
        return None
    data = _parse_json(raw)
    if data is None:
        return None
    rows = data.get("warnings", [])
    if not isinstance(rows, list):
        return None
    by_id = {hit.id: hit for hit in cited}
    out: list[tuple[Hit, list[str]]] = []
    claimed = 0
    kept = 0
    for row in rows:
        if not isinstance(row, dict):
            return None
        hit = by_id.get(row.get("id"))
        if hit is None:
            return None
        accepted: list[str] = []
        excerpts = row.get("excerpts") or []
        if not isinstance(excerpts, list):
            return None
        for excerpt in excerpts:
            if not isinstance(excerpt, str):
                return None
            claimed += 1
            if excerpt_in_source(excerpt, hit):
                accepted.append(excerpt)
                kept += 1
        out.append((hit, accepted))
    if claimed and kept == 0:
        return None
    return out


def print_safety_staple(warnings: list[tuple[Hit, list[str]]]) -> None:
    printed: set[tuple[str, str]] = set()
    for hit, excerpts in warnings:
        unique: list[str] = []
        for excerpt in excerpts:
            key = (hit.doc_id, excerpt)
            if key in printed:
                continue
            printed.add(key)
            unique.append(excerpt)
        if not unique:
            continue
        print(
            f"⚠ SAFETY — required warnings from {hit.title} ({hit.path}). "
            "Read every time. Not optional."
        )
        print()
        for excerpt in unique:
            print(excerpt)
            print()


def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
