"""Closed catalog fields copied onto each vector.

flagged_outdated is stored and is not a retrieve filter. Ask uses it as
the winner rule when an old revision and a current one disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["sales", "technician"]


@dataclass(frozen=True)
class ManifestRow:
    doc_id: str
    path: str
    title: str
    doc_type: str
    audience: str
    model: str
    last_updated: str
    version: str
    flagged_outdated: bool


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    text: str
    title: str
    path: str
    doc_id: str
    model: str
    doc_type: str
    flagged_outdated: bool
    contains_warning: bool
    warning_text: str
    # Check 5 cache. warnings_cached false means ingest never classified this
    # file, so ask runs the live pass for the whole cited set.
    warnings_cached: bool = False
    warning_excerpts: tuple[str, ...] = ()
    warning_audiences: tuple[str, ...] = ()


def load_manifest(corpus_dir: Path) -> list[ManifestRow]:
    payload = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[ManifestRow] = []
    for raw in payload["documents"]:
        rows.append(
            ManifestRow(
                doc_id=raw["doc_id"],
                path=raw["path"],
                title=raw["title"],
                doc_type=raw["doc_type"],
                audience=raw["audience"],
                model=raw["model"],
                last_updated=raw["last_updated"],
                version=raw["version"],
                flagged_outdated=bool(raw["flagged_outdated"]),
            )
        )
    return rows
