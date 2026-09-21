from __future__ import annotations

import json
import sys
from pathlib import Path

from mka.config import Config
from mka.types import ManifestRow, load_manifest


def run_scan(cfg: Config) -> int:
    manifest_path = cfg.corpus_dir / "manifest.json"
    if not cfg.corpus_dir.is_dir():
        print(f"error: corpus directory not found: {cfg.corpus_dir}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        rows = load_manifest(cfg.corpus_dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: cannot read manifest: {exc}", file=sys.stderr)
        return 1

    print("Conflict discovery is production-only.")
    print("Type 2 resolution is not agreed; this command does not search neighbors.")
    print()
    _print_flagged(rows, manifest_path)
    return 0


def _print_flagged(rows: list[ManifestRow], manifest_path: Path) -> None:
    flagged = [row for row in rows if row.flagged_outdated]
    print("Type 1 on this corpus is already flagged_outdated in the manifest:")
    if not flagged:
        print(f"- none listed in {manifest_path}")
        return
    for row in flagged:
        print(f"- {row.title} — {row.version} ({row.doc_id}, {row.path})")
