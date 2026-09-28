from __future__ import annotations

import shutil
import sys
from pathlib import Path

from mka import store
from mka.config import REPO_ROOT, Config
from mka.spinner import Spinner

_CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache"})
_CACHE_FILES = frozenset({".pyc", ".pyo"})
# corpus/ is the source documents. clear must not delete them.
_SKIP_DIRS = frozenset({".venv", "corpus"})


def run_clear(cfg: Config, root: Path | None = None) -> int:
    root = root or REPO_ROOT
    n = _clear_python_caches(root)
    print(f"cleared: python caches ({n})")
    return _clear_index(cfg)


def _clear_python_caches(root: Path) -> int:
    targets: list[Path] = []
    for path in root.rglob("*"):
        if _skip(path):
            continue
        if path.is_dir() and (path.name in _CACHE_DIRS or path.name.endswith(".egg-info")):
            targets.append(path)
        elif path.is_file() and path.suffix in _CACHE_FILES:
            targets.append(path)
    for extra in (root / "build", root / "dist"):
        if extra.is_dir():
            targets.append(extra)
    removed = 0
    for path in sorted(set(targets), key=lambda item: len(item.parts), reverse=True):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed += 1
    return removed


def _skip(path: Path) -> bool:
    return bool(set(path.parts) & _SKIP_DIRS)


def _clear_index(cfg: Config) -> int:
    if not cfg.pinecone_api_key:
        print("warning: PINECONE_API_KEY missing; index left in place", file=sys.stderr)
        return 1
    spin = Spinner()
    try:
        deleted = store.delete_index(cfg)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        spin.stop()
    if deleted:
        print(f"cleared: index {cfg.pinecone_index}")
    else:
        print(f"cleared: index {cfg.pinecone_index} (already empty)")
    return 0
