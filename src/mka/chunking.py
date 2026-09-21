from __future__ import annotations

from mka.types import Chunk, ManifestRow


def chunk_document(row: ManifestRow, text: str) -> list[Chunk]:
    del row, text
    raise NotImplementedError("chunking is not implemented yet")
