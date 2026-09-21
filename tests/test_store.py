from __future__ import annotations

from types import SimpleNamespace

from mka.store import query, upsert
from mka.types import Chunk, Hit

from conftest import make_config


class _FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[list[dict]] = []
        self.last_query: dict | None = None

    def upsert(self, vectors: list[dict], namespace: str) -> None:
        del namespace
        self.upserts.append(vectors)

    def query(self, **kwargs):
        self.last_query = kwargs
        return SimpleNamespace(
            matches=[
                SimpleNamespace(
                    id="spec_x::0",
                    score=0.91,
                    metadata={
                        "text": "body",
                        "title": "Spec",
                        "path": "docs/spec_x.md",
                        "doc_id": "spec_x",
                        "model": "MD-7000",
                        "doc_type": "spec",
                        "contains_warning": True,
                        "warning_text": "Never exceed 2100 psi",
                    },
                )
            ]
        )


def test_upsert_batches_of_100(tmp_path, monkeypatch) -> None:
    fake = _FakeIndex()
    monkeypatch.setattr("mka.store._index", lambda cfg: fake)
    chunks = [
        Chunk(id=f"doc::{i}", text="t", metadata={"n": i}) for i in range(101)
    ]
    upsert(make_config(tmp_path), chunks, [[0.0]] * 101)
    assert [len(batch) for batch in fake.upserts] == [100, 1]
    assert fake.upserts[0][0]["id"] == "doc::0"


def test_query_maps_hit_including_warning(tmp_path, monkeypatch) -> None:
    fake = _FakeIndex()
    monkeypatch.setattr("mka.store._index", lambda cfg: fake)
    hits = query(make_config(tmp_path), [0.1], {"flagged_outdated": {"$eq": False}}, 8)
    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, Hit)
    assert hit.id == "spec_x::0"
    assert hit.score == 0.91
    assert hit.contains_warning is True
    assert hit.warning_text == "Never exceed 2100 psi"
    assert fake.last_query["include_metadata"] is True
