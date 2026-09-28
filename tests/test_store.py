from __future__ import annotations

from types import SimpleNamespace

import pytest

from mka import usage
from mka.store import delete_index, ensure_index, query, upsert, wipe_namespace
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
                        "flagged_outdated": True,
                        "contains_warning": True,
                        "warning_text": "Never exceed 2100 psi",
                    },
                )
            ],
            usage=SimpleNamespace(read_units=1),
        )


def test_upsert_batches_of_100(tmp_path, monkeypatch) -> None:
    fake = _FakeIndex()
    monkeypatch.setattr("mka.store._index", lambda cfg: fake)
    chunks = [Chunk(id=f"doc::{i}", text="t", metadata={"n": i}) for i in range(101)]
    cfg = make_config(tmp_path)
    with usage.track("ingest") as ledger:
        upsert(cfg, chunks, [[0.0]] * 101)
    assert [len(batch) for batch in fake.upserts] == [100, 1]
    assert fake.upserts[0][0]["id"] == "doc::0"
    assert [call.detail["vectors"] for call in ledger.pinecone] == [100, 1]


def test_delete_index_skips_when_missing(tmp_path, monkeypatch) -> None:
    class _FakeClient:
        def has_index(self, name: str) -> bool:
            del name
            return False

        def delete_index(self, name: str) -> None:
            raise AssertionError(f"should not delete {name}")

    monkeypatch.setattr("mka.store._client", lambda cfg: _FakeClient())
    assert delete_index(make_config(tmp_path)) is False


def test_delete_index_waits_until_gone(tmp_path, monkeypatch) -> None:
    state = {"present": True}

    class _FakeClient:
        def has_index(self, name: str) -> bool:
            del name
            return state["present"]

        def delete_index(self, name: str) -> None:
            del name
            state["present"] = False

    monkeypatch.setattr("mka.store._client", lambda cfg: _FakeClient())
    assert delete_index(make_config(tmp_path), wait=1, poll=0) is True


def test_query_maps_hit_and_records_stats(tmp_path, monkeypatch) -> None:
    fake = _FakeIndex()
    monkeypatch.setattr("mka.store._index", lambda cfg: fake)
    cfg = make_config(tmp_path)
    with usage.track("ask") as ledger:
        hits = query(cfg, [0.1], {"flagged_outdated": {"$eq": False}}, 8)
    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, Hit)
    assert hit.id == "spec_x::0"
    assert hit.score == 0.91
    assert hit.contains_warning is True
    assert hit.flagged_outdated is True
    assert hit.warning_text == "Never exceed 2100 psi"
    assert fake.last_query["include_metadata"] is True
    call = ledger.pinecone[0]
    assert call.op == "query"
    assert call.detail == {"namespace": "poc", "top_k": 8, "matches": 1, "read_units": 1}


def test_query_missing_index_explains_ingest(tmp_path, monkeypatch) -> None:
    class Missing:
        def query(self, **kwargs):
            raise RuntimeError("[404 NOT_FOUND] Resource mka-poc not found")

    monkeypatch.setattr("mka.store._index", lambda cfg: Missing())
    with pytest.raises(RuntimeError, match="does not exist") as excinfo:
        query(make_config(tmp_path), [0.1], {"doc_type": {"$ne": "service"}}, 8)
    message = str(excinfo.value)
    assert "mka-poc" in message
    assert "mka ingest" in message
    assert "404" not in message


def test_failed_call_is_recorded_once_with_an_error(tmp_path, monkeypatch) -> None:
    class Boom:
        def query(self, **kwargs):
            raise RuntimeError("pinecone down")

    monkeypatch.setattr("mka.store._index", lambda cfg: Boom())
    cfg = make_config(tmp_path)
    with usage.track("ask") as ledger:
        with pytest.raises(RuntimeError):
            query(cfg, [0.1], {"doc_type": {"$ne": "service"}}, 8)
    assert len(ledger.pinecone) == 1
    call = ledger.pinecone[0]
    assert call.op == "query"
    assert call.detail["error"] == "pinecone down"
    assert "matches" not in call.detail


def test_wipe_tolerates_missing_namespace(tmp_path, monkeypatch) -> None:
    class Index:
        def delete(self, **kwargs):
            raise RuntimeError("Namespace not found")

    monkeypatch.setattr("mka.store._index", lambda cfg: Index())
    wipe_namespace(make_config(tmp_path))


def test_ensure_index_records_existing_index(tmp_path, monkeypatch) -> None:
    class Client:
        def has_index(self, name: str) -> bool:
            del name
            return True

        def create_index(self, **kwargs):
            raise AssertionError(f"should not create {kwargs}")

    monkeypatch.setattr("mka.store._client", lambda cfg: Client())
    with usage.track("ingest") as ledger:
        ensure_index(make_config(tmp_path))
    assert ledger.pinecone[0].op == "ensure_index"
    assert ledger.pinecone[0].detail["created"] is False
