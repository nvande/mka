from __future__ import annotations

from pathlib import Path

import pytest

from mka.config import Config

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def make_config(corpus_dir: Path, pinecone_api_key: str = "") -> Config:
    return Config(
        corpus_dir=corpus_dir,
        pinecone_api_key=pinecone_api_key,
        pinecone_index="mka-poc",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_namespace="poc",
        llm_provider="openai",
        chat_model="gpt-5.4-nano",
        decision_model="gpt-6-luna",
        embed_model="text-embedding-3-small",
        embed_dim=1536,
        top_k=8,
        retrieve_floor=0.20,
        max_query_chars=4000,
    )


@pytest.fixture
def fixture_corpus() -> Path:
    return FIXTURE_CORPUS
