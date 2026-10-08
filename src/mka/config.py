from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("pyproject.toml not found")


REPO_ROOT = _repo_root()
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus"


@dataclass(frozen=True)
class Config:
    corpus_dir: Path
    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    llm_provider: str
    chat_model: str
    decision_model: str
    embed_model: str
    embed_dim: int
    top_k: int
    retrieve_floor: float
    max_query_chars: int


def load_config() -> Config:
    load_dotenv(REPO_ROOT / ".env")
    return Config(
        corpus_dir=Path(os.getenv("CORPUS_DIR", str(DEFAULT_CORPUS_DIR)))
        .expanduser()
        .resolve(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index=os.getenv("PINECONE_INDEX", "mka-poc"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "poc"),
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-5.4-nano"),
        # Scope classification is a fixed choice, so it uses the Decisions API
        # model instead of a chat completion.
        decision_model=os.getenv("DECISION_MODEL", "gpt-6-luna"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        embed_dim=int(os.getenv("EMBED_DIM", "1536")),
        # How many neighbors the first search keeps. A question can need every
        # section of a document, so chunks are not dropped by document.
        top_k=int(os.getenv("TOP_K", "16")),
        retrieve_floor=float(os.getenv("RETRIEVE_FLOOR", "0.20")),
        max_query_chars=int(os.getenv("MAX_QUERY_CHARS", "4000")),
    )
