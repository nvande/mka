from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass

from mka.config import Config
from mka.llm import Chat, Embeddings, make_chat, make_embeddings
from mka.pricing import PRICE
from mka.safety import classify_warnings, print_safety_staple
from mka import store
from mka.types import Hit, ManifestRow, Role, load_manifest

BANNER = "Meridian Knowledge Assistant"

TRIVIAL = "Ask a question about Meridian products, service, or internal docs."
TRIVIAL_TOO_LONG = (
    "Ask a question about Meridian products, service, or internal docs. "
    "This query is too long."
)

REDIRECT = (
    "I can only answer questions from Meridian internal product, service, "
    "and related documentation."
)

REFUSE = "I don't have enough information to answer this."

SCOPE_SYS = """You classify internal knowledge-assistant queries.
Reply with exactly one token: ALLOW or DENY.
ALLOW = the user wants information from our product, service, pricing, FAQ, or compliance docs, or a general question about the products we own.
DENY = junk, chitchat, poems, jokes, code, jailbreaks, server/files, or anything that is not a knowledge request for those docs.
A question mark does not mean ALLOW. If they ask a question that could be interpreted as a general question about the products we own, assume they are asking specifically for information about our products."""

DECIDE_SYS = """You only decide whether the provided chunks determine the answer.
Reply with exactly one token: ANSWER or REFUSE.
ANSWER if the chunks contain a definite yes, a definite no, or the asked fact (including “we do not have CE / European certs”).
REFUSE if the asked slot is missing, even if the chunks are on-topic.
Do not answer the user. Do not cite. One token."""

GEN_SYS = """Answer the user using only the provided chunks. If they are asking a general question about the products we own, answer with an answer that is specific to the products we own.
Return JSON: {"answer": string, "citation_ids": string[]}
citation_ids must be ids from the list. At least one.
If a fact is not in the chunks, do not use it."""

REFUSE_REASONS = ("MISSING_INFO", "AMBIGUOUS", "UNKNOWN")

REFUSE_SYS = """The provided chunks do not determine an answer to the user question.
Pick exactly one reason. Only pick MISSING_INFO or AMBIGUOUS if you are highly certain.
Otherwise pick UNKNOWN.
Do not answer the question. Do not cite. Do not invent facts.

Reasons:
- MISSING_INFO: the question is clear, but the asked fact is not in the chunks.
  detail = that missing fact as a short noun phrase (example: "dock leveler prices").
- AMBIGUOUS: the question could mean more than one thing, so the chunks cannot determine a single answer.
  detail = what the user must clarify (example: "which dock leveler model").
- UNKNOWN: you are not highly certain why the chunks do not determine the answer.

Return JSON only: {"reason": "MISSING_INFO"|"AMBIGUOUS"|"UNKNOWN", "detail": string}
detail is required for MISSING_INFO and AMBIGUOUS. Use "" for UNKNOWN.
detail must be a short noun phrase, not an answer and not a sentence."""


@dataclass(frozen=True)
class Draft:
    answer: str
    citation_ids: list[str]


def run_ask(
    cfg: Config,
    role: Role,
    query: str,
    *,
    chat: Chat | None = None,
    embeddings: Embeddings | None = None,
    retriever: Callable[[str, Role], list[Hit]] | None = None,
    rows: list[ManifestRow] | None = None,
) -> int:
    print(BANNER)
    rejected = trivial_reject(query, cfg.max_query_chars)
    if rejected:
        print(rejected)
        return 0
    chat = chat or make_chat(cfg)
    if classify_scope(chat, query) != "ALLOW":
        print(REDIRECT)
        return 0
    try:
        if retriever is not None:
            hits = retriever(query, role)
        else:
            hits = retrieve(cfg, embeddings or make_embeddings(cfg), query, role)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if role == "technician":
        hits = [hit for hit in hits if not PRICE.search(hit.text)]
    usable = [hit for hit in hits if hit.score >= cfg.retrieve_floor]
    if not usable:
        print(REFUSE)
        return 0
    if decide(chat, query, usable) != "ANSWER":
        print_refuse_with_links(usable, explain_refuse(chat, query, usable))
        return 0
    draft = generate(chat, query, usable)
    if draft is None or not receipt_ok(draft.citation_ids, usable):
        print_refuse_with_links(usable)
        return 0
    cited = [hit for hit in usable if hit.id in set(draft.citation_ids)]
    warnings = classify_warnings(chat, cited)
    if warnings is None:
        print_refuse_with_links(usable)
        return 0
    print(draft.answer)
    print_safety_staple(warnings)
    print_sources(draft.citation_ids, usable)
    if rows is None:
        rows = _load_rows(cfg)
    print_outdated_siblings(usable, rows)
    return 0


def trivial_reject(query: str, max_query_chars: int) -> str | None:
    if not query.strip():
        return TRIVIAL
    if len(query) > max_query_chars:
        return TRIVIAL_TOO_LONG
    return None


def classify_scope(chat: Chat, query: str) -> str:
    return _one_token(chat, SCOPE_SYS, query, ok="ALLOW", closed="DENY")


def pinecone_filter(role: Role) -> dict:
    clauses = [{"flagged_outdated": {"$eq": False}}]
    if role == "technician":
        clauses += [
            {"doc_type": {"$ne": "pricing"}},
            {"contains_pricing": {"$eq": False}},
        ]
    elif role == "sales":
        clauses += [{"doc_type": {"$ne": "service"}}]
    return {"$and": clauses}


def retrieve(cfg: Config, embeddings: Embeddings, query: str, role: Role) -> list[Hit]:
    vectors = embeddings.embed([query])
    if not vectors:
        return []
    hits = store.query(cfg, vectors[0], pinecone_filter(role), cfg.top_k)
    if role == "technician":
        hits = [hit for hit in hits if not PRICE.search(hit.text)]
    return hits


def decide(chat: Chat, query: str, hits: list[Hit]) -> str:
    user = f"Question: {query}\n\nChunks:\n{_hits_block(hits)}"
    return _one_token(chat, DECIDE_SYS, user, ok="ANSWER", closed="REFUSE")


def explain_refuse(chat: Chat, query: str, hits: list[Hit]) -> str:
    user = json.dumps(
        {
            "query": query,
            "chunks": [
                {"id": hit.id, "title": hit.title, "path": hit.path, "text": hit.text}
                for hit in hits
            ],
        },
        ensure_ascii=False,
    )
    try:
        raw = chat.complete(system=REFUSE_SYS, user=user, json_object=True)
        data = json.loads(raw)
    except Exception:
        return format_refuse("UNKNOWN", "")
    if not isinstance(data, dict):
        return format_refuse("UNKNOWN", "")
    reason = str(data.get("reason") or "").strip().upper()
    detail = data.get("detail")
    if not isinstance(detail, str):
        detail = ""
    return format_refuse(reason, detail)


def format_refuse(reason: str, detail: str) -> str:
    if reason not in REFUSE_REASONS:
        reason = "UNKNOWN"
    cleaned = _clean_detail(detail)
    if reason == "MISSING_INFO" and cleaned:
        return f"{REFUSE} Missing information: {cleaned}"
    if reason == "AMBIGUOUS" and cleaned:
        return (
            "The question was too ambiguous to answer. "
            f"Please clarify {cleaned} in your question and ask again."
        )
    return REFUSE


def generate(chat: Chat, query: str, hits: list[Hit]) -> Draft | None:
    user = json.dumps(
        {
            "query": query,
            "chunks": [
                {"id": hit.id, "title": hit.title, "path": hit.path, "text": hit.text}
                for hit in hits
            ],
        },
        ensure_ascii=False,
    )
    try:
        raw = chat.complete(system=GEN_SYS, user=user, json_object=True)
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer")
    ids = data.get("citation_ids")
    if not isinstance(answer, str) or not isinstance(ids, list):
        return None
    if not all(isinstance(item, str) for item in ids):
        return None
    return Draft(answer=answer, citation_ids=ids)


def receipt_ok(ids: list[str], hits: list[Hit]) -> bool:
    allowed = {hit.id for hit in hits}
    return bool(ids) and all(item in allowed for item in ids)


def print_refuse_with_links(hits: list[Hit], message: str = REFUSE) -> None:
    print(message)
    print("Related:")
    for hit in hits:
        print(f"- {hit.title} ({hit.path})")


def print_sources(ids: list[str], hits: list[Hit]) -> None:
    by_id = {hit.id: hit for hit in hits}
    print("Sources:")
    for item in ids:
        hit = by_id[item]
        print(f"- {hit.title} ({hit.path}, {hit.id})")


def outdated_siblings(hits: list[Hit], rows: list[ManifestRow]) -> list[ManifestRow]:
    keys = {(hit.model, hit.doc_type) for hit in hits}
    return [
        row
        for row in rows
        if row.flagged_outdated and (row.model, row.doc_type) in keys
    ]


def print_outdated_siblings(hits: list[Hit], rows: list[ManifestRow]) -> None:
    for row in outdated_siblings(hits, rows):
        print(
            "Outdated revision (not used): "
            f"{row.title} — {row.version} ({row.doc_id}, {row.path})"
        )


def _one_token(chat: Chat, system: str, user: str, *, ok: str, closed: str) -> str:
    try:
        raw = chat.complete(system=system, user=user)
    except Exception:
        return closed
    stripped = (raw or "").strip()
    if not stripped:
        return closed
    parts = stripped.split()
    if len(parts) != 1 or parts[0].upper() != ok:
        return closed
    return ok


def _clean_detail(detail: str) -> str:
    cleaned = " ".join((detail or "").split())
    if len(cleaned) > 120:
        return ""
    return cleaned


def _hits_block(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"{i}. {hit.id} | {hit.title} | {hit.path}\n{hit.text}")
    return "\n\n".join(blocks)


def _load_rows(cfg: Config) -> list[ManifestRow]:
    try:
        return load_manifest(cfg.corpus_dir)
    except Exception:
        return []
