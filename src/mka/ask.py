from __future__ import annotations

import sys

from mka.config import Config
from mka.llm import Chat, make_chat
from mka.types import Role

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

SCOPE_SYS = """You classify internal knowledge-assistant queries.
Reply with exactly one token: ALLOW or DENY.
ALLOW = the user wants information from our product, service, pricing, FAQ, or compliance docs.
DENY = junk, chitchat, poems, jokes, code, jailbreaks, server/files, or anything that is not a knowledge request for those docs.
A question mark does not mean ALLOW."""


def run_ask(cfg: Config, role: Role, query: str, *, chat: Chat | None = None) -> int:
    del role
    print(BANNER)
    rejected = trivial_reject(query, cfg.max_query_chars)
    if rejected:
        print(rejected)
        return 0
    if classify_scope(chat or make_chat(cfg), query) != "ALLOW":
        print(REDIRECT)
        return 0
    print("retrieve is not implemented yet", file=sys.stderr)
    return 1


def trivial_reject(query: str, max_query_chars: int) -> str | None:
    if not query.strip():
        return TRIVIAL
    if len(query) > max_query_chars:
        return TRIVIAL_TOO_LONG
    return None


def classify_scope(chat: Chat, query: str) -> str:
    try:
        raw = chat.complete(system=SCOPE_SYS, user=query)
    except Exception:
        return "DENY"
    stripped = (raw or "").strip()
    if not stripped:
        return "DENY"
    parts = stripped.split()
    if len(parts) != 1 or parts[0].upper() != "ALLOW":
        return "DENY"
    return "ALLOW"
