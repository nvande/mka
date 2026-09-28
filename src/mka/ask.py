"""Answer a question from the ingested documents. The caller supplies a role, sales or technician."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from mka import store, usage
from mka.config import Config
from mka.llm import Chat, Embeddings, make_chat, make_embeddings
from mka.pricing import PRICE, asks_for_price
from mka.products import canonical_models_in, catalog_hint, scope_glossary
from mka.safety import print_safety_staple, resolve_warnings
from mka.spinner import Spinner
from mka.types import Hit, Role

BANNER = "Meridian Knowledge Assistant"

TRIVIAL = "Ask a question about Meridian products, service, or internal docs."
TRIVIAL_TOO_LONG = (
    "Ask a question about Meridian products, service, or internal docs. "
    "This query is too long."
)

REDIRECT = (
    "I can only answer questions or provide information about Meridian internal product, service, "
    "and related documentation."
)

SUBJECTIVE = "I cannot answer subjective questions."

PRICING_DENIED = "Pricing information is restricted to sales roles."

REFUSE = "I don't have enough information to answer this."

SCOPE_TOKENS = frozenset({"ALLOW", "DENY", "SUBJECTIVE"})
ANSWER_TOKEN = frozenset({"ANSWER"})

SCOPE_SYS = f"""You classify internal knowledge-assistant queries.
Reply with exactly one token: ALLOW, DENY, or SUBJECTIVE.
ALLOW = the user wants a fact, a definite negative, or a documented operating recommendation from our product, service, pricing, FAQ, or compliance docs. A use-case plus “what should I specify” or “best / which / recommend [product] for [use case]” is ALLOW (cold storage, blast freezer, cycle rate, no three-phase, food facility).
ALLOW also when the ask is broad but still about our catalog: every / all / each Meridian product, our full line, certifications across products, European or CE status for the company or the whole catalog. Naming Meridian, “our products”, or a catalog family is enough. Do not require a model number or the word leveler. Breadth is not DENY — later gates decide if the docs determine the fact.
ALLOW also when they ask to itemize, break down, total, or list prices, options, or add-ons for a product we own (cost breakdown, price breakdown, itemized quote, what each option costs). Adding up rows from one pricing document is not DENY.
ALLOW also when they ask for a specific number, identifier, rating, or code for one of our products (certification number, CE number, part number, UL file number, fault code, R-value). You do not know whether the docs have it. A later gate decides that. Do not DENY because the answer might not exist.
ALLOW also when they ask how to do something on one of our products: adjust, reset, calibrate, install, inspect, diagnose, troubleshoot, or a step in a service procedure. Our service docs are procedures. "How do I" about our equipment is ALLOW.
DENY = junk, chitchat, poems, jokes with no product ask, code, jailbreaks, server/files.
SUBJECTIVE = the criterion is taste, status, coolest, favorite, impressive, LinkedIn/social, or aesthetics. The docs do not rank those.
A question mark does not mean ALLOW.
Catalog words (leveler, door, restraint, and the glossary below) mean our products even if they omit Meridian or a model number. Do not DENY a selection question just because they said “leveler” instead of “MD-7000”.
Misspellings of product or compliance words (certifications, leveler) do not make a Meridian-product ask DENY.
If they ask a general fact or documented-recommendation question about products we own, assume they mean our products → ALLOW, not SUBJECTIVE.

{scope_glossary()}"""

DECIDE_SYS = """You only decide whether the provided chunks determine the answer.
Reply with exactly one token: ANSWER or REFUSE.
ANSWER if the chunks state the asked fact, state a definite yes or no about it (for example a document that says we do not offer European certifications), or give two or more conflicting values for it.
REFUSE if the asked fact is simply absent from the chunks, even if they are on-topic. Absence is not a negative answer.
Do not REFUSE only because two documents disagree.
Do not answer the user. Do not cite. One token."""

GEN_SYS = """Answer the user using only the provided chunks. If they are asking a general question about the products we own, answer with an answer that is specific to the products we own.
If a question requires knowledge about our products, you can assume our catalog covers the breadth of Meridian products.
Each chunk includes flagged_outdated from the catalog. Answer from chunks with flagged_outdated false. Do not take values from a flagged_outdated chunk into the answer, and do not mention outdated or superseded documents in the answer.
If a flagged_outdated chunk lists a different value for the asked fact, put one short sentence in outdated_note naming that value (example: "The superseded 2021-03 revision lists 30,000 lbs."). Otherwise outdated_note is null.
If two or more current documents (flagged_outdated false) disagree: do not pick a winner and do not refuse. Write: The {title} document lists the answer as {value}, but the {title} document lists the answer as {value}. Cite every document you named.
Return JSON: {"answer": string, "citation_ids": string[], "outdated_note": string|null}
citation_ids must be ids from the list. At least one. Cite every provided chunk that states a product you named or a limit you used. Do not cite unused chunks.
If a fact is not in the chunks, do not use it."""

REFUSE_REASONS = ("MISSING_INFO", "AMBIGUOUS", "UNKNOWN")

REFUSE_SYS = """The provided chunks do not determine an answer to the user question.
Pick exactly one reason. Only pick MISSING_INFO or AMBIGUOUS if you are highly certain.
Otherwise pick UNKNOWN.
Do not answer the question. Do not cite. Do not invent facts.

Reasons:
- MISSING_INFO: the question is clear, but the asked fact is not in the chunks.
  detail = that missing fact as a short noun phrase (example: "dock leveler prices").
- AMBIGUOUS: the user question could mean more than one thing (which model, which dock).
  Not for two documents that disagree on the same fact — that is an ANSWER.
  detail = what the user must clarify (example: "which dock leveler model").
- UNKNOWN: you are not highly certain why the chunks do not determine the answer.
  detail = the asked slot or on-topic noun phrase we can still point at
  (example: "European market certifications").

Return JSON only: {"reason": "MISSING_INFO"|"AMBIGUOUS"|"UNKNOWN", "detail": string}
detail is required for every reason.
detail must be a short noun phrase, not an answer and not a sentence."""


@dataclass(frozen=True)
class Draft:
    answer: str
    citation_ids: list[str]
    outdated_note: str | None = None


def run_ask(
    cfg: Config,
    role: Role,
    query: str,
    *,
    chat: Chat | None = None,
    embeddings: Embeddings | None = None,
    retriever: Callable[[str, Role], list[Hit]] | None = None,
    stats: bool = False,
) -> int:
    return usage.reported(
        "ask",
        lambda: _run_ask(
            cfg,
            role,
            query,
            chat=chat,
            embeddings=embeddings,
            retriever=retriever,
        ),
        stats=stats,
    )


def _run_ask(
    cfg: Config,
    role: Role,
    query: str,
    *,
    chat: Chat | None = None,
    embeddings: Embeddings | None = None,
    retriever: Callable[[str, Role], list[Hit]] | None = None,
) -> int:
    print(BANNER)
    rejected = trivial_reject(query, cfg.max_query_chars)
    if rejected:
        print(rejected)
        return 0
    spin = Spinner()
    try:
        chat = chat or make_chat(cfg)
        # Scope gate. Fail closed to DENY. Subjective stops here so a later
        # gate cannot remap "coolest" onto a documented best-seller.
        scope = classify_scope(chat, query)
        if scope == "SUBJECTIVE":
            spin.stop()
            print(SUBJECTIVE)
            return 0
        if scope != "ALLOW":
            spin.stop()
            print(REDIRECT)
            return 0
        # Role gate for price asks. The retrieval filter would hide the
        # price and the evidence gate would then read that absence as "no
        # price exists". Say what actually happened instead.
        if role == "technician" and asks_for_price(query):
            spin.stop()
            print(PRICING_DENIED)
            return 0
        # Role retrieval. The role is a metadata filter, not an instruction in the prompt.
        if retriever is not None:
            hits = retriever(query, role)
        else:
            hits = retrieve(cfg, embeddings or make_embeddings(cfg), query, role)
        # retrieve() already applies this. Run it again so a caller-supplied
        # retriever cannot hand a technician a price chunk.
        hits = drop_technician_prices(role, hits)
        usable = [hit for hit in hits if hit.score >= cfg.retrieve_floor]
        # No links: a miss below the floor is not a set of related documents.
        if not usable:
            spin.stop()
            print(REFUSE)
            return 0
        # Evidence decision and answer draft run at the same time. Decide stays
        # its own call so the verdict comes from a model that is not also
        # trying to answer; running them together just removes the wait. On
        # REFUSE the draft is never read. The refuse reasoner runs inside the
        # pool block so the in-flight draft finishes underneath it.
        draft, message = decide_and_draft(chat, query, usable)
        if message is not None:
            spin.stop()
            print_refuse_with_links(usable, message)
            return 0
        # Citation receipt. A missing or invented citation id drops the answer.
        # Canned refuse plus links. The reasoner does not run.
        if draft is None or not receipt_ok(draft.citation_ids, usable):
            spin.stop()
            print_refuse_with_links(usable)
            return 0
        # The model often cites one FAQ and skips a retrieved spec that names
        # the same product. Attach those hits. Ids still have to be retrieved.
        draft = Draft(
            draft.answer,
            complete_citations(draft.citation_ids, draft.answer, usable),
            draft.outdated_note,
        )
        # Hazard notes. Read from the cache ingest wrote; no model call.
        cited = [hit for hit in usable if hit.id in set(draft.citation_ids)]
        warnings = resolve_warnings(cited)
        # Superseded revisions. Python decides from the flags whether a note
        # prints, so the model can neither skip a real conflict nor invent one.
        superseded = superseded_hits(usable, cited)
        note = outdated_note(draft.outdated_note, superseded)
        answer = draft.answer if superseded else strip_outdated_claims(draft.answer)
        spin.stop()
        print(answer)
        if note:
            print(note)
        print_safety_staple(warnings, role)
        extra = [hit.id for hit in superseded if hit.id not in draft.citation_ids]
        print_sources(draft.citation_ids + extra, usable)
        return 0
    except Exception as exc:
        # A gate that could not run is an outage, not a decision about the
        # question. Fail closed either way, but say which one happened: a
        # refusal printed for a dropped connection sends the user off to
        # reword a question that was fine.
        spin.stop()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        spin.stop()


def trivial_reject(query: str, max_query_chars: int) -> str | None:
    if not query.strip():
        return TRIVIAL
    if len(query) > max_query_chars:
        return TRIVIAL_TOO_LONG
    return None


def one_token(chat: Chat, system: str, user: str, *, allowed: frozenset, closed: str) -> str:
    """Every gate call has this shape. A sentence, a hedge, a second token, or
    an empty reply is the closed path. The model gets no retry.

    A call that never completed is not a verdict, so it is left to raise. The
    closed token would otherwise tell the user their question was out of
    scope when what actually happened is that our API was down.
    """
    raw = chat.complete(system=system, user=user)
    parts = (raw or "").strip().split()
    if len(parts) != 1:
        return closed
    token = parts[0].upper()
    return token if token in allowed else closed


def classify_scope(chat: Chat, query: str) -> str:
    return one_token(
        chat, SCOPE_SYS, catalog_hint(query), allowed=SCOPE_TOKENS, closed="DENY"
    )


def pinecone_filter(role: Role) -> dict:
    # flagged_outdated is not in this filter. The citation receipt needs the old revision
    # in the hit list so it can prefer the current one and note the conflict.
    # contains_pricing is separate from doc_type because a FAQ row can still
    # carry a price in one question.
    if role == "technician":
        return {
            "$and": [
                {"doc_type": {"$ne": "pricing"}},
                {"contains_pricing": {"$eq": False}},
            ]
        }
    return {"doc_type": {"$ne": "service"}}


def drop_technician_prices(role: Role, hits: list[Hit]) -> list[Hit]:
    """Second price cut. The metadata filter misses a `$` ingest did not tag."""
    if role != "technician":
        return hits
    return [hit for hit in hits if not PRICE.search(hit.text)]


def retrieve(cfg: Config, embeddings: Embeddings, query: str, role: Role) -> list[Hit]:
    vectors = embeddings.embed([query])
    if not vectors:
        return []
    pool = max(cfg.retrieve_pool, cfg.top_k)
    hits = store.query(cfg, vectors[0], pinecone_filter(role), pool)
    hits = drop_technician_prices(role, hits)
    hits = cap_per_doc(hits, cfg.max_chunks_per_doc)[: cfg.top_k]
    hits = follow_up_specs(cfg, embeddings, role, hits)
    return drop_technician_prices(role, hits)


def cap_per_doc(hits: list[Hit], max_per_doc: int) -> list[Hit]:
    """Keep document order, but stop a split FAQ from occupying every slot."""
    if max_per_doc <= 0:
        return list(hits)
    counts: dict[str, int] = {}
    out: list[Hit] = []
    for hit in hits:
        used = counts.get(hit.doc_id, 0)
        if used >= max_per_doc:
            continue
        counts[hit.doc_id] = used + 1
        out.append(hit)
    return out


def follow_up_specs(
    cfg: Config, embeddings: Embeddings, role: Role, hits: list[Hit]
) -> list[Hit]:
    """Pull a spec chunk for catalog models the first pass named but did not retrieve."""
    missing = _models_missing_spec(hits)
    if not missing:
        return hits
    queries = [f"{model} product specification" for model in missing]
    vectors = embeddings.embed(queries)
    if len(vectors) != len(missing):
        return hits
    known = {hit.id for hit in hits}
    extra: list[Hit] = []
    role_filter = pinecone_filter(role)
    for vector in vectors:
        candidates = [
            hit
            for hit in store.query(cfg, vector, role_filter, 4)
            if hit.id not in known and hit.score >= cfg.retrieve_floor
        ]
        chosen = next((hit for hit in candidates if hit.doc_type == "spec"), None)
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is not None:
            extra.append(chosen)
            known.add(chosen.id)
    return hits + drop_technician_prices(role, extra)


def _models_missing_spec(hits: list[Hit]) -> list[str]:
    mentioned: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        for model in canonical_models_in(hit.text):
            if model not in seen:
                seen.add(model)
                mentioned.append(model)
    covered: set[str] = set()
    for hit in hits:
        if hit.doc_type != "spec":
            continue
        covered.update(canonical_models_in(hit.text))
        covered.update(canonical_models_in(hit.title))
    return [model for model in mentioned if model not in covered]


def complete_citations(ids: list[str], answer: str, hits: list[Hit]) -> list[str]:
    """Add retrieved hits that name a catalog model already in the answer.

    One extra id per document. The model is allowed to cite a single FAQ;
    Python still lists the other retrieved docs that support the same products.
    """
    named = set(canonical_models_in(answer))
    if not named:
        return list(ids)
    have = set(ids)
    cited_docs = {hit.doc_id for hit in hits if hit.id in have}
    extra: list[str] = []
    for hit in hits:
        if hit.id in have or hit.doc_id in cited_docs:
            continue
        if named.intersection(canonical_models_in(hit.text)):
            extra.append(hit.id)
            have.add(hit.id)
            cited_docs.add(hit.doc_id)
    return list(ids) + extra


def decide(chat: Chat, query: str, hits: list[Hit]) -> str:
    user = f"Question: {query}\n\nChunks:\n{_hits_block(hits)}"
    return one_token(chat, DECIDE_SYS, user, allowed=ANSWER_TOKEN, closed="REFUSE")


def decide_and_draft(chat: Chat, query: str, hits: list[Hit]) -> tuple[Draft | None, str | None]:
    """Run the evidence decision and the answer draft concurrently.

    Returns (draft, None) on ANSWER, or (None, refuse message) on REFUSE. A
    draft that errored is only surfaced when the verdict was ANSWER; on
    REFUSE it is discarded unread, error or not. A decide call that errored
    raises, since that is an outage rather than a verdict.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        verdict = pool.submit(decide, chat, query, hits)
        drafting = pool.submit(generate, chat, query, hits)
        if verdict.result() != "ANSWER":
            return None, explain_refuse(chat, query, hits)
        return drafting.result(), None


def explain_refuse(chat: Chat, query: str, hits: list[Hit]) -> str:
    user = json.dumps(
        {
            "query": query,
            "chunks": [_chunk_payload(hit) for hit in hits],
        },
        ensure_ascii=False,
    )
    raw = chat.complete(system=REFUSE_SYS, user=user, json_object=True)
    try:
        data = json.loads(raw)
    except ValueError:
        return format_refuse("UNKNOWN", "")
    if not isinstance(data, dict):
        return format_refuse("UNKNOWN", "")
    reason = str(data.get("reason") or "").strip().upper()
    detail = data.get("detail")
    if not isinstance(detail, str):
        detail = ""
    return format_refuse(reason, detail)


def format_refuse(reason: str, detail: str) -> str:
    # AMBIGUOUS is an unclear question, not two documents that disagree.
    # Disagreement is an ANSWER and never reaches this function.
    cleaned = _clean_detail(detail)
    if reason == "MISSING_INFO" and cleaned:
        return f"{REFUSE} Missing information: {cleaned}"
    if reason == "AMBIGUOUS" and cleaned:
        return (
            "The question was too ambiguous to answer. "
            f"Please clarify {cleaned} in your question and ask again."
        )
    if reason == "UNKNOWN" and cleaned:
        return f"{REFUSE} Related topic: {cleaned}"
    return REFUSE


def generate(chat: Chat, query: str, hits: list[Hit]) -> Draft | None:
    user = json.dumps(
        {
            "query": query,
            "chunks": [_chunk_payload(hit) for hit in hits],
        },
        ensure_ascii=False,
    )
    raw = chat.complete(system=GEN_SYS, user=user, json_object=True)
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer")
    ids = data.get("citation_ids")
    if not isinstance(answer, str) or not isinstance(ids, list):
        return None
    if not all(isinstance(item, str) for item in ids):
        return None
    note = data.get("outdated_note")
    return Draft(
        answer=answer,
        citation_ids=ids,
        outdated_note=note if isinstance(note, str) else None,
    )


def superseded_hits(usable: list[Hit], cited: list[Hit]) -> list[Hit]:
    """Outdated hits that are an older revision of a cited current document.

    Same model and doc_type as a cited current chunk. A legacy spec next to
    the current spec qualifies; a legacy spec next to a pricing sheet does not.
    """
    current = {(hit.model, hit.doc_type) for hit in cited if not hit.flagged_outdated}
    return [
        hit
        for hit in usable
        if hit.flagged_outdated and (hit.model, hit.doc_type) in current
    ]


_NUMBER = re.compile(r"\d[\d,.\-]*\d|\d")
_OUTDATED_SENTENCE = re.compile(r"[^.!?\n]*\boutdated\b[^.!?\n]*[.!?]?[ \t]*", re.I)


def outdated_note(note: str | None, superseded: list[Hit]) -> str | None:
    """The line under the answer when a superseded revision was retrieved.

    The model's sentence is used only if every number in it appears in the
    superseded text, so it cannot misquote the old value. Otherwise, or when
    the model gave no sentence, a canned line names the document. No
    superseded hit means no note, whatever the model said.
    """
    if not superseded:
        return None
    blob = " ".join(f"{hit.title} {hit.text}" for hit in superseded)
    cleaned = " ".join((note or "").split())
    if cleaned and all(number in blob for number in _NUMBER.findall(cleaned)):
        return cleaned
    titles = "; ".join(
        f"{hit.title} ({hit.path})" for hit in _unique_by_doc(superseded)
    )
    return (
        "Note: a superseded revision is also on file and may list different "
        f"values: {titles}. This answer uses the current document."
    )


def strip_outdated_claims(answer: str) -> str:
    # No superseded revision was retrieved, so any sentence about an
    # outdated document is the model inventing a conflict. Drop it.
    return _OUTDATED_SENTENCE.sub("", answer).strip()


def _unique_by_doc(hits: list[Hit]) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in hits:
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            out.append(hit)
    return out


def receipt_ok(ids: list[str], hits: list[Hit]) -> bool:
    # The model must point at chunks we actually retrieved. An empty list
    # or an id it invented fails the receipt and the answer is not shown.
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


def _clean_detail(detail: str) -> str:
    # A long detail is the model slipping into an answer. Drop it and use
    # the canned refuse line instead.
    cleaned = " ".join((detail or "").split())
    if len(cleaned) > 120:
        return ""
    return cleaned


def _chunk_payload(hit: Hit) -> dict:
    return {
        "id": hit.id,
        "title": hit.title,
        "path": hit.path,
        "text": hit.text,
        "flagged_outdated": hit.flagged_outdated,
    }


def _hits_block(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        flag = "outdated" if hit.flagged_outdated else "current"
        blocks.append(f"{i}. {hit.id} | {hit.title} | {hit.path} | {flag}\n{hit.text}")
    return "\n\n".join(blocks)
