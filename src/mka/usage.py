"""Record token counts, time, and vector store usage, and print them when --stats is set."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

# USD per 1M tokens, standard list price. A model not listed reports "unknown".
_CHAT_RATES: dict[str, tuple[float, float]] = {"gpt-5.4-nano": (0.20, 1.25)}  # (in, out)
_EMBED_RATES: dict[str, float] = {"text-embedding-3-small": 0.02}


@dataclass
class Call:
    kind: str  # "chat" | "embed"
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


@dataclass
class PineconeCall:
    op: str
    latency_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ledger:
    workflow: str
    calls: list[Call] = field(default_factory=list)
    pinecone: list[PineconeCall] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)


_current: Ledger | None = None


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def lookup(obj: object, *names: str) -> Any:
    """First present attribute or key. SDKs return objects or dicts by version."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif getattr(obj, name, None) is not None:
            return getattr(obj, name)
    return None


@contextmanager
def track(workflow: str) -> Iterator[Ledger]:
    global _current
    ledger = Ledger(workflow)
    _current = ledger
    try:
        yield ledger
    finally:
        _current = None


def reported(workflow: str, run: Callable[[], int], *, stats: bool) -> int:
    """Run a command; print the stats block after it when --stats is set."""
    if not stats:
        return run()
    with track(workflow) as ledger:
        try:
            return run()
        finally:
            print()
            print(render(ledger))


def record_chat(model: str, response: object, latency_ms: float) -> None:
    if _current is None:
        return
    usage = lookup(response, "usage")
    _current.calls.append(
        Call(
            "chat",
            model,
            _int(lookup(usage, "prompt_tokens", "input_tokens")),
            _int(lookup(usage, "completion_tokens", "output_tokens")),
            latency_ms,
        )
    )


def record_embed(model: str, response: object, latency_ms: float) -> None:
    if _current is None:
        return
    usage = lookup(response, "usage")
    tokens = _int(lookup(usage, "prompt_tokens", "total_tokens"))
    _current.calls.append(Call("embed", model, tokens, 0 if tokens is not None else None, latency_ms))


def record_pinecone(op: str, latency_ms: float, **detail: Any) -> None:
    if _current is None:
        return
    _current.pinecone.append(PineconeCall(op, latency_ms, detail))


def token_cost_usd(calls: list[Call]) -> float | None:
    """List-price total, or None if any call is unpriced or unreported."""
    total = 0.0
    for call in calls:
        if call.prompt_tokens is None or call.completion_tokens is None:
            return None
        if call.kind == "chat":
            rate = _CHAT_RATES.get(_base_model(call.model, _CHAT_RATES))
            if rate is None:
                return None
            total += call.prompt_tokens * rate[0] + call.completion_tokens * rate[1]
        else:
            rate_e = _EMBED_RATES.get(_base_model(call.model, _EMBED_RATES))
            if rate_e is None:
                return None
            total += call.prompt_tokens * rate_e
    return total / 1_000_000


def render(ledger: Ledger) -> str:
    lines = [
        "--- stats ---",
        f"workflow: {ledger.workflow}",
        f"latency_ms: {elapsed_ms(ledger.started):.0f}",
    ]
    for kind in ("chat", "embed"):
        group = [call for call in ledger.calls if call.kind == kind]
        models = ",".join(dict.fromkeys(call.model for call in group)) or "-"
        prompt = _sum(call.prompt_tokens for call in group)
        ms = sum(call.latency_ms for call in group)
        line = f"{kind}_calls: {len(group)} model={models} prompt_tokens={_n(prompt)}"
        if kind == "chat":
            line += f" completion_tokens={_n(_sum(call.completion_tokens for call in group))}"
        lines.append(f"{line} latency_ms={ms:.0f}")
    cost = token_cost_usd(ledger.calls)
    lines.append(f"token_cost_usd: {'unknown' if cost is None else f'{cost:.8f}'}")
    lines.append(f"pinecone_calls: {len(ledger.pinecone)}")
    for call in ledger.pinecone:
        detail = " ".join(f"{k}={v}" for k, v in call.detail.items())
        lines.append(f"  {call.op}: latency_ms={call.latency_ms:.0f} {detail}".rstrip())
    return "\n".join(lines)


def _base_model(model: str, table: dict) -> str:
    # "gpt-5.4-nano-2026-03-17" bills at the gpt-5.4-nano rate.
    for name in sorted(table, key=len, reverse=True):
        if model == name or model.startswith(name + "-"):
            return name
    return model


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _sum(values: Iterator[int | None] | Any) -> int | None:
    items = list(values)
    if any(item is None for item in items):
        return None
    return sum(items)


def _n(value: int | None) -> str:
    return "unreported" if value is None else str(value)
