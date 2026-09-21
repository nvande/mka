from __future__ import annotations

import pytest

from mka.ask import (
    BANNER,
    REDIRECT,
    SCOPE_SYS,
    TRIVIAL,
    TRIVIAL_TOO_LONG,
    classify_scope,
    run_ask,
    trivial_reject,
)

from conftest import make_config


class FakeChat:
    def __init__(self, reply: str | BaseException) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def test_trivial_empty_and_whitespace() -> None:
    assert trivial_reject("", 4000) == TRIVIAL
    assert trivial_reject("   \n", 4000) == TRIVIAL


def test_trivial_too_long() -> None:
    assert trivial_reject("x" * 5, 4) == TRIVIAL_TOO_LONG


def test_trivial_ok() -> None:
    assert trivial_reject("What is MD-9000 capacity?", 4000) is None


def test_scope_allow_only_exact_token() -> None:
    chat = FakeChat("ALLOW")
    assert classify_scope(chat, "MD-9000 capacity?") == "ALLOW"


def test_scope_deny_extra_prose_and_punctuation() -> None:
    assert classify_scope(FakeChat("ALLOW extra"), "q") == "DENY"
    assert classify_scope(FakeChat("ALLOW."), "q") == "DENY"
    assert classify_scope(FakeChat("DENY"), "q") == "DENY"
    assert classify_scope(FakeChat(""), "q") == "DENY"
    assert classify_scope(FakeChat(RuntimeError("boom")), "q") == "DENY"


def test_run_ask_trivial_prints_banner_no_llm(tmp_path, capsys) -> None:
    chat = FakeChat("ALLOW")
    code = run_ask(make_config(tmp_path), "sales", "  ", chat=chat)
    out = capsys.readouterr().out
    assert code == 0
    assert chat.calls == []
    assert BANNER in out
    assert TRIVIAL in out


def test_run_ask_redirect_on_deny(tmp_path, capsys) -> None:
    chat = FakeChat("DENY")
    code = run_ask(make_config(tmp_path), "sales", "Write me a poem", chat=chat)
    out = capsys.readouterr().out
    assert code == 0
    assert BANNER in out
    assert REDIRECT in out
    assert chat.calls[0][0] == SCOPE_SYS
    assert chat.calls[0][1] == "Write me a poem"
    assert "sales" not in chat.calls[0][0].lower()


def test_run_ask_allow_stops_before_retrieve(tmp_path, capsys) -> None:
    chat = FakeChat("allow")
    code = run_ask(make_config(tmp_path), "technician", "MD-9000 capacity?", chat=chat)
    captured = capsys.readouterr()
    assert code == 1
    assert BANNER in captured.out
    assert REDIRECT not in captured.out
    assert "retrieve is not implemented yet" in captured.err
