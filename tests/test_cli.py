from __future__ import annotations

from pathlib import Path

import pytest

from mka.cli import main
from mka.config import DEFAULT_CORPUS_DIR, REPO_ROOT, load_config


def test_no_command_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_ask_rejects_unknown_role() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--role", "admin", "hello"])
    assert exc.value.code == 2


def test_ask_short_flags_match_role(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_ask(cfg, role: str, query: str, stats: bool = False) -> int:
        del cfg
        seen["role"] = role
        seen["query"] = query
        seen["stats"] = stats
        return 0

    monkeypatch.setattr("mka.cli.run_ask", fake_ask)
    monkeypatch.setattr("mka.cli.load_config", lambda: object())

    with pytest.raises(SystemExit) as sales:
        main(["ask", "-s", "capacity?"])
    assert sales.value.code == 0
    assert seen == {"role": "sales", "query": "capacity?", "stats": False}

    with pytest.raises(SystemExit) as tech:
        main(["ask", "-t", "reset pressure?"])
    assert tech.value.code == 0
    assert seen == {"role": "technician", "query": "reset pressure?", "stats": False}

    with pytest.raises(SystemExit) as long_sales:
        main(["ask", "--sales", "list price?"])
    assert long_sales.value.code == 0
    assert seen["role"] == "sales"

    with pytest.raises(SystemExit) as long_tech:
        main(["ask", "--technician", "lockout?"])
    assert long_tech.value.code == 0
    assert seen["role"] == "technician"


def test_stats_flag_reaches_ask_ingest_and_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}

    def fake_ask(cfg, role: str, query: str, stats: bool = False) -> int:
        del cfg, role, query
        seen["ask"] = stats
        return 0

    def fake_ingest(cfg, stats: bool = False) -> int:
        del cfg
        seen["ingest"] = stats
        return 0

    def fake_eval(cfg, stats: bool = False) -> int:
        del cfg
        seen["eval"] = stats
        return 0

    monkeypatch.setattr("mka.cli.run_ask", fake_ask)
    monkeypatch.setattr("mka.cli.run_ingest", fake_ingest)
    monkeypatch.setattr("mka.cli.run_eval", fake_eval)
    monkeypatch.setattr("mka.cli.load_config", lambda: object())

    with pytest.raises(SystemExit) as ask:
        main(["ask", "-s", "--stats", "capacity?"])
    assert ask.value.code == 0
    assert seen["ask"] is True

    with pytest.raises(SystemExit) as ingest:
        main(["ingest", "--stats"])
    assert ingest.value.code == 0
    assert seen["ingest"] is True

    with pytest.raises(SystemExit) as eval_cmd:
        main(["eval", "--stats"])
    assert eval_cmd.value.code == 0
    assert seen["eval"] is True


def test_ask_rejects_combined_role_flags() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "-s", "-t", "hello"])
    assert exc.value.code == 2


def test_ingest_via_cli(
    fixture_corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    monkeypatch.setenv("CORPUS_DIR", str(fixture_corpus))
    monkeypatch.setattr("mka.ingest._write_index", lambda cfg, chunks: None)
    with pytest.raises(SystemExit) as exc:
        main(["ingest"])
    assert exc.value.code == 0
    assert "ingest complete: 2 chunks" in capsys.readouterr().out


def test_ask_empty_via_cli(capsys: object) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--role", "sales", "   "])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Meridian Knowledge Assistant" in out
    assert "Ask a question about Meridian products" in out


def test_ask_redirect_via_cli(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    class DenyChat:
        def complete(self, *, system: str, user: str) -> str:
            del system, user
            return "DENY"

    monkeypatch.setattr("mka.ask.make_chat", lambda cfg: DenyChat())
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--role", "sales", "Tell me a joke"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Meridian Knowledge Assistant" in out
    assert "I can only answer questions or provide information about Meridian internal" in out


def test_ask_subjective_via_cli(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    class SubjectiveChat:
        def complete(self, *, system: str, user: str) -> str:
            del system, user
            return "SUBJECTIVE"

    monkeypatch.setattr("mka.ask.make_chat", lambda cfg: SubjectiveChat())
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "ask",
                "--role",
                "technician",
                "What Meridian dock leveler will make me the coolest with my friends on linkedin?",
            ]
        )
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Meridian Knowledge Assistant" in out
    assert "I cannot answer subjective questions." in out
    assert "I can only answer questions or provide information" not in out


def test_scan_is_not_a_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["scan"])
    assert exc.value.code == 2


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORPUS_DIR", raising=False)
    monkeypatch.delenv("PINECONE_INDEX", raising=False)
    cfg = load_config()
    assert cfg.corpus_dir == DEFAULT_CORPUS_DIR.resolve()
    assert cfg.pinecone_index == "mka-poc"
    assert REPO_ROOT.name == "mka"


def test_load_config_corpus_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path))
    cfg = load_config()
    assert cfg.corpus_dir == tmp_path.resolve()
