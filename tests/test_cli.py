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


def test_ingest_is_stub(capsys: object) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ingest"])
    assert exc.value.code == 1
    assert "ingest is not implemented yet" in capsys.readouterr().err


def test_ask_is_stub(capsys: object) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--role", "sales", "capacity?"])
    assert exc.value.code == 1
    assert "ask is not implemented yet" in capsys.readouterr().err


def test_scan_via_cli(fixture_corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    monkeypatch.setenv("CORPUS_DIR", str(fixture_corpus))
    with pytest.raises(SystemExit) as exc:
        main(["scan"])
    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "spec_legacy" in out


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
