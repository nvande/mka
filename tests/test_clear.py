from __future__ import annotations

from pathlib import Path

import pytest

from mka.clear import run_clear
from mka.cli import main

from conftest import make_config


def _layout(root: Path) -> dict[str, Path]:
    cache = root / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.cpython-311.pyc").write_bytes(b"x")
    stale = root / "stale.pyc"
    stale.write_bytes(b"x")
    pytest_cache = root / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "v").write_text("1", encoding="utf-8")
    egg = root / "mka.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("n", encoding="utf-8")
    venv_cache = root / ".venv" / "__pycache__"
    venv_cache.mkdir(parents=True)
    (venv_cache / "keep.pyc").write_bytes(b"x")
    corpus_cache = root / "corpus" / "__pycache__"
    corpus_cache.mkdir(parents=True)
    (corpus_cache / "keep.pyc").write_bytes(b"x")
    return {
        "cache": cache,
        "stale": stale,
        "pytest_cache": pytest_cache,
        "egg": egg,
        "venv_cache": venv_cache,
        "corpus_cache": corpus_cache,
    }


def test_clear_wipes_python_caches_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _layout(tmp_path)
    seen: list[str] = []

    def fake_delete(cfg) -> bool:
        seen.append(cfg.pinecone_index)
        return True

    monkeypatch.setattr("mka.clear.store.delete_index", fake_delete)
    cfg = make_config(tmp_path, pinecone_api_key="pc-test")
    assert run_clear(cfg, root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "cleared: python caches" in out
    assert "cleared: index mka-poc" in out
    assert seen == ["mka-poc"]
    assert not paths["cache"].exists()
    assert not paths["stale"].exists()
    assert not paths["pytest_cache"].exists()
    assert not paths["egg"].exists()
    assert paths["venv_cache"].is_dir()
    assert paths["corpus_cache"].is_dir()


def test_clear_without_key_still_clears_local(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _layout(tmp_path)
    cfg = make_config(tmp_path)
    assert run_clear(cfg, root=tmp_path) == 1
    captured = capsys.readouterr()
    assert "PINECONE_API_KEY missing" in captured.err
    assert not paths["cache"].exists()


def test_clear_missing_index_is_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("mka.clear.store.delete_index", lambda cfg: False)
    cfg = make_config(tmp_path, pinecone_api_key="pc-test")
    assert run_clear(cfg, root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "already empty" in out


def test_clear_via_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mka.cli.run_clear", lambda cfg: 0)
    with pytest.raises(SystemExit) as exc:
        main(["clear"])
    assert exc.value.code == 0
