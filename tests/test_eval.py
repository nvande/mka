from __future__ import annotations

import json
from pathlib import Path

import pytest

from mka.cli import main
from mka.eval import cited_ids, run_eval

from conftest import make_config

MANIFEST = {
    "documents": [
        {
            "doc_id": doc_id,
            "path": f"docs/{doc_id}.md",
            "title": doc_id,
            "doc_type": doc_type,
            "audience": "all",
            "model": "MD-7000",
            "last_updated": "2025-01-01",
            "version": "2025-01",
            "flagged_outdated": False,
        }
        for doc_id, doc_type in [
            ("spec_md9000", "spec"),
            ("pricing_md7000_2026", "pricing"),
            ("service_md7000_hydraulic_reset", "service"),
            ("spec_md7000", "spec"),
        ]
    ]
}

QUESTIONS = {
    "questions": [
        {
            "id": "q1_simple_lookup",
            "query": "MD-9000 capacity?",
            "expected_sources": ["spec_md9000"],
        },
        {
            "id": "q5_role_scoped",
            "query": "MD-7000 price?",
            "expected_sources_sales": ["pricing_md7000_2026"],
            "expected_sources_technician": [],
        },
        {
            "id": "q6_procedural_detail",
            "query": "relief valve psi?",
            "expected_sources": ["service_md7000_hydraulic_reset", "spec_md7000"],
        },
    ]
}

ANSWERS = {
    ("sales", "MD-9000 capacity?"): "40,000 lbs.\nSources:\n- S (docs/spec_md9000.md, spec_md9000::0)\n",
    ("technician", "MD-9000 capacity?"): "It is 40,000 lbs.\nSources:\n- S (docs/spec_md9000.md, spec_md9000::0)\n",
    ("sales", "MD-7000 price?"): "$10,550.\nSources:\n- P (docs/pricing_md7000_2026.md, pricing_md7000_2026::0)\n",
    # The bug the harness should catch: a technician told the price is $10,550.
    ("technician", "MD-7000 price?"): "$10,550.\nSources:\n- P (docs/pricing_md7000_2026.md, pricing_md7000_2026::0)\n",
    ("technician", "relief valve psi?"): (
        "1,800 psi.\nSources:\n"
        "- R (docs/service_md7000_hydraulic_reset.md, service_md7000_hydraulic_reset::0)\n"
        "- S (docs/spec_md7000.md, spec_md7000::0)\n"
    ),
    # Sales cannot see the service doc, so only spec_md7000 is expected here.
    ("sales", "relief valve psi?"): "1,800 psi.\nSources:\n- S (docs/spec_md7000.md, spec_md7000::0)\n",
}


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (tmp_path / "questions.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")

    def fake_ask(cfg, role, query, **kwargs):
        print(ANSWERS[(role, query)], end="")
        return 0

    monkeypatch.setattr("mka.eval.run_ask", fake_ask)
    return tmp_path


def test_cited_ids_parses_sources_block() -> None:
    out = "answer\nSources:\n- Title, with comma (docs/a.md, a::q1)\n- B (docs/b.md, b::0)\n"
    assert cited_ids(out) == ["a::q1", "b::0"]
    assert cited_ids("I don't have enough information.\nRelated:\n- A (docs/a.md)\n") == []


def test_eval_grades_both_roles_and_filters_hidden_sources(corpus: Path, capsys) -> None:
    code = run_eval(make_config(corpus))
    out = capsys.readouterr().out
    assert code == 1
    assert "q1_simple_lookup [sales]  PASS" in out
    assert "q1_simple_lookup [technician]  PASS" in out
    assert "q5_role_scoped [sales]  PASS" in out
    assert "q5_role_scoped [technician]  FAIL" in out
    assert "forbidden text '$'" in out
    assert "missing text 'not available'" in out
    assert "q6_procedural_detail [sales]  PASS" in out
    assert "q6_procedural_detail [technician]  PASS" in out
    assert "5/6 passed" in out


def test_eval_via_cli(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    with pytest.raises(SystemExit) as exc:
        main(["eval"])
    assert exc.value.code == 1
