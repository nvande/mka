from __future__ import annotations

import json
from pathlib import Path

import pytest

from mka import usage
from mka.cli import main
from mka.eval import grade, judge, run_eval

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
            "expected_answer_summary": "40,000 lbs",
            "expected_sources": ["spec_md9000"],
            "grading_notes": "Cite spec_md9000 and return the exact figure.",
        },
        {
            "id": "q5_role_scoped",
            "query": "MD-7000 price?",
            "expected_answer_summary": "Sales gets the list price. Technician is refused.",
            "expected_sources_sales": ["pricing_md7000_2026"],
            "expected_sources_technician": [],
            "grading_notes": "Sales should get the numeric answer; technician should be refused.",
        },
        {
            "id": "q6_procedural_detail",
            "query": "relief valve psi?",
            "expected_answer_summary": "1,800 psi",
            "expected_sources": ["service_md7000_hydraulic_reset", "spec_md7000"],
            "grading_notes": "Return 1,800 psi and cite the sources this role can see.",
        },
    ]
}

ANSWERS = {
    ("sales", "MD-9000 capacity?"): "40,000 lbs.\nSources:\n- S (docs/spec_md9000.md, spec_md9000::0)\n",
    ("technician", "MD-9000 capacity?"): "It is 40,000 lbs.\nSources:\n- S (docs/spec_md9000.md, spec_md9000::0)\n",
    ("sales", "MD-7000 price?"): "$10,550.\nSources:\n- P (docs/pricing_md7000_2026.md, pricing_md7000_2026::0)\n",
    # A technician told the price. The grader, not a substring check, scores this.
    ("technician", "MD-7000 price?"): "$10,550.\nSources:\n- P (docs/pricing_md7000_2026.md, pricing_md7000_2026::0)\n",
    ("technician", "relief valve psi?"): (
        "1,800 psi.\nSources:\n"
        "- R (docs/service_md7000_hydraulic_reset.md, service_md7000_hydraulic_reset::0)\n"
        "- S (docs/spec_md7000.md, spec_md7000::0)\n"
    ),
    # Sales cannot see the service doc, so only spec_md7000 is on the rubric.
    ("sales", "relief valve psi?"): "1,800 psi.\nSources:\n- S (docs/spec_md7000.md, spec_md7000::0)\n",
}


class FakeChat:
    def __init__(self, raw: str | None = None) -> None:
        self.raw = raw
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, json_object: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_object": json_object})
        usage.record_chat(
            "gpt-5.4-nano",
            {"usage": {"prompt_tokens": 8, "completion_tokens": 4}},
            1.0,
        )
        if self.raw is not None:
            return self.raw
        if "Role: technician" in user and "Question: MD-7000 price?" in user:
            return json.dumps({"score": 0, "reason": "technician was given a price"})
        return json.dumps({"score": 100, "reason": "meets the notes"})


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (tmp_path / "questions.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")

    def fake_ask(cfg, role, query, **kwargs):
        print(ANSWERS[(role, query)], end="")
        return 0

    monkeypatch.setattr("mka.eval.run_ask", fake_ask)
    monkeypatch.setattr("mka.eval.make_chat", lambda cfg: FakeChat())
    return tmp_path


def test_judge_sends_the_rubric_and_reads_the_score() -> None:
    chat = FakeChat()
    result = judge(
        chat,
        role="sales",
        query="MD-9000 capacity?",
        answer="40,000 lbs.",
        summary="40,000 lbs",
        sources=["spec_md9000"],
        notes="Cite spec_md9000 and return the exact figure.",
    )
    assert result.score == 100
    assert result.reason == "meets the notes"
    sent = chat.calls[0]
    assert sent["json_object"] is True
    assert "grader scoring a class assignment" in sent["system"]
    assert "0 to 100" in sent["system"]
    assert "Expected answer summary:\n40,000 lbs" in sent["user"]
    assert "Expected sources for this role:\n- spec_md9000" in sent["user"]
    assert "Grading notes:\nCite spec_md9000 and return the exact figure." in sent["user"]
    assert "Assistant answer:\n40,000 lbs." in sent["user"]


def test_judge_scores_an_unreadable_reply_as_zero() -> None:
    result = judge(
        FakeChat("not json"),
        role="sales",
        query="q",
        answer="a",
        summary="s",
        sources=[],
        notes="n",
    )
    assert result.score == 0
    assert result.reason == "grader returned an unreadable score"


def test_judge_rejects_a_score_outside_0_to_100() -> None:
    result = judge(
        FakeChat(json.dumps({"score": 140, "reason": "too high"})),
        role="sales",
        query="q",
        answer="a",
        summary="s",
        sources=[],
        notes="n",
    )
    assert result.score == 0
    assert result.reason == "too high"


def test_grade_drops_sources_the_role_cannot_see(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mka.eval.run_ask",
        lambda cfg, role, query, **kwargs: print("1,800 psi.") or 0,
    )
    chat = FakeChat()
    question = {
        "id": "q6_procedural_detail",
        "query": "relief valve psi?",
        "expected_answer_summary": "1,800 psi",
        "expected_sources": ["service_md7000_hydraulic_reset", "spec_md7000"],
        "grading_notes": "Return 1,800 psi and cite the sources this role can see.",
    }
    doc_type = {
        "service_md7000_hydraulic_reset": "service",
        "spec_md7000": "spec",
    }
    sales = grade(make_config(tmp_path), question, "sales", doc_type, chat)
    assert sales.score == 100
    user = chat.calls[0]["user"]
    assert "Expected answer summary:\n1,800 psi" in user
    assert "- spec_md7000" in user
    assert "service_md7000_hydraulic_reset" not in user
    assert "Return 1,800 psi and cite the sources this role can see." in user
    grade(make_config(tmp_path), question, "technician", doc_type, chat)
    tech = chat.calls[1]["user"]
    assert "- service_md7000_hydraulic_reset" in tech
    assert "- spec_md7000" in tech


def test_eval_grades_both_roles_and_filters_hidden_sources(corpus: Path, capsys) -> None:
    code = run_eval(make_config(corpus))
    out = capsys.readouterr().out
    assert code == 1
    assert "q1_simple_lookup [sales]  100" in out
    assert "q1_simple_lookup [technician]  100" in out
    assert "q5_role_scoped [sales]  100" in out
    assert "q5_role_scoped [technician]    0" in out
    assert "technician was given a price" in out
    assert "q6_procedural_detail [sales]  100" in out
    assert "q6_procedural_detail [technician]  100" in out
    # five 100s and a 0 → mean 83
    assert "score: 83" in out
    assert "--- stats ---" not in out


def test_eval_via_cli(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    with pytest.raises(SystemExit) as exc:
        main(["eval"])
    assert exc.value.code == 1


def test_run_eval_stats_after_scores_and_records_nested_asks(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def fake_ask(cfg, role, query, **kwargs):
        del cfg, kwargs
        usage.record_chat(
            "gpt-5.4-nano",
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            1.0,
        )
        print(ANSWERS[(role, query)], end="")
        return 0

    monkeypatch.setattr("mka.eval.run_ask", fake_ask)
    code = run_eval(make_config(corpus), stats=True)
    out = capsys.readouterr().out
    assert code == 1
    assert out.index("score: 83") < out.index("--- stats ---")
    assert "workflow: eval" in out
    # Six answers plus six grader calls.
    assert "chat_calls: 12 model=gpt-5.4-nano prompt_tokens=108 completion_tokens=54" in out
    assert "pinecone_calls: 0" in out
    assert "latency_ms:" in out
    # Nested ask must not print its own stats block into the score lines.
    assert out.count("--- stats ---") == 1


def test_eval_stats_via_cli(
    corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--stats"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "score: 83" in out
    assert out.index("score: 83") < out.index("--- stats ---")
    assert "workflow: eval" in out
