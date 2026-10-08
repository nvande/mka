"""The eval command. Runs every question in corpus/questions.json as both roles.

A grader model scores each answer from 0 to 100, the way a teacher scores a
short assignment. The rubric is that question's expected answer summary, the
expected sources for the role, and the grading notes. The overall score is
the mean of the cases.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import dataclass

from mka import usage
from mka.ask import run_ask
from mka.config import Config
from mka.llm import Chat, make_chat
from mka.types import Role, load_manifest

ROLES: tuple[Role, ...] = ("sales", "technician")
HIDDEN = {"sales": "service", "technician": "pricing"}  # doc_type a role cannot see

GRADER_SYS = """You are a grader scoring a class assignment.
Read the question, the assistant's answer, the expected answer summary, the expected sources, and the grading notes.
Give the answer a score from 0 to 100 for how closely it meets the grading notes.
Use the expected answer summary and the expected sources as the criteria next to those notes.
0 means the answer misses the assignment. 100 means it meets the assignment.
The expected sources list already leaves out documents this role cannot see. Do not mark the answer down for omitting those.
Reply with a JSON object and nothing else: {"score": <integer 0-100>, "reason": "<one sentence>"}
"""


@dataclass(frozen=True)
class Grade:
    score: int
    reason: str


def run_eval(cfg: Config, *, stats: bool = False) -> int:
    return usage.reported("eval", lambda: _run_eval(cfg), stats=stats)


def _run_eval(cfg: Config) -> int:
    questions = json.loads((cfg.corpus_dir / "questions.json").read_text(encoding="utf-8"))
    doc_type = {row.doc_id: row.doc_type for row in load_manifest(cfg.corpus_dir)}
    chat = make_chat(cfg)
    grades: list[Grade] = []
    for q in questions["questions"]:
        for role in ROLES:
            result = grade(cfg, q, role, doc_type, chat)
            grades.append(result)
            line = f"{q['id']} [{role}]  {result.score:3d}"
            if result.reason:
                line += "  " + result.reason
            print(line)
    overall = round(sum(item.score for item in grades) / len(grades)) if grades else 0
    print(f"score: {overall}")
    return 0 if overall == 100 else 1


def grade(
    cfg: Config,
    q: dict,
    role: Role,
    doc_type: dict[str, str],
    chat: Chat | None = None,
) -> Grade:
    chat = chat or make_chat(cfg)
    return judge(
        chat,
        role=role,
        query=q["query"],
        answer=ask_capture(cfg, role, q["query"]),
        summary=str(q.get("expected_answer_summary") or ""),
        sources=_sources_for(q, role, doc_type),
        notes=str(q.get("grading_notes") or ""),
    )


def judge(
    chat: Chat,
    *,
    role: str,
    query: str,
    answer: str,
    summary: str,
    sources: list[str],
    notes: str,
) -> Grade:
    raw = chat.complete(
        system=GRADER_SYS,
        user=_grader_user(role, query, answer, summary, sources, notes),
        json_object=True,
    )
    return _parse_grade(raw)


def _sources_for(q: dict, role: Role, doc_type: dict[str, str]) -> list[str]:
    expected = q.get(f"expected_sources_{role}", q.get("expected_sources", []))
    return [doc for doc in expected if doc_type.get(doc) != HIDDEN[role]]


def _grader_user(
    role: str,
    query: str,
    answer: str,
    summary: str,
    sources: list[str],
    notes: str,
) -> str:
    listed = "\n".join(f"- {doc}" for doc in sources) or "(none)"
    return (
        f"Role: {role}\n"
        f"Question: {query}\n\n"
        f"Expected answer summary:\n{summary or '(none)'}\n\n"
        f"Expected sources for this role:\n{listed}\n\n"
        f"Grading notes:\n{notes or '(none)'}\n\n"
        f"Assistant answer:\n{answer}"
    )


def _parse_grade(raw: str) -> Grade:
    try:
        data = json.loads(raw)
    except ValueError:
        return Grade(0, "grader returned an unreadable score")
    if not isinstance(data, dict):
        return Grade(0, "grader returned an unreadable score")
    reason = " ".join(str(data.get("reason") or "").split())
    score = data.get("score")
    # bool is an int subclass and is not a score.
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return Grade(0, reason or "grader returned an unreadable score")
    score_i = round(score)
    if score_i < 0 or score_i > 100:
        return Grade(0, reason or "grader returned a score outside 0-100")
    return Grade(score_i, reason)


def ask_capture(cfg: Config, role: Role, query: str) -> str:
    # Leave stats off. The eval ledger already wraps the whole run; a nested
    # --stats ask would print into this buffer and wipe the parent ledger.
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_ask(cfg, role, query)
    return buffer.getvalue()
