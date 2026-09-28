"""The eval command. Runs every question in corpus/questions.json as both roles.

Each case gets a 0–100 score from three independent checks: the expected
sources were cited, the must-contain strings showed up, and the
must-not-contain strings didn't. Expected sources come from questions.json,
minus any document that role can't see. The string checks live in CHECKS
below. The overall score is the mean of the cases.
"""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from dataclasses import dataclass

from mka import usage
from mka.ask import run_ask
from mka.config import Config
from mka.types import Role, load_manifest

ROLES: tuple[Role, ...] = ("sales", "technician")
HIDDEN = {"sales": "service", "technician": "pricing"}  # doc_type a role cannot see
_SOURCE_ID = re.compile(r", ([^,()]+)\)$")
_ROLE_KEYS = frozenset(ROLES)

# contain / avoid are substrings of the whole printed output, keyed by question
# id. A nested sales/technician dict overrides those lists for that role.
CHECKS: dict[str, dict] = {
    "q1_simple_lookup": {"contain": ["40,000"]},
    "q2_cross_doc_synthesis": {
        "contain": ["MD-9000", "ThermaGuard 600", "RapidRoll 400"],
    },
    # The legacy source in Sources is guaranteed by Python; the old value in
    # the note is the model's when grounded, so it is not asserted here.
    "q3_contradiction_handling": {"contain": ["35,000"]},
    "q4_unanswerable": {"contain": ["enough information"], "avoid": ["Sources:"]},
    "q5_role_scoped": {
        "sales": {"contain": ["$10,550"]},
        "technician": {"contain": ["restricted to sales"], "avoid": ["$"]},
    },
    "q6_procedural_detail": {"contain": ["1,800"]},
    "q7_near_miss_boundary": {
        "contain": ["MD-9000"],
        "avoid": ["outdated"],
    },
}


@dataclass(frozen=True)
class Grade:
    score: int
    earned: int
    possible: int
    problems: list[str]


def run_eval(cfg: Config, *, stats: bool = False) -> int:
    return usage.reported("eval", lambda: _run_eval(cfg), stats=stats)


def _run_eval(cfg: Config) -> int:
    questions = json.loads((cfg.corpus_dir / "questions.json").read_text(encoding="utf-8"))
    doc_type = {row.doc_id: row.doc_type for row in load_manifest(cfg.corpus_dir)}
    grades: list[Grade] = []
    for q in questions["questions"]:
        for role in ROLES:
            result = grade(cfg, q, role, doc_type)
            grades.append(result)
            line = f"{q['id']} [{role}]  {result.score:3d}"
            if result.problems:
                line += "  " + "; ".join(result.problems)
            print(line)
    overall = round(sum(item.score for item in grades) / len(grades)) if grades else 0
    print(f"score: {overall}")
    return 0 if overall == 100 else 1


def grade(cfg: Config, q: dict, role: Role, doc_type: dict[str, str]) -> Grade:
    checks = _checks_for(q["id"], role)
    expected = q.get(f"expected_sources_{role}", q.get("expected_sources", []))
    expected = [doc for doc in expected if doc_type.get(doc) != HIDDEN[role]]
    out = ask_capture(cfg, role, q["query"])
    return score_output(out, expected, checks.get("contain", []), checks.get("avoid", []))


def score_output(
    out: str,
    expected: list[str],
    contain: list[str],
    avoid: list[str],
) -> Grade:
    cited = {sid.split("::")[0] for sid in cited_ids(out)}
    problems: list[str] = []
    earned = 0
    possible = len(expected) + len(contain) + len(avoid)
    for doc in expected:
        if doc in cited:
            earned += 1
        else:
            problems.append(f"missing source {doc}")
    for text in contain:
        if text in out:
            earned += 1
        else:
            problems.append(f"missing text {text!r}")
    for text in avoid:
        if text not in out:
            earned += 1
        else:
            problems.append(f"forbidden text {text!r}")
    score = 100 if possible == 0 else round(100 * earned / possible)
    return Grade(score, earned, possible, problems)


def _checks_for(qid: str, role: Role) -> dict:
    row = dict(CHECKS.get(qid, {}))
    overlay = row.pop(role, None) if role in row else None
    for key in _ROLE_KEYS:
        row.pop(key, None)
    if isinstance(overlay, dict):
        row.update(overlay)
    return row


def ask_capture(cfg: Config, role: Role, query: str) -> str:
    # Leave stats off. The eval ledger already wraps the whole run; a nested
    # --stats ask would print into this buffer and wipe the parent ledger.
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_ask(cfg, role, query)
    return buffer.getvalue()


def cited_ids(out: str) -> list[str]:
    # Sources lines look like: - {title} ({path}, {chunk_id})
    if "Sources:" not in out:
        return []
    tail = out.split("Sources:", 1)[1]
    return [m.group(1) for line in tail.splitlines() if (m := _SOURCE_ID.search(line))]
