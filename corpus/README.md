# Meridian Industrial — Test Corpus for RAG Evaluation

Synthetic corpus of 22 documents simulating internal knowledge at a fictional industrial equipment manufacturer (dock levelers, industrial doors, vehicle restraints). Built for the AI Engineer Screening Exercise as a reference dataset and grading aid.

## Contents

```
data/
├── README.md               # this file
├── manifest.json           # metadata for all 22 documents
├── questions.json          # 7 evaluation queries with grading notes
└── docs/                   # 22 documents (.md)
    ├── spec_*.md           # 6 product specifications
    ├── service_*.md        # 6 service procedures
    ├── faq_*.md            # 5 FAQ documents
    ├── pricing_*.md        # 2 pricing documents (sales audience)
    ├── install_requirements.md
    ├── compliance_certifications.md
    └── operator_quickstart.md
```

## Design intent

This corpus is engineered to stress-test a RAG system in the specific ways an enterprise deployment would fail in the real world:

### 1. Deliberate contradiction
`spec_md7000.md` (current) and `spec_md7000_legacy.md` (superseded) disagree on lifting capacity (35,000 lbs vs 30,000 lbs), motor HP (2 vs 1.5), and relief pressure (1,800 vs 1,600 psi). The current spec explicitly references Engineering Bulletin EB-2024-003 and the legacy spec is flagged as superseded in its own header and in `manifest.json` (`flagged_outdated: true`). A strong system prefers the current revision; a weak system returns conflicting numbers or the wrong one.

### 2. Unanswerable question
`compliance_certifications.md` explicitly states Meridian does not hold European market certifications. Question `q4_unanswerable` asks for a CE certification number that doesn't exist. Systems that fabricate an answer fail hallucination mitigation.

### 3. Role-based access
Two pricing documents (`pricing_md7000_2026.md`, `pricing_service_contracts.md`) are tagged `audience: sales`. Service procedures are tagged `audience: technician`. Candidates implementing the access-scoping guardrail can demonstrate correct role-based filtering against question `q5_role_scoped`.

### 4. Realistic multi-hop synthesis
`q2_cross_doc_synthesis` (cold storage dock recommendation) requires combining information from at least 3 documents — a product selection FAQ, a cold storage FAQ, and individual spec sheets — to construct a correct recommendation.

## Audience roles

| Role | Can access | Cannot access |
|---|---|---|
| `technician` | spec, service, faq, reference | pricing |
| `sales` | spec, faq, pricing, reference | service (technician procedures) |
| `all` (no filter) | everything | — |

## Using this corpus for grading

When evaluating a candidate's submission:

1. Replace the candidate's `data/` directory with this one
2. Re-run their ingestion script against this corpus
3. Run their system against `questions.json`
4. Compare outputs against the `grading_notes` and `expected_sources` in each question
5. Pay special attention to:
   - **Q3** — does the system surface the contradiction, or silently return one value?
   - **Q4** — does the system fabricate a CE number, or correctly say "not in documentation"?
   - **Q5** — does role-based filtering actually filter, or just pass through?

## Known characteristics (intentional)

- Document lengths vary from ~250 to ~700 words — tests chunking robustness
- Some documents are tabular/structured (pricing, PM checklist), others are narrative (service procedures) — tests handling of mixed content shapes
- Technical specs include specific part numbers (`MD7-HV-42`, `RR4-PE-SET`) — good recall signal for precise retrieval
- No PII, no real company or product names, no copyrighted content — safe for external distribution

## Version

Generated 2026-04-21. Corpus version 1.0.0.
