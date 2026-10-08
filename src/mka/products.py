from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Product catalog for the scope gate. Users say "leveler", never "MD-7000".
#
# catalog.json is committed and hand-maintained: families, models, synonyms,
# named components, and universal terms that belong to every family. It is
# loaded once at import. Nothing writes to it.

CATALOG_PATH = Path(__file__).with_name("catalog.json")


@dataclass(frozen=True)
class ProductFamily:
    family: str
    models: tuple[str, ...]
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    families: tuple[ProductFamily, ...]
    components: tuple[tuple[str, str], ...]
    universal: tuple[str, ...] = ()


def load_catalog(path: Path) -> Catalog:
    """Read and validate a catalog file, or raise.

    This does not degrade quietly: without a catalog the scope gate stops
    recognizing our own products and starts denying real questions.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        families = tuple(
            ProductFamily(
                family=str(row["family"]),
                models=tuple(str(item) for item in row.get("models", ())),
                synonyms=tuple(str(item) for item in row["synonyms"]),
            )
            for row in payload["families"]
        )
        components = tuple(
            (str(row["term"]), str(row["family"])) for row in payload["components"]
        )
        universal = tuple(str(item) for item in payload.get("universal", ()))
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"cannot read product catalog {path}: {exc}") from exc
    if not families:
        raise RuntimeError(f"{path} declares no product families")
    # A component in an undeclared family is a typo, not a new family.
    names = {family.family for family in families}
    unknown = sorted({fam for _term, fam in components if fam not in names})
    if unknown:
        raise RuntimeError(f"{path} has components in undeclared families: {unknown}")
    return Catalog(families, components, universal)


# Loaded once at import: ask.py bakes scope_glossary() into the classifier
# instructions, so the catalog must not change under a running process.
_CATALOG = load_catalog(CATALOG_PATH)
FAMILIES = _CATALOG.families
COMPONENTS = _CATALOG.components
UNIVERSAL = _CATALOG.universal


def _model_aliases(model: str) -> list[str]:
    compact = re.sub(r"[\s\-]+", "", model).lower()
    spaced = re.sub(r"[\-]+", " ", model).lower()
    return [model.lower(), compact, spaced]


def _pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [("meridian", "Meridian")]
    for family in FAMILIES:
        for model in family.models:
            for alias in _model_aliases(model):
                pairs.append((alias, family.family))
        for syn in family.synonyms:
            pairs.append((syn.lower(), family.family))
    for term in UNIVERSAL:
        for family in FAMILIES:
            pairs.append((term.lower(), family.family))
    pairs.extend(COMPONENTS)
    return pairs


def _build_patterns() -> list[tuple[re.Pattern[str], str, tuple[str, ...]]]:
    # Longest first, so "dock leveler" matches before "leveler" and the
    # shorter span is skipped. The lookaround keeps the match off the
    # inside of a longer token. A term listed for several families, such as
    # a universal term, keeps every family.
    grouped: dict[str, tuple[str, tuple[str, ...]]] = {}
    for term, family in sorted(_pairs(), key=lambda item: len(item[0]), reverse=True):
        key = term.casefold()
        saved = grouped.get(key)
        if saved is None:
            grouped[key] = (term, (family,))
            continue
        if family not in saved[1]:
            grouped[key] = (saved[0], saved[1] + (family,))
    return [
        (re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", re.I), term, families)
        for term, families in grouped.values()
    ]


_PATTERNS = _build_patterns()


def _build_model_patterns() -> list[tuple[re.Pattern[str], str]]:
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for family in FAMILIES:
        for model in family.models:
            for alias in _model_aliases(model):
                key = alias.casefold()
                if key in seen:
                    continue
                seen.add(key)
                unique.append((alias, model))
    unique.sort(key=lambda item: len(item[0]), reverse=True)
    return [
        (re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.I), model)
        for alias, model in unique
    ]


_MODEL_PATTERNS = _build_model_patterns()


def canonical_models_in(text: str) -> list[str]:
    """Canonical catalog model names mentioned in text, longest match first.

    Family synonyms such as "leveler" do not count. Follow-up retrieve and
    citation completion use this so a FAQ that names RapidRoll 400 can pull
    that spec without embedding every dock leveler.
    """
    found: list[str] = []
    seen: set[str] = set()
    seen_spans: list[tuple[int, int]] = []
    for pattern, model in _MODEL_PATTERNS:
        for hit in pattern.finditer(text):
            span = hit.span()
            if any(span[0] >= start and span[1] <= end for start, end in seen_spans):
                continue
            seen_spans.append(span)
            if model not in seen:
                seen.add(model)
                found.append(model)
    return found


def match_product_terms(query: str) -> list[tuple[str, str]]:
    """Return (matched term, family) for catalog words in the query.

    Patterns are longest-first. A span already covered by a longer match
    is skipped, so "leveler" does not also fire inside "dock leveler".
    """
    found: list[tuple[str, str]] = []
    seen_spans: list[tuple[int, int]] = []
    for pattern, term, families in _PATTERNS:
        for hit in pattern.finditer(query):
            span = hit.span()
            if any(span[0] >= start and span[1] <= end for start, end in seen_spans):
                continue
            seen_spans.append(span)
            found.extend((term, family) for family in families)
    return found


def catalog_hint(query: str) -> str:
    """Append matched catalog terms so the scope gate can treat them as our products."""
    matched = match_product_terms(query)
    if not matched:
        return query
    labels = ", ".join(f"{term} → {family}" for term, family in matched)
    return (
        f"{query}\n\n"
        f"Catalog terms in this query (treat as Meridian products): {labels}"
    )


def scope_glossary() -> str:
    lines = [
        "These words refer to our products even if the user does not say Meridian or a model number:"
    ]
    for family in FAMILIES:
        models = f" ({', '.join(family.models)})" if family.models else ""
        syns = ", ".join(family.synonyms[:8])
        lines.append(f"- {family.family}{models}: {syns}")
    if UNIVERSAL:
        lines.append(f"- every product family: {', '.join(UNIVERSAL)}")
    return "\n".join(lines)
