from __future__ import annotations

import json
from pathlib import Path

import pytest

from mka.products import (
    CATALOG_PATH,
    FAMILIES,
    canonical_models_in,
    catalog_hint,
    load_catalog,
    match_product_terms,
    scope_glossary,
)


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


GOOD_CATALOG = {
    "families": [
        {"family": "dock leveler", "models": ["MD-7000"], "synonyms": ["leveler"]}
    ],
    "components": [{"term": "air bag", "family": "dock leveler"}],
}


def test_shipped_catalog_loads() -> None:
    catalog = load_catalog(CATALOG_PATH)
    assert catalog.families == FAMILIES
    assert ("velocity fuse", "dock leveler") in catalog.components


def test_load_catalog_rejects_a_component_in_an_undeclared_family(tmp_path: Path) -> None:
    payload = {
        "families": GOOD_CATALOG["families"],
        "components": [{"term": "photo-eye", "family": "industrial door"}],
    }
    path = _write_catalog(tmp_path / "catalog.json", payload)
    with pytest.raises(RuntimeError, match="undeclared families"):
        load_catalog(path)


def test_load_catalog_rejects_an_empty_or_broken_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cannot read product catalog"):
        load_catalog(tmp_path / "missing.json")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot read product catalog"):
        load_catalog(bad_json)
    no_key = _write_catalog(tmp_path / "nokey.json", {"families": [{"family": "x"}]})
    with pytest.raises(RuntimeError, match="cannot read product catalog"):
        load_catalog(no_key)
    empty = _write_catalog(tmp_path / "empty.json", {"families": [], "components": []})
    with pytest.raises(RuntimeError, match="no product families"):
        load_catalog(empty)


def test_load_catalog_defaults_models_to_empty(tmp_path: Path) -> None:
    payload = {
        "families": [{"family": "dock equipment", "synonyms": ["loading dock"]}],
        "components": [],
    }
    catalog = load_catalog(_write_catalog(tmp_path / "c.json", payload))
    assert catalog.families[0].models == ()
    assert catalog.families[0].synonyms == ("loading dock",)
    assert catalog.components == ()


def _families(query: str) -> set[str]:
    return {family for _term, family in match_product_terms(query)}


def test_leveler_means_dock_leveler() -> None:
    assert "dock leveler" in _families("What is the best leveler for a freezer?")
    assert "dock leveler" in _families("best leveller for high traffic")
    assert "dock leveler" in _families("MD-9000 capacity?")


def test_door_and_restraint_families() -> None:
    assert "industrial door" in _families("Which high-speed door for a vestibule?")
    assert "industrial door" in _families("ThermaGuard 600 R-value")
    assert "vehicle restraint" in _families("How do I reset a DockGuard fault?")


def test_longest_match_wins() -> None:
    # "dock leveler" covers the span, so "leveler" does not fire a second time.
    assert match_product_terms("dock leveler capacity") == [("dock leveler", "dock leveler")]


def test_bare_door_is_not_a_catalog_hit() -> None:
    assert match_product_terms("Please close the door") == []
    assert match_product_terms("Write me a poem") == []


def test_hint_only_when_matched() -> None:
    query = "What is the best leveler for a blast freezer?"
    hinted = catalog_hint(query)
    assert hinted.startswith(query)
    assert "leveler → dock leveler" in hinted
    assert catalog_hint("Tell me a joke") == "Tell me a joke"


def test_canonical_models_in_ignores_family_synonyms() -> None:
    assert canonical_models_in("Which leveler for a freezer?") == []
    assert set(canonical_models_in("Pair MD-9000 with ThermaGuard 600 and RapidRoll 400")) == {
        "MD-9000",
        "ThermaGuard 600",
        "RapidRoll 400",
    }
    assert canonical_models_in("md9000 capacity") == ["MD-9000"]


def test_glossary_lists_families() -> None:
    text = scope_glossary()
    assert "dock leveler" in text
    assert "MD-7000" in text
    assert "industrial door" in text
    assert "vehicle restraint" in text
