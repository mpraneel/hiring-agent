"""Tests for the skill ontology and the normalizer built on top of it."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from core.normalize.normalizer import SkillNormalizer

ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "core" / "normalize" / "skill_ontology.json"


@pytest.fixture
def normalizer() -> SkillNormalizer:
    return SkillNormalizer()


# --------------------------------------------------------------------------
# Behaviour that already works and must keep working
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("JS", "javascript"),
        ("javascript", "javascript"),
        ("py", "python"),
        ("Python", "python"),
        ("k8s", "kubernetes"),
        ("golang", "go"),
        ("sklearn", "scikit-learn"),
        ("ts", "typescript"),
    ],
)
def test_known_variants_normalize_to_canonical(normalizer, raw, expected):
    assert normalizer.normalize_skill(raw) == expected


def test_unknown_skill_passes_through_unchanged(normalizer):
    assert normalizer.normalize_skill("totallyunknownskill") == "totallyunknownskill"


def test_empty_skill_returns_empty_string(normalizer):
    assert normalizer.normalize_skill("") == ""


def test_normalize_skills_deduplicates_preserving_order(normalizer):
    assert normalizer.normalize_skills(["JS", "javascript", "py", "Python"]) == [
        "javascript",
        "python",
    ]


def test_get_all_canonical_skills_is_non_empty(normalizer):
    canonicals = normalizer.get_all_canonical_skills()
    assert "python" in canonicals
    assert len(canonicals) > 100


def test_get_skill_variants_returns_variants(normalizer):
    assert "k8s" in normalizer.get_skill_variants("kubernetes")


def test_elastic_variant_resolves_to_elasticsearch(normalizer):
    """Guards the duplicate-key fix.

    This currently passes only because the substring fuzzy match rescues it.
    Phase 2 deletes that fuzzy match, so it will pass afterwards only if the
    duplicate elasticsearch key is merged and the 'elastic' variant restored.
    """
    assert normalizer.normalize_skill("elastic") == "elasticsearch"


# --------------------------------------------------------------------------
# Ontology data integrity
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2: 'elasticsearch' is declared twice, so the first entry is discarded",
    strict=True,
)
def test_ontology_has_no_duplicate_canonical_keys():
    raw = ONTOLOGY_PATH.read_text(encoding="utf-8")
    keys = re.findall(r'^\s{2}"([^"]+)":', raw, re.MULTILINE)
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    assert duplicates == []


@pytest.mark.xfail(
    reason="Phase 2: several generic variants are claimed by more than one canonical",
    strict=True,
)
def test_every_variant_maps_to_exactly_one_canonical():
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    owners: dict[str, list[str]] = defaultdict(list)
    for canonical, variants in ontology.items():
        for variant in variants:
            owners[variant.lower()].append(canonical)
    ambiguous = {v: c for v, c in owners.items() if len(c) > 1}
    assert ambiguous == {}


@pytest.mark.xfail(
    reason="Phase 2: the normalizer has no startup validation of the ontology",
    strict=True,
)
def test_duplicate_variant_in_ontology_raises_on_load(tmp_path):
    bad = tmp_path / "bad_ontology.json"
    bad.write_text(
        json.dumps({"redis": ["redis", "cache"], "caching": ["caching", "cache"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache"):
        SkillNormalizer(ontology_path=str(bad))


def test_canonical_is_always_one_of_its_own_variants():
    """Every canonical name should be reachable by typing the canonical name."""
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    missing = [c for c, variants in ontology.items() if c not in [v.lower() for v in variants]]
    assert missing == [], f"canonicals not listed among their own variants: {missing}"


# --------------------------------------------------------------------------
# Substring fuzzy matching produces wrong answers (Phase 2 deletes it)
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2: substring fuzzy match maps 'scikit learn' to jenkins via the 'ci' variant",
    strict=True,
)
def test_scikit_learn_is_not_mangled_by_substring_match(normalizer):
    assert normalizer.normalize_skill("scikit-learn") == "scikit-learn"


@pytest.mark.xfail(
    reason="Phase 2: substring fuzzy match maps 'react js' to javascript via the 'js' variant",
    strict=True,
)
def test_react_dot_js_normalizes_to_react(normalizer):
    assert normalizer.normalize_skill("react.js") == "react"


@pytest.mark.xfail(
    reason="Phase 2: single letters substring-match arbitrary variants",
    strict=True,
)
@pytest.mark.parametrize("token", ["r", "c"])
def test_single_letter_tokens_are_not_normalized(normalizer, token):
    assert normalizer.normalize_skill(token) == token


@pytest.mark.xfail(
    reason="Phase 2: the _fuzzy_match elif chain duplicating the ontology should be deleted",
    strict=True,
)
def test_fuzzy_match_helper_is_gone(normalizer):
    assert not hasattr(normalizer, "_fuzzy_match")


# --------------------------------------------------------------------------
# Ambiguous generic variants should be dropped, not resolved arbitrarily
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2: ambiguous generic variants should be removed from the ontology",
    strict=True,
)
@pytest.mark.parametrize(
    "token",
    ["password hashing", "cache", "design", "requirements", "api", "ai"],
)
def test_ambiguous_generic_tokens_are_not_normalized(normalizer, token):
    assert normalizer.normalize_skill(token) == token
