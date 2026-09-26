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

    Before Phase 2 this passed only because the substring fuzzy match rescued
    it. With that match deleted, it passes only because the duplicate
    elasticsearch key was merged and the 'elastic' variant restored.
    """
    assert normalizer.normalize_skill("elastic") == "elasticsearch"


# --------------------------------------------------------------------------
# Ontology data integrity
# --------------------------------------------------------------------------


def test_ontology_has_no_duplicate_canonical_keys():
    raw = ONTOLOGY_PATH.read_text(encoding="utf-8")
    keys = re.findall(r'^\s{2}"([^"]+)":', raw, re.MULTILINE)
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    assert duplicates == []


def test_every_variant_maps_to_exactly_one_canonical():
    ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    owners: dict[str, list[str]] = defaultdict(list)
    for canonical, variants in ontology.items():
        for variant in variants:
            owners[variant.lower()].append(canonical)
    ambiguous = {v: c for v, c in owners.items() if len(c) > 1}
    assert ambiguous == {}


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


def test_scikit_learn_is_not_mangled_by_substring_match(normalizer):
    assert normalizer.normalize_skill("scikit-learn") == "scikit-learn"


def test_react_dot_js_normalizes_to_react(normalizer):
    assert normalizer.normalize_skill("react.js") == "react"


@pytest.mark.parametrize("token", ["r", "c"])
def test_single_letter_tokens_are_not_normalized(normalizer, token):
    assert normalizer.normalize_skill(token) == token


def test_fuzzy_match_helper_is_gone(normalizer):
    assert not hasattr(normalizer, "_fuzzy_match")


# --------------------------------------------------------------------------
# Ambiguous generic variants should be dropped, not resolved arbitrarily
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["password hashing", "cache", "design", "requirements", "api", "ai"],
)
def test_ambiguous_generic_tokens_are_not_normalized(normalizer, token):
    assert normalizer.normalize_skill(token) == token


# --------------------------------------------------------------------------
# Ontology load-time validation
# --------------------------------------------------------------------------


def test_duplicate_canonical_key_raises_on_load(tmp_path):
    """json.loads keeps the last duplicate, which is how 'elastic' was lost."""
    bad = tmp_path / "dupe_key.json"
    bad.write_text(
        '{"elasticsearch": ["elasticsearch", "elastic"], "redis": ["redis"], '
        '"elasticsearch": ["elasticsearch", "es"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate canonical keys"):
        SkillNormalizer(ontology_path=str(bad))


def test_non_list_variants_raise_on_load(tmp_path):
    bad = tmp_path / "bad_shape.json"
    bad.write_text('{"python": "python"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        SkillNormalizer(ontology_path=str(bad))


def test_a_clean_ontology_loads(tmp_path):
    good = tmp_path / "ok.json"
    good.write_text('{"python": ["python", "py"], "go": ["go", "golang"]}', encoding="utf-8")
    normalizer = SkillNormalizer(ontology_path=str(good))
    assert normalizer.normalize_skill("py") == "python"


def test_real_ontology_loads_without_error():
    """The shipped ontology must satisfy its own validation."""
    assert SkillNormalizer().get_all_canonical_skills()


# --------------------------------------------------------------------------
# Coverage of mainstream skills
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PostgreSQL", "postgresql"),
        ("Postgres", "postgresql"),
        ("FastAPI", "fastapi"),
        ("Django", "django"),
        ("pytest", "pytest"),
        ("Vite", "vite"),
        ("webpack", "webpack"),
        ("Storybook", "storybook"),
        ("TailwindCSS", "tailwind"),
        ("Node", "node.js"),
        ("Azure", "azure"),
        ("GitHub Actions", "github actions"),
    ],
)
def test_mainstream_skills_are_covered(normalizer, raw, expected):
    assert normalizer.normalize_skill(raw) == expected


def test_specific_databases_are_not_folded_into_generic_sql(normalizer):
    """Postgres and PostgreSQL previously normalized differently from each other."""
    assert normalizer.normalize_skill("postgres") == normalizer.normalize_skill("postgresql")
    assert normalizer.normalize_skill("mysql") != "sql"
    assert normalizer.normalize_skill("sql") == "sql"


def test_hyphen_and_punctuation_bearing_names_survive_cleaning(normalizer):
    for raw, expected in [("C++", "c++"), ("C#", "c#"), ("socket.io", "socket.io")]:
        assert normalizer.normalize_skill(raw) == expected


@pytest.mark.parametrize(
    "generic",
    ["deployment", "dashboard", "authentication", "container", "crm", "paas", "ci"],
)
def test_generic_category_words_do_not_resolve_to_one_product(normalizer, generic):
    """'deployment' is not vercel and 'authentication' is not oauth."""
    assert normalizer.normalize_skill(generic) == generic
