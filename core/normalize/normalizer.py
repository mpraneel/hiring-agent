"""Canonical skill normalization backed by a curated ontology.

Design notes:

* Normalization is exact-match only. The previous implementation fell back to a
  substring match ("is this variant a substring of that skill, or vice versa"),
  which produced confidently wrong answers: "scikit learn" became jenkins via
  the "ci" variant, "react js" became javascript via "js", and single letters
  matched arbitrary variants. An unknown skill passing through unnormalized is
  strictly better than a wrong canonical, so unknown skills pass through.

* The ontology is validated on load. A variant claimed by two canonicals has no
  correct answer, so it is a startup error rather than a silent last-one-wins.

* Some tokens are real skill names but also ordinary English or ambiguous
  abbreviations ("go", "rest", "find"). They are listed in AMBIGUOUS_TOKENS and
  callers that scan free text must require a disambiguating neighbour or a
  skill-list context before accepting them. Normalizing such a token when it is
  handed over explicitly is fine; guessing it out of prose is not.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

DEFAULT_ONTOLOGY_PATH = Path(__file__).parent / "skill_ontology.json"

# Tokens that match ordinary English or are too short to be unambiguous. They
# only count as a skill in a list context or next to a disambiguating word.
AMBIGUOUS_TOKENS: frozenset[str] = frozenset(
    {"go", "rest", "find", "ping", "shell", "es", "ga", "bi", "tf", "pd", "np", "r", "c"}
)

# Words that, adjacent to an ambiguous token, make the skill reading the obvious
# one. Matched against the words immediately around the token.
DISAMBIGUATORS: Dict[str, frozenset[str]] = {
    "go": frozenset({"language", "lang", "golang", "programming", "routine", "routines", "module", "modules"}),
    "rest": frozenset({"api", "apis", "endpoint", "endpoints", "service", "services", "ful", "restful"}),
    "find": frozenset({"command", "utility", "unix"}),
    "ping": frozenset({"command", "utility", "icmp", "latency"}),
    "shell": frozenset({"script", "scripts", "scripting", "bash", "unix"}),
    "es": frozenset({"cluster", "index", "indices", "elasticsearch"}),
    "ga": frozenset({"analytics", "google"}),
    "bi": frozenset({"tool", "tools", "dashboard", "dashboards", "reporting"}),
    "tf": frozenset({"tensorflow", "idf", "keras"}),
    "pd": frozenset({"pandas", "dataframe", "dataframes"}),
    "np": frozenset({"numpy", "array", "arrays"}),
    "r": frozenset({"language", "lang", "statistical", "cran", "rstudio"}),
    "c": frozenset({"language", "lang", "programming", "embedded"}),
}


class OntologyError(ValueError):
    """Raised when the ontology file cannot yield an unambiguous mapping."""


class SkillNormalizer:
    def __init__(self, ontology_path: Optional[str] = None) -> None:
        """Load and validate the skill ontology."""
        path = Path(ontology_path) if ontology_path else DEFAULT_ONTOLOGY_PATH
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()

        self.ontology: Dict[str, List[str]] = self._load_validated(raw, path)

        self.variant_to_canonical: Dict[str, str] = {}
        for canonical, variants in self.ontology.items():
            for variant in variants:
                self.variant_to_canonical[variant.lower()] = canonical

    # -- loading ----------------------------------------------------------

    @staticmethod
    def _load_validated(raw: str, path: Path) -> Dict[str, List[str]]:
        """Parse the ontology, rejecting duplicate keys and ambiguous variants.

        json.loads normally keeps the last of a set of duplicate keys, which is
        how the 'elastic' and 'es' variants were silently lost. object_pairs_hook
        exposes every declaration so a duplicate can be reported instead.
        """
        pairs = json.loads(raw, object_pairs_hook=lambda kv: kv)

        seen_keys: Set[str] = set()
        duplicate_keys: Set[str] = set()
        ontology: Dict[str, List[str]] = {}
        for canonical, variants in pairs:
            if canonical in seen_keys:
                duplicate_keys.add(canonical)
            seen_keys.add(canonical)
            if not isinstance(variants, list):
                raise OntologyError(f"{path}: variants for {canonical!r} must be a list")
            ontology[canonical] = list(variants)

        if duplicate_keys:
            raise OntologyError(
                f"{path}: duplicate canonical keys {sorted(duplicate_keys)}. "
                "Merge them, otherwise the earlier variants are discarded."
            )

        owners: Dict[str, List[str]] = {}
        for canonical, variants in ontology.items():
            for variant in variants:
                owners.setdefault(variant.lower(), []).append(canonical)
        ambiguous = {v: sorted(c) for v, c in owners.items() if len(c) > 1}
        if ambiguous:
            details = ", ".join(f"{v!r} -> {c}" for v, c in sorted(ambiguous.items()))
            raise OntologyError(
                f"{path}: these variants are claimed by more than one canonical, "
                f"so normalization would depend on ordering: {details}"
            )

        return ontology

    # -- normalization ----------------------------------------------------

    def normalize_skill(self, skill: str) -> str:
        """Return the canonical form of a skill, or the input if it is unknown."""
        if not skill:
            return ""
        cleaned = self._clean_skill_string(skill)
        if not cleaned:
            return ""
        return self.variant_to_canonical.get(cleaned, cleaned)

    def normalize_skills(self, skills: Iterable[str]) -> List[str]:
        """Normalize a list of skills, de-duplicating and preserving order."""
        normalized: List[str] = []
        seen: Set[str] = set()
        for skill in skills:
            canonical = self.normalize_skill(skill)
            if canonical and canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)
        return normalized

    @staticmethod
    def _clean_skill_string(skill: str) -> str:
        """Lowercase and strip punctuation that is never part of a skill name.

        Characters that carry meaning inside real skill names are preserved:
        '+' for c++, '#' for c#, '.' for socket.io and node.js, and '-' for
        scikit-learn and ci-cd. Splitting on those is what broke hyphenated
        skills previously.
        """
        cleaned = skill.lower().strip()
        cleaned = re.sub(r"[^\w\s+#.\-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Trailing or leading separators are punctuation, not part of the name.
        return cleaned.strip(".-")

    # -- introspection ----------------------------------------------------

    def get_all_canonical_skills(self) -> Set[str]:
        """Every canonical skill name in the ontology."""
        return set(self.ontology.keys())

    def get_skill_variants(self, canonical_skill: str) -> List[str]:
        """Every variant registered for a canonical skill."""
        return list(self.ontology.get(canonical_skill, []))

    def is_ambiguous(self, token: str) -> bool:
        """Whether a token needs contextual support before it counts as a skill."""
        return token.lower() in AMBIGUOUS_TOKENS

    @staticmethod
    def disambiguators_for(token: str) -> frozenset[str]:
        """Words whose presence beside an ambiguous token confirms the skill."""
        return DISAMBIGUATORS.get(token.lower(), frozenset())
