#!/usr/bin/env python3
"""Measure extraction quality against hand written gold labels.

Reports precision, recall and F1 per field, micro-averaged over cases, plus
must/nice classification accuracy. Writes eval/results/<mode>.json and prints
a markdown table.

Usage:
    python eval/run_eval.py --mode deterministic
    python eval/run_eval.py --mode hybrid --min-f1 0.55
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.parsers.jd_parser import JDParser  # noqa: E402
from core.parsers.resume_parser import ResumeParser  # noqa: E402

Mode = Literal["deterministic", "llm", "hybrid"]
MODES: tuple[Mode, ...] = ("deterministic", "llm", "hybrid")

FIELDS: tuple[str, ...] = ("jd_must_have", "jd_nice_to_have", "resume_skills")

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Counts:
    """True positive, false positive and false negative tallies for one field."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> Optional[float]:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else None

    @property
    def recall(self) -> Optional[float]:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else None

    @property
    def f1(self) -> Optional[float]:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def score_field(predicted: set[str], gold: set[str]) -> Counts:
    return Counts(
        tp=len(predicted & gold),
        fp=len(predicted - gold),
        fn=len(gold - predicted),
    )


def priority_map(must: set[str], nice: set[str]) -> dict[str, str]:
    """Map each skill to its priority. A skill in both lists counts as must."""
    mapping = {skill: "nice" for skill in nice}
    mapping.update({skill: "must" for skill in must})
    return mapping


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    case_id: str
    jd_text: str
    resume_text: str
    gold: dict[str, set[str]]

    @property
    def is_labeled(self) -> bool:
        """A case counts as labeled once any gold list is non-empty."""
        return any(self.gold[field] for field in FIELDS)


@dataclass
class CaseOutcome:
    case_id: str
    predicted: dict[str, set[str]]
    counts: dict[str, Counts] = field(default_factory=dict)
    classification_correct: int = 0
    classification_total: int = 0


def normalize_labels(values) -> set[str]:
    """Gold labels are compared case-insensitively after trimming."""
    if values is None:
        return set()
    if not isinstance(values, list):
        raise ValueError(f"expected a list of skills, got {type(values).__name__}")
    return {str(v).strip().lower() for v in values if str(v).strip()}


def load_cases(data_dir: Path) -> list[Case]:
    cases: list[Case] = []
    for case_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        gold_path = case_dir / "gold.json"
        jd_path = case_dir / "jd.txt"
        resume_path = case_dir / "resume.txt"
        missing = [p.name for p in (gold_path, jd_path, resume_path) if not p.exists()]
        if missing:
            print(f"  skipping {case_dir.name}: missing {', '.join(missing)}", file=sys.stderr)
            continue

        raw_gold = json.loads(gold_path.read_text(encoding="utf-8"))
        unknown = set(raw_gold) - set(FIELDS) - {"notes"}
        if unknown:
            raise ValueError(f"{gold_path}: unknown keys {sorted(unknown)}")

        cases.append(
            Case(
                case_id=case_dir.name,
                jd_text=jd_path.read_text(encoding="utf-8"),
                resume_text=resume_path.read_text(encoding="utf-8"),
                gold={f: normalize_labels(raw_gold.get(f)) for f in FIELDS},
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Extraction modes
# ---------------------------------------------------------------------------


class Extractor:
    """Runs one extraction mode over a case and returns canonical skill sets."""

    def __init__(self, mode: Mode) -> None:
        self.mode = mode
        self.jd_parser = JDParser()
        self.resume_parser = ResumeParser()
        self._llm = None
        if mode in ("llm", "hybrid"):
            self._llm = self._load_llm_extractor()

    @staticmethod
    def _load_llm_extractor():
        try:
            from core.extraction.llm_extractor import LLMExtractor
        except ImportError as exc:
            raise SystemExit(
                "LLM extraction is not available yet (Phase 3 adds "
                "core/extraction/llm_extractor.py). Run with "
                "--mode deterministic for now.\n"
                f"  underlying import error: {exc}"
            ) from exc
        return LLMExtractor()

    def extract(self, case: Case) -> dict[str, set[str]]:
        deterministic = self._extract_deterministic(case)
        if self.mode == "deterministic":
            return deterministic

        llm = self._extract_llm(case)
        if self.mode == "llm":
            return llm

        return self._merge(llm, deterministic)

    def _extract_deterministic(self, case: Case) -> dict[str, set[str]]:
        jd = self.jd_parser.parse_jd(case.jd_text)
        resume = self.resume_parser.parse_text(case.resume_text)
        must = {s.strip().lower() for s in jd.must_haves_raw if s.strip()}
        nice = {s.strip().lower() for s in jd.nice_to_haves_raw if s.strip()}
        return {
            "jd_must_have": must,
            "jd_nice_to_have": nice - must,
            "resume_skills": {s.strip().lower() for s in resume.skills_norm if s.strip()},
        }

    def _extract_llm(self, case: Case) -> dict[str, set[str]]:
        assert self._llm is not None
        jd = self._llm.extract_jd(case.jd_text)
        resume = self._llm.extract_resume(case.resume_text)
        must = {r.skill.lower() for r in jd.requirements if r.priority == "must"}
        nice = {r.skill.lower() for r in jd.requirements if r.priority == "nice"}
        return {
            "jd_must_have": must,
            "jd_nice_to_have": nice - must,
            "resume_skills": {s.skill.lower() for s in resume.skills},
        }

    @staticmethod
    def _merge(primary: dict[str, set[str]], secondary: dict[str, set[str]]) -> dict[str, set[str]]:
        """Union of both extractors, with the primary winning priority conflicts."""
        must = primary["jd_must_have"] | secondary["jd_must_have"]
        nice = primary["jd_nice_to_have"] | secondary["jd_nice_to_have"]
        # A skill the primary calls nice stays nice even if the secondary said must.
        must -= primary["jd_nice_to_have"]
        return {
            "jd_must_have": must,
            "jd_nice_to_have": nice - must,
            "resume_skills": primary["resume_skills"] | secondary["resume_skills"],
        }


# ---------------------------------------------------------------------------
# Driving the evaluation
# ---------------------------------------------------------------------------


def evaluate(cases: list[Case], mode: Mode) -> tuple[list[CaseOutcome], dict[str, Counts], int, int]:
    extractor = Extractor(mode)
    outcomes: list[CaseOutcome] = []
    totals = {f: Counts() for f in FIELDS}
    correct = 0
    total = 0

    for case in cases:
        predicted = extractor.extract(case)
        outcome = CaseOutcome(case_id=case.case_id, predicted=predicted)

        if case.is_labeled:
            for name in FIELDS:
                counts = score_field(predicted[name], case.gold[name])
                outcome.counts[name] = counts
                totals[name] = totals[name] + counts

            gold_priority = priority_map(case.gold["jd_must_have"], case.gold["jd_nice_to_have"])
            pred_priority = priority_map(predicted["jd_must_have"], predicted["jd_nice_to_have"])
            shared = set(gold_priority) & set(pred_priority)
            outcome.classification_total = len(shared)
            outcome.classification_correct = sum(
                1 for s in shared if gold_priority[s] == pred_priority[s]
            )
            correct += outcome.classification_correct
            total += outcome.classification_total

        outcomes.append(outcome)

    return outcomes, totals, correct, total


def format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _display_path(path: Path) -> str:
    """Show a repo-relative path when possible, else the path as given."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_markdown(
    mode: Mode,
    totals: dict[str, Counts],
    n_cases: int,
    n_labeled: int,
    correct: int,
    total: int,
) -> str:
    lines = [
        f"### Extraction quality, mode `{mode}`",
        "",
        f"Cases discovered: {n_cases}. Cases with gold labels: {n_labeled}.",
        "",
    ]

    if not n_labeled:
        lines += [
            "No labeled cases found, so no metrics can be computed.",
            "Fill in the gold lists described in eval/README.md, then rerun.",
        ]
        return "\n".join(lines)

    lines += [
        "| Field | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in FIELDS:
        counts = totals[name]
        lines.append(
            f"| {name} | {counts.tp} | {counts.fp} | {counts.fn} | "
            f"{format_metric(counts.precision)} | {format_metric(counts.recall)} | "
            f"{format_metric(counts.f1)} |"
        )

    overall = Counts()
    for name in FIELDS:
        overall = overall + totals[name]
    lines.append(
        f"| **micro average** | {overall.tp} | {overall.fp} | {overall.fn} | "
        f"{format_metric(overall.precision)} | {format_metric(overall.recall)} | "
        f"**{format_metric(overall.f1)}** |"
    )

    accuracy = correct / total if total else None
    lines += [
        "",
        f"Must/nice classification accuracy: {format_metric(accuracy)} "
        f"({correct} of {total} skills present in both gold and prediction).",
    ]
    return "\n".join(lines)


def build_report(
    mode: Mode,
    outcomes: list[CaseOutcome],
    totals: dict[str, Counts],
    cases: list[Case],
    correct: int,
    total: int,
) -> dict:
    overall = Counts()
    for name in FIELDS:
        overall = overall + totals[name]
    labeled = [c for c in cases if c.is_labeled]

    return {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases_discovered": len(cases),
        "cases_labeled": len(labeled),
        "per_field": {name: totals[name].to_dict() for name in FIELDS},
        "micro_average": overall.to_dict(),
        "classification": {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else None,
        },
        "per_case": [
            {
                "case_id": outcome.case_id,
                "labeled": bool(outcome.counts),
                "predicted": {name: sorted(outcome.predicted[name]) for name in FIELDS},
                "counts": {name: c.to_dict() for name, c in outcome.counts.items()},
                "classification": {
                    "correct": outcome.classification_correct,
                    "total": outcome.classification_total,
                },
            }
            for outcome in outcomes
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="deterministic")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--min-f1",
        type=float,
        default=None,
        help="exit non-zero if the micro-averaged F1 falls below this value",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the report without writing the JSON file",
    )
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        print(f"error: data directory not found: {args.data_dir}", file=sys.stderr)
        return 2

    cases = load_cases(args.data_dir)
    if not cases:
        print(f"error: no usable cases in {args.data_dir}", file=sys.stderr)
        return 2

    outcomes, totals, correct, total = evaluate(cases, args.mode)
    labeled = [c for c in cases if c.is_labeled]

    print(render_markdown(args.mode, totals, len(cases), len(labeled), correct, total))

    report = build_report(args.mode, outcomes, totals, cases, correct, total)
    if not args.no_write:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.results_dir / f"{args.mode}.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {_display_path(out_path)}")

    if args.min_f1 is not None:
        micro_f1 = report["micro_average"]["f1"]
        if not labeled:
            print(
                "\nerror: --min-f1 was given but there are no labeled cases to measure",
                file=sys.stderr,
            )
            return 1
        if micro_f1 is None or micro_f1 < args.min_f1:
            print(
                f"\nerror: micro F1 {format_metric(micro_f1)} is below the "
                f"threshold {args.min_f1:.3f}",
                file=sys.stderr,
            )
            return 1
        print(f"Micro F1 {micro_f1:.3f} meets the threshold {args.min_f1:.3f}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
