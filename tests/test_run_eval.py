"""Tests for the evaluation harness.

CI gates on run_eval.py, so its arithmetic and its exit codes are themselves
worth testing. Nothing here touches the network or the real eval/data cases.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# eval/ is a script directory rather than an importable package, so load the
# harness by path. It must be registered in sys.modules before exec_module
# because @dataclass resolves annotations through sys.modules at class creation.
_spec = importlib.util.spec_from_file_location("run_eval", REPO_ROOT / "eval" / "run_eval.py")
assert _spec is not None and _spec.loader is not None
run_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = run_eval
_spec.loader.exec_module(run_eval)


# ---------------------------------------------------------------------------
# Metric arithmetic
# ---------------------------------------------------------------------------


def test_counts_of_a_perfect_prediction():
    counts = run_eval.score_field({"python", "docker"}, {"python", "docker"})
    assert (counts.tp, counts.fp, counts.fn) == (2, 0, 0)
    assert counts.precision == 1.0
    assert counts.recall == 1.0
    assert counts.f1 == 1.0


def test_counts_split_false_positives_and_negatives():
    counts = run_eval.score_field({"python", "java"}, {"python", "docker"})
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)
    assert counts.f1 == pytest.approx(0.5)


def test_f1_is_the_harmonic_mean():
    counts = run_eval.Counts(tp=3, fp=1, fn=3)
    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(0.5)
    assert counts.f1 == pytest.approx(2 * 0.75 * 0.5 / 1.25)


def test_metrics_are_none_rather_than_zero_when_undefined():
    """An empty prediction against empty truth has no precision to report."""
    empty = run_eval.score_field(set(), set())
    assert empty.precision is None
    assert empty.recall is None
    assert empty.f1 is None


def test_no_prediction_gives_zero_recall_and_undefined_precision():
    counts = run_eval.score_field(set(), {"python"})
    assert counts.precision is None
    assert counts.recall == 0.0
    assert counts.f1 is None


def test_counts_add():
    total = run_eval.Counts(1, 2, 3) + run_eval.Counts(10, 20, 30)
    assert (total.tp, total.fp, total.fn) == (11, 22, 33)


def test_priority_map_prefers_must_when_a_skill_is_in_both():
    mapping = run_eval.priority_map({"docker"}, {"docker", "go"})
    assert mapping["docker"] == "must"
    assert mapping["go"] == "nice"


def test_label_normalization_lowercases_and_trims():
    assert run_eval.normalize_labels([" Python ", "DOCKER", ""]) == {"python", "docker"}


def test_label_normalization_of_none_is_empty():
    assert run_eval.normalize_labels(None) == set()


def test_label_normalization_rejects_a_non_list():
    with pytest.raises(ValueError, match="list of skills"):
        run_eval.normalize_labels("python")


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def write_case(root: Path, case_id: str, gold: dict, jd="Required: Python", resume="Skills\nPython"):
    case = root / case_id
    case.mkdir(parents=True)
    (case / "jd.txt").write_text(jd, encoding="utf-8")
    (case / "resume.txt").write_text(resume, encoding="utf-8")
    (case / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    return case


def test_case_with_all_empty_gold_lists_is_unlabeled(tmp_path):
    write_case(tmp_path, "c1", {"jd_must_have": [], "jd_nice_to_have": [], "resume_skills": []})
    cases = run_eval.load_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].is_labeled is False


def test_case_with_any_gold_label_is_labeled(tmp_path):
    write_case(tmp_path, "c1", {"jd_must_have": ["python"]})
    assert run_eval.load_cases(tmp_path)[0].is_labeled is True


def test_notes_key_is_allowed(tmp_path):
    write_case(tmp_path, "c1", {"notes": "hello", "jd_must_have": ["python"]})
    assert len(run_eval.load_cases(tmp_path)) == 1


def test_unknown_gold_key_is_an_error(tmp_path):
    write_case(tmp_path, "c1", {"jd_must_haves": ["python"]})
    with pytest.raises(ValueError, match="unknown keys"):
        run_eval.load_cases(tmp_path)


def test_case_missing_a_file_is_skipped(tmp_path):
    case = tmp_path / "broken"
    case.mkdir()
    (case / "gold.json").write_text("{}", encoding="utf-8")
    assert run_eval.load_cases(tmp_path) == []


# ---------------------------------------------------------------------------
# Hybrid merge rules
# ---------------------------------------------------------------------------


def test_hybrid_merge_is_a_union():
    primary = {"jd_must_have": {"python"}, "jd_nice_to_have": {"go"}, "resume_skills": {"docker"}}
    secondary = {"jd_must_have": {"rust"}, "jd_nice_to_have": set(), "resume_skills": {"git"}}
    merged = run_eval.Extractor._merge(primary, secondary)
    assert merged["jd_must_have"] == {"python", "rust"}
    assert merged["jd_nice_to_have"] == {"go"}
    assert merged["resume_skills"] == {"docker", "git"}


def test_hybrid_merge_lets_the_primary_win_a_priority_conflict():
    """The LLM calling something nice-to-have beats the regex calling it must."""
    primary = {"jd_must_have": set(), "jd_nice_to_have": {"docker"}, "resume_skills": set()}
    secondary = {"jd_must_have": {"docker"}, "jd_nice_to_have": set(), "resume_skills": set()}
    merged = run_eval.Extractor._merge(primary, secondary)
    assert merged["jd_nice_to_have"] == {"docker"}
    assert merged["jd_must_have"] == set()


def test_merged_lists_are_always_disjoint():
    primary = {"jd_must_have": {"python"}, "jd_nice_to_have": {"go"}, "resume_skills": set()}
    secondary = {"jd_must_have": {"go"}, "jd_nice_to_have": {"python"}, "resume_skills": set()}
    merged = run_eval.Extractor._merge(primary, secondary)
    assert merged["jd_must_have"] & merged["jd_nice_to_have"] == set()


# ---------------------------------------------------------------------------
# Deterministic extraction never reports a skill as both priorities
# ---------------------------------------------------------------------------


def test_deterministic_extraction_yields_disjoint_priorities(tmp_path):
    write_case(
        tmp_path,
        "c1",
        {"jd_must_have": ["python"]},
        jd="Requirements:\n- Docker required\n\nNice to have:\n- Docker at scale\n",
    )
    case = run_eval.load_cases(tmp_path)[0]
    predicted = run_eval.Extractor("deterministic").extract(case)
    assert predicted["jd_must_have"] & predicted["jd_nice_to_have"] == set()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_a_results_file(tmp_path, capsys):
    data = tmp_path / "data"
    write_case(data, "c1", {"jd_must_have": ["python"], "resume_skills": ["python"]})
    results = tmp_path / "results"

    exit_code = run_eval.main(
        ["--mode", "deterministic", "--data-dir", str(data), "--results-dir", str(results)]
    )
    assert exit_code == 0

    report = json.loads((results / "deterministic.json").read_text(encoding="utf-8"))
    assert report["mode"] == "deterministic"
    assert report["cases_discovered"] == 1
    assert report["cases_labeled"] == 1
    assert "micro_average" in report
    assert len(report["per_case"]) == 1

    assert "| Field | TP | FP | FN |" in capsys.readouterr().out


def test_cli_reports_unlabeled_cases_without_metrics(tmp_path, capsys):
    data = tmp_path / "data"
    write_case(data, "c1", {"jd_must_have": [], "jd_nice_to_have": [], "resume_skills": []})
    exit_code = run_eval.main(
        ["--data-dir", str(data), "--results-dir", str(tmp_path / "r"), "--no-write"]
    )
    assert exit_code == 0
    assert "No labeled cases found" in capsys.readouterr().out


def test_cli_no_write_leaves_no_file(tmp_path):
    data = tmp_path / "data"
    write_case(data, "c1", {"jd_must_have": ["python"]})
    results = tmp_path / "results"
    run_eval.main(["--data-dir", str(data), "--results-dir", str(results), "--no-write"])
    assert not results.exists()


def test_cli_errors_on_a_missing_data_dir(tmp_path):
    assert run_eval.main(["--data-dir", str(tmp_path / "nope")]) == 2


def test_cli_errors_when_no_cases_are_usable(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    assert run_eval.main(["--data-dir", str(data)]) == 2


def test_min_f1_gate_passes_when_extraction_is_good(tmp_path):
    data = tmp_path / "data"
    write_case(
        data,
        "c1",
        {"jd_must_have": ["python"], "resume_skills": ["python"]},
        jd="Required: Python",
        resume="Skills\nPython",
    )
    exit_code = run_eval.main(
        ["--data-dir", str(data), "--results-dir", str(tmp_path / "r"), "--min-f1", "0.0"]
    )
    assert exit_code == 0


def test_min_f1_gate_fails_when_below_threshold(tmp_path):
    data = tmp_path / "data"
    write_case(
        data,
        "c1",
        {"jd_must_have": ["cobol"], "resume_skills": ["fortran"]},
        jd="Required: Python",
        resume="Skills\nPython",
    )
    exit_code = run_eval.main(
        ["--data-dir", str(data), "--results-dir", str(tmp_path / "r"), "--min-f1", "0.9"]
    )
    assert exit_code == 1


def test_min_f1_gate_fails_loudly_when_nothing_is_labeled(tmp_path):
    """CI must not read "no labels" as a pass."""
    data = tmp_path / "data"
    write_case(data, "c1", {"jd_must_have": [], "jd_nice_to_have": [], "resume_skills": []})
    exit_code = run_eval.main(
        ["--data-dir", str(data), "--results-dir", str(tmp_path / "r"), "--min-f1", "0.5"]
    )
    assert exit_code == 1
