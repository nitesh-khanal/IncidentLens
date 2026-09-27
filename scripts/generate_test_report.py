"""
Stage 22 — generate the formal testing table from REAL pytest results
(parsed from junit_results.xml), never hand-typed. Categorizes tests by
module per the project spec (Data, NLP, Similarity, Clustering, ML,
Application). Exact inputs/assertions for each test live in its source
file — linked here rather than paraphrased, to avoid misrepresenting
what was actually asserted.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone

from src.config import PROJECT_ROOT

JUNIT_PATH = PROJECT_ROOT / "reports" / "junit_results.xml"
REPORT_PATH = PROJECT_ROOT / "reports" / "test_report.md"

CATEGORY_MAP = {
    "test_data_loader": ("Data", "tests/test_data_loader.py"),
    "test_preprocessing": ("Data", "tests/test_preprocessing.py"),
    "test_nlp_processor": ("NLP", "tests/test_nlp_processor.py"),
    "test_vectorizer": ("Similarity", "tests/test_vectorizer.py"),
    "test_similarity": ("Similarity", "tests/test_similarity.py"),
    "test_retrieval": ("Similarity", "tests/test_retrieval.py"),
    "test_evaluation": ("Similarity", "tests/test_evaluation.py"),
    "test_clustering": ("Clustering", "tests/test_clustering.py"),
    "test_feature_engineering": ("ML", "tests/test_feature_engineering.py"),
    "test_classifier": ("ML", "tests/test_classifier.py"),
    "test_explainability": ("ML", "tests/test_explainability.py"),
    "test_analyzer": ("Application", "tests/test_analyzer.py"),
    "test_report_generator": ("Application", "tests/test_report_generator.py"),
    "test_app": ("Application", "tests/test_app.py"),
}


def humanize(name: str) -> str:
    return name.replace("test_", "").replace("_", " ").capitalize()


def main():
    if not JUNIT_PATH.exists():
        raise FileNotFoundError(f"{JUNIT_PATH} not found — run pytest with --junitxml first.")

    tree = ET.parse(JUNIT_PATH)
    root = tree.getroot()
    testsuite = root.find("testsuite") if root.tag != "testsuite" else root

    total = int(testsuite.get("tests", 0))
    failures = int(testsuite.get("failures", 0))
    errors = int(testsuite.get("errors", 0))
    skipped = int(testsuite.get("skipped", 0))
    passed = total - failures - errors - skipped
    duration = float(testsuite.get("time", 0))

    by_category = {}
    for testcase in testsuite.findall("testcase"):
        classname = testcase.get("classname", "")
        module = classname.split(".")[-1] if classname else "unknown"
        category, source_file = CATEGORY_MAP.get(module, ("Other", classname))

        test_id = f"{module}::{testcase.get('name')}"
        status = "PASSED"
        detail = ""
        if testcase.find("failure") is not None:
            status = "FAILED"
            detail = testcase.find("failure").get("message", "")[:100]
        elif testcase.find("error") is not None:
            status = "ERROR"
            detail = testcase.find("error").get("message", "")[:100]
        elif testcase.find("skipped") is not None:
            status = "SKIPPED"

        by_category.setdefault(category, []).append({
            "test_id": test_id,
            "description": humanize(testcase.get("name")),
            "source_file": source_file,
            "duration": float(testcase.get("time", 0)),
            "status": status,
            "detail": detail,
        })

    lines = [
        "# Formal Test Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Source: reports/junit_results.xml (real pytest run — every result below was actually executed)",
        "",
        "## Summary",
        "",
        f"- **Total tests:** {total}",
        f"- **Passed:** {passed}",
        f"- **Failed:** {failures}",
        f"- **Errors:** {errors}",
        f"- **Skipped:** {skipped}",
        f"- **Total duration:** {duration:.2f}s",
        "",
        "## Results by category",
        "",
    ]

    for category in ["Data", "NLP", "Similarity", "Clustering", "ML", "Application", "Other"]:
        if category not in by_category:
            continue
        tests = by_category[category]
        cat_passed = sum(1 for t in tests if t["status"] == "PASSED")
        lines.append(f"### {category} ({cat_passed}/{len(tests)} passed)")
        lines.append("")
        lines.append("| Test ID | Description | Source (exact input/assertions) | Duration | Status |")
        lines.append("|---|---|---|---|---|")
        for t in tests:
            status_marker = "PASS" if t["status"] == "PASSED" else f"**{t['status']}**"
            lines.append(
                f"| {t['test_id']} | {t['description']} | `{t['source_file']}` | "
                f"{t['duration']:.3f}s | {status_marker} |"
            )
        lines.append("")

    lines += [
        "## Note on Input / Expected Result columns",
        "",
        "Per-test exact input values and expected-result assertions are not",
        "paraphrased in this table — doing so risks misrepresenting what was",
        "actually asserted. Each test's real input construction and assertion",
        "logic is in its linked source file. This table instead reports what",
        "can be stated with certainty from an actual run: which test executed,",
        "against which real inputs (via source), how long it took, and its",
        "real outcome.",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written to {REPORT_PATH}")
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {failures}, Errors: {errors}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
