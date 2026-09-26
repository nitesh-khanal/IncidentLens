"""
Stage 10 — run the controlled evaluation set against the real retrieval
engine and produce a real evaluation table + report. No numbers here
are invented; everything is computed from an actual run.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.evaluation import hit_at_k, precision_at_k
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "data" / "test"))
from controlled_eval_set import TEST_CASES  # noqa: E402

DATA_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"
VECTORIZER_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = DATA_PROCESSED_DIR / "tfidf_matrix.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "retrieval_evaluation.md"


def main():
    df = pd.read_csv(DATA_PATH)
    vectorizer = IncidentVectorizer.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)

    engine = SimilarityEngine(vectorizer, matrix, df)
    retrieval = RetrievalEngine(engine)

    rows = []
    p1_scores, p3_scores, p5_scores = [], [], []

    for case in TEST_CASES:
        results = retrieval.retrieve(case["query"], top_n=5)
        retrieved_types = [r["issue_type"] for r in results]
        expected = case["expected_issue_type"]

        h1 = hit_at_k(retrieved_types, expected, 1)
        h3 = hit_at_k(retrieved_types, expected, 3)
        h5 = hit_at_k(retrieved_types, expected, 5)
        p1 = precision_at_k(retrieved_types, expected, 1)
        p3 = precision_at_k(retrieved_types, expected, 3)
        p5 = precision_at_k(retrieved_types, expected, 5)

        if p1 is not None:
            p1_scores.append(p1)
            p3_scores.append(p3)
            p5_scores.append(p5)

        rows.append({
            "id": case["id"],
            "category": case["category"],
            "query": case["query"][:50] + ("..." if len(case["query"]) > 50 else ""),
            "expected": expected or "N/A",
            "n_results": len(results),
            "hit@1": h1,
            "hit@3": h3,
            "hit@5": h5,
            "top_result_types": retrieved_types[:3],
        })

        print(f"[{case['id']}] {case['category']}: {len(results)} results, "
              f"hit@1={h1}, hit@3={h3}, hit@5={h5}")

    avg_p1 = sum(p1_scores) / len(p1_scores) if p1_scores else 0
    avg_p3 = sum(p3_scores) / len(p3_scores) if p3_scores else 0
    avg_p5 = sum(p5_scores) / len(p5_scores) if p5_scores else 0

    print(f"\nAverage Precision@1: {avg_p1:.3f}")
    print(f"Average Precision@3: {avg_p3:.3f}")
    print(f"Average Precision@5: {avg_p5:.3f}")
    print(f"(computed over {len(p1_scores)} test cases with a defined expected category)")

    report_lines = [
        "# Retrieval Evaluation Report",
        "",
        "## Relevance definition",
        "A retrieved result is considered relevant if its `issue_type` matches the",
        "query's known source category. This is a proxy for true relevance (no",
        "human-annotated ground truth exists for this dataset) — see",
        "docs/retrieval_evaluation.md for the full rationale and limitations.",
        "",
        "## Aggregate metrics",
        f"- Average Precision@1: {avg_p1:.3f}",
        f"- Average Precision@3: {avg_p3:.3f}",
        f"- Average Precision@5: {avg_p5:.3f}",
        f"- Computed over {len(p1_scores)} of {len(TEST_CASES)} test cases (cases without",
        "  a defined expected category are evaluated for crash-safety, not precision)",
        "",
        "## Per-test-case results",
        "",
        "| ID | Category | Query | Expected | # Results | Hit@1 | Hit@3 | Hit@5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['id']} | {row['category']} | {row['query']} | {row['expected']} | "
            f"{row['n_results']} | {row['hit@1']} | {row['hit@3']} | {row['hit@5']} |"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
