"""
IncidentLens — retrieval evaluation metrics.

Relevance is defined as issue_type match with the query's known source
category — a proxy for true relevance, not human-annotated ground truth.
See docs/retrieval_evaluation.md for the full rationale and limitations.
"""


def hit_at_k(retrieved_issue_types: list, expected_issue_type: str, k: int) -> bool:
    """True if a relevant (issue_type-matching) result appears in the top k."""
    if expected_issue_type is None:
        return None
    return expected_issue_type in retrieved_issue_types[:k]


def precision_at_k(retrieved_issue_types: list, expected_issue_type: str, k: int):
    """Fraction of the top-k results that are relevant. None if no expected category."""
    if expected_issue_type is None:
        return None
    top_k = retrieved_issue_types[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for t in top_k if t == expected_issue_type)
    return relevant_count / len(top_k)
