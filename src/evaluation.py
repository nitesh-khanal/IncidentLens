"""
IncidentLens — retrieval evaluation metrics.

Relevance is defined as issue_type match with the query's known source
category — a proxy for true relevance, not human-annotated ground truth.
See docs/retrieval_evaluation.md for the full rationale and limitations.
"""


def hit_at_k(retrieved_issue_types: list, expected_issue_type: str, k: int) -> bool:
    """True if a relevant (issue_type-matching) result appears in the top k."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if expected_issue_type is None:
        return None
    return expected_issue_type in retrieved_issue_types[:k]


def precision_at_k(retrieved_issue_types: list, expected_issue_type: str, k: int):
    """Relevant hits divided by requested k; missing results count as misses."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if expected_issue_type is None:
        return None
    top_k = retrieved_issue_types[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for t in top_k if t == expected_issue_type)
    return relevant_count / k
