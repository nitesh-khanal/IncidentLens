"""
IncidentLens — controlled evaluation dataset (Stage 10).

Deliberately designed test cases covering the categories from the
project spec: near-identical, paraphrased, related-but-different,
unrelated, duplicate, short, noisy, and missing-information queries.
expected_issue_type is None where no meaningful relevance judgment
applies (e.g. missing-information queries) — those cases are evaluated
for crash-safety and output sanity, not precision/recall.
"""

TEST_CASES = [
    {
        "id": "T1",
        "category": "near_identical",
        "query": "Queries in the api integration module are timing out.",
        "expected_issue_type": "performance",
        "notes": "Verbatim text of a real, common templated ticket in the corpus.",
    },
    {
        "id": "T2",
        "category": "paraphrase",
        "query": "Payment API is returning 502 errors and database connections are timing out after a deployment.",
        "expected_issue_type": "performance",
        "notes": "Hand-written paraphrase of the spec's canonical example query.",
    },
    {
        "id": "T3",
        "category": "related_different",
        "query": "The checkout page takes a very long time to load and sometimes times out.",
        "expected_issue_type": "performance",
        "notes": "Different specific symptom, same underlying category (performance).",
    },
    {
        "id": "T4",
        "category": "unrelated",
        "query": "I was charged twice for my monthly subscription this month.",
        "expected_issue_type": "billing_problem",
        "notes": "Should NOT match performance-category incidents.",
    },
    {
        "id": "T5",
        "category": "duplicate",
        "query": "I cannot log in; the system says my password is incorrect.",
        "expected_issue_type": "account_access",
        "notes": "Verbatim text of a known real ticket (also used in Stage 5 EDA examples).",
    },
    {
        "id": "T6",
        "category": "short",
        "query": "Login broken.",
        "expected_issue_type": "account_access",
        "notes": "Minimal-length query — tests whether short text still retrieves sensibly.",
    },
    {
        "id": "T7",
        "category": "noisy",
        "query": "cant LOGIN???!! password keeps saying WRONG!!! pls help",
        "expected_issue_type": "account_access",
        "notes": "Informal casing/punctuation/spelling — tests robustness of preprocessing.",
    },
    {
        "id": "T8",
        "category": "missing_information",
        "query": "It doesn't work.",
        "expected_issue_type": None,
        "notes": "Too vague to have a defensible expected category — evaluated for crash-safety, not precision.",
    },
    {
        "id": "T9",
        "category": "empty",
        "query": "",
        "expected_issue_type": None,
        "notes": "Edge case — must return an empty result list without error.",
    },
    {
        "id": "T10",
        "category": "unseen_vocabulary",
        "query": "quixotic flibbertigibbet zorptastic malfunction",
        "expected_issue_type": None,
        "notes": "No overlap with corpus vocabulary — must return empty results, not crash or hallucinate a match.",
    },
]
