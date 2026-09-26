"""
Unit tests for src.similarity_engine.

Includes a test that proves, rather than hides, a real TF-IDF limitation:
a pure paraphrase sharing no vocabulary with the query scores low, since
TF-IDF is lexical, not semantic (see docs/tfidf_representation.md).
"""
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.nlp_processor import default_preprocess, ensure_nltk_data

RAW_TEXTS = [
    "payment api returning 502 errors database connection timeout deployment",  # 0
    "payment api returns 502 error database connection times out after deployment",  # 1: near-identical
    "checkout service failing users cannot complete transactions after release",  # 2: paraphrase, no shared words
    "database connection pool exhausted causing timeouts",  # 3: related
    "printer offline no ink cartridge replacement needed",  # 4: unrelated
]

QUERY = "Payment API is returning 502 errors and database connections are timing out after a deployment."


def build_engine():
    ensure_nltk_data()
    processed = [default_preprocess(t) for t in RAW_TEXTS]

    vec = IncidentVectorizer()
    matrix = vec.fit_transform(processed)

    metadata = pd.DataFrame({
        "ticket_id": [f"TCKT_{i}" for i in range(len(RAW_TEXTS))],
        "initial_message": RAW_TEXTS,
        "issue_type": ["bug"] * len(RAW_TEXTS),
        "resolution_summary": ["fixed it"] * len(RAW_TEXTS),
        "has_resolution": [True] * len(RAW_TEXTS),
    })
    return SimilarityEngine(vec, matrix, metadata)


def test_near_identical_ranks_highest():
    # Either TCKT_0 or TCKT_1 is an acceptable top match — both are
    # near-duplicates of the query. Exact winner depends on lemmatizer
    # verb-form handling (see docs/tfidf_representation.md limitations),
    # which is not something this test should hard-code an assumption about.
    engine = build_engine()
    results = engine.find_similar(QUERY, top_n=5)
    assert results[0]["ticket_id"] in {"TCKT_0", "TCKT_1"}
    assert results[0]["similarity"] > 0.5


def test_related_incident_ranks_above_unrelated():
    # TCKT_4 shares zero vocabulary with the query, so it legitimately
    # gets excluded entirely (0.0 similarity) rather than ranked low —
    # that is a stronger "not similar" signal than a poor rank, and is
    # correct engine behavior, not a bug.
    engine = build_engine()
    results = engine.find_similar(QUERY, top_n=5)
    ids = [r["ticket_id"] for r in results]
    assert "TCKT_3" in ids
    assert "TCKT_4" not in ids


def test_unrelated_incident_scores_low_or_absent():
    engine = build_engine()
    results = engine.find_similar(QUERY, top_n=5)
    unrelated = [r for r in results if r["ticket_id"] == "TCKT_4"]
    if unrelated:
        assert unrelated[0]["similarity"] < 0.2


def test_paraphrase_limitation_documented():
    engine = build_engine()
    results = engine.find_similar(QUERY, top_n=5)
    paraphrase = [r for r in results if r["ticket_id"] == "TCKT_2"]
    if paraphrase:
        assert paraphrase[0]["similarity"] < 0.3


def test_empty_query_returns_empty_list():
    engine = build_engine()
    assert engine.find_similar("", top_n=5) == []
    assert engine.find_similar("   ", top_n=5) == []


def test_very_short_query():
    engine = build_engine()
    results = engine.find_similar("payment", top_n=5)
    assert isinstance(results, list)


def test_top_n_respected():
    engine = build_engine()
    results = engine.find_similar(QUERY, top_n=2)
    assert len(results) <= 2


def test_result_fields_present():
    engine = build_engine()
    results = engine.find_similar(QUERY, top_n=1)
    if results:
        r = results[0]
        for field in ["ticket_id", "similarity", "issue_type", "initial_message", "has_resolution"]:
            assert field in r
