"""Unit tests for src.retrieval."""
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.nlp_processor import default_preprocess, ensure_nltk_data

RAW_TEXTS = [
    "payment api returning 502 errors database connection timeout deployment",
    "checkout service failing users cannot complete transactions after release",
    "database connection pool exhausted causing timeouts",
    "printer offline no ink cartridge replacement needed",
]


def build_retrieval_engine():
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
        "created_at": ["2024-01-01"] * len(RAW_TEXTS),
        "priority": ["low"] * len(RAW_TEXTS),
        "status": ["resolved"] * len(RAW_TEXTS),
        "customer_segment": ["individual"] * len(RAW_TEXTS),
        "channel": ["email"] * len(RAW_TEXTS),
        "region": ["eu"] * len(RAW_TEXTS),
    })
    engine = SimilarityEngine(vec, matrix, metadata)
    return RetrievalEngine(engine)


def test_empty_query_returns_empty_list():
    retrieval = build_retrieval_engine()
    assert retrieval.retrieve("", top_n=5) == []


def test_very_short_query():
    retrieval = build_retrieval_engine()
    results = retrieval.retrieve("payment", top_n=5)
    assert isinstance(results, list)


def test_unseen_words_returns_empty_no_crash():
    retrieval = build_retrieval_engine()
    results = retrieval.retrieve("qzxjklw plorvantix flerbidoo", top_n=5)
    assert results == []


def test_unrelated_query_matches_correct_incident():
    retrieval = build_retrieval_engine()
    results = retrieval.retrieve("printer ink cartridge offline", top_n=5)
    assert results[0]["ticket_id"] == "TCKT_3"


def test_duplicate_incident_retrieves_itself_top():
    retrieval = build_retrieval_engine()
    results = retrieval.retrieve(RAW_TEXTS[0], top_n=5)
    assert results[0]["ticket_id"] == "TCKT_0"
    assert results[0]["similarity"] > 0.95


def test_valid_incident_returns_enriched_metadata():
    retrieval = build_retrieval_engine()
    results = retrieval.retrieve(RAW_TEXTS[2], top_n=3)
    assert len(results) >= 1
    for field in ["priority", "status", "region", "created_at"]:
        assert field in results[0]


def test_deterministic_results():
    retrieval = build_retrieval_engine()
    r1 = retrieval.retrieve(RAW_TEXTS[0], top_n=5)
    r2 = retrieval.retrieve(RAW_TEXTS[0], top_n=5)
    assert r1 == r2
