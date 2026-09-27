"""Unit tests for src.analyzer."""
import pandas as pd
from scipy.sparse import csr_matrix

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.clustering import IncidentClusterer
from src.classifier import train_model
from src.feature_engineering import FeatureBuilder
from src.analyzer import IncidentIntelligenceEngine, format_intelligence_report
from src.nlp_processor import default_preprocess, ensure_nltk_data

RAW_TEXTS = [
    "payment api returning 502 errors database connection timeout deployment",
    "account locked after failed login password incorrect",
    "checkout page slow loading crashing frequently",
    "invoice charged twice billing subscription monthly",
]
ISSUE_TYPES = ["performance", "account_access", "bug", "billing_problem"]


def build_full_engine():
    ensure_nltk_data()
    processed = [default_preprocess(t) for t in RAW_TEXTS]

    vec = IncidentVectorizer()
    matrix = vec.fit_transform(processed)

    metadata = pd.DataFrame({
        "ticket_id": [f"TCKT_{i}" for i in range(len(RAW_TEXTS))],
        "initial_message": RAW_TEXTS,
        "issue_type": ISSUE_TYPES,
        "resolution_summary": ["fixed it"] * len(RAW_TEXTS),
        "has_resolution": [True, False, True, True],
        "resolution_time_hours": [5.0, None, 3.0, 2.0],
        "created_at": ["2024-01-01"] * len(RAW_TEXTS),
        "priority": ["low"] * len(RAW_TEXTS),
        "status": ["resolved"] * len(RAW_TEXTS),
        "customer_segment": ["individual"] * len(RAW_TEXTS),
        "channel": ["email"] * len(RAW_TEXTS),
        "product_area": ["billing"] * len(RAW_TEXTS),
        "sla_plan": ["standard"] * len(RAW_TEXTS),
        "platform": ["web"] * len(RAW_TEXTS),
        "region": ["eu"] * len(RAW_TEXTS),
        "has_attachment": [0, 0, 0, 0],
    })

    sim_engine = SimilarityEngine(vec, matrix, metadata)
    retrieval = RetrievalEngine(sim_engine)

    clusterer = IncidentClusterer(n_clusters=2)
    clusterer.fit(matrix)
    top_terms = clusterer.top_terms_per_cluster(matrix, vec.get_feature_names(), top_n=5)

    builder = FeatureBuilder()
    X = builder.fit_transform(metadata, matrix)
    classifier = train_model("decision_tree", X, metadata["issue_type"].values)

    engine = IncidentIntelligenceEngine(
        retrieval_engine=retrieval, clusterer=clusterer, cluster_top_terms=top_terms,
        classifier_model=classifier, feature_builder=builder, vectorizer=vec,
        metadata_df=metadata,
    )
    return engine


def test_analyze_returns_all_sections():
    engine = build_full_engine()
    result = engine.analyze(RAW_TEXTS[0], top_n=2)
    assert set(result.keys()) == {"query", "classification", "similar_incidents", "cluster", "recurring_patterns"}


def test_classification_flags_missing_metadata():
    engine = build_full_engine()
    result = engine.analyze(RAW_TEXTS[0], metadata=None, top_n=2)
    assert result["classification"]["metadata_provided"] is False
    assert "NOT provided" in result["classification"]["note"]


def test_classification_flags_provided_metadata():
    engine = build_full_engine()
    result = engine.analyze(RAW_TEXTS[0], metadata={"priority": "high"}, top_n=2)
    assert result["classification"]["metadata_provided"] is True
    assert "provided by the caller" in result["classification"]["note"]


def test_account_access_prediction_includes_limitation_note():
    engine = build_full_engine()
    result = engine.analyze(RAW_TEXTS[1], top_n=2)  # account_access-like text
    if result["classification"]["predicted_issue_type"] == "account_access":
        assert "Known limitation" in result["classification"]["note"]


def test_empty_query_returns_none_sections_gracefully():
    engine = build_full_engine()
    result = engine.analyze("", top_n=2)
    assert result["classification"] is None
    assert result["cluster"] is None
    assert result["recurring_patterns"] is None
    assert result["similar_incidents"] == []


def test_cluster_note_present_and_not_a_root_cause_claim():
    engine = build_full_engine()
    result = engine.analyze(RAW_TEXTS[0], top_n=2)
    assert "NOT a" in result["cluster"]["note"]


def test_recurring_patterns_frequency_computed_correctly():
    engine = build_full_engine()
    result = engine.analyze(RAW_TEXTS[0], top_n=2)
    rp = result["recurring_patterns"]
    # 1 of 4 rows has each issue_type in this tiny synthetic set -> 25%
    assert rp["issue_type_frequency_pct"] == 25.0
    assert rp["historical_incident_count"] == 1


def test_format_report_produces_readable_string():
    engine = build_full_engine()
    result = engine.analyze(RAW_TEXTS[0], top_n=2)
    report = format_intelligence_report(result)
    assert "NEW INCIDENT" in report
    assert "CLASSIFICATION" in report
    assert "SIMILAR HISTORICAL INCIDENTS" in report
    assert "CLUSTER MEMBERSHIP" in report
    assert "RECURRING PATTERNS" in report
