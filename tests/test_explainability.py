"""Unit tests for src.explainability."""
import numpy as np
from scipy.sparse import csr_matrix

from src.explainability import explain_similarity, explain_classification, explain_shared_metadata
from src.vectorizer import IncidentVectorizer

TEXTS = [
    "payment api error database timeout",
    "payment api error database timeout deployment",
    "printer offline ink cartridge",
]


def build_vectors():
    vec = IncidentVectorizer()
    matrix = vec.fit_transform(TEXTS)
    return vec, matrix


def test_similarity_explanation_shared_terms_identified():
    vec, matrix = build_vectors()
    result = explain_similarity(matrix[0], matrix[1], vec.get_feature_names())
    terms = {t["term"] for t in result["contributing_terms"]}
    assert "payment" in terms
    assert "database" in terms


def test_similarity_explanation_contributions_sum_to_total():
    vec, matrix = build_vectors()
    result = explain_similarity(matrix[0], matrix[1], vec.get_feature_names(), top_k=100)
    total_from_terms = sum(t["contribution"] for t in result["contributing_terms"])
    assert abs(total_from_terms - result["total_similarity"]) < 1e-6


def test_similarity_explanation_unrelated_has_no_contributing_terms():
    vec, matrix = build_vectors()
    result = explain_similarity(matrix[0], matrix[2], vec.get_feature_names())
    assert result["total_similarity"] == 0.0
    assert result["contributing_terms"] == []


def test_classification_explanation_logistic_regression_is_class_specific():
    from sklearn.linear_model import LogisticRegression
    X = csr_matrix(np.random.rand(20, 5))
    y = ["a"] * 10 + ["b"] * 10
    model = LogisticRegression().fit(X, y)
    result = explain_classification(model, [f"f{i}" for i in range(5)], "a")
    assert result["mechanism"] == "logistic_regression_coefficients"
    assert result["class_specific"] is True


def test_classification_explanation_tree_is_global_not_class_specific():
    from sklearn.tree import DecisionTreeClassifier
    X = csr_matrix(np.random.rand(20, 5))
    y = ["a"] * 10 + ["b"] * 10
    model = DecisionTreeClassifier().fit(X, y)
    result = explain_classification(model, [f"f{i}" for i in range(5)], "a")
    assert result["mechanism"] == "tree_feature_importances"
    assert result["class_specific"] is False
    assert "NOT" in result["note"]


def test_shared_metadata_identifies_matches():
    import pandas as pd
    query_meta = {"customer_segment": "individual", "priority": "high", "region": "eu"}
    candidate_row = pd.Series({"customer_segment": "individual", "priority": "low", "region": "eu"})
    shared = explain_shared_metadata(query_meta, candidate_row)
    assert "customer_segment" in shared
    assert "region" in shared
    assert "priority" not in shared
