"""Unit tests for src.clustering."""
import pytest
import numpy as np

from src.vectorizer import IncidentVectorizer
from src.clustering import IncidentClusterer

# Two clearly separated groups by vocabulary — a well-behaved clusterer
# should recover this grouping.
GROUP_A = [
    "payment billing invoice charge subscription",
    "billing invoice payment charge monthly",
    "subscription payment charge billing account",
]
GROUP_B = [
    "server crash database timeout error",
    "database error timeout server crash",
    "timeout database crash server error",
]


def test_expected_grouping_recovered():
    texts = GROUP_A + GROUP_B
    vec = IncidentVectorizer()
    matrix = vec.fit_transform(texts)

    clusterer = IncidentClusterer(n_clusters=2)
    labels = clusterer.fit_predict(matrix)

    # Group A's 3 items should share one label; Group B's 3 should share
    # the other (label numbers themselves are arbitrary).
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_predict_before_fit_raises():
    vec = IncidentVectorizer()
    matrix = vec.fit_transform(GROUP_A + GROUP_B)
    clusterer = IncidentClusterer(n_clusters=2)
    with pytest.raises(RuntimeError):
        clusterer.predict(matrix)


def test_top_terms_per_cluster_returns_real_terms():
    texts = GROUP_A + GROUP_B
    vec = IncidentVectorizer()
    matrix = vec.fit_transform(texts)
    feature_names = vec.get_feature_names()

    clusterer = IncidentClusterer(n_clusters=2)
    clusterer.fit(matrix)
    top_terms = clusterer.top_terms_per_cluster(matrix, feature_names, top_n=5)

    assert len(top_terms) == 2
    for terms in top_terms.values():
        assert len(terms) == 5
        assert all(isinstance(t, str) for t in terms)


def test_inertia_before_fit_raises():
    clusterer = IncidentClusterer(n_clusters=2)
    with pytest.raises(RuntimeError):
        clusterer.inertia()


def test_inertia_decreases_with_more_clusters():
    texts = GROUP_A + GROUP_B
    vec = IncidentVectorizer()
    matrix = vec.fit_transform(texts)

    c2 = IncidentClusterer(n_clusters=2).fit(matrix)
    c4 = IncidentClusterer(n_clusters=4).fit(matrix)
    # More clusters should never increase inertia (a known property of k-means)
    assert c4.inertia() <= c2.inertia()


def test_single_cluster_on_uniform_data_handles_gracefully():
    # All-identical text — clustering should not crash even though
    # meaningful separation isn't possible.
    texts = ["error error error"] * 5
    vec = IncidentVectorizer()
    matrix = vec.fit_transform(texts)

    clusterer = IncidentClusterer(n_clusters=2)
    labels = clusterer.fit_predict(matrix)
    assert len(labels) == 5
