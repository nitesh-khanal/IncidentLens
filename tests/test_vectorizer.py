"""Unit tests for src.vectorizer."""
import pytest

from src.vectorizer import IncidentVectorizer

SAMPLE_TEXTS = [
    "payment api return 502 error",
    "database connection timeout deployment",
    "user unable login password incorrect",
    "payment api 502 error database",
]


def test_fit_transform_shape():
    vec = IncidentVectorizer()
    matrix = vec.fit_transform(SAMPLE_TEXTS)
    assert matrix.shape[0] == len(SAMPLE_TEXTS)
    assert matrix.shape[1] == vec.vocabulary_size()


def test_transform_before_fit_raises():
    vec = IncidentVectorizer()
    with pytest.raises(RuntimeError):
        vec.transform(SAMPLE_TEXTS)


def test_vocabulary_contains_expected_terms():
    vec = IncidentVectorizer()
    vec.fit_transform(SAMPLE_TEXTS)
    features = set(vec.get_feature_names())
    assert "payment" in features
    assert "database" in features
    assert "timeout" in features


def test_shared_terms_produce_higher_similarity():
    from sklearn.metrics.pairwise import cosine_similarity

    vec = IncidentVectorizer()
    matrix = vec.fit_transform(SAMPLE_TEXTS)

    sim_0_3 = cosine_similarity(matrix[0], matrix[3])[0][0]
    sim_0_2 = cosine_similarity(matrix[0], matrix[2])[0][0]

    assert sim_0_3 > sim_0_2


def test_empty_corpus_raises():
    vec = IncidentVectorizer()
    with pytest.raises(ValueError):
        vec.fit_transform([])


def test_save_and_load_roundtrip(tmp_path):
    vec = IncidentVectorizer()
    vec.fit_transform(SAMPLE_TEXTS)
    path = tmp_path / "vectorizer.joblib"
    vec.save(path)

    loaded = IncidentVectorizer.load(path)
    assert loaded.vocabulary_size() == vec.vocabulary_size()
    matrix = loaded.transform(SAMPLE_TEXTS)
    assert matrix.shape[0] == len(SAMPLE_TEXTS)
