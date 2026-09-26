"""Unit tests for src.feature_engineering."""
import pandas as pd
from scipy.sparse import csr_matrix

from src.feature_engineering import (
    FeatureBuilder, add_temporal_features, add_text_length_feature, LEAKAGE_FIELDS,
)

SAMPLE_DF = pd.DataFrame({
    "created_at": ["2024-03-15T14:30:00", "2024-07-01T08:00:00"],
    "initial_message": ["short msg", "a somewhat longer incident description here"],
    "customer_segment": ["individual", "small_business"],
    "channel": ["email", "chat"],
    "product_area": ["billing", "api_integration"],
    "priority": ["low", "high"],
    "sla_plan": ["standard", "standard"],
    "platform": ["web", "android"],
    "region": ["eu", "apac"],
    "has_attachment": [0, 1],
})


def dummy_tfidf():
    return csr_matrix([[0.1, 0.2], [0.3, 0.4]])


def test_temporal_features_extracted_correctly():
    df = add_temporal_features(SAMPLE_DF)
    assert df.loc[0, "hour_of_day"] == 14
    assert df.loc[0, "month"] == 3
    assert df.loc[1, "hour_of_day"] == 8
    assert df.loc[1, "month"] == 7


def test_message_length_computed():
    df = add_text_length_feature(SAMPLE_DF)
    assert df.loc[0, "message_length"] == len("short msg")


def test_fit_transform_produces_correct_shape():
    builder = FeatureBuilder()
    X = builder.fit_transform(SAMPLE_DF, dummy_tfidf())
    assert X.shape[0] == 2
    # 2 tfidf + one-hot categorical + 5 numeric >= at least 2 + 5 = 7
    assert X.shape[1] >= 7


def test_transform_before_fit_raises():
    builder = FeatureBuilder()
    try:
        builder.transform(SAMPLE_DF, dummy_tfidf())
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_no_leakage_fields_in_feature_names():
    builder = FeatureBuilder()
    builder.fit_transform(SAMPLE_DF, dummy_tfidf())
    feature_names = builder.get_feature_names(["tfidf_a", "tfidf_b"])
    overlap = set(LEAKAGE_FIELDS) & set(feature_names)
    assert not overlap, f"Leakage fields found: {overlap}"


def test_unseen_category_handled_without_crash():
    builder = FeatureBuilder()
    builder.fit_transform(SAMPLE_DF, dummy_tfidf())

    new_df = SAMPLE_DF.copy()
    new_df.loc[0, "region"] = "never_seen_before_region"
    X = builder.transform(new_df, dummy_tfidf())
    assert X.shape[0] == 2


def test_feature_names_length_matches_matrix_columns():
    builder = FeatureBuilder()
    X = builder.fit_transform(SAMPLE_DF, dummy_tfidf())
    feature_names = builder.get_feature_names(["tfidf_a", "tfidf_b"])
    assert X.shape[1] == len(feature_names)
