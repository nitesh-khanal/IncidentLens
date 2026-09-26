"""
Stage 14 — build the real combined feature matrix from the actual dataset.
"""
import joblib
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.feature_engineering import FeatureBuilder, LEAKAGE_FIELDS
from src.config import DATA_PROCESSED_DIR

DATA_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"
VECTORIZER_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = DATA_PROCESSED_DIR / "tfidf_matrix.joblib"
FEATURES_PATH = DATA_PROCESSED_DIR / "feature_matrix.joblib"
BUILDER_PATH = DATA_PROCESSED_DIR / "feature_builder.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    vectorizer = IncidentVectorizer.load(VECTORIZER_PATH)
    tfidf_matrix = joblib.load(MATRIX_PATH)

    builder = FeatureBuilder()
    X = builder.fit_transform(df, tfidf_matrix)
    feature_names = builder.get_feature_names(vectorizer.get_feature_names())

    print(f"Combined feature matrix shape: {X.shape}")
    print(f"Total features: {len(feature_names)}")
    print(f"  TF-IDF text features: {tfidf_matrix.shape[1]}")
    print(f"  Categorical (one-hot) features: {len(feature_names) - tfidf_matrix.shape[1] - len(builder.numeric_columns)}")
    print(f"  Numeric/temporal features: {len(builder.numeric_columns)}")

    print(f"\nExcluded as leakage (verified NOT in feature set): {LEAKAGE_FIELDS}")
    overlap = set(LEAKAGE_FIELDS) & set(feature_names)
    assert not overlap, f"LEAKAGE DETECTED: {overlap} found in feature names!"
    print("Confirmed: no leakage fields present in the feature matrix.")

    joblib.dump(X, FEATURES_PATH)
    joblib.dump(builder, BUILDER_PATH)
    print(f"\nSaved feature matrix to {FEATURES_PATH}")
    print(f"Saved feature builder to {BUILDER_PATH}")


if __name__ == "__main__":
    main()
