"""
Stage 7 — build the TF-IDF representation from the real, preprocessed
dataset (Stage 6's output) and inspect its real shape and vocabulary.
"""
import pandas as pd
import joblib

from src.vectorizer import IncidentVectorizer
from src.config import DATA_PROCESSED_DIR

INPUT_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"
VECTORIZER_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = DATA_PROCESSED_DIR / "tfidf_matrix.joblib"


def main():
    df = pd.read_csv(INPUT_PATH)
    texts = df["processed_message"].fillna("")

    vectorizer = IncidentVectorizer()
    matrix = vectorizer.fit_transform(texts)

    print(f"Documents: {matrix.shape[0]}")
    print(f"Vocabulary size: {vectorizer.vocabulary_size()}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero entries: {matrix.nnz}")
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
    print(f"Density: {density:.4f}%  (i.e. {100 - density:.4f}% sparse)")

    print("\nTop 15 highest-IDF (most distinctive) terms:")
    feature_names = vectorizer.get_feature_names()
    idf_scores = vectorizer.vectorizer.idf_
    top_idf = sorted(zip(feature_names, idf_scores), key=lambda x: -x[1])[:15]
    for term, score in top_idf:
        print(f"  {term}: {score:.3f}")

    print("\nTop 15 lowest-IDF (most common) terms:")
    bottom_idf = sorted(zip(feature_names, idf_scores), key=lambda x: x[1])[:15]
    for term, score in bottom_idf:
        print(f"  {term}: {score:.3f}")

    joblib.dump(matrix, MATRIX_PATH)
    vectorizer.save(VECTORIZER_PATH)
    print(f"\nSaved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved matrix to {MATRIX_PATH}")


if __name__ == "__main__":
    main()
