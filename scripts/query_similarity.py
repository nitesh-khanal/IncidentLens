"""
Stage 8 — demonstrate the similarity engine against the real dataset.
"""
import joblib
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.config import DATA_PROCESSED_DIR

DATA_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"
VECTORIZER_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = DATA_PROCESSED_DIR / "tfidf_matrix.joblib"

QUERY = "Payment API is returning 502 errors and database connections are timing out after a deployment."


def main():
    df = pd.read_csv(DATA_PATH)
    vectorizer = IncidentVectorizer.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)

    engine = SimilarityEngine(vectorizer, matrix, df)

    print(f"Query: {QUERY}\n")
    results = engine.find_similar(QUERY, top_n=5)

    if not results:
        print("No similar incidents found.")
        return

    for i, r in enumerate(results, 1):
        print(f"#{i} - {r['ticket_id']}  (similarity: {r['similarity']:.3f})")
        print(f"   Issue type: {r['issue_type']}")
        print(f"   Description: {r['initial_message']}")
        if r["has_resolution"]:
            print(f"   Historical resolution: {r['resolution_summary']}")
        else:
            print("   Historical resolution: none recorded")
        print()


if __name__ == "__main__":
    main()
