"""
Stage 9 — demonstrate the full retrieval layer, real dataset, formatted report.
"""
import joblib
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine, format_retrieval_report
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
    retrieval = RetrievalEngine(engine)

    results = retrieval.retrieve(QUERY, top_n=5)
    print(format_retrieval_report(QUERY, results))

    # Determinism check: same query run twice must give identical results
    results_again = retrieval.retrieve(QUERY, top_n=5)
    print(f"Deterministic: {results == results_again}")


if __name__ == "__main__":
    main()
