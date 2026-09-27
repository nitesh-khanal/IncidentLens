"""
Stage 18 — demonstrate the full intelligence engine against the real
dataset and all previously trained artifacts.
"""
import joblib
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.analyzer import IncidentIntelligenceEngine, format_intelligence_report
from src.config import DATA_PROCESSED_DIR

DATA_PATH = DATA_PROCESSED_DIR / "tickets_clustered.csv"
VECTORIZER_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = DATA_PROCESSED_DIR / "tfidf_matrix.joblib"
CLUSTERER_PATH = DATA_PROCESSED_DIR / "clusterer.joblib"
FEATURE_BUILDER_PATH = DATA_PROCESSED_DIR / "feature_builder.joblib"
MODEL_PATH = DATA_PROCESSED_DIR / "models" / "random_forest.joblib"

QUERY = "Payment API is returning 502 errors and database connections are timing out after a deployment."


def main():
    df = pd.read_csv(DATA_PATH)
    vectorizer = IncidentVectorizer.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    clusterer = joblib.load(CLUSTERER_PATH)
    feature_builder = joblib.load(FEATURE_BUILDER_PATH)
    classifier = joblib.load(MODEL_PATH)

    # Rebuild top terms per cluster (from Stage 11's clusterer)
    top_terms = clusterer.top_terms_per_cluster(matrix, vectorizer.get_feature_names(), top_n=10)

    sim_engine = SimilarityEngine(vectorizer, matrix, df)
    retrieval = RetrievalEngine(sim_engine)

    engine = IncidentIntelligenceEngine(
        retrieval_engine=retrieval,
        clusterer=clusterer,
        cluster_top_terms=top_terms,
        classifier_model=classifier,
        feature_builder=feature_builder,
        vectorizer=vectorizer,
        metadata_df=df,
    )

    print("### Demo 1: no metadata provided ###\n")
    result1 = engine.analyze(QUERY, metadata=None, top_n=3)
    print(format_intelligence_report(result1))

    print("\n\n### Demo 2: with metadata provided ###\n")
    result2 = engine.analyze(
        QUERY,
        metadata={
            "customer_segment": "small_business", "channel": "email",
            "product_area": "api_integration", "priority": "high",
            "sla_plan": "standard", "platform": "web", "region": "eu",
            "has_attachment": 0,
        },
        top_n=3,
    )
    print(format_intelligence_report(result2))

    print("\n\n### Demo 3: query likely to hit account_access (known limitation case) ###\n")
    result3 = engine.analyze("I forgot my password and got locked out of my account", top_n=3)
    print(format_intelligence_report(result3))


if __name__ == "__main__":
    main()
