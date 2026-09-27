"""
Stage 19 — demonstrate real explainability against the trained models.
"""
import joblib
import pandas as pd

from src.vectorizer import IncidentVectorizer
from src.nlp_processor import default_preprocess, ensure_nltk_data
from src.explainability import explain_similarity, explain_classification
from src.config import DATA_PROCESSED_DIR

DATA_PATH = DATA_PROCESSED_DIR / "tickets_clustered.csv"
VECTORIZER_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"

QUERY = "Payment API is returning 502 errors and database connections are timing out after a deployment."


def main():
    ensure_nltk_data()
    df = pd.read_csv(DATA_PATH)
    vectorizer = IncidentVectorizer.load(VECTORIZER_PATH)

    # --- Similarity explanation: query vs. one real similar historical incident ---
    processed_query = default_preprocess(QUERY)
    query_vec = vectorizer.transform([processed_query])

    # Pick a genuinely close match (the templated "queries timing out" ticket
    # from Stage 8/9's demos, known to score ~50% similarity) rather than an
    # arbitrary row, so the explanation actually has something to show.
    performance_texts = df[df["issue_type"] == "performance"]["processed_message"].dropna()
    candidate_text = performance_texts[
        performance_texts.str.contains("timing", na=False)
    ].iloc[0]
    candidate_vec = vectorizer.transform([candidate_text])

    print("=== Similarity Explanation ===")
    print(f"Query: {QUERY}")
    print(f"Candidate: {candidate_text}\n")
    explanation = explain_similarity(query_vec, candidate_vec, vectorizer.get_feature_names())
    print(f"Total cosine similarity: {explanation['total_similarity']:.4f}\n")
    print("Top contributing terms:")
    for term in explanation["contributing_terms"]:
        print(f"  {term['term']:15s} contribution={term['contribution']:.4f} "
              f"({term['pct_of_total']:.1f}% of total)")

    # --- Classification explanation: Logistic Regression (class-specific) ---
    print("\n\n=== Classification Explanation: Logistic Regression ===")
    lr_model = joblib.load(DATA_PROCESSED_DIR / "models" / "logistic_regression.joblib")
    builder = joblib.load(DATA_PROCESSED_DIR / "feature_builder.joblib")
    feature_names = builder.get_feature_names(vectorizer.get_feature_names())

    lr_explanation = explain_classification(lr_model, feature_names, "performance", top_k=10)
    print(f"Mechanism: {lr_explanation['mechanism']}")
    print(f"Note: {lr_explanation['note']}")
    for f in lr_explanation["top_features"]:
        print(f"  {f['feature']:25s} weight={f['weight']:.4f}")

    # --- Classification explanation: Random Forest (global, not class-specific) ---
    print("\n\n=== Classification Explanation: Random Forest ===")
    rf_model = joblib.load(DATA_PROCESSED_DIR / "models" / "random_forest.joblib")
    rf_explanation = explain_classification(rf_model, feature_names, "performance", top_k=10)
    print(f"Mechanism: {rf_explanation['mechanism']}")
    print(f"Note: {rf_explanation['note']}")
    for f in rf_explanation["top_features"]:
        print(f"  {f['feature']:25s} importance={f['importance']:.4f}")


if __name__ == "__main__":
    main()
