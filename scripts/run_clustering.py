"""
Stage 11 — real k-selection experiment and final clustering run against
the full dataset. No cluster count or quality number is assumed in
advance; everything here is measured.
"""
import joblib
import pandas as pd

from src.clustering import IncidentClusterer, evaluate_k_range
from src.vectorizer import IncidentVectorizer
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

DATA_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"
VECTORIZER_PATH = DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib"
MATRIX_PATH = DATA_PROCESSED_DIR / "tfidf_matrix.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "clustering_report.md"

K_RANGE = [4, 6, 8, 10, 12, 15]
CHOSEN_K = 8  # matches issue_type's 8 known classes — a deliberate choice to
              # test whether unsupervised clustering rediscovers that structure


def main():
    df = pd.read_csv(DATA_PATH)
    vectorizer = IncidentVectorizer.load(VECTORIZER_PATH)
    matrix = joblib.load(MATRIX_PATH)
    feature_names = vectorizer.get_feature_names()

    print("Evaluating k range (this samples for silhouette; may take a minute)...")
    k_results = evaluate_k_range(matrix, K_RANGE)

    print(f"\nFitting final model with k={CHOSEN_K}...")
    clusterer = IncidentClusterer(n_clusters=CHOSEN_K)
    labels = clusterer.fit_predict(matrix)
    df["cluster"] = labels

    top_terms = clusterer.top_terms_per_cluster(matrix, feature_names, top_n=10)

    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())

    print("\nTop terms per cluster:")
    for cluster_id, terms in top_terms.items():
        print(f"  Cluster {cluster_id}: {', '.join(terms)}")

    # Sanity check (NOT validation) against issue_type — clustering is
    # unsupervised and doesn't use this label; we only check afterward
    # whether the discovered structure resembles the known categories.
    print("\nCluster vs issue_type crosstab (sanity check only):")
    crosstab = pd.crosstab(df["cluster"], df["issue_type"])
    print(crosstab)

    report_lines = [
        "# Incident Clustering Report",
        "",
        "## k-selection results (real measurements)",
        "",
        "| k | Inertia | Silhouette (5K sample) |",
        "|---|---|---|",
    ]
    for r in k_results:
        sil_str = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "N/A"
        report_lines.append(f"| {r['k']} | {r['inertia']:.2f} | {sil_str} |")

    report_lines += [
        "",
        f"## Chosen k = {CHOSEN_K}",
        "",
        "## Cluster sizes",
        "",
        df["cluster"].value_counts().sort_index().to_markdown(),
        "",
        "## Top terms per cluster",
        "",
    ]
    for cluster_id, terms in top_terms.items():
        report_lines.append(f"- **Cluster {cluster_id}:** {', '.join(terms)}")

    report_lines += [
        "",
        "## Cluster vs issue_type crosstab (sanity check, NOT validation)",
        "",
        crosstab.to_markdown(),
        "",
        "Clustering is unsupervised and does not use issue_type as input. This",
        "crosstab is shown only to check, after the fact, whether discovered",
        "clusters resemble known categories — a strong alignment would suggest",
        "the clusters are picking up on real, meaningful structure; a weak",
        "alignment would not mean the clustering is wrong, since clusters can",
        "legitimately capture different structure than the labeled categories.",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines))
    print(f"\nReport written to {REPORT_PATH}")

    df.to_csv(DATA_PROCESSED_DIR / "tickets_clustered.csv", index=False)
    joblib.dump(clusterer, DATA_PROCESSED_DIR / "clusterer.joblib")


if __name__ == "__main__":
    main()
