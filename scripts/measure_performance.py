"""
Stage 23 — real performance measurement across the pipeline. Every
number here comes from an actual timed run against real data/artifacts,
at multiple dataset sizes where the operation scales with size.
"""
import time
import statistics

import joblib
import pandas as pd

from src.data_loader import load_primary_dataset
from src.preprocessing import clean_tickets
from src.nlp_processor import default_preprocess, ensure_nltk_data
from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.clustering import IncidentClusterer
from src.analyzer import IncidentIntelligenceEngine, DEFAULT_METADATA
from src.report_generator import generate_html_report
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "performance.md"
SIZES = [1000, 10000, 50000, 100000]
QUERY = "Payment API is returning 502 errors and database connections are timing out after a deployment."


def timeit(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def main():
    ensure_nltk_data()
    results = {"sizes": {}}

    # --- Data loading (fixed cost, full dataset only) ---
    raw, t_load = timeit(load_primary_dataset)
    print(f"Data loading (100,000 rows): {t_load:.3f}s")
    results["load_time"] = t_load

    # --- Preprocessing + NLP + vectorization at multiple sizes ---
    for size in SIZES:
        sample = raw.head(size).copy()

        cleaned, t_clean = timeit(clean_tickets, sample)

        start = time.perf_counter()
        cleaned["processed_message"] = cleaned["initial_message"].apply(default_preprocess)
        t_nlp = time.perf_counter() - start

        vec = IncidentVectorizer()
        matrix, t_vectorize = timeit(vec.fit_transform, cleaned["processed_message"].fillna(""))

        clusterer = IncidentClusterer(n_clusters=8)
        _, t_cluster = timeit(clusterer.fit, matrix)

        print(f"\nSize {size}:")
        print(f"  Cleaning: {t_clean:.3f}s ({t_clean/size*1000:.4f}ms/row)")
        print(f"  NLP preprocessing: {t_nlp:.3f}s ({t_nlp/size*1000:.4f}ms/row)")
        print(f"  TF-IDF vectorization: {t_vectorize:.3f}s ({t_vectorize/size*1000:.4f}ms/row)")
        print(f"  Clustering (k=8): {t_cluster:.3f}s")

        results["sizes"][size] = {
            "clean": t_clean, "nlp": t_nlp, "vectorize": t_vectorize, "cluster": t_cluster,
        }

    # --- Retrieval latency (real, full-scale artifacts, repeated for stability) ---
    print("\n--- Retrieval latency (full 100K dataset, 20 repeated queries) ---")
    df = pd.read_csv(DATA_PROCESSED_DIR / "tickets_clustered.csv")
    vectorizer = IncidentVectorizer.load(DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(DATA_PROCESSED_DIR / "tfidf_matrix.joblib")
    sim_engine = SimilarityEngine(vectorizer, matrix, df)
    retrieval = RetrievalEngine(sim_engine)

    retrieval_times = []
    for _ in range(20):
        _, t = timeit(retrieval.retrieve, QUERY, 5)
        retrieval_times.append(t)
    print(f"  Mean: {statistics.mean(retrieval_times)*1000:.2f}ms, "
          f"Median: {statistics.median(retrieval_times)*1000:.2f}ms, "
          f"Max: {max(retrieval_times)*1000:.2f}ms")
    results["retrieval_ms"] = {
        "mean": statistics.mean(retrieval_times) * 1000,
        "median": statistics.median(retrieval_times) * 1000,
        "max": max(retrieval_times) * 1000,
    }

    # --- Model inference latency (all 3 models, real trained artifacts) ---
    print("\n--- Model inference latency (single prediction, 20 repeats each) ---")
    builder = joblib.load(DATA_PROCESSED_DIR / "feature_builder.joblib")
    processed = default_preprocess(QUERY)
    tfidf_vec = vectorizer.transform([processed])
    row = pd.DataFrame([{"created_at": pd.Timestamp.now().isoformat(), "initial_message": QUERY, **DEFAULT_METADATA}])
    X = builder.transform(row, tfidf_vec)

    results["inference_ms"] = {}
    for name in ["logistic_regression", "decision_tree", "random_forest"]:
        model = joblib.load(DATA_PROCESSED_DIR / "models" / f"{name}.joblib")
        times = []
        for _ in range(20):
            _, t = timeit(model.predict, X)
            times.append(t)
        mean_ms = statistics.mean(times) * 1000
        print(f"  {name}: mean {mean_ms:.3f}ms")
        results["inference_ms"][name] = mean_ms

    # --- Full intelligence engine + report generation (end-to-end) ---
    print("\n--- End-to-end intelligence engine + report generation ---")
    clusterer_full = joblib.load(DATA_PROCESSED_DIR / "clusterer.joblib")
    top_terms = clusterer_full.top_terms_per_cluster(matrix, vectorizer.get_feature_names(), top_n=10)
    rf_model = joblib.load(DATA_PROCESSED_DIR / "models" / "random_forest.joblib")
    engine = IncidentIntelligenceEngine(
        retrieval_engine=retrieval, clusterer=clusterer_full, cluster_top_terms=top_terms,
        classifier_model=rf_model, feature_builder=builder, vectorizer=vectorizer, metadata_df=df,
    )

    result, t_analyze = timeit(engine.analyze, QUERY, None, 5)
    print(f"  Full analyze(): {t_analyze*1000:.2f}ms")

    _, t_report = timeit(generate_html_report, result)
    print(f"  HTML report generation: {t_report*1000:.2f}ms")

    results["analyze_ms"] = t_analyze * 1000
    results["report_ms"] = t_report * 1000

    # --- Write report ---
    lines = [
        "# Performance Report",
        "",
        f"Data loading (100,000 rows, real): **{t_load:.3f}s**",
        "",
        "## Pipeline stages by dataset size (real measurements)",
        "",
        "| Size | Cleaning | NLP preprocessing | TF-IDF vectorization | Clustering (k=8) |",
        "|---|---|---|---|---|",
    ]
    for size, r in results["sizes"].items():
        lines.append(f"| {size:,} | {r['clean']:.3f}s | {r['nlp']:.3f}s | {r['vectorize']:.3f}s | {r['cluster']:.3f}s |")

    lines += [
        "",
        "## Query-time latency (real, full 100K dataset)",
        "",
        f"- Retrieval (20 repeats): mean {results['retrieval_ms']['mean']:.2f}ms, "
        f"median {results['retrieval_ms']['median']:.2f}ms, max {results['retrieval_ms']['max']:.2f}ms",
        "",
        "| Model | Mean inference latency |",
        "|---|---|",
    ]
    for name, ms in results["inference_ms"].items():
        lines.append(f"| {name} | {ms:.3f}ms |")

    lines += [
        "",
        f"- Full `analyze()` (classification + retrieval + cluster + patterns): **{results['analyze_ms']:.2f}ms**",
        f"- HTML report generation: **{results['report_ms']:.2f}ms**",
        "",
        "## Interpretation",
        "_Filled in after reviewing real numbers below — not assumed in advance._",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
