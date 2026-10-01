"""
IncidentLens — Streamlit dashboard (Stage 20).

Sections: Dashboard, Analyze New Incident, Recurring Patterns,
Incident Trends, ML Performance, Reports.
"""
import joblib
import pandas as pd
import streamlit as st

from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.analyzer import IncidentIntelligenceEngine, format_intelligence_report, DEFAULT_METADATA
from src.explainability import explain_similarity, explain_classification
from src.report_generator import generate_html_report
from src.nlp_processor import default_preprocess, ensure_nltk_data
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

st.set_page_config(page_title="IncidentLens", page_icon="🔍", layout="wide")

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest"]
PRIMARY_MODEL = "random_forest"


@st.cache_resource
def load_artifacts():
    ensure_nltk_data()
    df = pd.read_csv(DATA_PROCESSED_DIR / "tickets_clustered.csv")
    vectorizer = IncidentVectorizer.load(DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(DATA_PROCESSED_DIR / "tfidf_matrix.joblib")
    clusterer = joblib.load(DATA_PROCESSED_DIR / "clusterer.joblib")
    feature_builder = joblib.load(DATA_PROCESSED_DIR / "feature_builder.joblib")
    models = {name: joblib.load(DATA_PROCESSED_DIR / "models" / f"{name}.joblib") for name in MODEL_NAMES}
    top_terms = clusterer.top_terms_per_cluster(matrix, vectorizer.get_feature_names(), top_n=10)

    sim_engine = SimilarityEngine(vectorizer, matrix, df)
    retrieval = RetrievalEngine(sim_engine)
    engine = IncidentIntelligenceEngine(
        retrieval_engine=retrieval, clusterer=clusterer, cluster_top_terms=top_terms,
        classifier_model=models[PRIMARY_MODEL], feature_builder=feature_builder,
        vectorizer=vectorizer, metadata_df=df,
    )
    return {
        "df": df, "vectorizer": vectorizer, "matrix": matrix, "clusterer": clusterer,
        "feature_builder": feature_builder, "models": models, "top_terms": top_terms,
        "engine": engine, "retrieval": retrieval,
    }


artifacts = load_artifacts()
df = artifacts["df"]

st.sidebar.title("🔍 IncidentLens")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard", "🔎 Analyze New Incident", "🧩 Recurring Patterns",
    "📈 Incident Trends", "🤖 ML Performance", "📄 Reports",
])

# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.title("IncidentLens Dashboard")
    st.caption("An analytical decision-support tool — not a definitive root-cause diagnosis system.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total incidents", f"{len(df):,}")
    col2.metric("Resolution coverage", f"{df['has_resolution'].mean()*100:.1f}%")
    col3.metric("Issue types", df["issue_type"].nunique())
    rated = df[df["csat_score"] > 0]
    unrated_pct = (df["csat_score"] == 0).mean() * 100
    col4.metric(
        "Avg CSAT (scores 1-5)",
        f"{rated['csat_score'].mean():.2f} / 5",
        help=(
            f"Excludes the {unrated_pct:.1f}% of tickets with a score of 0. The dataset "
            "does not document 0; it is treated as 'no rating' because the zero rate is "
            "about 30% in every ticket status, including resolved."
        ),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Volume by issue type")
        st.bar_chart(df["issue_type"].value_counts())
    with c2:
        st.subheader("Volume by priority")
        st.bar_chart(df["priority"].value_counts())

# ============================================================
# ANALYZE NEW INCIDENT
# ============================================================
elif page == "🔎 Analyze New Incident":
    st.title("Analyze a New Incident")
    st.caption(
        "Results are historical evidence and model output — not a confirmed root cause. "
        "See each section's label for what kind of evidence it is."
    )

    query = st.text_area("Incident description", height=100,
                          placeholder="e.g. Payment API is returning 502 errors and database connections are timing out after a deployment.")

    provide_metadata = st.checkbox("Provide additional metadata (improves classification accuracy)")
    metadata = None
    if provide_metadata:
        metadata = {}
        c1, c2, c3, c4 = st.columns(4)
        metadata["customer_segment"] = c1.selectbox("Customer segment", sorted(df["customer_segment"].dropna().unique()))
        metadata["channel"] = c2.selectbox("Channel", sorted(df["channel"].dropna().unique()))
        metadata["product_area"] = c3.selectbox("Product area", sorted(df["product_area"].dropna().unique()))
        metadata["priority"] = c4.selectbox("Priority", sorted(df["priority"].dropna().unique()))
        c5, c6, c7, c8 = st.columns(4)
        metadata["sla_plan"] = c5.selectbox("SLA plan", sorted(df["sla_plan"].dropna().unique()))
        metadata["platform"] = c6.selectbox("Platform", sorted(df["platform"].dropna().unique()))
        metadata["region"] = c7.selectbox("Region", sorted(df["region"].dropna().unique()))
        metadata["has_attachment"] = int(c8.checkbox("Has attachment"))

    if st.button("Analyze Incident", type="primary") and query.strip():
        result = artifacts["engine"].analyze(query, metadata=metadata, top_n=5)

        st.divider()

        # --- Classification: primary model + comparison across all 3 ---
        st.subheader("Classification (model output)")
        if result["classification"]:
            st.info(f"**Predicted issue type:** {result['classification']['predicted_issue_type']}")
            st.caption(result["classification"]["note"])

            processed = default_preprocess(query)
            if processed:
                meta_for_row = dict(DEFAULT_METADATA)
                if metadata:
                    meta_for_row.update(metadata)
                row = pd.DataFrame([{"created_at": pd.Timestamp.now().isoformat(),
                                      "initial_message": query, **meta_for_row}])
                tfidf_vec = artifacts["vectorizer"].transform([processed])
                X = artifacts["feature_builder"].transform(row, tfidf_vec)
                comparison = {name: model.predict(X)[0] for name, model in artifacts["models"].items()}

                matched_terms = int(tfidf_vec.nnz)
                st.write("**Comparison across all 3 trained models:**")
                if matched_terms == 0:
                    st.warning(
                        "None of this text's words appear in the training vocabulary "
                        "(99 terms). All three predictions below are driven almost "
                        "entirely by metadata defaults, not the incident text — treat "
                        "them as unreliable."
                    )
                elif matched_terms <= 2:
                    st.caption(
                        f"Only {matched_terms} word(s) in this text matched the training "
                        "vocabulary — treat these predictions as weakly supported."
                    )
                st.table(pd.DataFrame([comparison]))
        else:
            st.warning("No classification available for this input.")

        # --- Similar historical incidents ---
        st.subheader("Similar Historical Incidents (historical evidence)")
        if result["similar_incidents"]:
            for i, r in enumerate(result["similar_incidents"]):
                shared = r.get("shared_terms", 0)
                with st.expander(f"#{i+1} {r['ticket_id']} — {r['issue_type']} (similarity {r['similarity']*100:.0f}%)"):
                    st.write(f"**Description:** {r['initial_message']}")
                    if shared <= 2:
                        st.caption(
                            f"⚠️ Based on only {shared} shared word(s) — this match may be "
                            "weakly supported despite the similarity score shown above."
                        )
                    if r["has_resolution"]:
                        st.success(f"**Historical resolution:** {r['resolution_summary']}")
                    else:
                        st.warning("No resolution recorded for this historical incident.")
                    if i == 0:
                        processed = default_preprocess(query)
                        if processed:
                            candidate_row = df[df["ticket_id"] == r["ticket_id"]].iloc[0]
                            candidate_text = candidate_row["processed_message"]
                            q_vec = artifacts["vectorizer"].transform([processed])
                            c_vec = artifacts["vectorizer"].transform([str(candidate_text)])
                            explanation = explain_similarity(q_vec, c_vec, artifacts["vectorizer"].get_feature_names())
                            st.caption("**Why similar?** (real, computed term contributions)")
                            if explanation["contributing_terms"]:
                                st.dataframe(pd.DataFrame(explanation["contributing_terms"]), hide_index=True)
                            else:
                                st.caption("No shared vocabulary contributed to this score.")
        else:
            st.info("No historically similar incidents found for this description.")

        # --- Cluster membership ---
        st.subheader("Cluster Membership (unsupervised inference)")
        if result["cluster"]:
            st.write(f"**Cluster #{result['cluster']['cluster_id']}** — characterized by: "
                     f"{', '.join(result['cluster']['top_terms'][:8])}")
            st.caption(result["cluster"]["note"])

        # --- Recurring patterns ---
        st.subheader("Recurring Patterns (observed historical data)")
        rp = result["recurring_patterns"]
        if rp:
            c1, c2, c3 = st.columns(3)
            c1.metric("Category frequency", f"{rp['issue_type_frequency_pct']}%")
            c2.metric("Historical incidents", f"{rp['historical_incident_count']:,}")
            c3.metric("Avg resolution time",
                      f"{rp['avg_resolution_time_hours']}h" if rp["avg_resolution_time_hours"] else "N/A")

        st.session_state["last_report"] = format_intelligence_report(result)
        st.session_state["last_html_report"] = generate_html_report(result)
        st.session_state["last_query"] = query

# ============================================================
# RECURRING PATTERNS (cluster overview)
# ============================================================
elif page == "🧩 Recurring Patterns":
    st.title("Recurring Patterns (Clustering)")
    st.caption("K-Means clusters group incidents by language similarity — NOT a claim of shared root cause.")

    cluster_sizes = df["cluster"].value_counts().sort_index()
    for cluster_id, size in cluster_sizes.items():
        terms = artifacts["top_terms"].get(cluster_id, [])
        with st.expander(f"Cluster #{cluster_id} — {size:,} incidents — {', '.join(terms[:6])}"):
            st.write(f"**Top terms:** {', '.join(terms)}")
            st.write("**issue_type breakdown in this cluster:**")
            st.bar_chart(df[df["cluster"] == cluster_id]["issue_type"].value_counts())

# ============================================================
# INCIDENT TRENDS
# ============================================================
elif page == "📈 Incident Trends":
    st.title("Incident Trends")
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    if figures_dir.exists():
        pngs = sorted(figures_dir.glob("*.png"))
        for png in pngs:
            st.image(str(png), caption=png.stem)
    else:
        st.warning("No trend figures found — run scripts/run_trend_analysis.py first.")

# ============================================================
# ML PERFORMANCE
# ============================================================
elif page == "🤖 ML Performance":
    st.title("ML Model Performance")
    eval_report = PROJECT_ROOT / "reports" / "model_evaluation.md"
    if eval_report.exists():
        st.markdown(eval_report.read_text())
    else:
        st.warning("No evaluation report found — run scripts/evaluate_models.py first.")

# ============================================================
# REPORTS
# ============================================================
elif page == "📄 Reports":
    st.title("Analysis Reports")
    if "last_report" in st.session_state:
        st.caption(f"Last analyzed: {st.session_state['last_query']}")
        st.download_button("Download full report (.html, with chart)",
                            st.session_state["last_html_report"],
                            file_name="incident_report.html", mime="text/html")
        st.download_button("Download plain-text report (.txt)", st.session_state["last_report"],
                            file_name="incident_report.txt")
        st.code(st.session_state["last_report"], language=None)
    else:
        st.info("No analysis run yet — go to 'Analyze New Incident' first.")
