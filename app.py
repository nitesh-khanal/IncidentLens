"""
IncidentLens — Streamlit dashboard (Stage 20).

Sections: Dashboard, Analyze New Incident, Recurring Patterns,
Incident Trends, ML Performance, Reports.
"""
from datetime import datetime, timezone
import json

import joblib
import pandas as pd
import streamlit as st

from src.classifier import PRIMARY_MODEL
from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.analyzer import IncidentIntelligenceEngine, format_intelligence_report, DEFAULT_METADATA
from src.explainability import explain_similarity
from src.report_generator import generate_html_report
from src.nlp_processor import default_preprocess, ensure_nltk_data
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

st.set_page_config(page_title="IncidentLens · Incident intelligence", page_icon="🔍", layout="wide")

# Fixed presentation styles; incident text is always rendered through safe widgets.
st.markdown("""
<style>
.stApp { background: #f7f9fc; }
[data-testid="stSidebar"] { background: #edf2f8; }
[data-testid="stMetric"] {
    background: white; border: 1px solid #dbe4ef; border-radius: 14px;
    padding: 18px; box-shadow: 0 3px 12px rgba(15, 23, 42, .03);
}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) { flex-wrap: wrap; }
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {
    min-width: 160px; flex: 1 1 160px;
}
[data-testid="stMetricValue"] { font-size: 1.8rem; }
[data-testid="stMetricValue"] > div { white-space: normal; overflow-wrap: anywhere; }
[data-testid="stMetricLabel"] p { white-space: normal; }
h1, h2, h3 { color: #16324f; }
.stButton > button { border-radius: 9px; }
[data-testid="stExpander"] { background: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def readable(value):
    return str(value).replace("_", " ").title()


DEMO_EXAMPLES = {
    "Billing · duplicate charge": "I was charged twice for my monthly subscription. Please check the invoice and refund the duplicate charge.",
    "Performance · slow loading": "The application is slow and pages take too long to load.",
    "Account access · unfamiliar phrasing": "I cannot sign in to my account after resetting my password.",
    "Unrecognized text · evidence boundary": "quasar nebula starlight",
}


MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest"]


@st.cache_resource
def load_artifacts():
    ensure_nltk_data()
    df = pd.read_csv(DATA_PROCESSED_DIR / "tickets_clustered.csv")
    vectorizer = IncidentVectorizer.load(DATA_PROCESSED_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(DATA_PROCESSED_DIR / "tfidf_matrix.joblib")
    clusterer = joblib.load(DATA_PROCESSED_DIR / "clusterer.joblib")
    feature_builder = joblib.load(DATA_PROCESSED_DIR / "feature_builder.joblib")
    models = {name: joblib.load(DATA_PROCESSED_DIR / "models" / f"{name}.joblib") for name in MODEL_NAMES}
    if matrix.shape != (len(df), vectorizer.vocabulary_size()):
        raise ValueError("Historical rows, vocabulary, and TF-IDF matrix do not match. Rebuild the demonstration artifacts together.")
    feature_count = len(feature_builder.get_feature_names(vectorizer.get_feature_names()))
    if any(model.n_features_in_ != feature_count for model in models.values()):
        raise ValueError("The classifiers and feature builder do not match. Rebuild the demonstration artifacts together.")
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


try:
    with st.spinner("Preparing the incident dataset and trained models…"):
        artifacts = load_artifacts()
except (OSError, ValueError, LookupError, EOFError) as exc:
    st.title("IncidentLens")
    st.error("The demonstration data or language resources could not be loaded.")
    st.info("Check that the committed data/processed files are available locally and that the bundled data/nltk resources are included in the clone.")
    with st.expander("Startup details"):
        st.code(str(exc), language=None)
    st.stop()
df = artifacts["df"]

st.sidebar.title("🔍 IncidentLens")
st.sidebar.caption("Historical evidence. Clearer incident decisions.")
page = st.sidebar.radio("Navigate", [
    "🏠 Dashboard", "🔎 Analyze New Incident", "🧩 Recurring Patterns",
    "📈 Incident Trends", "🤖 ML Performance", "📄 Reports",
])

st.sidebar.divider()
st.sidebar.caption("DEMONSTRATION WORKFLOW")
st.sidebar.markdown("1. Explore the historical dataset\n2. Analyze an incident\n3. Explain the matching terms\n4. Download the evidence report")
st.sidebar.caption("Synthetic dataset · TF-IDF lexical matching · Decision support")

# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.title("From past incidents to useful evidence")
    st.caption("IncidentLens · IT incident similarity and historical resolution analysis")
    st.info("Describe an incident to find related historical tickets, inspect recorded resolutions, and compare three classification models. Every result distinguishes historical evidence from model output.")

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
        counts = df["issue_type"].value_counts().rename(index=readable)
        st.bar_chart(counts, color="#2675bd")
    with c2:
        st.subheader("Volume by priority")
        priority_counts = df["priority"].value_counts().rename(index=readable)
        st.bar_chart(priority_counts, color="#159a91")

    with st.expander("What this demonstration can and cannot establish", expanded=True):
        st.markdown("**100,000 synthetic tickets contain only 96 unique descriptions.** Similarity measures shared words; it is not a probability of a shared cause. The template-holdout evaluation is more informative than the perfect row-split accuracy. Missing resolutions are shown explicitly.")
        st.caption("Start with Analyze New Incident, choose an example, and inspect the first historical match.")

# ============================================================
# ANALYZE NEW INCIDENT
# ============================================================
elif page == "🔎 Analyze New Incident":
    st.title("Analyze a New Incident")
    st.caption(
        "Results are historical evidence and model output — not a confirmed root cause. "
        "See each section's label for what kind of evidence it is."
    )

    if "pending_example" in st.session_state:
        st.session_state["incident_description"] = st.session_state.pop("pending_example")
    example = st.selectbox("Demonstration example", list(DEMO_EXAMPLES))
    query = st.text_area("Incident description", key="incident_description", height=120,
                          placeholder="e.g. Payment API is returning 502 errors and database connections are timing out after a deployment.")

    provide_metadata = st.checkbox("Add ticket context", help="Uses the supplied context in classification. It does not guarantee a more accurate prediction.")
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

    action, example_action = st.columns([1, 3])
    analyze_clicked = action.button("Analyze Incident", type="primary")
    if example_action.button("Use selected example"):
        # Apply on the next run before the text area is instantiated.
        st.session_state["pending_example"] = DEMO_EXAMPLES[example]
        st.rerun()

    if analyze_clicked:
        if not query.strip():
            st.warning("Enter an incident description or load a demonstration example first.")
        else:
            with st.spinner("Finding historical evidence and comparing the trained models…"):
                result = artifacts["engine"].analyze(query, metadata=metadata, top_n=5)
                processed = default_preprocess(query)
                comparison = {}
                if processed:
                    meta_for_row = dict(DEFAULT_METADATA)
                    if metadata:
                        meta_for_row.update(metadata)
                    row = pd.DataFrame([{"created_at": datetime.now(timezone.utc).isoformat(),
                                          "initial_message": query, **meta_for_row}])
                    tfidf_vec = artifacts["vectorizer"].transform([processed])
                    X = artifacts["feature_builder"].transform(row, tfidf_vec)
                    comparison = {name: model.predict(X)[0] for name, model in artifacts["models"].items()}
                st.session_state["last_comparison"] = comparison
                st.session_state["last_result"] = result
                st.session_state["last_metadata"] = metadata
                st.session_state["last_report"] = format_intelligence_report(result)
                st.session_state["last_html_report"] = generate_html_report(result)
                st.session_state["last_query"] = query

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        analyzed_query = result["query"]
        analyzed_metadata = st.session_state["last_metadata"]
        st.divider()
        st.caption(f"Showing saved analysis: {analyzed_query}")
        if query != analyzed_query or metadata != analyzed_metadata:
            st.info("The draft has changed. Select Analyze Incident to update the results below.")
        summary = st.columns(3)
        category = result["classification"]
        summary[0].metric("Suggested category", readable(category["predicted_issue_type"]) if category and category.get("supported", True) else "Insufficient evidence")
        summary[1].metric("Historical matches", len(result["similar_incidents"]))
        summary[2].metric("Matched vocabulary terms", category["matched_vocabulary_terms"] if category else 0)

        # --- Classification: primary model + comparison across all 3 ---
        st.subheader("Classification (model output)")
        if result["classification"]:
            if result["classification"].get("supported", True):
                st.info(f"**Suggested issue type:** {readable(result['classification']['predicted_issue_type'])}")
            else:
                st.warning("Insufficient evidence to suggest an issue type. Add specific symptoms and ticket context; the labels below are diagnostic model candidates.")
            st.caption(result["classification"]["note"])

            comparison = st.session_state["last_comparison"]
            if comparison:
                matched_terms = result["classification"]["matched_vocabulary_terms"]
                st.write("**Comparison across all 3 trained models:**")
                if matched_terms == 0:
                    st.warning(
                        "None of this text's words appear in the training vocabulary "
                        f"({artifacts['vectorizer'].vocabulary_size()} terms). All three predictions below are driven almost "
                        "entirely by supplied metadata or defaults, not the incident text — treat "
                        "them as unreliable."
                    )
                elif matched_terms <= 2:
                    st.caption(
                        f"Only {matched_terms} word(s) in this text matched the training "
                        "vocabulary — treat these predictions as weakly supported."
                    )
                st.dataframe(pd.DataFrame([
                    {"Model": readable(name), "Model candidate": readable(prediction)}
                    for name, prediction in comparison.items()
                ]), hide_index=True, width="stretch")
                if len(set(comparison.values())) > 1:
                    st.warning("The models disagree on this description. Review historical evidence before deciding how to route the incident.")
        else:
            st.warning("No classification available for this input.")

        # --- Similar historical incidents ---
        st.subheader("Similar Historical Incidents (historical evidence)")
        if result["similar_incidents"]:
            for i, r in enumerate(result["similar_incidents"]):
                shared = r.get("shared_terms", 0)
                with st.expander(f"#{i+1} {r['ticket_id']} — {readable(r['issue_type'])} · lexical similarity {r['similarity']*100:.0f}%", expanded=i == 0):
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
                        processed = default_preprocess(analyzed_query)
                        if processed:
                            candidate_row = df[df["ticket_id"] == r["ticket_id"]].iloc[0]
                            candidate_text = candidate_row["processed_message"]
                            q_vec = artifacts["vectorizer"].transform([processed])
                            c_vec = artifacts["vectorizer"].transform([str(candidate_text)])
                            explanation = explain_similarity(q_vec, c_vec, artifacts["vectorizer"].get_feature_names())
                            st.caption("**Why similar?** (real, computed term contributions)")
                            if explanation["contributing_terms"]:
                                terms = pd.DataFrame(explanation["contributing_terms"]).rename(columns={
                                    "term": "Shared word", "query_weight": "Query weight",
                                    "candidate_weight": "Historical weight", "contribution": "Score contribution",
                                    "pct_of_total": "Share of similarity (%)",
                                }).round(3)
                                st.dataframe(terms, hide_index=True)
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

        else:
            st.info("No cluster assigned: this description has no usable overlap with the training vocabulary.")

        # --- Recurring patterns ---
        st.subheader("Recurring Patterns (observed historical data)")
        rp = result["recurring_patterns"]
        if rp:
            c1, c2, c3 = st.columns(3)
            c1.metric("Category frequency", f"{rp['issue_type_frequency_pct']}%")
            c2.metric("Historical incidents", f"{rp['historical_incident_count']:,}")
            c3.metric("Avg resolution time",
                      f"{rp['avg_resolution_time_hours']}h" if rp["avg_resolution_time_hours"] is not None else "N/A")

        elif category and not category.get("supported", True):
            st.info("Category statistics are withheld because the category suggestion has insufficient evidence.")

        st.caption("A similarity percentage measures lexical overlap; it is not confidence in a diagnosis. Download this analysis from Reports.")

# ============================================================
# RECURRING PATTERNS (cluster overview)
# ============================================================
elif page == "🧩 Recurring Patterns":
    st.title("Recurring Patterns (Clustering)")
    st.caption("K-Means clusters group incidents by language similarity — NOT a claim of shared root cause.")

    cluster_sizes = df["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_sizes.rename(index=lambda value: f"Cluster {value}"), color="#2675bd")
    st.caption("Large mixed clusters reflect the limited, templated dataset. Cluster numbers are regenerated after normalization and do not identify confirmed causes.")
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
    st.caption("Observed dataset patterns · synthetic ticket trends and a separate incident workflow event log")
    st.caption("The saved cluster-distribution trend chart is from the original clustering run; use Recurring Patterns for the regenerated clusters.")
    st.info("The synthetic ticket dataset shows no meaningful time trends. Workflow charts use a separate dataset; the two datasets are never joined.")
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    pngs = sorted(figures_dir.glob("*.png"))
    if pngs:
        for offset in range(0, len(pngs), 2):
            columns = st.columns(2)
            for column, png in zip(columns, pngs[offset:offset + 2]):
                with column:
                    st.image(str(png), caption=readable(png.stem.split("_", 1)[-1]))
    else:
        st.info("Trend charts are not available in this copy. The analysis and reporting workflow remains available.")

# ============================================================
# ML PERFORMANCE
# ============================================================
elif page == "🤖 ML Performance":
    st.title("How well do the models generalize?")
    st.caption("Evaluation context matters more than a perfect headline score.")
    validation_path = PROJECT_ROOT / "reports" / "validated_evaluation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        row_metrics = validation["row_split"]["models"][PRIMARY_MODEL]
        holdout_metrics = validation["template_holdout"]["models"][PRIMARY_MODEL]
        grouped_metrics = validation["grouped_summary"][PRIMARY_MODEL]
        c1, c2, c3 = st.columns(3)
        c1.metric("Row-split accuracy", f"{row_metrics['accuracy']:.2%}", help="Repeated text templates can appear in both training and test rows.")
        c2.metric("Unseen-template accuracy", f"{holdout_metrics['accuracy']:.2%}", help="TF-IDF and categorical encoders are fitted on training rows only.")
        c3.metric("Grouped mean accuracy", f"{grouped_metrics['mean_accuracy']:.2%}", help="Three folds with no normalized template overlap; every fold refits preprocessing.")
        st.info(f"Primary model: {readable(PRIMARY_MODEL)}, selected using grouped mean macro F1. The challenge examples were excluded from fitting and model selection.")
        st.warning("The dataset is synthetic and contains only 96 unique descriptions. These results measure unseen templates, not performance on real-world incidents. Some small categories have only three templates.")
        model_rows = [{"Model": readable(name), "Grouped mean accuracy": metrics["mean_accuracy"],
                       "Accuracy standard deviation": metrics["std_accuracy"], "Grouped mean macro F1": metrics["mean_macro_f1"]}
                      for name, metrics in validation["grouped_summary"].items()]
        st.dataframe(pd.DataFrame(model_rows).round(4), hide_index=True)
    else:
        st.info("Train-only validation results have not been generated for this copy.")

    challenge_path = PROJECT_ROOT / "reports" / "challenge_results.json"
    if challenge_path.exists():
        challenge = json.loads(challenge_path.read_text())
        with st.expander("Exploratory challenge set · coverage and evidence boundaries"):
            st.caption("48 assistant-authored labelled examples and 6 boundary inputs. Labels are not independently human-reviewed. This is exploratory validation, not a real-world benchmark.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Suggestion coverage", f"{challenge['coverage']:.1%}")
            c2.metric("Correct among suggestions", f"{challenge['accepted_accuracy']:.1%}" if challenge['accepted_accuracy'] is not None else "N/A")
            c3.metric("Boundary inputs withheld", f"{challenge['boundary_abstentions']}/{challenge['boundary_cases']}")
            st.write(f"Raw classifier accuracy on the labelled examples: {challenge['raw_classifier_accuracy']:.1%}. Withheld inputs are not counted as correct predictions.")
            st.caption("A suggestion needs at least three matching vocabulary terms and a model score of 0.5. This conservative gate is a heuristic; its score is not calibrated confidence.")
            with st.expander("Inspect every challenge case"):
                st.dataframe(pd.DataFrame([{
                    "Description": case["query"], "Intended category": case["expected_issue_type"] or "Boundary input",
                    "Model candidate": case["predicted"] or "None", "Suggested": case["accepted"],
                    "Matched terms": case["matched_terms"],
                } for case in challenge["cases"]]), hide_index=True)

    eval_report = PROJECT_ROOT / "reports" / "model_evaluation.md"
    if eval_report.exists():
        with st.expander("Full evaluation report"):
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
        st.subheader("Report preview")
        st.code(st.session_state["last_report"], language=None)
    else:
        st.info("No analysis run yet — go to 'Analyze New Incident' first.")
