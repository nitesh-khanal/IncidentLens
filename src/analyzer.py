"""
IncidentLens — Incident Intelligence Engine (Stage 18).

Integrates classification, similarity retrieval, and clustering into one
structured view per incident. Every section is explicitly labeled by its
kind of evidence — Observed data, Model output, Historical evidence, or
Inference — per the project's evidential language requirement. Never
claims a definitive root cause.
"""
from datetime import datetime, timezone

import pandas as pd

from src.nlp_processor import default_preprocess, ensure_nltk_data

DEFAULT_METADATA = {
    "customer_segment": "unknown",
    "channel": "unknown",
    "product_area": "unknown",
    "priority": "unknown",
    "sla_plan": "unknown",
    "platform": "unknown",
    "region": "unknown",
    "has_attachment": 0,
}


class IncidentIntelligenceEngine:
    """Combines retrieval, clustering, and classification into one report."""

    def __init__(self, retrieval_engine, clusterer, cluster_top_terms,
                 classifier_model, feature_builder, vectorizer, metadata_df):
        self.retrieval = retrieval_engine
        self.clusterer = clusterer
        self.cluster_top_terms = cluster_top_terms  # {cluster_id: [terms]}
        self.classifier = classifier_model
        self.feature_builder = feature_builder
        self.vectorizer = vectorizer
        self.metadata_df = metadata_df
        ensure_nltk_data()

    def analyze(self, text: str, metadata: dict = None, top_n: int = 5) -> dict:
        metadata_provided = metadata is not None
        meta = dict(DEFAULT_METADATA)
        if metadata:
            meta.update(metadata)

        similar = self.retrieval.retrieve(text, top_n=top_n)

        processed = default_preprocess(text)
        classification = None
        cluster_info = None

        if processed:
            tfidf_vec = self.vectorizer.transform([processed])
            matched_terms = int(tfidf_vec.nnz)

            row = pd.DataFrame([{
                "created_at": datetime.now(timezone.utc).isoformat(),
                "initial_message": text,
                **meta,
            }])
            X = self.feature_builder.transform(row, tfidf_vec)
            predicted = self.classifier.predict(X)[0]

            confidence = None
            margin = 0.0
            if hasattr(self.classifier, "predict_proba"):
                proba = self.classifier.predict_proba(X)[0]
                confidence = float(proba.max())
                ranked = sorted(proba)
                margin = float(ranked[-1] - ranked[-2]) if len(ranked) > 1 else 0.0

            note = (
                "Model output, not a confirmed category. Metadata was "
                + ("provided by the caller." if metadata_provided else
                   "NOT provided — defaults used, which may reduce accuracy.")
            )
            if predicted == "account_access":
                note += (
                    " Known limitation: this category has only 3 historical "
                    "text templates in the training data (see "
                    "docs/ml_problem_definition.md) — confidence is lower for "
                    "phrasing unlike those templates."
                )
            if matched_terms == 0:
                note += (
                    " WARNING: none of this text's words appear in the "
                    f"training vocabulary ({self.vectorizer.vocabulary_size()} terms). This prediction is "
                    "driven almost entirely by supplied metadata or defaults, not the "
                    "incident text — treat it as unreliable."
                )
            elif matched_terms <= 2:
                note += (
                    f" Only {matched_terms} word(s) in this text matched the "
                    "training vocabulary — treat this prediction as weakly "
                    "supported, even though the model reports a definite "
                    "category."
                )
            elif confidence is not None and confidence < 0.3:
                note += (
                    f" Low model score ({confidence*100:.0f}%, vs. a "
                    f"{100/len(self.classifier.classes_):.0f}% random "
                    f"baseline across {len(self.classifier.classes_)} categories) — the text gave the "
                    "model little to distinguish this category from others."
                )

            # This is a conservative evidence gate, not calibrated certainty.
            # Keep raw candidates for diagnostics, but do not endorse weak output.
            supported = matched_terms >= 3 and confidence is not None and confidence >= 0.5
            # A short phrase needs stronger model AND historical agreement.
            # No added synonym can turn a one-term keyword into support here.
            short_supported = (
                matched_terms == 2 and confidence is not None and confidence >= 0.8
                and margin >= 0.5 and bool(similar)
                and similar[0]["issue_type"] == predicted
                and similar[0].get("shared_terms", 0) >= 2
                and similar[0]["similarity"] >= 0.6
            )
            supported = supported or short_supported
            if short_supported:
                note += " Tentative short-description suggestion: a high model score and a historical match agree. Add incident details before acting; this gate is heuristic, not calibrated certainty."
            if not supported:
                note += " Insufficient evidence: provide a more specific description. The candidate label is diagnostic only; the standard 3-term/0.5-score gate and stricter two-term agreement gate are heuristics, not calibrated guarantees."
            classification = {
                "supported": supported,
                "evidence_status": ("short_description_supported" if short_supported else
                                    "candidate_supported" if supported else "insufficient_evidence"),
                "predicted_issue_type": predicted,
                "metadata_provided": metadata_provided,
                "matched_vocabulary_terms": matched_terms,
                "confidence": confidence,
                "note": note,
            }

            # A zero vector has no lexical evidence for any cluster.
            if matched_terms > 0:
                cluster_id = int(self.clusterer.predict(tfidf_vec)[0])
                cluster_info = {
                    "cluster_id": cluster_id,
                    "top_terms": self.cluster_top_terms.get(cluster_id, []),
                    "note": (
                        "Unsupervised grouping by language similarity — NOT a "
                        "claim about a shared root cause (see docs/clustering.md)."
                    ),
                }

        recurring = None
        if classification and classification["supported"]:
            same_type = self.metadata_df[
                self.metadata_df["issue_type"] == classification["predicted_issue_type"]
            ]
            freq_pct = len(same_type) / len(self.metadata_df) * 100
            resolved = same_type[same_type["has_resolution"] == True]
            avg_res_time = resolved["resolution_time_hours"].mean() if len(resolved) else None
            recurring = {
                "issue_type_frequency_pct": round(freq_pct, 2),
                "historical_incident_count": int(len(same_type)),
                "avg_resolution_time_hours": (
                    round(float(avg_res_time), 2) if pd.notna(avg_res_time) else None
                ),
                "note": "Observed data from the historical dataset for the predicted category.",
            }

        return {
            "query": text,
            "classification": classification,
            "similar_incidents": similar,
            "cluster": cluster_info,
            "recurring_patterns": recurring,
        }


def format_intelligence_report(result: dict) -> str:
    lines = ["NEW INCIDENT", "", result["query"].strip(), ""]

    lines += ["=" * 50, "CLASSIFICATION (model output)", "=" * 50]
    c = result["classification"]
    if c:
        if c.get("supported", True):
            lines.append(f"Suggested issue type: {c['predicted_issue_type']}")
        else:
            lines.append("Insufficient evidence for a category suggestion.")
            lines.append(f"Diagnostic model candidate: {c['predicted_issue_type']}")
        lines.append(f"Note: {c['note']}")
    else:
        lines.append("No classification available (empty/unusable query text).")
    lines.append("")

    lines += ["=" * 50, "SIMILAR HISTORICAL INCIDENTS (historical evidence)", "=" * 50]
    similar = result["similar_incidents"]
    if not similar:
        lines.append("No historically similar incidents found.")
    else:
        for r in similar:
            lines.append(f"- {r['ticket_id']} (similarity {r['similarity']*100:.0f}%) — {r['issue_type']}")
            lines.append(f"    Description: {r['initial_message']}")
            lines.append(f"    Same wording in {r.get('description_occurrences', 1):,} tickets; one representative shown.")
            if r["has_resolution"]:
                lines.append(f"    Historical resolution: {r['resolution_summary']}")
            else:
                lines.append("    No resolution recorded.")
    lines.append("")

    lines += ["=" * 50, "CLUSTER MEMBERSHIP (unsupervised inference)", "=" * 50]
    cl = result["cluster"]
    if cl:
        lines.append(f"Cluster #{cl['cluster_id']} — characterized by: {', '.join(cl['top_terms'][:8])}")
        lines.append(f"Note: {cl['note']}")
    else:
        lines.append("No cluster assignment available.")
    lines.append("")

    lines += ["=" * 50, "RECURRING PATTERNS (observed historical data)", "=" * 50]
    rp = result["recurring_patterns"]
    if rp:
        lines.append(f"This predicted category represents {rp['issue_type_frequency_pct']}% "
                     f"of the historical dataset ({rp['historical_incident_count']:,} incidents in this category).")
        if rp["avg_resolution_time_hours"] is not None:
            lines.append(f"Average historical resolution time: {rp['avg_resolution_time_hours']}h")
        else:
            lines.append("No resolved historical incidents in this category to compute average time.")
    else:
        lines.append("No recurring pattern data available.")

    return "\n".join(lines)
