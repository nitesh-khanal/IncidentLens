"""
IncidentLens — explainability (Stage 19).

Answers "why similar?" and "why this category?" with real, computed
evidence — never an approximation presented as certainty, and never a
claim of semantic understanding the underlying models don't have.
"""
import numpy as np


def explain_similarity(query_vector, candidate_vector, feature_names, top_k=10):
    """
    Decompose cosine similarity between two TF-IDF vectors into real,
    per-term contributions. Returns the top_k terms by contribution,
    each with its query weight, candidate weight, and share of the
    total similarity score. Purely lexical — shared terms, not meaning.
    """
    q = query_vector.toarray().flatten()
    c = candidate_vector.toarray().flatten()

    q_norm = np.linalg.norm(q)
    c_norm = np.linalg.norm(c)
    if q_norm == 0 or c_norm == 0:
        return {"total_similarity": 0.0, "contributing_terms": []}

    # Per-term contribution to the cosine similarity numerator, normalized
    # by both vector norms so contributions sum to the actual cosine score.
    term_contributions = (q * c) / (q_norm * c_norm)
    total_similarity = float(term_contributions.sum())

    nonzero_idx = np.where(term_contributions > 0)[0]
    sorted_idx = nonzero_idx[np.argsort(-term_contributions[nonzero_idx])][:top_k]

    contributing_terms = [
        {
            "term": feature_names[i],
            "query_weight": float(q[i]),
            "candidate_weight": float(c[i]),
            "contribution": float(term_contributions[i]),
            "pct_of_total": float(term_contributions[i] / total_similarity * 100) if total_similarity > 0 else 0.0,
        }
        for i in sorted_idx
    ]

    return {"total_similarity": total_similarity, "contributing_terms": contributing_terms}


def explain_classification(model, feature_names, predicted_class, top_k=15):
    """
    Real feature-importance explanation, mechanism stated explicitly:
    - Logistic Regression: per-class coefficients (class-specific, signed)
    - Decision Tree / Random Forest: global feature_importances_ (NOT
      class-specific — stated explicitly, not glossed over as if it were)
    """
    if hasattr(model, "coef_"):
        classes = list(model.classes_)
        if predicted_class not in classes:
            return {"mechanism": "logistic_regression_coefficients", "supported": False,
                     "reason": f"'{predicted_class}' not in model.classes_"}
        class_idx = classes.index(predicted_class)
        coefs = model.coef_[class_idx]
        top_idx = np.argsort(-coefs)[:top_k]
        return {
            "mechanism": "logistic_regression_coefficients",
            "supported": True,
            "class_specific": True,
            "note": f"Top features pushing the prediction TOWARD '{predicted_class}' specifically.",
            "top_features": [
                {"feature": feature_names[i], "weight": float(coefs[i])} for i in top_idx
            ],
        }

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(-importances)[:top_k]
        return {
            "mechanism": "tree_feature_importances",
            "supported": True,
            "class_specific": False,
            "note": (
                "GLOBAL feature importances across ALL classes, NOT specific "
                f"to the predicted class '{predicted_class}'. Decision Tree/"
                "Random Forest do not natively expose per-class, per-prediction "
                "importance the way Logistic Regression's coefficients do — "
                "stated explicitly rather than implied otherwise."
            ),
            "top_features": [
                {"feature": feature_names[i], "importance": float(importances[i])} for i in top_idx
            ],
        }

    return {"mechanism": "unknown", "supported": False, "reason": "Model type not recognized."}


def explain_shared_metadata(query_metadata: dict, candidate_row) -> list:
    """Real, simple evidence: which metadata fields match between query and candidate."""
    shared = []
    for field in ["customer_segment", "channel", "product_area", "priority", "region"]:
        if field in query_metadata and field in candidate_row.index:
            if str(query_metadata[field]) == str(candidate_row[field]):
                shared.append(field)
    return shared
