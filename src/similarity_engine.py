"""
IncidentLens — similarity engine.

Given a new incident description, retrieves the top-N most similar
historical incidents using TF-IDF + cosine similarity. Purely lexical
(see docs/tfidf_representation.md) — does not claim semantic understanding
or a definitive root cause; results are historical evidence, not proof.
"""
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp_processor import default_preprocess, ensure_nltk_data


class SimilarityEngine:
    """Retrieves similar historical incidents via TF-IDF cosine similarity."""

    def __init__(self, vectorizer, matrix, metadata_df):
        """
        vectorizer: fitted IncidentVectorizer
        matrix: TF-IDF matrix, row-aligned with metadata_df
        metadata_df: DataFrame with ticket_id, initial_message, issue_type,
                     resolution_summary, has_resolution (at minimum)
        """
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.metadata_df = metadata_df.reset_index(drop=True)
        ensure_nltk_data()

    def find_similar(self, query_text: str, top_n: int = 5):
        """Return up to top_n similar incidents, ranked by cosine similarity,
        excluding zero-similarity matches. Empty/whitespace query -> []."""
        processed = default_preprocess(query_text)
        if not processed:
            return []

        query_vector = self.vectorizer.transform([processed])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        top_indices = scores.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            row = self.metadata_df.iloc[idx]
            results.append({
                "ticket_id": row["ticket_id"],
                "similarity": float(scores[idx]),
                "issue_type": row["issue_type"],
                "initial_message": row["initial_message"],
                "resolution_summary": row["resolution_summary"] if row["has_resolution"] else None,
                "has_resolution": bool(row["has_resolution"]),
            })
        return results
