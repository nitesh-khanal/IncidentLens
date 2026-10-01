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
        # Repeated synthetic templates must not crowd distinct descriptions out.
        keys = self.metadata_df["initial_message"].fillna("").astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
        self.description_counts = keys.value_counts()
        self.description_keys = keys
        candidates = self.metadata_df.assign(_description=keys).sort_values(
            "has_resolution", ascending=False, kind="stable")
        self.representative_indices = candidates.drop_duplicates("_description").index.to_numpy()


    def find_similar(self, query_text: str, top_n: int = 5):
        """Return up to top_n similar incidents, ranked by cosine similarity,
        excluding zero-similarity matches and repeated descriptions.
        Prefer a representative with a recorded resolution. Empty query -> []."""
        if top_n <= 0:
            return []
        processed = default_preprocess(query_text)
        if not processed:
            return []

        query_vector = self.vectorizer.transform([processed])
        scores = cosine_similarity(query_vector, self.matrix[self.representative_indices])[0]
        top_positions = (-scores).argsort(kind="stable")[:top_n]

        results = []
        for position in top_positions:
            idx = self.representative_indices[position]
            if scores[position] <= 0:
                continue
            row = self.metadata_df.iloc[idx]
            # Count terms where BOTH the query and this candidate have a
            # nonzero TF-IDF weight — the real number of shared words
            # behind this similarity score, not just the score itself.
            shared_terms = int((query_vector.multiply(self.matrix[idx]) != 0).nnz)
            results.append({
                "ticket_id": row["ticket_id"],
                "similarity": float(scores[position]),
                "description_occurrences": int(self.description_counts[self.description_keys.iloc[idx]]),
                "issue_type": row["issue_type"],
                "initial_message": row["initial_message"],
                "resolution_summary": row["resolution_summary"] if row["has_resolution"] else None,
                "has_resolution": bool(row["has_resolution"]),
                "shared_terms": shared_terms,
            })
        return results
