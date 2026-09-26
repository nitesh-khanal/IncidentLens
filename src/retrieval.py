"""
IncidentLens — historical incident retrieval layer.

Wraps SimilarityEngine with fuller metadata and a human-readable report
format. Language stays evidential throughout ("historical resolution",
never "root cause proven") per the project's ethical framing.
"""
from src.similarity_engine import SimilarityEngine

RETRIEVAL_FIELDS = [
    "created_at", "priority", "status", "customer_segment",
    "channel", "region",
]


class RetrievalEngine:
    """Composes a SimilarityEngine to produce enriched, reportable results."""

    def __init__(self, similarity_engine: SimilarityEngine):
        self.engine = similarity_engine

    def retrieve(self, query_text: str, top_n: int = 5):
        """Return up to top_n similar incidents enriched with metadata.
        Deterministic for a fixed vectorizer/matrix/query (no randomness)."""
        results = self.engine.find_similar(query_text, top_n=top_n)

        enriched = []
        for r in results:
            match = self.engine.metadata_df[
                self.engine.metadata_df["ticket_id"] == r["ticket_id"]
            ]
            full = dict(r)
            if not match.empty:
                row = match.iloc[0]
                for field in RETRIEVAL_FIELDS:
                    if field in row.index:
                        full[field] = row[field]
            enriched.append(full)
        return enriched


def format_retrieval_report(query: str, results: list) -> str:
    """Format retrieval results as a human-readable report."""
    lines = ["NEW INCIDENT", "", query.strip(), "", "SIMILAR HISTORICAL INCIDENTS", ""]

    if not results:
        lines.append("No historically similar incidents found for this description.")
        return "\n".join(lines)

    for r in results:
        lines.append(f"Incident #{r['ticket_id']}")
        lines.append(f"Similarity: {r['similarity'] * 100:.0f}%")
        lines.append("")
        lines.append(f"Issue type: {r.get('issue_type', 'unknown')}")
        lines.append(f"Priority: {r.get('priority', 'unknown')}  |  Status: {r.get('status', 'unknown')}")
        lines.append("")
        if r["has_resolution"]:
            lines.append("Historical resolution (evidence, not a confirmed root cause):")
            lines.append(r["resolution_summary"])
        else:
            lines.append("No resolution recorded for this historical incident.")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines)
