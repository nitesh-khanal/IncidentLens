"""
IncidentLens — report generation (Stage 21).

Produces a self-contained HTML report (chart embedded as base64 PNG, no
external files needed) from an IncidentIntelligenceEngine result. Never
generates recommendations beyond what the evidence shows — language stays
evidential throughout, and an explicit limitations section is always
included, not optional.
"""
import base64
import html as html_module
import io
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIMITATIONS = [
    "TF-IDF similarity is purely lexical (shared words), not semantic — "
    "two incidents describing the same problem in different words may "
    "score low similarity (see docs/tfidf_representation.md).",
    "The primary dataset contains only 96 unique text templates across "
    "100,000 rows; classifier accuracy figures reported elsewhere reflect "
    "this (see docs/ml_problem_definition.md).",
    "The account_access category has only 3 historical text templates, "
    "reducing classification confidence for novel phrasing of "
    "account-lockout issues specifically.",
    "Cluster membership reflects language similarity, not a confirmed "
    "shared root cause (see docs/clustering.md).",
    "This is a decision-support tool. It reports historical patterns and "
    "model output as evidence — it does not determine the true root "
    "cause of any incident.",
]


def _similarity_chart_base64(similar_incidents: list) -> str:
    """Real bar chart of similarity scores, embedded as base64 PNG."""
    if not similar_incidents:
        return ""
    labels = [r["ticket_id"] for r in similar_incidents]
    scores = [r["similarity"] * 100 for r in similar_incidents]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, scores, color="steelblue")
    ax.set_xlabel("Similarity (%)")
    ax.set_title("Similarity scores of retrieved incidents")
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_html_report(result: dict) -> str:
    """Build a self-contained HTML report string from an analyze() result."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    query = result["query"]
    classification = result["classification"]
    similar = result["similar_incidents"]
    cluster = result["cluster"]
    recurring = result["recurring_patterns"]

    chart_b64 = _similarity_chart_base64(similar)
    chart_html = (
        f'<img src="data:image/png;base64,{chart_b64}" alt="Lexical similarity scores for retrieved historical incidents" style="max-width:100%;">'
        if chart_b64 else "<p><em>No similar incidents to chart.</em></p>"
    )

    classification_html = "<p><em>No classification available.</em></p>"
    if classification:
        status = "Suggested issue type" if classification.get("supported", True) else "Insufficient evidence; diagnostic candidate only"
        classification_html = f"""
        <p><strong>{status}:</strong> {html_module.escape(str(classification['predicted_issue_type']))}</p>
        <p class="note">{html_module.escape(str(classification['note']))}</p>
        """

    similar_html = "<p><em>No historically similar incidents found.</em></p>"
    if similar:
        rows = ""
        for r in similar:
            safe_resolution = (
                html_module.escape(r["resolution_summary"]) if r["has_resolution"]
                else "<em>No resolution recorded.</em>"
            )
            rows += f"""
            <tr>
                <td>{html_module.escape(str(r['ticket_id']))}</td>
                <td>{r['similarity']*100:.0f}%</td>
                <td>{html_module.escape(str(r['issue_type']))}</td>
                <td>{html_module.escape(str(r['initial_message']))}</td>
                <td>{safe_resolution}</td>
            </tr>"""
        similar_html = f"""
        <table>
            <tr><th>Ticket</th><th>Similarity</th><th>Issue Type</th><th>Description</th><th>Historical Resolution</th></tr>
            {rows}
        </table>"""

    cluster_html = "<p><em>No cluster assignment available.</em></p>"
    if cluster:
        cluster_html = f"""
        <p><strong>Cluster #{html_module.escape(str(cluster['cluster_id']))}</strong> — characterized by: {html_module.escape(', '.join(str(t) for t in cluster['top_terms'][:8]))}</p>
        <p class="note">{html_module.escape(str(cluster['note']))}</p>
        """

    recurring_html = "<p><em>No recurring pattern data available.</em></p>"
    if recurring:
        avg_time = f"{recurring['avg_resolution_time_hours']}h" if recurring["avg_resolution_time_hours"] is not None else "N/A"
        recurring_html = f"""
        <p>This predicted category represents <strong>{recurring['issue_type_frequency_pct']}%</strong>
        of the historical dataset (<strong>{recurring['historical_incident_count']:,}</strong> incidents in this category).</p>
        <p>Average historical resolution time: <strong>{avg_time}</strong></p>
        """

    limitations_html = "".join(f"<li>{l}</li>" for l in LIMITATIONS)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>IncidentLens Report</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 24px; color: #16324f; line-height: 1.6; }}
td {{ overflow-wrap: anywhere; }}
@media print {{ body {{ margin: 0; }} tr {{ break-inside: avoid; }} }}
h1 {{ color: #1a4d8f; }}
h2 {{ border-bottom: 2px solid #eee; padding-bottom: 4px; margin-top: 30px; }}
.note {{ color: #666; font-size: 0.9em; font-style: italic; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.9em; }}
th {{ background: #f5f5f5; }}
.limitations {{ background: #fff8e1; padding: 15px; border-left: 4px solid #f9a825; }}
</style></head>
<body>
<h1>IncidentLens Analysis Report</h1>
<p class="note">Generated {timestamp}. This report is a decision-support tool — see Limitations below.</p>

<h2>New Incident</h2>
<p>{html_module.escape(query)}</p>

<h2>Classification (model output)</h2>
{classification_html}

<h2>Similar Historical Incidents (historical evidence)</h2>
{similar_html}
{chart_html}

<h2>Cluster Membership (unsupervised inference)</h2>
{cluster_html}

<h2>Recurring Patterns (observed historical data)</h2>
{recurring_html}

<h2>Analytical Limitations</h2>
<div class="limitations"><ul>{limitations_html}</ul></div>

</body></html>"""
