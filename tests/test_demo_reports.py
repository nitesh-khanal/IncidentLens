from src.report_generator import generate_html_report


def test_report_escapes_model_and_cluster_notes_and_preserves_zero_duration():
    result = {
        "query": "test",
        "classification": {"predicted_issue_type": "<script>alert(1)</script>", "note": "<img src=x onerror=alert(1)>"},
        "similar_incidents": [],
        "cluster": {"cluster_id": 0, "top_terms": [], "note": "<script>alert(2)</script>"},
        "recurring_patterns": {"avg_resolution_time_hours": 0.0, "issue_type_frequency_pct": 25, "historical_incident_count": 10},
    }
    report = generate_html_report(result)
    assert "<script>" not in report
    assert "<img src=x" not in report
    assert "&lt;script&gt;" in report
    assert "<strong>0.0h</strong>" in report
    assert "of the historical dataset" in report
