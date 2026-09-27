"""Unit tests for src.report_generator."""
from src.report_generator import generate_html_report, LIMITATIONS

SAMPLE_RESULT = {
    "query": "Payment API is returning 502 errors after deployment.",
    "classification": {
        "predicted_issue_type": "performance",
        "metadata_provided": True,
        "note": "Model output, not a confirmed category.",
    },
    "similar_incidents": [
        {"ticket_id": "TCKT_001", "similarity": 0.85, "issue_type": "performance",
         "initial_message": "API timing out", "resolution_summary": "Restarted service",
         "has_resolution": True},
        {"ticket_id": "TCKT_002", "similarity": 0.60, "issue_type": "performance",
         "initial_message": "Slow response", "resolution_summary": None,
         "has_resolution": False},
    ],
    "cluster": {"cluster_id": 4, "top_terms": ["slow", "crash", "load"],
                "note": "Unsupervised grouping — NOT a root cause claim."},
    "recurring_patterns": {"issue_type_frequency_pct": 12.5, "historical_incident_count": 12500,
                            "avg_resolution_time_hours": 14.7},
}

EMPTY_RESULT = {
    "query": "", "classification": None, "similar_incidents": [],
    "cluster": None, "recurring_patterns": None,
}


def test_report_contains_query_text():
    html = generate_html_report(SAMPLE_RESULT)
    assert SAMPLE_RESULT["query"] in html


def test_report_contains_ticket_ids():
    html = generate_html_report(SAMPLE_RESULT)
    assert "TCKT_001" in html
    assert "TCKT_002" in html


def test_report_shows_resolution_and_no_resolution_correctly():
    html = generate_html_report(SAMPLE_RESULT)
    assert "Restarted service" in html
    assert "No resolution recorded" in html


def test_report_always_includes_limitations():
    html = generate_html_report(SAMPLE_RESULT)
    for limitation in LIMITATIONS:
        assert limitation in html


def test_report_includes_embedded_chart_when_results_exist():
    html = generate_html_report(SAMPLE_RESULT)
    assert "data:image/png;base64," in html


def test_empty_result_does_not_crash_and_still_has_limitations():
    html = generate_html_report(EMPTY_RESULT)
    assert "<html" in html.lower()
    for limitation in LIMITATIONS:
        assert limitation in html
    assert "No historically similar incidents found" in html


def test_no_chart_when_no_similar_incidents():
    html = generate_html_report(EMPTY_RESULT)
    assert "data:image/png;base64," not in html


def test_html_injection_in_query_is_escaped():
    """Regression test for Stage 24: a real XSS vulnerability was found
    and fixed — raw HTML/script tags in user-supplied query text must
    never appear unescaped in the generated report."""
    malicious_result = dict(EMPTY_RESULT)
    malicious_result["query"] = '<script>alert("xss")</script> Payment API is down'
    html = generate_html_report(malicious_result)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_html_injection_in_resolution_summary_is_escaped():
    """Similar_incidents fields come from the dataset, not direct user
    input, but are escaped defensively — this must hold too."""
    malicious_result = dict(SAMPLE_RESULT)
    malicious_result["similar_incidents"] = [{
        "ticket_id": "TCKT_999", "similarity": 0.9, "issue_type": "bug",
        "initial_message": "normal text",
        "resolution_summary": '<img src=x onerror="alert(1)">',
        "has_resolution": True,
    }]
    html = generate_html_report(malicious_result)
    assert "<img src=x onerror" not in html
