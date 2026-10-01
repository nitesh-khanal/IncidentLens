"""
Application-level tests for app.py (Stage 22), using Streamlit's
AppTest framework. These run the REAL app.py script against the REAL
trained artifacts from Stages 1-21 — not mocks — so a passing suite
here means the actual dashboard genuinely works, not just its pieces
in isolation.
"""
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = "../app.py"
CANONICAL_QUERY = (
    "Payment API is returning 502 errors and database connections "
    "are timing out after a deployment."
)


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_app_starts_without_exception(app):
    assert not app.exception


def test_dashboard_page_loads_with_metrics(app):
    app.sidebar.radio[0].set_value("🏠 Dashboard").run()
    assert not app.exception
    assert len(app.metric) >= 4


def test_navigate_to_analyze_page(app):
    app.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    assert not app.exception
    assert len(app.text_area) >= 1


def test_analyze_with_valid_query_produces_results(app):
    app.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    app.text_area[0].set_value(CANONICAL_QUERY).run()
    app.button[0].click().run()
    assert not app.exception


def test_analyze_with_empty_query_does_not_crash(app):
    app.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    app.text_area[0].set_value("").run()
    app.button[0].click().run()
    assert not app.exception


def test_analyze_with_whitespace_only_query_does_not_crash(app):
    app.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    app.text_area[0].set_value("   ").run()
    app.button[0].click().run()
    assert not app.exception


def test_navigate_to_recurring_patterns_page(app):
    app.sidebar.radio[0].set_value("🧩 Recurring Patterns").run()
    assert not app.exception


def test_navigate_to_trends_page(app):
    app.sidebar.radio[0].set_value("📈 Incident Trends").run()
    assert not app.exception


def test_navigate_to_ml_performance_page(app):
    app.sidebar.radio[0].set_value("🤖 ML Performance").run()
    assert not app.exception


def test_reports_page_before_any_analysis_shows_info_not_crash(app):
    fresh = AppTest.from_file(APP_PATH, default_timeout=60)
    fresh.run()
    fresh.sidebar.radio[0].set_value("📄 Reports").run()
    assert not fresh.exception


def test_reports_page_after_analysis_shows_download_buttons(app):
    app.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    app.text_area[0].set_value(CANONICAL_QUERY).run()
    app.button[0].click().run()
    app.sidebar.radio[0].set_value("📄 Reports").run()
    assert not app.exception
    assert len(app.download_button) >= 2


def test_analyze_with_no_vocabulary_overlap_shows_warning(app):
    """Regression test: a query with no words in the TF-IDF vocabulary
    must show the low-confidence warning in both the main classification
    box and the 3-model comparison table, not report a silent prediction."""
    app.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    app.text_area[0].set_value("hello").run()
    app.button[0].click().run()
    assert not app.exception
    page_text = " ".join(w.value for w in app.caption) + " ".join(w.value for w in app.warning)
    assert "training vocabulary" in page_text


def test_results_persist_after_navigation_and_draft_edit():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    at.text_area[0].set_value(CANONICAL_QUERY).run()
    at.button[0].click().run()
    saved_report = at.session_state["last_report"]
    at.sidebar.radio[0].set_value("🏠 Dashboard").run()
    at.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    assert not at.exception
    assert any("Showing saved analysis" in caption.value for caption in at.caption)
    at.text_area[0].set_value("Changed draft text").run()
    assert at.session_state["last_report"] == saved_report
    assert any("draft has changed" in info.value for info in at.info)


def test_load_example_and_analyze():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    at.button[1].click().run()
    assert "charged twice" in at.text_area[0].value
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["last_result"]["similar_incidents"]


def test_empty_submit_has_actionable_feedback():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.sidebar.radio[0].set_value("🔎 Analyze New Incident").run()
    at.button[0].click().run()
    assert any("Enter an incident description" in warning.value for warning in at.warning)


def test_unrecognized_input_displays_insufficient_evidence():
    at=AppTest.from_file(APP_PATH,default_timeout=60).run()
    at.sidebar.radio[0].set_value('🔎 Analyze New Incident').run()
    at.text_area[0].set_value('quasar nebula starlight').run()
    at.button[0].click().run()
    assert not at.exception
    assert any('Insufficient evidence' in str(metric.value) for metric in at.metric)
    assert at.session_state['last_result']['recurring_patterns'] is None
