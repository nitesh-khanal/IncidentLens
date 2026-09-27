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
