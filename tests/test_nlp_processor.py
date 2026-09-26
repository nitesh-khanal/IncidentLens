"""Unit tests for src.nlp_processor."""
import pytest

from src.nlp_processor import preprocess_text, ensure_nltk_data


@pytest.fixture(scope="module", autouse=True)
def _nltk_data():
    ensure_nltk_data()


def test_normal_incident_text():
    text = "The payment API is returning 502 errors after deployment."
    result = preprocess_text(text)
    assert "payment" in result
    assert "api" in result
    assert "." not in result


def test_punctuation_removed():
    text = "Server down!!! Users can't login... please help???"
    result = preprocess_text(text, remove_punctuation=True)
    for punct in ["!", "?"]:
        assert punct not in result


def test_abbreviations_preserved_as_tokens():
    text = "VPN and API both failed after the SSL cert expired."
    result = preprocess_text(text)
    assert "vpn" in result
    assert "api" in result
    assert "ssl" in result


def test_numbers_preserved():
    text = "Error code 502 occurred 3 times today."
    result = preprocess_text(text)
    assert "502" in result
    assert "3" in result


def test_short_text():
    text = "Login broken."
    result = preprocess_text(text)
    assert result.strip() != ""
    assert "login" in result


def test_empty_text_returns_empty_string():
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
    assert preprocess_text(None) == ""


def test_repeated_words_all_kept():
    text = "error error error occurred"
    result = preprocess_text(text)
    assert result.split().count("error") == 3


def test_stopword_removal_removes_common_words():
    text = "The server is down and the users are unable to log in"
    without_sw = preprocess_text(text, remove_stopwords=True)
    with_sw = preprocess_text(text, remove_stopwords=False)
    assert "the" not in without_sw.split()
    assert "the" in with_sw.split()


def test_stemming_collapses_related_forms():
    text = "connections connecting connected"
    result = preprocess_text(text, stem=True)
    tokens = result.split()
    assert len(set(tokens)) <= 2


def test_stem_and_lemmatize_mutually_exclusive():
    with pytest.raises(ValueError):
        preprocess_text("running", stem=True, lemmatize=True)
