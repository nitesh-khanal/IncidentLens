"""Semantic regressions against WordNet and the shipped deployment artifacts."""
import joblib
import pandas as pd
import pytest
from nltk.corpus import wordnet
from src.config import DATA_PROCESSED_DIR
from src.nlp_processor import default_preprocess, ensure_nltk_data
from src.synonyms import REVIEWED_WORDNET
from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.analyzer import IncidentIntelligenceEngine


@pytest.fixture(scope='module')
def deployed():
    ensure_nltk_data()
    p = DATA_PROCESSED_DIR
    df = pd.read_csv(p / 'tickets_clustered.csv')
    vec = IncidentVectorizer.load(p / 'tfidf_vectorizer.joblib')
    matrix = joblib.load(p / 'tfidf_matrix.joblib')
    engine = IncidentIntelligenceEngine(
        RetrievalEngine(SimilarityEngine(vec, matrix, df)),
        joblib.load(p / 'clusterer.joblib'), {},
        joblib.load(p / 'models/logistic_regression.joblib'),
        joblib.load(p / 'feature_builder.joblib'), vec, df)
    return engine, df, matrix


def test_reviewed_senses_are_real_wordnet_synonyms(deployed):
    for variant, (canonical, sense) in REVIEWED_WORDNET.items():
        assert {variant, canonical} <= set(wordnet.synset(sense).lemma_names())


def test_existing_historical_vectors_remain_consistent(deployed):
    engine, df, matrix = deployed
    rows = df.drop_duplicates('initial_message')
    cleaned = rows.initial_message.map(default_preprocess)
    assert cleaned.tolist() == rows.processed_message.tolist()
    actual = engine.vectorizer.transform(cleaned)
    assert (actual != matrix[rows.index]).nnz == 0
    assert cleaned.nunique() == 96


@pytest.mark.parametrize('variant,canonical', [
    ('The data export page is sluggish', 'The data export page is slow'),
    ('My invoice amount is wrong', 'My invoice amount is incorrect'),
    ('Can I customise notifications', 'Can I customize notifications'),
    ('I was billed twice for my subscription', 'I was charged twice for my subscription'),
    ('I paid two times for my subscription', 'I was charged twice for my subscription'),
    ('A duplicate payment for my subscription', 'charged twice for my subscription'),
    ('The data export page loads slowly', 'The data export page loads slow'),
    ('Data exportation failed', 'Data export failed'),
    ('The computer code has a computer error', 'The code has an error'),
    ('A mouse click on the dashboard', 'A click on the dashboard'),
    ('A MOUSE-CLICK on the dashboard', 'A click on the dashboard'),
    ('A mouse_click on the dashboard', 'A click on the dashboard'),
    ('My two-factor authentication code is not working', 'My 2FA code is not working'),
    ('My TWO FACTOR AUTHENTICATION code is not working', 'My 2FA code is not working'),
    ('The application programming interface integration is slow', 'The API integration is slow'),
    ('The mobile application page is slow', 'The mobile app page is slow'),
    ('The sign-in authentication page is slow', 'The login auth page is slow'),
])
def test_equivalent_wording_has_identical_retrieval(deployed, variant, canonical):
    engine, _, _ = deployed
    assert default_preprocess(variant) == default_preprocess(canonical)
    assert engine.retrieval.retrieve(variant) == engine.retrieval.retrieve(canonical)


@pytest.mark.parametrize('query', ['payment', 'subscription', 'twice', 'billing page', 'payment failed', 'double', 'charge'])
def test_ambiguous_short_inputs_still_abstain(deployed, query):
    engine, _, _ = deployed
    result = engine.analyze(query)
    assert not result['classification']['supported']
    assert result['recurring_patterns'] is None


def test_subscription_twice_has_agreeing_short_evidence(deployed):
    engine, _, _ = deployed
    result = engine.analyze('subscription twice')
    c = result['classification']
    assert c['matched_vocabulary_terms'] == 2
    assert c['supported']
    assert c['predicted_issue_type'] == 'billing_problem'
    assert 'Tentative short-description' in c['note']
    assert result['similar_incidents'][0]['issue_type'] == 'billing_problem'


def test_no_global_rewrite_of_ambiguous_billing_words(deployed):
    assert default_preprocess('payment subscription account bill return charge') == 'payment subscription account bill return charge'


def test_extended_synonyms_preserve_boundaries_and_unrelated_meanings():
    from src.synonyms import normalize_support_synonyms
    text = 'computer science mouse pointer encoding exposure easy slowlyish exportations'
    assert normalize_support_synonyms(text) == text
    normalized = normalize_support_synonyms('No computer error after a mouse click')
    assert normalized == 'No error after a click'
    assert normalize_support_synonyms(normalized) == normalized


@pytest.mark.parametrize('text', [
    'two factors affect authentication', 'multi-factor authentication',
    'application programming course', 'mobile applicationform',
    'desktop application', 'authentication failed', 'not working',
    'instructions for enabling two-factor authentication',
    'slow connection', 'request failed', 'payment declined',
])
def test_technical_phrase_aliases_leave_ambiguous_text_unchanged(text):
    from src.synonyms import normalize_support_synonyms
    assert normalize_support_synonyms(text) == text


def test_technical_phrase_aliases_preserve_negation_and_are_idempotent():
    from src.synonyms import normalize_support_synonyms
    normalized = normalize_support_synonyms('Do not share your two-factor authentication code')
    assert normalized == 'Do not share your 2fa code'
    assert normalize_support_synonyms(normalized) == normalized


def test_short_suggestion_requires_historical_agreement(deployed, monkeypatch):
    engine, _, _ = deployed
    monkeypatch.setattr(engine.retrieval, 'retrieve', lambda *a, **kw: [])
    assert not engine.analyze('subscription twice')['classification']['supported']
