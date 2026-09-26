"""
IncidentLens — NLP preprocessing pipeline.

Provides configurable text-cleaning operations (lowercase, punctuation
removal, stopword removal, stemming, lemmatization). Every operation is
optional and independently testable — Stage 6 exists to determine, via
a real experiment, which combination actually helps before Stage 7
commits to one. The original text is never modified by this module;
callers store the result separately.
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()


def ensure_nltk_data():
    """Download required NLTK corpora if not already present locally."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def _stopwords_set():
    return set(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    """Tokenize text into words. Empty/whitespace-only/None text returns []."""
    if not text or not str(text).strip():
        return []
    return word_tokenize(str(text))


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = False,
    stem: bool = False,
    lemmatize: bool = False,
) -> str:
    """
    Apply a configurable NLP preprocessing pipeline to a single text.

    Returns the processed text as a space-joined string ("" for
    empty/None input). Does not mutate the input.
    """
    if not text or not str(text).strip():
        return ""

    if stem and lemmatize:
        raise ValueError("Choose stem or lemmatize, not both — they conflict.")

    working = str(text).lower() if lowercase else str(text)
    tokens = tokenize(working)

    if remove_punctuation:
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = [t for t in tokens if re.search(r"[a-zA-Z0-9]", t)]

    if remove_stopwords:
        sw = _stopwords_set()
        tokens = [t for t in tokens if t not in sw]

    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
    elif lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# Chosen configuration for IncidentLens, based on the Stage 6 experiment
# (see docs/nlp_preprocessing.md): stopword removal cuts already-short
# descriptions roughly in half but keeps distinctive terms; lemmatization
# over stemming because the vocabulary-size difference is negligible
# (98 vs 100 tokens) while lemmatization keeps real, explainable words.
DEFAULT_CONFIG = dict(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    lemmatize=True,
)


def default_preprocess(text: str) -> str:
    """Apply IncidentLens's chosen NLP preprocessing configuration."""
    return preprocess_text(text, **DEFAULT_CONFIG)
