"""
IncidentLens — TF-IDF text representation.

Wraps scikit-learn's TfidfVectorizer with IncidentLens defaults,
built on Stage 6's already-preprocessed `processed_message` field.
TF-IDF captures lexical overlap, not semantic meaning — see
docs/tfidf_representation.md for what it can and cannot do.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class IncidentVectorizer:
    """Thin, testable wrapper around scikit-learn's TfidfVectorizer."""

    def __init__(self, max_features=None, ngram_range=(1, 1), min_df=1):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        self._fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts):
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fit before calling transform().")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix

    def vocabulary_size(self) -> int:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fit first.")
        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self):
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fit first.")
        return self.vectorizer.get_feature_names_out()

    def save(self, path):
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted vectorizer.")
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.vectorizer = joblib.load(path)
        obj._fitted = True
        return obj
