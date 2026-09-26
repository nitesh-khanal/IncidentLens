"""
IncidentLens — feature engineering for the Stage 15-17 classifier.

Builds features available at ticket-creation time only. Explicitly
excludes anything determined after resolution (resolution info, status,
reopened, csat_score, customer_sentiment) to avoid label leakage — see
docs/feature_engineering.md for the full rationale.
"""
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = [
    "customer_segment", "channel", "product_area",
    "priority", "sla_plan", "platform", "region",
]

# Fields that exist in the dataset but must NEVER be used as features for
# issue_type prediction — they are only known after (or during) resolution,
# not at ticket-creation time when this classifier would actually run.
LEAKAGE_FIELDS = [
    "resolution_summary", "resolution_time_hours", "has_resolution",
    "status", "reopened", "csat_score", "customer_sentiment",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day, day_of_week, month from created_at. Does not mutate input."""
    df = df.copy()
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["hour_of_day"] = created.dt.hour
    df["day_of_week"] = created.dt.dayofweek
    df["month"] = created.dt.month
    return df


def add_text_length_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add message_length (character count of initial_message). Does not mutate input."""
    df = df.copy()
    df["message_length"] = df["initial_message"].fillna("").str.len()
    return df


class FeatureBuilder:
    """Builds the combined (TF-IDF + categorical + numeric) feature matrix."""

    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self._fitted = False
        self.numeric_columns = ["hour_of_day", "day_of_week", "month",
                                 "message_length", "has_attachment"]

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_temporal_features(df)
        df = add_text_length_feature(df)
        return df

    def fit_transform(self, df: pd.DataFrame, tfidf_matrix):
        df = self._prepare(df)
        cat_matrix = self.encoder.fit_transform(df[CATEGORICAL_FEATURES].astype(str))
        self._fitted = True
        numeric_matrix = csr_matrix(df[self.numeric_columns].fillna(0).values)
        return hstack([tfidf_matrix, cat_matrix, numeric_matrix]).tocsr()

    def transform(self, df: pd.DataFrame, tfidf_matrix):
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fit before calling transform().")
        df = self._prepare(df)
        cat_matrix = self.encoder.transform(df[CATEGORICAL_FEATURES].astype(str))
        numeric_matrix = csr_matrix(df[self.numeric_columns].fillna(0).values)
        return hstack([tfidf_matrix, cat_matrix, numeric_matrix]).tocsr()

    def get_feature_names(self, tfidf_feature_names) -> list:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder must be fit first.")
        cat_names = list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return list(tfidf_feature_names) + cat_names + self.numeric_columns
