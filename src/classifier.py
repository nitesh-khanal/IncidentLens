"""
IncidentLens — issue_type classifiers.

Trains three baseline models (Logistic Regression, Decision Tree, Random
Forest) on the Stage 14 feature matrix. Training only — evaluation
(precision/recall/F1/confusion matrix/cross-validation) is Stage 17's
responsibility, kept separate so it's clear which stage produced which
number.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
}


def make_train_test_split(X, y, test_size=0.2):
    """Stratified split, fixed random_state for reproducibility."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def train_model(name: str, X_train, y_train):
    """Train one model by name from MODEL_REGISTRY."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}. Choices: {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name]()
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train):
    """Train all three registered models. Returns {name: fitted_model}."""
    return {name: train_model(name, X_train, y_train) for name in MODEL_REGISTRY}
