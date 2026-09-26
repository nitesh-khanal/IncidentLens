"""
IncidentLens — model evaluation utilities.

Computes standard classification metrics AND a template-holdout split,
which is the more honest generalization measure for this specific
dataset (see docs/ml_problem_definition.md — Stage 16 found only 96
unique text templates mapping deterministically to labels).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import cross_val_score


def compute_metrics(model, X_test, y_test) -> dict:
    """Real metrics for one fitted model on one test set. Macro-averaged
    precision/recall/F1, appropriate given issue_type's balanced classes."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "labels": sorted(set(y_test)),
    }


def cross_validate_model(model, X, y, cv=5) -> dict:
    """Real k-fold cross-validation accuracy scores on the training set."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


def template_holdout_split(df, X, template_col="processed_message", test_frac=0.2, random_state=42):
    """
    Split by UNIQUE TEMPLATE, not by row. All rows sharing a template go
    entirely into either train or test — never both. This measures
    whether a model can classify a template it has genuinely never seen,
    which the standard row-level split cannot measure on this dataset
    (Stage 16: only 96 unique templates, memorized perfectly otherwise).
    """
    unique_templates = np.array(df[template_col].unique(), dtype=object)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_templates)

    n_test_templates = max(1, int(len(unique_templates) * test_frac))
    test_templates = set(unique_templates[:n_test_templates])
    train_templates = set(unique_templates[n_test_templates:])

    train_mask = df[template_col].isin(train_templates).values
    test_mask = df[template_col].isin(test_templates).values

    X_train, X_test = X[train_mask], X[test_mask]
    y_train = df.loc[train_mask, "issue_type"].astype(str).to_numpy()
    y_test = df.loc[test_mask, "issue_type"].astype(str).to_numpy()

    return X_train, X_test, y_train, y_test, len(train_templates), len(test_templates)
