"""Unit tests for src.classifier."""
import numpy as np
from scipy.sparse import csr_matrix

from src.classifier import make_train_test_split, train_model, train_all_models, RANDOM_STATE

# Small synthetic dataset with a clear pattern each model should learn easily
N = 100
np.random.seed(0)
X_dense = np.random.rand(N, 5)
y = np.where(X_dense[:, 0] > 0.5, "class_a", "class_b")
X = csr_matrix(X_dense)


def test_train_test_split_reproducible():
    split1 = make_train_test_split(X, y)
    split2 = make_train_test_split(X, y)
    # Same random_state -> identical split every time
    np.testing.assert_array_equal(split1[2], split2[2])  # y_train


def test_split_is_stratified():
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.2)
    train_ratio = (y_train == "class_a").mean()
    test_ratio = (y_test == "class_a").mean()
    overall_ratio = (y == "class_a").mean()
    # Stratified split should keep ratios close across train/test/overall
    assert abs(train_ratio - overall_ratio) < 0.1
    assert abs(test_ratio - overall_ratio) < 0.1


def test_train_model_unknown_name_raises():
    try:
        train_model("not_a_real_model", X, y)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_train_all_models_returns_three_fitted_models():
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    models = train_all_models(X_train, y_train)
    assert set(models.keys()) == {"logistic_regression", "decision_tree", "random_forest"}
    for name, model in models.items():
        preds = model.predict(X_test)
        assert len(preds) == X_test.shape[0]


def test_each_model_beats_random_baseline_on_easy_synthetic_data():
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    models = train_all_models(X_train, y_train)
    for name, model in models.items():
        acc = model.score(X_test, y_test)
        # Baseline for a 2-class, roughly-balanced problem is ~0.5;
        # this synthetic data has an easy linear pattern, so every
        # model should clear that easily.
        assert acc > 0.7, f"{name} scored {acc:.3f}, expected clearly better than random"
