"""
Stage 16 — train the three models on the real dataset with a reproducible
stratified split. Saves models + the split itself (for Stage 17 to reuse
the identical test set, not a freshly re-split one).
"""
import time

import joblib
import pandas as pd

from src.classifier import make_train_test_split, train_all_models, RANDOM_STATE
from src.config import DATA_PROCESSED_DIR

DATA_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"
FEATURES_PATH = DATA_PROCESSED_DIR / "feature_matrix.joblib"
SPLIT_PATH = DATA_PROCESSED_DIR / "train_test_split.joblib"
MODELS_DIR = DATA_PROCESSED_DIR / "models"


def main():
    df = pd.read_csv(DATA_PATH)
    X = joblib.load(FEATURES_PATH)
    y = df["issue_type"].astype(str).to_numpy()  # force plain numpy, not pyarrow-backed

    print(f"Feature matrix: {X.shape}, labels: {len(y)}")
    print(f"Random state: {RANDOM_STATE}")

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Confirm stratification actually preserved class balance
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    print("\nClass balance check (train vs test, should be near-identical):")
    print(pd.DataFrame({"train": train_dist, "test": test_dist}).round(4))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), SPLIT_PATH)

    print("\nTraining models...")
    timings = {}
    models = {}
    for name in ["logistic_regression", "decision_tree", "random_forest"]:
        start = time.time()
        from src.classifier import train_model
        model = train_model(name, X_train, y_train)
        elapsed = time.time() - start
        timings[name] = elapsed
        models[name] = model
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  {name}: trained in {elapsed:.2f}s | "
              f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    print(f"\nModels saved to {MODELS_DIR}")
    print(f"Train/test split saved to {SPLIT_PATH}")


if __name__ == "__main__":
    main()
