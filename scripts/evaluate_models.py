"""
Stage 17 — full model evaluation: standard metrics + cross-validation
(on the original split) AND the template-holdout test (the honest
generalization measure for this specific dataset).
"""
import joblib
import pandas as pd

from src.model_evaluation import compute_metrics, cross_validate_model, template_holdout_split
from src.classifier import train_model
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

DATA_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"
FEATURES_PATH = DATA_PROCESSED_DIR / "feature_matrix.joblib"
SPLIT_PATH = DATA_PROCESSED_DIR / "train_test_split.joblib"
MODELS_DIR = DATA_PROCESSED_DIR / "models"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_evaluation.md"

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest"]


def main():
    df = pd.read_csv(DATA_PATH)
    X = joblib.load(FEATURES_PATH)
    X_train, X_test, y_train, y_test = joblib.load(SPLIT_PATH)

    report_lines = ["# Model Evaluation Report", ""]

    # --- Part 1: standard evaluation on the original row-level split ---
    report_lines += [
        "## Part 1: Standard evaluation (row-level train/test split)",
        "",
        "**Caution:** Stage 16 found this dataset contains only 96 unique text",
        "templates mapping deterministically to labels. High scores here reflect",
        "memorization, not necessarily generalization. See Part 2 for a fairer test.",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
        "|---|---|---|---|---|",
    ]

    all_metrics = {}
    for name in MODEL_NAMES:
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        metrics = compute_metrics(model, X_test, y_test)
        all_metrics[name] = metrics
        print(f"\n=== {name} (row-level split) ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")
        print("\nConfusion matrix:")
        print(metrics["confusion_matrix"])
        print("\nClassification report:")
        print(metrics["classification_report"])

        report_lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
            f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} |"
        )

    # --- Cross-validation on training set ---
    print("\n\n=== Cross-validation (5-fold, on training set) ===")
    report_lines += ["", "## Cross-validation (5-fold, training set)", "", "| Model | Mean Accuracy | Std |", "|---|---|---|"]
    for name in MODEL_NAMES:
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        cv_result = cross_validate_model(model, X_train, y_train, cv=5)
        print(f"{name}: mean={cv_result['mean']:.4f}, std={cv_result['std']:.4f}, scores={cv_result['scores']}")
        report_lines.append(f"| {name} | {cv_result['mean']:.4f} | {cv_result['std']:.4f} |")

    # --- Part 2: template-holdout test (the honest generalization measure) ---
    print("\n\n=== Part 2: Template-holdout test ===")
    Xh_train, Xh_test, yh_train, yh_test, n_train_t, n_test_t = template_holdout_split(df, X)
    print(f"Templates in train: {n_train_t}, Templates in test: {n_test_t}")
    print(f"Rows in train: {Xh_train.shape[0]}, Rows in test: {Xh_test.shape[0]}")

    report_lines += [
        "",
        "## Part 2: Template-holdout test (honest generalization measure)",
        "",
        f"Templates: {n_train_t} in train, {n_test_t} in test (zero overlap).",
        f"Rows: {Xh_train.shape[0]} in train, {Xh_test.shape[0]} in test.",
        "",
        "This retrains each model on templates it has never seen at test time —",
        "a genuinely fair test of whether the model learned transferable patterns",
        "or just memorized the 96-template lookup table.",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
        "|---|---|---|---|---|",
    ]

    for name in MODEL_NAMES:
        model = train_model(name, Xh_train, yh_train)
        metrics = compute_metrics(model, Xh_test, yh_test)
        print(f"\n{name} (template holdout): accuracy={metrics['accuracy']:.4f}, "
              f"f1_macro={metrics['f1_macro']:.4f}")
        report_lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
            f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} |"
        )

    report_lines += [
        "",
        "## Interpretation",
        "_Filled in after reviewing real Part 1 vs Part 2 results — a large drop",
        "from Part 1 to Part 2 confirms memorization; a small drop would suggest",
        "genuine generalization despite the templated data. Not assumed here._",
    ]

    REPORT_PATH.write_text("\n".join(report_lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
