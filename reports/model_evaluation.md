# Model Evaluation Report

## Part 1: Standard evaluation (row-level train/test split)

**Caution:** Stage 16 found this dataset contains only 96 unique text
templates mapping deterministically to labels. High scores here reflect
memorization, not necessarily generalization. See Part 2 for a fairer test.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| decision_tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Cross-validation (5-fold, training set)

| Model | Mean Accuracy | Std |
|---|---|---|
| logistic_regression | 1.0000 | 0.0000 |
| decision_tree | 1.0000 | 0.0000 |
| random_forest | 1.0000 | 0.0000 |

## Part 2: Template-holdout test (honest generalization measure)

Templates: 77 in train, 19 in test (zero overlap).
Rows: 81651 in train, 18349 in test.

This retrains each model on templates it has never seen at test time —
a genuinely fair test of whether the model learned transferable patterns
or just memorized the 96-template lookup table.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| logistic_regression | 0.5447 | 0.6030 | 0.7143 | 0.6231 |
| decision_tree | 0.5447 | 0.6874 | 0.8333 | 0.7036 |
| random_forest | 0.5447 | 0.5744 | 0.8333 | 0.6313 |

## Interpretation
_Filled in after reviewing real Part 1 vs Part 2 results — a large drop
from Part 1 to Part 2 confirms memorization; a small drop would suggest
genuine generalization despite the templated data. Not assumed here._