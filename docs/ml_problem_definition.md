# Machine Learning Problem Definition

## Task
**Multi-class classification: predict `issue_type` from a new incident's
available information at creation time.**

## Target
`issue_type` — 8 classes: how_to, account_access, performance,
feature_request, other, security_concern, billing_problem, bug.
Confirmed in Stage 5 EDA as nearly perfectly balanced (12.35-12.74% each),
so no class-imbalance handling (resampling, class weighting) is required
for this dataset — stated as a property of this specific dataset, not a
general claim about incident classification.

## Inputs
The 138-feature matrix built in Stage 14: TF-IDF text vector (99 dims) +
one-hot categorical fields (34 dims: customer_segment, channel,
product_area, priority, sla_plan, platform, region) + numeric/temporal
features (5 dims: hour_of_day, day_of_week, month, message_length,
has_attachment). All inputs are available at ticket-creation time —
Stage 14's leakage exclusion (resolution info, status, reopened,
csat_score, customer_sentiment) is a hard requirement, not optional.

## Why this prediction is useful
Automatic issue_type classification at ticket creation could route
incidents to the right team/queue immediately, without waiting for a
human to triage. It's also the input trigger for the rest of
IncidentLens's intelligence engine (Stage 18) — classification result
displayed alongside similarity-based retrieval, each clearly labeled as
a distinct kind of evidence.

## Training data
The 100,000-row primary dataset (`data/processed/tickets_nlp.csv` +
`feature_matrix.joblib`), split into train/test sets in Stage 16 with a
fixed `random_state` for reproducibility. No separate labeled dataset is
used — issue_type is the label already present in the source data.

## Evaluation strategy
- **Train/test split** (Stage 16), stratified by issue_type to preserve
  class balance in both sets.
- **Cross-validation** on the training set to check stability across folds,
  not just a single train/test split.
- **Metrics** (Stage 17): accuracy, precision, recall, F1-score (macro-
  averaged, appropriate given balanced classes), confusion matrix,
  full classification report. Accuracy alone is not treated as sufficient,
  per the project's evaluation standard — even though this dataset happens
  to be balanced, reporting only accuracy would set a bad precedent and
  hide any per-class weaknesses.
- **Three models compared** (Stage 16): Logistic Regression, Decision Tree,
  Random Forest — chosen as a spread from simple/linear/interpretable to
  ensemble, per the original spec and the user's explicit request to
  compare exactly three models.

## Limitations (stated in advance, not discovered after training)
- The classifier can only be as good as the 99-word TF-IDF vocabulary
  (Stage 7) allows — a real constraint on how much the text features alone
  can distinguish between categories.
- Stage 11's clustering already found that many incidents share generic,
  category-non-distinguishing vocabulary (the 41,802-incident mixed
  cluster) — the classifier may lean heavily on the categorical/temporal
  features rather than text to separate classes; this will be checked
  empirically in Stage 19 (explainability/feature importance), not
  assumed here.
- Balanced classes on this synthetic dataset should not be read as
  evidence that a real-world incident dataset would also be balanced —
  this is a property of how the dataset was generated.
