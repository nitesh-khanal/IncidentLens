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

## Critical finding during training (Stage 16)

All three models achieved 100.00% accuracy on both train and test sets.
Diagnosed, not accepted at face value: the primary dataset contains only
**96 unique `processed_message` text values across all 100,000 rows**, and
every one of those 96 texts maps to exactly one `issue_type` with zero
exceptions. This means TF-IDF features alone form a perfect lookup table —
the models are memorizing a 96-entry mapping, not learning to generalize
from incident language. Metadata features (product_area, priority, sla_plan,
channel, platform, region, customer_segment) were checked and ruled out as
the leak source (max purity ~13%, consistent with random chance across 8
balanced classes).

### Implication
100% accuracy on this dataset is a property of its synthetic template
generation, not evidence the underlying approach would work on real,
non-templated incident text. Stage 17 will report this finding prominently
rather than presenting 100% as a genuine result, and will run an additional
"template holdout" test — training and testing on disjoint sets of the 96
templates — to measure whether the model can classify a genuinely unseen
phrasing, which is a fairer, more honest test of generalization than the
standard row-level train/test split on this particular dataset.

## Scope of the account_access generalization limitation (finalized)

This limitation is narrow and well-understood, not systemic:

**Affected:** Only the classifier's (Stage 15-17) ability to correctly
categorize a genuinely novel phrasing of an account_access incident —
one that doesn't resemble any of the 3 existing templates for that
category in the training data.

**NOT affected:**
- The similarity/retrieval engine (Stages 8-10) — retrieval relies on
  lexical overlap with the existing corpus, not generalization to unseen
  phrasing in the same way classification does. A query resembling any
  of the existing account_access templates retrieves normally.
- Clustering (Stage 11) — unsupervised, no train/test generalization
  concept applies.
- Classification of any other issue_type category — all 7 other
  categories had 7+ unique templates and generalized correctly (100%
  accuracy) in the Stage 17 template-holdout test.
- The reported 100% row-level accuracy — separately known to reflect
  memorization (Stage 16), unaffected by this finding either way.

## Considered fix (not implemented — documented for future work)

A concrete augmentation path was investigated: `ameau01/synthetic-it-support-tickets`
(Hugging Face, MIT license, 745 records, evaluated during dataset selection)
was checked for real text diversity relevant to account_access. Verified
findings: 514 of 745 records match account/authentication-related keywords
(login, password, lockout, MFA, SSO, sign-in, access denied), with 327
genuinely unique title phrasings among them (not another template set) —
confirmed via `root_cause` field uniqueness (745 of 745 unique) and manual
inspection of sample titles (BitLocker, GlobalProtect VPN, Conditional
Access, Software Center — naturally varied IT terminology).

This was not implemented in the current system because it would require
reopening and re-running four already-completed stages (6, 7, 14, 16) on
an augmented, differently-sourced dataset, which was judged out of scope
for this stage of the certification project. It remains a concrete,
evidenced next step (see Stage 27's Future Work) rather than a vague
aspiration — the exact dataset, filter criteria, and real diversity
numbers needed to execute it are recorded here.
