# Model evaluation — train-only preprocessing

Raw rows split before fitting TF-IDF/IDF and categorical encoders. Grouped tests share zero normalized templates. Challenge set excluded from training.

The corpus is synthetic and templated. These results do not establish production accuracy.
The legacy run used full-corpus precomputed features. Its accuracy happens to match the corrected holdout here; the corrected metrics below supersede that run.

## Row split (repeated templates may cross the split)

Training rows: 80,000; test rows: 20,000.
Training templates: 96; test templates: 96; shared templates: 96.
Vocabulary fitted on training rows: 97 terms.

- logistic_regression: accuracy 100.00%; macro F1 1.0000.
- decision_tree: accuracy 100.00%; macro F1 1.0000.
- random_forest: accuracy 100.00%; macro F1 1.0000.
## Unseen-template holdout

Training rows: 81,651; test rows: 18,349.
Training templates: 77; test templates: 19; shared templates: 0.
Vocabulary fitted on training rows: 90 terms.

- logistic_regression: accuracy 54.47%; macro F1 0.5452.
- decision_tree: accuracy 54.47%; macro F1 0.5277.
- random_forest: accuracy 54.47%; macro F1 0.5277.

## Three-fold grouped validation

Every fold refits both TF-IDF and the categorical encoder on its training rows. Normalized descriptions form groups. Small categories have only three templates, so fold results can vary substantially.

- logistic_regression: mean accuracy 58.45%; standard deviation 5.91%; mean macro F1 0.4921.
- decision_tree: mean accuracy 52.31%; standard deviation 5.79%; mean macro F1 0.4383.
- random_forest: mean accuracy 46.30%; standard deviation 6.15%; mean macro F1 0.3718.

Primary model: logistic_regression, selected by highest grouped mean macro F1.

## Provenance and scope

The JSON companion contains fold-level counts and per-category accuracy. Deployed components are fitted on the full historical corpus only after evaluation; they are not the held-out evaluation models.
No challenge-set query or label is used to fit vocabulary, encoders, clusters, or models.
