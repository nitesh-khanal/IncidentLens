# Dataset limitations and validation improvements

## What changed

The historical corpus still contains 100,000 synthetic tickets with 96 unique descriptions. No extra rows, generated resolutions, or invented real-world tickets were added. The system now handles this limitation more carefully:

- Equivalent authentication phrases (`log in`, `login`, `sign in`, `sign-in`) are normalized before stopword removal. Historical vectors, clusters, and deployment models were rebuilt together; the original descriptions and resolutions were preserved.
- Validation starts with raw rows. Each split fits TF-IDF vocabulary/IDF and the categorical encoder on its training rows only. Whole-corpus precomputed features and deployed models are never used to score held-out classification.
- A fixed unseen-template holdout and three-fold stratified grouped validation keep normalized templates together. Macro F1 uses the complete class universe; missing categories are explicit in the JSON results.
- Logistic Regression is the primary model because it has the best mean macro F1 across grouped folds. The exploratory challenge set is not used for model selection.
- Fewer than three matching vocabulary terms, a missing probability score, or a score below 0.5 produce `insufficient_evidence`. Raw candidates remain diagnostic, and category statistics are withheld. This is a conservative heuristic, not calibrated uncertainty or a guarantee of correctness. Historical matches remain available with lexical-overlap warnings.

## Exploratory examples and their provenance

`data/test/challenge_set.json` contains 48 assistant-authored descriptions across eight intended categories and six boundary inputs. It is frozen in source, labelled explicitly, and excluded from fitting. Author-assigned labels are judgements; some natural-language examples are ambiguous. Changes were motivated by known development failures, so this is not an untouched blind benchmark. It contains no independently collected real tickets and has not been independently human-reviewed.

Reports show raw classifier accuracy, accepted-suggestion accuracy, suggestion coverage, boundary abstention, and retrieval category Precision@K. A higher accepted-suggestion accuracy can result from withholding hard cases. Always show coverage and the raw result alongside it. The before/after comparison changes normalization, deployment fitting scope, primary model, and the evidence gate together; it is not a causal ablation experiment.

For genuine independent validation, ask a reviewer to label additional anonymized real tickets before seeing predictions. Keep all tuning separate from that set, verify licensing/privacy, and review whether retrieved resolutions apply to each query. Synthetic augmentation or embeddings alone cannot supply missing resolution evidence or establish real-world accuracy.

## Reproduce

Use Python 3.12 and the pinned requirements. Prepare the NLTK resources, then:

```bash
python -m scripts.rebuild_validated_demo
python -m scripts.evaluate_retrieval
python -m pytest tests/ -q
```

The rebuild reads the committed historical CSV, recomputes normalized text from original descriptions, evaluates from train-only features, rebuilds all deployment components on the historical corpus, and writes measured JSON and readable reports. Challenge cases are only read after fitting finishes.

To refresh classification evaluation without changing deployment artifacts:

```bash
python -m scripts.evaluate_models
```

## Reading the results

`reports/validated_evaluation.json` records row/template counts, overlap, per-category accuracy, grouped folds, selected model, and deployment dimensions. `reports/model_evaluation.md` summarizes the measured run. `reports/challenge_baseline.json` is the pre-change random-forest baseline; `reports/challenge_results.json` is the current primary-model evaluation. The older reports and screenshots may contain original-run results; the new validation reports and the app's current metrics take precedence.

A college demonstration can establish that the system works, explains lexical matches, evaluates honestly, and detects some evidence boundaries. It does not establish that the classifier or historical resolutions are reliable for a production IT team.

Precision@K divides relevant hits by the requested K; missing result slots count as misses.
