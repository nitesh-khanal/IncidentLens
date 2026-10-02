# Reviewed synonym handling

IncidentLens uses a small reviewed set of WordNet senses, not unrestricted synonym expansion. `src/synonyms.py` records the approved variants and source senses. Generic words such as payment, account, bill, charge and return are deliberately not globally rewritten. Phrase rules normalize duplicate billing wording only when the phrase supplies context. Negation words are still subject to the existing stopword processing; these rules do not interpret intent or prove a duplicate charge occurred.

Normalization runs in `preprocess_text`, shared by training, classification, retrieval and explanation. The mappings leave all 96 existing cleaned templates unchanged, so the historical vectors and trained artifacts remain consistent; no retraining or invented training examples are necessary for this update. Future mappings that change historical text require running the validated rebuild before release.

A two-term description may receive a tentative category when model score is at least 0.8, the lead over the runner-up is at least 0.5, and the first historical match agrees with the category, shares at least two terms, and has cosine similarity at least 0.6. The normal three-term / 0.5-score gate remains. One-term inputs abstain. These are conservative heuristics, not calibrated probabilities. Short suggestions need human review.

This improves specific vocabulary coverage, not general semantic understanding or dataset diversity. New synonyms should target existing historical vocabulary, use a reviewed sense, and pass regression tests against ambiguous language. Tests validate the reviewed lemmas against the shipped WordNet corpus.

## Validation for this change

The full suite passed (151 tests). Deployment regression tests verify identical stored historical vectors, reviewed WordNet senses, equivalent-query retrieval, short billing evidence, and abstention for ambiguous keywords. On the existing assistant-authored exploratory challenge set, the updated run classified 36/48 cases correctly, supported 25/48 (24 correct), and withheld all six boundary cases. The before-change snapshot is `reports/challenge_before_synonyms.json`; current results are `reports/challenge_results.json`. These cases are exploratory and do not establish independent real-world accuracy.
