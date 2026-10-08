# Reviewed synonym handling

IncidentLens uses a small reviewed set of WordNet senses, not unrestricted synonym expansion. `src/synonyms.py` records the approved variants and source senses. Generic words such as payment, account, bill, charge and return are deliberately not globally rewritten. Phrase rules normalize duplicate billing wording only when the phrase supplies context. Negation words are still subject to the existing stopword processing; these rules do not interpret intent or prove a duplicate charge occurred.

Normalization runs in `preprocess_text`, shared by training, classification, retrieval and explanation. The mappings leave all 96 existing cleaned templates unchanged, so the historical vectors and trained artifacts remain consistent; no retraining or invented training examples are necessary for this update. Future mappings that change historical text require running the validated rebuild before release.

A two-term description may receive a tentative category when model score is at least 0.8, the lead over the runner-up is at least 0.5, and the first historical match agrees with the category, shares at least two terms, and has cosine similarity at least 0.6. The normal three-term / 0.5-score gate remains. One-term inputs abstain. These are conservative heuristics, not calibrated probabilities. Short suggestions need human review.

This improves specific vocabulary coverage, not general semantic understanding or dataset diversity. New synonyms should target existing historical vocabulary, use a reviewed sense, and pass regression tests against ambiguous language. Tests validate the reviewed lemmas against the shipped WordNet corpus.

## Conservative vocabulary extension

Five additional reviewed WordNet lemmas normalize `slowly` to `slow`,
`exportation` to `export`, `computer code` to `code`, `computer error` to
`error`, and `mouse click` to `click`. Multiword WordNet lemmas accept spaces,
hyphens, or underscores. Existing vocabulary words are not globally redirected
to other vocabulary words; ambiguous alternatives such as `easy` (also listed
under the slow adverb sense), `encoding`, and `exposure` remain unchanged.

Direct regression checks confirmed all 96 historical processed descriptions
and their stored TF-IDF vectors remain identical. On the frozen exploratory
challenge set, 47 of 48 labelled cases were unchanged; the remaining query,
"The mobile app responds slowly when I switch between screens", improved from
`bug` to `performance`, with top-one retrieval changing from incorrect to
correct. All six boundary cases still abstained. Raw category accuracy improved
from 36/48 to 37/48 and top-one retrieval from 39/48 to 40/48. Accepted accuracy
and coverage were unchanged. This is exploratory evidence, not a guarantee of
real-world performance. The full suite passed: **175 tests**. Regression cases in
`tests/test_synonyms.py` verify the extension against the deployed artifacts.

## Validation for this change

### Conservative technical phrases

Additional domain rules recognize `application programming interface` as `api`,
`mobile application` as `mobile app`, `login authentication` as `login auth`,
and `two-factor authentication code` as `2fa code`. These also accept spaces or
hyphens and mixed case. The authentication rule requires the following word
`code`: the unrestricted phrase caused worse retrieval for an existing setup
question, so it was narrowed before acceptance. Generic failure phrases such
as `not working`, `request failed`, and `payment declined` remain unchanged.
These are explicit domain rules, not WordNet-derived sentence understanding.

Direct checks passed all equivalent-query and ambiguity cases, and confirmed
the 96 historical descriptions and stored vectors remain unchanged. Across
the 54 exploratory queries, category predictions and support decisions stayed
unchanged, and no individual case lost top-one or top-five retrieval precision.
The full suite passed: **175 tests**. Runtime preflight and the offline WordNet
audit also passed. Two warnings were non-failing: core-count detection and an
intentional duplicate-point clustering test.

The current full suite passed (175 tests). Deployment regression tests verify identical stored historical vectors, reviewed WordNet senses, equivalent-query retrieval, short billing evidence, and abstention for ambiguous keywords. On the existing assistant-authored exploratory challenge set, the updated run classified 37/48 cases correctly, supported 25/48 (24 correct), retrieved a correct top-one category for 40/48, and withheld all six boundary cases. The snapshot before the original synonym implementation is `reports/challenge_before_synonyms.json`; current results are `reports/challenge_results.json`. These cases are exploratory and do not establish independent real-world accuracy.
