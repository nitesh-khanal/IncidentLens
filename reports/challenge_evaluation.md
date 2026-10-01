# Exploratory challenge evaluation

Assistant-authored exploratory challenge set; not independent real-world validation.

48 category-labelled cases and 6 boundary cases. Labels were assigned by the assistant and have not been independently reviewed. This set is excluded from fitting but is not a blind, independent test; changes address known development failures. Treat the results as exploratory and collect human-reviewed real tickets before making generalization claims.

## Before improvements

- Raw classifier accuracy: 75.00%.
- Category suggestion coverage: 100.00%.
- Accuracy among suggested cases: 75.00%.
- Boundary cases withheld: 1/6.
- Retrieval category Precision@1: 81.25%.
- Retrieval category Precision@5: 81.25%.

## After improvements

- Raw classifier accuracy: 72.92%.
- Category suggestion coverage: 43.75%.
- Accuracy among suggested cases: 95.24%.
- Boundary cases withheld: 6/6.
- Retrieval category Precision@1: 81.25%.
- Retrieval category Precision@5: 81.25%.

## Interpretation

Abstention reduces the number of category suggestions. Accuracy among accepted suggestions must always be read together with coverage; it does not replace overall accuracy.
The evidence gate requires at least three vocabulary terms and a model probability score of at least 0.5. It is a heuristic and is not statistically calibrated.
Similarity relevance here means matching the intended category, not confirming that a recorded resolution is appropriate.
Training scope, primary model selection, normalization, and the evidence gate changed together, so this comparison does not isolate the causal effect of one change.
