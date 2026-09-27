# Explainability Methodology

## Similarity explanations
Cosine similarity is decomposed per-term directly from the TF-IDF vectors:
each term's contribution is `(query_weight * candidate_weight) / (||query|| * ||candidate||)`,
which sums exactly to the real cosine similarity score (verified by unit test).
This is exact, not approximated — every contributing term is traceable back
to real TF-IDF weights.

### Real example (Stage 19 demo)
Query: "Payment API is returning 502 errors and database connections are
timing out after a deployment."
Candidate: a real historical `performance` ticket ("query mobile app module
timing").

Total similarity: **0.2832**. A single shared token, "timing", accounts for
**100% of that score** — no other vocabulary overlaps between the two texts.
This is a genuinely useful, honest illustration: it shows precisely why this
pair scored moderately-low rather than high, rather than showcasing an
artificially strong match. It also reinforces Stage 7's documented
limitation directly: TF-IDF only credits exact shared words, so two
incidents about a related-but-differently-worded problem can score low
even when a human would consider them related.

## Classification explanations
Two genuinely different mechanisms are used, stated explicitly rather than
presented as equivalent:

- **Logistic Regression**: per-class coefficients — real, class-specific
  evidence for "these words pushed toward THIS category." Real output
  (predicting `performance`): top weighted terms were module, query, timing,
  long, slow, take, time, load — all genuinely performance-related
  vocabulary, which is reassuring evidence the model learned meaningful
  signal, not just noise.
- **Decision Tree / Random Forest**: only expose global `feature_importances_`,
  not class-specific, per-prediction importance — explicitly labeled as such
  rather than presented misleadingly as if it explained one specific
  prediction. Real output was itself an important finding: Random Forest's
  top global features were `option`, `plan`, `message_length` — categorical
  and numeric features, NOT text vocabulary. This is direct, concrete
  confirmation of a concern raised back in Stage 15: given the templated
  text (Stage 16's 96-template finding), the tree-based model may lean on
  metadata rather than language to separate classes. Now verified, not just
  hypothesized.
