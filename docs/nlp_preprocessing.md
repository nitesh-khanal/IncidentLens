# NLP Preprocessing Decision

## Experiment (Stage 6, run on 2,000-message sample of the real dataset)

| Config | Avg tokens | Min tokens | Vocabulary size |
|---|---|---|---|
| lowercase only | 11.40 | 9 | 146 |
| + punctuation removed | 10.32 | 8 | 142 |
| + stopwords removed | 5.30 | 3 | 102 |
| + lemmatization | 5.30 | 3 | 100 |
| + stemming (alt.) | 5.30 | 3 | 98 |

## Chosen configuration
**lowercase → remove punctuation → remove stopwords → lemmatize**

## Rationale
- Descriptions are already short (Stage 5 EDA: mean 56 characters). Stopword removal cuts average tokens roughly in half (11.4 → 5.3), but the removed tokens (articles, pronouns, auxiliary verbs) carry no discriminating signal for similarity — the remaining tokens are the ones that actually distinguish one incident from another.
- Stemming vs. lemmatization: near-identical vocabulary size (98 vs. 100 — a 2-token difference), so there is no meaningful information-density argument for stemming.
- Lemmatization produces real words (`notice`, `change`) vs. stemming's truncated forms (`notic`, `chang`), which matters directly for Stage 19's explainability requirement — the dashboard must show users recognizable shared terms, not fragments.
- No zero-token messages were produced under this config on the full run (see below) — the pipeline doesn't over-strip content even on the shortest descriptions.

## Full-dataset application
Applied via `scripts/apply_nlp_preprocessing.py`, which preserves the original `initial_message` and adds `processed_message` as a new column, saved to `data/processed/tickets_nlp.csv`.
