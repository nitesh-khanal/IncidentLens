# Performance Report

Data loading (100,000 rows, real): **0.308s**

## Pipeline stages by dataset size (real measurements)

| Size | Cleaning | NLP preprocessing | TF-IDF vectorization | Clustering (k=8) |
|---|---|---|---|---|
| 1,000 | 0.018s | 1.672s | 0.024s | 0.061s |
| 10,000 | 0.012s | 4.238s | 0.027s | 0.070s |
| 50,000 | 0.038s | 21.174s | 0.134s | 0.261s |
| 100,000 | 0.086s | 42.892s | 0.270s | 0.480s |

## Query-time latency (real, full 100K dataset)

- Retrieval (20 repeats): mean 11.16ms, median 10.64ms, max 20.41ms

| Model | Mean inference latency |
|---|---|
| logistic_regression | 0.067ms |
| decision_tree | 0.073ms |
| random_forest | 2.531ms |

- Full `analyze()` (classification + retrieval + cluster + patterns): **27.79ms**
- HTML report generation: **77.28ms**

## Interpretation

**NLP preprocessing is the only real bottleneck** at 42.9s for the full
100,000-row dataset (~0.43ms/row, consistent linear scaling across all
tested sizes — not degrading, just an inherent per-row cost from NLTK
tokenization and lemmatization calls). All other pipeline stages (cleaning,
TF-IDF vectorization, clustering) complete in well under a second even at
full scale.

**This cost does not matter for real usage.** NLP preprocessing is a
one-time, offline batch step (Stage 6) — its output is cached to
`tickets_nlp.csv` and reused by every subsequent stage. It is never
re-run as part of answering a user's query.

**Query-time (interactive) latency is fast across the board**: retrieval
averages 11.16ms even against the full 100K-document corpus, all three
models predict a single incident in under 3ms, and the full intelligence
engine (classification + retrieval + clustering + recurring patterns)
completes in 27.79ms end-to-end. Even generating a full HTML report with
an embedded chart takes only 77.28ms.

**Conclusion: no optimization is needed.** Per the project standard of
optimizing only where a real measurement shows a problem, none of the
measured numbers here warrant further work — the one genuinely slow
operation (NLP preprocessing) is already a one-time offline cost with no
impact on interactive use, and everything a user actually experiences in
real time is well under 100ms.