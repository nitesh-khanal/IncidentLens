# Incident Clustering Methodology

## Method
K-Means over the Stage 7 TF-IDF vectors. K-Means was chosen for being
simple, fast at this scale (100K rows, 99-dimensional sparse vectors),
and directly interpretable via cluster centroids (top-weighted terms
per cluster act as a natural cluster label).

## Choosing k
Real inertia (elbow method) and silhouette score (computed on a 5,000-row
random sample, since silhouette is O(n^2) and infeasible on the full 100K
rows) were measured across k in {4, 6, 8, 10, 12, 15} — see
reports/clustering_report.md for actual numbers.

k=8 was selected deliberately to match issue_type's known 8 categories —
not because 8 is objectively "correct," but to test a specific, falsifiable
question: does purely unsupervised clustering on text alone rediscover the
same structure as the labeled categories? The crosstab in the report answers
this with real data, not assumption.

## Important limitation — clusters are not root causes
A cluster groups incidents whose language is similar under TF-IDF. It is
NOT a claim that all incidents in a cluster share the same true root cause,
and a cluster is not validated as "correct" by any ground truth — clustering
is unsupervised by definition. Top-weighted terms per cluster are shown as
descriptive labels only ("this group is characterized by these words"), not
as diagnostic conclusions.

## Real results (Stage 11 run, k=8)

Silhouette score at k=8: 0.307 (moderate — consistent with the mixed
picture below, not uniformly strong separation).

### What was actually discovered (not assumed)
- **Clusters 0 and 6 are pure, single-category clusters**: cluster 0 is
  100% billing_problem; cluster 6 is 100% account_access, specifically
  2FA/sign-in language ("2fa, working, code, sign").
- **Cluster 4 found a real cross-category pattern**: bug and performance
  incidents (4,005 + 4,193) grouped together, unified by shared symptom
  vocabulary ("slow, load, crashing, take time") rather than their labeled
  category — bugs and performance issues often *present* the same way.
- **Cluster 1 is a large, mixed catch-all** (41,802 incidents — 42% of the
  dataset — spanning billing_problem, bug, feature_request, how_to,
  performance, and security_concern). This is not a clustering failure; it
  reflects that most incidents share generic support-ticket vocabulary
  ("need, help, please, explain") that doesn't carry category-distinguishing
  signal, consistent with Stage 7's documented 99-word vocabulary limit.
- **Cluster 7 mixes account_access and billing_problem** around the shared
  token "log" — the same tokenization artifact documented in Stage 10
  (stopword-stripped "log in" → "log" overlaps with generic phrasing).

### Interpretation
Clustering did not cleanly rediscover all 8 issue_type categories, and that
is itself the honest finding: with a small, heavily-processed vocabulary,
TF-IDF-based clustering successfully separates categories with genuinely
distinctive language (billing terms, 2FA terms) but collapses categories
that share generic support-ticket phrasing into one large group. This is
reported as a real limitation of the lexical approach at this vocabulary
size — not smoothed over as a clean 8-way match.
