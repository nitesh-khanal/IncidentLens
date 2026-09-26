# Incident Clustering Report

## k-selection results (real measurements)

| k | Inertia | Silhouette (5K sample) |
|---|---|---|
| 4 | 79622.39 | 0.1558 |
| 6 | 70539.92 | 0.2394 |
| 8 | 62574.94 | 0.3066 |
| 10 | 54610.04 | 0.3721 |
| 12 | 48002.11 | 0.4367 |
| 15 | 37172.60 | 0.5372 |

## Chosen k = 8

## Cluster sizes

|   cluster |   count |
|----------:|--------:|
|         0 |    4263 |
|         1 |   41802 |
|         2 |   12449 |
|         3 |   12388 |
|         4 |    8198 |
|         5 |    8438 |
|         6 |    4255 |
|         7 |    8207 |

## Top terms per cluster

- **Cluster 0:** downgraded, higher, tier, billed, still, plan, would, feature, degraded, detail
- **Cluster 1:** data, export, customize, add, detail, encryption, compliance, properly, explain, configure
- **Cluster 2:** account, login, general, question, noticed, suspicious, attempt, multiple, failed, locked
- **Cluster 3:** need, team, help, setting, advanced, sure, category, fit, option, notification
- **Cluster 4:** page, take, time, slow, load, long, open, crashing, whenever, keep
- **Cluster 5:** seems, vulnerability, functionality, check, please, workspace, something, billing, notification, mobile
- **Cluster 6:** 2fa, working, code, sign, try, feature, degraded, detail, downgraded, encryption
- **Cluster 7:** incorrect, say, log, system, password, invoice, amount, selected, compared, plan

## Cluster vs issue_type crosstab (sanity check, NOT validation)

|   cluster |   account_access |   billing_problem |   bug |   feature_request |   how_to |   other |   performance |   security_concern |
|----------:|-----------------:|------------------:|------:|------------------:|---------:|--------:|--------------:|-------------------:|
|         0 |                0 |              4263 |     0 |                 0 |        0 |       0 |             0 |                  0 |
|         1 |                0 |              4066 |  8344 |              8381 |     8489 |       0 |          8320 |               4202 |
|         2 |             4200 |                 0 |     0 |                 0 |        0 |    4152 |             0 |               4097 |
|         3 |                0 |                 0 |     0 |              4106 |     4250 |    4032 |             0 |                  0 |
|         4 |                0 |                 0 |  4005 |                 0 |        0 |       0 |          4193 |                  0 |
|         5 |                0 |                 0 |     0 |                 0 |        0 |    4279 |             0 |               4159 |
|         6 |             4255 |                 0 |     0 |                 0 |        0 |       0 |             0 |                  0 |
|         7 |             4154 |              4053 |     0 |                 0 |        0 |       0 |             0 |                  0 |

Clustering is unsupervised and does not use issue_type as input. This
crosstab is shown only to check, after the fact, whether discovered
clusters resemble known categories — a strong alignment would suggest
the clusters are picking up on real, meaningful structure; a weak
alignment would not mean the clustering is wrong, since clusters can
legitimately capture different structure than the labeled categories.