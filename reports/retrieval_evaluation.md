# Retrieval Evaluation Report

## Relevance definition
A retrieved result is considered relevant if its `issue_type` matches the
query's known source category. This is a proxy for true relevance (no
human-annotated ground truth exists for this dataset) — see
docs/retrieval_evaluation.md for the full rationale and limitations.

## Aggregate metrics
- Average Precision@1: 0.714
- Average Precision@3: 0.714
- Average Precision@5: 0.714
- Computed over 7 of 10 test cases (cases without
  a defined expected category are evaluated for crash-safety, not precision)

## Per-test-case results

| ID | Category | Query | Expected | # Results | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|---|---|---|---|
| T1 | near_identical | Queries in the api integration module are timing o... | performance | 5 | True | True | True |
| T2 | paraphrase | Payment API is returning 502 errors and database c... | performance | 5 | True | True | True |
| T3 | related_different | The checkout page takes a very long time to load a... | performance | 5 | True | True | True |
| T4 | unrelated | I was charged twice for my monthly subscription th... | billing_problem | 5 | True | True | True |
| T5 | duplicate | I cannot log in; the system says my password is in... | account_access | 5 | True | True | True |
| T6 | short | Login broken. | account_access | 5 | False | False | False |
| T7 | noisy | cant LOGIN???!! password keeps saying WRONG!!! pls... | account_access | 5 | False | False | False |
| T8 | missing_information | It doesn't work. | N/A | 0 | None | None | None |
| T9 | empty |  | N/A | 0 | None | None | None |
| T10 | unseen_vocabulary | quixotic flibbertigibbet zorptastic malfunction | N/A | 0 | None | None | None |