# Dataset: Synthetic IT Support Tickets (Primary)

## Source
- Kaggle: https://www.kaggle.com/datasets/ahsanneural/synthetic-it-support-tickets
- File: `synthetic_it_support_tickets.csv`
- Downloaded via Kaggle CLI on 2026-09-23

## License & Attribution
- License: **CC BY 4.0** (Attribution 4.0 International)
- Permitted: share and adapt, including commercially
- Required: give appropriate credit to the dataset author (ahsanneural on Kaggle), link to the license (https://creativecommons.org/licenses/by/4.0/), and indicate if changes were made
- No additional restrictions may be applied on top of this license

## Real Statistics (measured via `scripts/inspect_dataset.py`)

- **Rows:** 100,000
- **Columns:** 20
- **Exact duplicate rows:** 0
- **In-memory size (as loaded by pandas):** ~45.7 MB

### Columns

| Column | Type | Missing | Notes |
|---|---|---|---|
| ticket_id | str | 0 | unique identifier, e.g. `TCKT_000001` |
| created_at | str (ISO datetime) | 0 | |
| customer_id | str | 0 | |
| customer_segment | str | 0 | e.g. individual, small_business |
| channel | str | 0 | e.g. email, chat, in_app |
| product_area | str | 0 | e.g. data_export, billing, api_integration |
| issue_type | str | 0 | e.g. account_access, security_concern, bug — closest thing to an incident category |
| priority | str | 0 | e.g. low, medium |
| status | str | 0 | e.g. resolved, closed_no_action, in_progress |
| sla_plan | str | 0 | e.g. standard |
| initial_message | str | 0 | the incident description — primary text field for NLP/similarity |
| agent_first_reply | str | 0 | first agent response text |
| resolution_summary | str | **39,887** | free-text resolution — **no separate root_cause field exists** |
| resolution_time_hours | float64 | 39,887 | missing wherever resolution_summary is missing (unresolved tickets) |
| reopened | int64 | 0 | 0/1 flag |
| customer_sentiment | str | 0 | e.g. very_negative, neutral |
| csat_score | int64 | 0 | numeric satisfaction score |
| has_attachment | int64 | 0 | 0/1 flag |
| platform | str | 0 | e.g. android, web |
| region | str | **19,997** | ~20% missing |

## Limitations

- **No explicit root-cause field.** `resolution_summary` doubles as the closest proxy but is free text and missing on ~40% of tickets (tickets with `status` values like `closed_no_action` or `in_progress` appear to lack a resolution). IncidentLens will need to treat "historical resolution" as sourced from `resolution_summary` where present, and clearly show "no recorded resolution" otherwise — never infer one.
- `region` is missing on ~20% of rows; any region-based trend analysis (Stage 12/13) must account for this.
- This is a **synthetic** dataset — patterns reflect the generation process, not necessarily real-world incident distributions. This will be stated explicitly in the final report's limitations section.
- `issue_type` is the closest existing field to an "incident category" and is the natural candidate label for the Stage 15 classification task — to be finalized when we get there, not assumed now.

## Attribution statement (for README/report use)

> Contains data from "IT Support Tickets" by ahsanneural, sourced from Kaggle (https://www.kaggle.com/datasets/ahsanneural/synthetic-it-support-tickets), licensed under CC BY 4.0. No modifications have been made to the raw data; derived/processed versions are noted where applicable.
