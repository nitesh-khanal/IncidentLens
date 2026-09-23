# Data Quality Report

Generated: 2026-09-23T06:52:17.131272+00:00

## Row counts
- Raw rows: 100000
- Cleaned rows: 100000
- Duplicate ticket_id rows removed: 0

## Field-level findings
- Rows with empty/whitespace-only `initial_message`: 0
- Rows with no recorded resolution (`has_resolution` = False): 39887
- Rows where `created_at` failed to parse: 0

## Transformations applied
- Whitespace stripped from all text/categorical fields
- Categorical fields normalized to lowercase
- `created_at` parsed to UTC datetime (unparseable -> NaT, not dropped)
- `region` missing values filled with "unknown" (documented in docs/dataset.md)
- `resolution_summary` missing values left as-is (NOT fabricated) — flagged via `has_resolution`
- Duplicate `ticket_id` rows removed (kept first occurrence)

## Notes
- Raw data at data/raw/ was not modified.
- Cleaned data written to data/processed/tickets_clean.csv
