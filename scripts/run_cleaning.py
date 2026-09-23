"""
Stage 4 — run the cleaning pipeline against the real primary dataset,
save the cleaned output to data/processed/, and write a real data
quality report to reports/data_quality_report.md. All numbers here
are measured by actually running this script, never invented.
"""
from datetime import datetime, timezone

from src.data_loader import load_primary_dataset
from src.preprocessing import clean_tickets
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_report.md"
OUTPUT_PATH = DATA_PROCESSED_DIR / "tickets_clean.csv"


def main():
    raw = load_primary_dataset()
    clean = clean_tickets(raw)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_PATH, index=False)

    invalid_messages = int((~clean["has_valid_message"]).sum())
    missing_resolution = int((~clean["has_resolution"]).sum())
    invalid_timestamps = int(clean["created_at"].isna().sum())
    duplicates_removed = len(raw) - len(clean)

    report = f"""# Data Quality Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Row counts
- Raw rows: {len(raw)}
- Cleaned rows: {len(clean)}
- Duplicate ticket_id rows removed: {duplicates_removed}

## Field-level findings
- Rows with empty/whitespace-only `initial_message`: {invalid_messages}
- Rows with no recorded resolution (`has_resolution` = False): {missing_resolution}
- Rows where `created_at` failed to parse: {invalid_timestamps}

## Transformations applied
- Whitespace stripped from all text/categorical fields
- Categorical fields normalized to lowercase
- `created_at` parsed to UTC datetime (unparseable -> NaT, not dropped)
- `region` missing values filled with "unknown" (documented in docs/dataset.md)
- `resolution_summary` missing values left as-is (NOT fabricated) — flagged via `has_resolution`
- Duplicate `ticket_id` rows removed (kept first occurrence)

## Notes
- Raw data at data/raw/ was not modified.
- Cleaned data written to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
