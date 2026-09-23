"""
IncidentLens — data cleaning / preprocessing pipeline.

Transforms raw ticket data into a cleaned, analysis-ready form.
Never mutates raw data in place — always returns a new DataFrame.
"""
import pandas as pd

STRING_COLUMNS = [
    "ticket_id", "customer_id", "customer_segment", "channel",
    "product_area", "issue_type", "priority", "status", "sla_plan",
    "initial_message", "agent_first_reply", "resolution_summary",
    "customer_sentiment", "platform", "region",
]

CATEGORICAL_COLUMNS = [
    "customer_segment", "channel", "product_area", "issue_type",
    "priority", "status", "sla_plan", "customer_sentiment",
    "platform", "region",
]


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw ticket data. Does not mutate the input DataFrame.

    Steps: strip whitespace, normalize categorical casing, parse
    timestamps, fill non-critical missing values explicitly, flag
    (never fabricate) missing resolutions, remove duplicate ticket_ids.
    """
    df = df.copy()

    # 1. Strip whitespace from all text/categorical fields
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # 2. Normalize categorical casing (defensive — data already looks
    #    consistent, but guards against future dataset revisions)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].str.lower()

    # 3. Parse timestamps; invalid ones become NaT, not a crash
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # 4. region missing ~20% (docs/dataset.md) — explicit fill, not drop
    if "region" in df.columns:
        df["region"] = df["region"].fillna("unknown")

    # 5. Flag empty/whitespace-only messages rather than drop them
    if "initial_message" in df.columns:
        df["has_valid_message"] = df["initial_message"].str.len().fillna(0) > 0

    # 6. Missing resolution is a real fact, not filled in — just flagged
    if "resolution_summary" in df.columns:
        df["has_resolution"] = df["resolution_summary"].notna()

    # 7. Remove duplicate ticket_id rows, keep first occurrence
    if "ticket_id" in df.columns:
        df = df.drop_duplicates(subset=["ticket_id"], keep="first")

    return df
