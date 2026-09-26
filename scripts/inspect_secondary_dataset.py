"""
Stage 13 — inspect the secondary (event log) dataset for real. No schema,
record count, or field meaning is assumed in advance.
"""
import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR


def main():
    csv_files = list(DATA_RAW_DIR.glob("*.csv"))
    # Exclude the primary dataset file, only inspect newly downloaded ones
    csv_files = [f for f in csv_files if f.name != "synthetic_it_support_tickets.csv"]

    if not csv_files:
        print(f"No new CSV files found in {DATA_RAW_DIR}.")
        sys.exit(1)

    for path in csv_files:
        print("=" * 70)
        print(f"File: {path.name}")
        print("=" * 70)

        df = pd.read_csv(path, sep=";", low_memory=False)

        print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")
        print("Columns and dtypes:")
        print(df.dtypes)

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

        print("\nMissing values per column (top 10):")
        print(df.isna().sum().sort_values(ascending=False).head(10))

        # Try to identify the incident/case identifier column
        id_candidates = [c for c in df.columns if "number" in c.lower() or "incident" in c.lower() or c.lower() == "id"]
        print(f"\nLikely case/incident ID column(s): {id_candidates}")
        if id_candidates:
            print(f"Unique values in '{id_candidates[0]}': {df[id_candidates[0]].nunique()}")


if __name__ == "__main__":
    main()
