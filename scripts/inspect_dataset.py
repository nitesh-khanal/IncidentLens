"""
Stage 2 — one-off dataset inspection script.
Run this after downloading the dataset into data/raw/.
Reports REAL statistics only — nothing here is assumed.
"""
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RAW_DIR}. Did the download succeed?")
        sys.exit(1)

    for path in csv_files:
        print("=" * 70)
        print(f"File: {path.name}")
        print("=" * 70)

        df = pd.read_csv(path)

        print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n")

        print("Columns and dtypes:")
        print(df.dtypes)

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

        print("\nMissing values per column:")
        print(df.isna().sum())

        print(f"\nExact duplicate rows: {df.duplicated().sum()}")

        print("\nMemory usage:")
        print(df.memory_usage(deep=True).sum() / (1024 * 1024), "MB")


if __name__ == "__main__":
    main()
