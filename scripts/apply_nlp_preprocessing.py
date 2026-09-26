"""
Stage 6 (final step) — apply the chosen NLP config to the full dataset.

Preserves the original initial_message untouched; adds a new
processed_message column. Saves as a new file so tickets_clean.csv
(Stage 4's output) stays as-is.
"""
from src.data_loader import load_primary_dataset
from src.preprocessing import clean_tickets
from src.nlp_processor import default_preprocess, ensure_nltk_data, DEFAULT_CONFIG
from src.config import DATA_PROCESSED_DIR

OUTPUT_PATH = DATA_PROCESSED_DIR / "tickets_nlp.csv"


def main():
    ensure_nltk_data()

    raw = load_primary_dataset()
    df = clean_tickets(raw)

    print(f"Applying config {DEFAULT_CONFIG} to {len(df)} descriptions...")
    df["processed_message"] = df["initial_message"].apply(default_preprocess)

    empty_after = (df["processed_message"].str.len() == 0).sum()
    print(f"Messages that became empty after processing: {empty_after}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
