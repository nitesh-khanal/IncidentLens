"""
Stage 6 — controlled NLP preprocessing experiment.

Runs several preprocessing configurations against a sample of the real
dataset and reports measurable differences (token counts, vocabulary
size, zero-token messages) — evidence for Stage 7's pipeline choice,
not an assumption made in advance.
"""
from collections import Counter

from src.data_loader import load_primary_dataset
from src.preprocessing import clean_tickets
from src.nlp_processor import preprocess_text, ensure_nltk_data

CONFIGS = {
    "lowercase_only": dict(lowercase=True, remove_punctuation=False),
    "punct_removed": dict(lowercase=True, remove_punctuation=True),
    "punct_stopwords": dict(lowercase=True, remove_punctuation=True, remove_stopwords=True),
    "punct_stopwords_lemma": dict(
        lowercase=True, remove_punctuation=True, remove_stopwords=True, lemmatize=True
    ),
    "punct_stopwords_stem": dict(
        lowercase=True, remove_punctuation=True, remove_stopwords=True, stem=True
    ),
}


def main():
    ensure_nltk_data()

    raw = load_primary_dataset()
    df = clean_tickets(raw)

    sample = df["initial_message"].dropna().head(2000)
    print(f"Running on a sample of {len(sample)} incident descriptions\n")
    print("=" * 70)

    for name, kwargs in CONFIGS.items():
        processed = sample.apply(lambda t: preprocess_text(t, **kwargs))
        token_counts = processed.str.split().apply(len)
        vocab = Counter()
        for text in processed:
            vocab.update(text.split())

        print(f"\nConfig: {name}")
        print(f"  Avg tokens per message: {token_counts.mean():.2f}")
        print(f"  Min tokens: {token_counts.min()}  |  Zero-token messages: {(token_counts == 0).sum()}")
        print(f"  Vocabulary size (unique tokens): {len(vocab)}")

    print("\n" + "=" * 70)
    print("\nSame 3 incidents under each config:\n")
    for i, text in enumerate(sample.head(3)):
        print(f"\n--- Incident {i + 1} (raw) ---")
        print(text)
        for name, kwargs in CONFIGS.items():
            print(f"  [{name}]: {preprocess_text(text, **kwargs)}")


if __name__ == "__main__":
    main()
