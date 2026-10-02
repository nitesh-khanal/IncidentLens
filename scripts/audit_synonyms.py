"""Verify reviewed WordNet aliases against the historical description vocabulary.

Run offline: python -m scripts.audit_synonyms
This audits the frozen allowlist; it never automatically accepts new synonyms.
"""
import pandas as pd
from nltk.corpus import wordnet
from src.config import DATA_PROCESSED_DIR
from src.nlp_processor import default_preprocess, ensure_nltk_data
from src.synonyms import REVIEWED_WORDNET


def main():
    ensure_nltk_data()
    df = pd.read_csv(DATA_PROCESSED_DIR / 'tickets_clustered.csv',
                     usecols=['initial_message', 'processed_message'])
    descriptions = df.drop_duplicates('initial_message')
    cleaned = descriptions.initial_message.map(default_preprocess)
    vocabulary = set(' '.join(cleaned).split())
    for variant, (canonical, sense) in REVIEWED_WORDNET.items():
        assert canonical in vocabulary, f'{canonical} is outside the historical vocabulary'
        assert {variant, canonical} <= set(wordnet.synset(sense).lemma_names())
        print(f'{variant} -> {canonical} (reviewed WordNet sense {sense})')
    assert cleaned.tolist() == descriptions.processed_message.tolist(), 'Historical preprocessing changed: rebuild artifacts before release'
    print(f'{cleaned.nunique()} historical templates unchanged; deployed features remain consistent.')


if __name__ == '__main__':
    main()
