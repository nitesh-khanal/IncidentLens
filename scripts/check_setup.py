"""Offline preflight: shipped file integrity, runtime versions and model loading."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_files(root=ROOT):
    manifest = json.loads((root / 'runtime-manifest.json').read_text())
    failures = []
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file():
            failures.append(f'missing {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f'changed or incomplete {name}')
    if failures:
        raise ValueError('Shipped data/models are incomplete: ' + '; '.join(failures)
                         + '. Download a complete clone/ZIP, or regenerate the manifest after intentionally rebuilding artifacts.')
    return manifest


def main():
    import importlib.metadata
    try:
        manifest = verify_files()
        if sys.version_info[:2] != (3, 12):
            raise ValueError('Use Python 3.12 with the shipped model artifacts')
        mismatch = [f'{name}: expected {version}, found {importlib.metadata.version(name)}'
                    for name, version in manifest['packages'].items()
                    if importlib.metadata.version(name) != version]
        if mismatch:
            raise ValueError('Runtime package mismatch: ' + '; '.join(mismatch) + '. Run python run.py to install the pinned environment.')
        from src.nlp_processor import ensure_nltk_data, default_preprocess
        ensure_nltk_data()
        assert default_preprocess('I cannot log in; my password is incorrect.')
        import joblib
        import pandas as pd
        from src.vectorizer import IncidentVectorizer
        from src.analyzer import DEFAULT_METADATA
        from src.classifier import MODEL_REGISTRY
        data = ROOT / 'data' / 'processed'
        df = pd.read_csv(data / 'tickets_clustered.csv')
        vec = IncidentVectorizer.load(data / 'tfidf_vectorizer.joblib')
        matrix = joblib.load(data / 'tfidf_matrix.joblib')
        if matrix.shape != (len(df), vec.vocabulary_size()):
            raise ValueError('Historical rows and vector dimensions do not match')
        builder = joblib.load(data / 'feature_builder.joblib')
        row = pd.DataFrame([{'initial_message': 'My subscription was charged twice.',
                            'created_at': '2026-01-01T12:00:00+00:00', **DEFAULT_METADATA}])
        vector = vec.transform([default_preprocess(row.iloc[0]['initial_message'])])
        X = builder.transform(row, vector)
        for name in MODEL_REGISTRY:
            model = joblib.load(data / 'models' / f'{name}.joblib')
            model.predict(X)
        clusterer = joblib.load(data / 'clusterer.joblib')
        clusterer.predict(vector)
        print(f'Ready: {len(df):,} historical tickets, all three models, clusters, and bundled English resources. No account, API key, or runtime download required.')
        return 0
    except (OSError, ValueError, LookupError, ImportError) as exc:
        print(f'Preflight failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
