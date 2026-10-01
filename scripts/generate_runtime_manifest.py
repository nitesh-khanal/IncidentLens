"""Record exact runtime packages and hashes after deliberately rebuilding data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [ROOT / 'data' / 'processed' / name for name in
             ['tickets_clustered.csv', 'tfidf_vectorizer.joblib', 'tfidf_matrix.joblib',
              'clusterer.joblib', 'feature_builder.joblib', 'models/logistic_regression.joblib',
              'models/decision_tree.joblib', 'models/random_forest.joblib']]
    files += [ROOT / 'data' / 'nltk' / name for name in
              ['corpora/wordnet.zip', 'corpora/stopwords/english',
               *['tokenizers/punkt_tab/english/' + name for name in
                 ['abbrev_types.txt', 'collocations.tab', 'ortho_context.tab', 'sent_starters.txt']]]]
    packages = dict(line.split('==', 1) for line in (ROOT / 'requirements.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#'))
    result = {'schema_version': 1, 'python': '3.12', 'packages': packages,
              'files': {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in files}}
    (ROOT / 'runtime-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Runtime manifest refreshed.')


if __name__ == '__main__':
    main()
