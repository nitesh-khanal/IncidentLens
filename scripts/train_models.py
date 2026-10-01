"""Fit deployment classifiers on the historical corpus.

In-sample accuracy is diagnostic only. Evaluate raw-row splits separately via
scripts.evaluate_models; it never loads these deployment classifiers/features.
Use scripts.rebuild_validated_demo to regenerate a consistent artifact bundle.
"""
import time
import joblib
import pandas as pd
from src.classifier import MODEL_REGISTRY, train_model
from src.config import DATA_PROCESSED_DIR


def main():
    df=pd.read_csv(DATA_PROCESSED_DIR/'tickets_nlp.csv')
    X=joblib.load(DATA_PROCESSED_DIR/'feature_matrix.joblib')
    y=df['issue_type'].astype(str).to_numpy()
    if X.shape[0]!=len(df):
        raise ValueError('Feature rows and historical ticket rows do not match')
    destination=DATA_PROCESSED_DIR/'models';destination.mkdir(exist_ok=True)
    for name in MODEL_REGISTRY:
        started=time.perf_counter()
        model=train_model(name,X,y)
        joblib.dump(model,destination/f'{name}.joblib')
        print(f'{name}: fitted {len(df):,} historical rows in {time.perf_counter()-started:.2f}s; in-sample accuracy {model.score(X,y):.4f} (not validation).',flush=True)
    print('Use python -m scripts.evaluate_models for train-only, grouped validation.')

if __name__=='__main__':
    main()
