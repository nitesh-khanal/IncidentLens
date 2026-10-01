"""Rebuild consistent artifacts and evaluate from train-only features.

Run from project root: python -m scripts.rebuild_validated_demo
No challenge-set query or label is used in fitting any component.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT
from src.validation import prepare_texts, fit_features, evaluate_split, split_template_indices, grouped_validation
from src.classifier import MODEL_REGISTRY, train_model
from src.clustering import IncidentClusterer
from src.nlp_processor import ensure_nltk_data


def main(evaluate_only=False):
    ensure_nltk_data()
    df=pd.read_csv(DATA_PROCESSED_DIR/'tickets_clustered.csv')
    print('Evaluate row split with train-only preprocessing',flush=True)
    train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=0.2,random_state=42,stratify=df['issue_type'])
    row=evaluate_split(df,train_idx,test_idx)
    print('Evaluate unseen-template split with train-only preprocessing',flush=True)
    train_idx,test_idx=split_template_indices(df)
    holdout=evaluate_split(df,train_idx,test_idx)
    folds=grouped_validation(df,folds=3)
    summary={'methodology':'Raw rows split before fitting TF-IDF/IDF and categorical encoders. Grouped tests share zero normalized templates. Challenge set excluded from training.',
        'row_split':row,'template_holdout':holdout,'grouped_folds':folds,
        'grouped_summary':{name:{'mean_accuracy':float(np.mean([f['models'][name]['accuracy'] for f in folds])),
            'std_accuracy':float(np.std([f['models'][name]['accuracy'] for f in folds])),
            'mean_macro_f1':float(np.mean([f['models'][name]['macro_f1'] for f in folds]))} for name in MODEL_REGISTRY}}
    reports=PROJECT_ROOT/'reports';reports.mkdir(exist_ok=True)
    if evaluate_only and (reports/'validated_evaluation.json').exists():
        previous=json.loads((reports/'validated_evaluation.json').read_text())
        if 'deployment' in previous:
            summary['deployment']=previous['deployment']
    summary['selected_primary_model']=max(summary['grouped_summary'],key=lambda name: summary['grouped_summary'][name]['mean_macro_f1'])
    (reports/'validated_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    if evaluate_only:
        return
    print('Fit deployment components on the historical corpus after evaluation',flush=True)
    vec,builder,X,_=fit_features(df,df.iloc[:1])
    df['processed_message']=prepare_texts(df)
    matrix=vec.transform(df['processed_message'])
    clusterer=IncidentClusterer(n_clusters=8)
    df['cluster']=clusterer.fit_predict(matrix)
    rng=np.random.RandomState(42);indices=rng.choice(len(df),min(5000,len(df)),replace=False)
    silhouette=float(silhouette_score(matrix[indices],df['cluster'].to_numpy()[indices]))
    summary['deployment']={'rows':len(df),'vocabulary_size':vec.vocabulary_size(),
        'features':X.shape[1],'silhouette_k8':silhouette,'cluster_sizes':{str(k):int(v) for k,v in df['cluster'].value_counts().sort_index().items()},
        'fit_scope':'Full historical corpus after separate evaluation; no challenge cases.'}
    (reports/'validated_evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    vec.save(DATA_PROCESSED_DIR/'tfidf_vectorizer.joblib')
    joblib.dump(matrix,DATA_PROCESSED_DIR/'tfidf_matrix.joblib')
    joblib.dump(builder,DATA_PROCESSED_DIR/'feature_builder.joblib')
    joblib.dump(clusterer,DATA_PROCESSED_DIR/'clusterer.joblib')
    joblib.dump(X,DATA_PROCESSED_DIR/'feature_matrix.joblib')
    df.to_csv(DATA_PROCESSED_DIR/'tickets_clustered.csv',index=False)
    df.drop(columns='cluster').to_csv(DATA_PROCESSED_DIR/'tickets_nlp.csv',index=False)
    (DATA_PROCESSED_DIR/'models').mkdir(exist_ok=True)
    for name in MODEL_REGISTRY:
        model=train_model(name,X,df['issue_type'].astype(str).to_numpy())
        joblib.dump(model,DATA_PROCESSED_DIR/'models'/f'{name}.joblib')
    from scripts.measure_challenge import measure
    challenge=measure();(reports/'challenge_results.json').write_text(json.dumps(challenge,indent=2)+'\n')
    from scripts.write_validation_report import main as write_report
    write_report()
    print(json.dumps({'template_holdout':holdout['models'],'grouped_summary':summary['grouped_summary'],
        'deployment':summary['deployment'],'challenge':{k:v for k,v in challenge.items() if k!='cases'}},indent=2),flush=True)

if __name__=='__main__':
    main()
