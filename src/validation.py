"""Train-only preprocessing and grouped validation from raw ticket rows.

Evaluation never loads precomputed whole-corpus features or deployed models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from src.classifier import train_model, MODEL_REGISTRY
from src.feature_engineering import FeatureBuilder
from src.nlp_processor import default_preprocess
from src.vectorizer import IncidentVectorizer


def prepare_texts(df):
    texts=df['initial_message'].fillna('').astype(str)
    cache={text:default_preprocess(text) for text in texts.unique()}
    return texts.map(cache)


def split_template_indices(df, test_frac=0.2, random_state=42):
    if not 0 < test_frac < 1:
        raise ValueError('test_frac must be between 0 and 1')
    groups=prepare_texts(df)
    templates=np.asarray(groups.unique(), dtype=object).copy()
    if len(templates)<2:
        raise ValueError('At least two distinct templates are required')
    np.random.RandomState(random_state).shuffle(templates)
    count=min(len(templates)-1,max(1,int(len(templates)*test_frac)))
    test=groups.isin(set(templates[:count])).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def fit_features(train_df, test_df):
    """Fit TF-IDF/IDF and the categorical encoder on training rows only."""
    vectorizer=IncidentVectorizer()
    train_matrix=vectorizer.fit_transform(prepare_texts(train_df))
    test_matrix=vectorizer.transform(prepare_texts(test_df))
    builder=FeatureBuilder()
    X_train=builder.fit_transform(train_df,train_matrix)
    X_test=builder.transform(test_df,test_matrix)
    return vectorizer,builder,X_train,X_test


def evaluate_split(df, train_idx, test_idx):
    train_df=df.iloc[train_idx];test_df=df.iloc[test_idx]
    vectorizer,builder,X_train,X_test=fit_features(train_df,test_df)
    y_train=train_df['issue_type'].astype(str).to_numpy()
    y_test=test_df['issue_type'].astype(str).to_numpy()
    metrics={}
    for name in MODEL_REGISTRY:
        model=train_model(name,X_train,y_train)
        predicted=model.predict(X_test)
        labels=sorted(set(y_train)|set(y_test))
        metrics[name]={'accuracy':float(accuracy_score(y_test,predicted)),
            'macro_f1':float(f1_score(y_test,predicted,labels=labels,average='macro',zero_division=0)),
            'per_category':{label:{'cases':int(sum(y_test==label)),
                'accuracy':float(np.mean(predicted[y_test==label]==label)) if sum(y_test==label) else None}
                for label in labels}}
    train_groups=set(prepare_texts(train_df));test_groups=set(prepare_texts(test_df))
    return {'train_rows':len(train_df),'test_rows':len(test_df),
        'train_templates':len(train_groups),'test_templates':len(test_groups),
        'template_overlap':len(train_groups&test_groups),
        'vocabulary_size':vectorizer.vocabulary_size(),'models':metrics}


def grouped_validation(df, folds=3):
    groups=prepare_texts(df)
    y=df['issue_type'].astype(str).to_numpy()
    splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=42)
    reports=[]
    for number,(train_idx,test_idx) in enumerate(splitter.split(df,y,groups),1):
        print(f'Grouped fold {number}/{folds}',flush=True)
        reports.append(evaluate_split(df,train_idx,test_idx))
    return reports
