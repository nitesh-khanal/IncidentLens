import pandas as pd
import pytest
from src.validation import fit_features, split_template_indices, prepare_texts


def rows(texts,channels=None):
    return pd.DataFrame({'initial_message':texts,'created_at':['2024-01-01']*len(texts),
        'customer_segment':['individual']*len(texts),'channel':channels or ['email']*len(texts),
        'product_area':['billing']*len(texts),'priority':['low']*len(texts),
        'sla_plan':['standard']*len(texts),'platform':['web']*len(texts),
        'region':['eu']*len(texts),'has_attachment':[0]*len(texts),
        'issue_type':['billing_problem']*len(texts)})


def test_holdout_only_words_and_categories_are_not_learned():
    train=rows(['billing invoice charged','invoice subscription payment'])
    test=rows(['quasar supernova'],channels=['never-seen-channel'])
    vec,builder,X_train,X_test=fit_features(train,test)
    assert 'quasar' not in vec.get_feature_names()
    assert 'supernova' not in vec.get_feature_names()
    assert 'never-seen-channel' not in builder.encoder.categories_[1]
    assert X_train.shape[1]==X_test.shape[1]
    assert vec.transform(prepare_texts(test)).nnz==0


def test_duplicate_and_normalized_templates_never_cross_split():
    df=rows(['cannot log in password incorrect','cannot login password incorrect',
        'invoice charged twice','invoice charged twice','page loading slow','page loading slow'])
    train,test=split_template_indices(df,test_frac=0.34)
    groups=prepare_texts(df)
    assert not set(groups.iloc[train]) & set(groups.iloc[test])
    assert len(train)+len(test)==len(df)


def test_single_template_cannot_be_evaluated_as_unseen_templates():
    with pytest.raises(ValueError,match='two distinct'):
        split_template_indices(rows(['billing invoice','billing invoice']))
