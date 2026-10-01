"""Evaluate a frozen exploratory challenge set against the running artifacts."""
import json
from pathlib import Path
import joblib
import pandas as pd
from src.config import DATA_PROCESSED_DIR, PROJECT_ROOT
from src.classifier import PRIMARY_MODEL
from src.vectorizer import IncidentVectorizer
from src.similarity_engine import SimilarityEngine
from src.retrieval import RetrievalEngine
from src.analyzer import IncidentIntelligenceEngine
from src.evaluation import precision_at_k


def measure():
    cases=json.loads((PROJECT_ROOT/'data/test/challenge_set.json').read_text())['cases']
    df=pd.read_csv(DATA_PROCESSED_DIR/'tickets_clustered.csv')
    vec=IncidentVectorizer.load(DATA_PROCESSED_DIR/'tfidf_vectorizer.joblib')
    matrix=joblib.load(DATA_PROCESSED_DIR/'tfidf_matrix.joblib')
    clusterer=joblib.load(DATA_PROCESSED_DIR/'clusterer.joblib')
    engine=IncidentIntelligenceEngine(RetrievalEngine(SimilarityEngine(vec,matrix,df)),clusterer,
        clusterer.top_terms_per_cluster(matrix,vec.get_feature_names()),
        joblib.load(DATA_PROCESSED_DIR/'models'/f'{PRIMARY_MODEL}.joblib'),
        joblib.load(DATA_PROCESSED_DIR/'feature_builder.joblib'),vec,df)
    outcomes=[]
    for case in cases:
        r=engine.analyze(case['query']);c=r['classification'];expected=case['expected_issue_type']
        accepted=bool(c and c.get('supported',True))
        outcomes.append({**case,'predicted':c['predicted_issue_type'] if c else None,
            'accepted':accepted, 'matched_terms':c['matched_vocabulary_terms'] if c else 0,
            'confidence':c['confidence'] if c else None,
            'precision_at_1':precision_at_k([x['issue_type'] for x in r['similar_incidents']],expected,1),
            'precision_at_5':precision_at_k([x['issue_type'] for x in r['similar_incidents']],expected,5)})
    labelled=[r for r in outcomes if r['expected_issue_type'] is not None]
    accepted=[r for r in labelled if r['accepted']]
    boundaries=[r for r in outcomes if r['expected_issue_type'] is None]
    return {'primary_model':PRIMARY_MODEL,'provenance':'Assistant-authored exploratory challenge set; not independent real-world validation.',
        'labelled_cases':len(labelled),'boundary_cases':len(boundaries),
        'raw_classifier_accuracy':sum(r['predicted']==r['expected_issue_type'] for r in labelled)/len(labelled),
        'coverage':len(accepted)/len(labelled),
        'accepted_accuracy':sum(r['predicted']==r['expected_issue_type'] for r in accepted)/len(accepted) if accepted else None,
        'boundary_abstentions':sum(not r['accepted'] for r in boundaries),
        'retrieval_precision_at_1':sum(r['precision_at_1'] for r in labelled)/len(labelled),
        'retrieval_precision_at_5':sum(r['precision_at_5'] for r in labelled)/len(labelled),'cases':outcomes}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/challenge_results.json')
    args=parser.parse_args();result=measure();path=PROJECT_ROOT/args.output
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
