"""Write readable reports from measured validation results, never fixed scores."""
import json
from pathlib import Path
from src.config import PROJECT_ROOT


def main():
    reports=PROJECT_ROOT/'reports'
    summary=json.loads((reports/'validated_evaluation.json').read_text())
    lines=['# Model evaluation — train-only preprocessing','',summary['methodology'],'',
        'The corpus is synthetic and templated. These results do not establish production accuracy.',
        'The legacy run used full-corpus precomputed features. Its accuracy happens to match the corrected holdout here; the corrected metrics below supersede that run.','']
    for title,key in [('Row split (repeated templates may cross the split)','row_split'),('Unseen-template holdout','template_holdout')]:
        result=summary[key];lines += [f'## {title}','',f"Training rows: {result['train_rows']:,}; test rows: {result['test_rows']:,}.",
            f"Training templates: {result['train_templates']}; test templates: {result['test_templates']}; shared templates: {result['template_overlap']}.",
            f"Vocabulary fitted on training rows: {result['vocabulary_size']} terms.",'']
        for name,metrics in result['models'].items():
            lines += [f"- {name}: accuracy {metrics['accuracy']:.2%}; macro F1 {metrics['macro_f1']:.4f}."]
    lines += ['','## Three-fold grouped validation','',
        'Every fold refits both TF-IDF and the categorical encoder on its training rows. Normalized descriptions form groups. Small categories have only three templates, so fold results can vary substantially.','']
    for name,metrics in summary['grouped_summary'].items():
        lines += [f"- {name}: mean accuracy {metrics['mean_accuracy']:.2%}; standard deviation {metrics['std_accuracy']:.2%}; mean macro F1 {metrics['mean_macro_f1']:.4f}."]
    lines += ['',f"Primary model: {summary.get('selected_primary_model', 'logistic_regression')}, selected by highest grouped mean macro F1.",'','## Provenance and scope','',
        'The JSON companion contains fold-level counts and per-category accuracy. Deployed components are fitted on the full historical corpus only after evaluation; they are not the held-out evaluation models.',
        'No challenge-set query or label is used to fit vocabulary, encoders, clusters, or models.','']
    (reports/'model_evaluation.md').write_text('\n'.join(lines))
    if (reports/'challenge_results.json').exists():
        old=json.loads((reports/'challenge_baseline.json').read_text());new=json.loads((reports/'challenge_results.json').read_text())
        lines=['# Exploratory challenge evaluation','',new['provenance'],'',
            '48 category-labelled cases and 6 boundary cases. Labels were assigned by the assistant and have not been independently reviewed. This set is excluded from fitting but is not a blind, independent test; changes address known development failures. Treat the results as exploratory and collect human-reviewed real tickets before making generalization claims.','']
        for name,r in [('Before improvements',old),('After improvements',new)]:
            lines += [f'## {name}','',f"- Raw classifier accuracy: {r['raw_classifier_accuracy']:.2%}.",
                f"- Category suggestion coverage: {r['coverage']:.2%}.",
                f"- Accuracy among suggested cases: {r['accepted_accuracy']:.2%}." if r['accepted_accuracy'] is not None else '- No supported category suggestions.',
                f"- Boundary cases withheld: {r['boundary_abstentions']}/{r['boundary_cases']}.",
                f"- Retrieval category Precision@1: {r['retrieval_precision_at_1']:.2%}.",
                f"- Retrieval category Precision@5: {r['retrieval_precision_at_5']:.2%}.",'']
        lines += ['## Interpretation','',
            'Abstention reduces the number of category suggestions. Accuracy among accepted suggestions must always be read together with coverage; it does not replace overall accuracy.',
            'The evidence gate requires at least three vocabulary terms and a model probability score of at least 0.5. It is a heuristic and is not statistically calibrated.',
            'Similarity relevance here means matching the intended category, not confirming that a recorded resolution is appropriate.',
            'Training scope, primary model selection, normalization, and the evidence gate changed together, so this comparison does not isolate the causal effect of one change.','']
        (reports/'challenge_evaluation.md').write_text('\n'.join(lines))

if __name__=='__main__':
    main()
