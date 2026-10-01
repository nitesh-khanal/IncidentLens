"""Evaluate raw rows with train-only preprocessing and grouped splits.

This deliberately does not load the precomputed whole-corpus feature matrix.
The deployment artifacts are preserved; results go to validated_evaluation.json.
Run scripts.write_validation_report afterwards to refresh the readable report.
"""
from scripts.rebuild_validated_demo import main as evaluate


def main():
    evaluate(evaluate_only=True)
    from scripts.write_validation_report import main as write_report
    write_report()


if __name__ == '__main__':
    main()
