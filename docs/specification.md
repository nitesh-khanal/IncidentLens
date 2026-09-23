# IncidentLens: An Intelligent IT Incident Similarity and Historical Resolution Analysis System

## Abstract

IncidentLens is a decision-support system for IT operations teams. Given a new incident description, it retrieves historically similar incidents using text-based similarity, surfaces their recorded root causes and resolutions, and highlights recurring incident patterns through clustering and exploratory analysis. A supervised classifier predicts incident category as a secondary, clearly-labeled signal. The system is built entirely with free, open-source, locally-runnable tools, and is designed to make the boundary between observed data, model inference, and historical evidence explicit at every step — it never claims to determine a definitive root cause.

## Introduction

IT support teams repeatedly encounter incidents that resemble past ones, but the knowledge of "what this was" and "what fixed it" is usually locked in unsearchable ticket text, tribal memory, or scattered documentation. IncidentLens applies NLP and classical machine learning to make that historical knowledge retrievable on demand.

## Problem Statement

When a new IT incident occurs, responders often cannot quickly determine whether a similar incident has happened before, what its root cause was, and how it was resolved — because historical incident data is unstructured, siloed, and not searchable by meaning rather than exact keywords.

## Motivation

- Faster mean-time-to-resolution by surfacing prior fixes.
- Reduced duplicated diagnostic effort across a support team.
- A realistic, well-scoped application of NLP/ML skills.
- A portfolio piece that demonstrates a full pipeline (data → NLP → retrieval → ML → dashboard → tests), not just a notebook.

## Main Objective

Build a working, tested, documented Python system that retrieves historically similar IT incidents for a new incident and presents their causes, resolutions, and recurring patterns as evidence-based, non-definitive analytical support.

## Specific Objectives

1. Acquire and document a real, verified incident dataset.
2. Build a reproducible cleaning and NLP preprocessing pipeline.
3. Implement a TF-IDF + cosine-similarity retrieval engine.
4. Evaluate retrieval quality with Precision@K / Recall@K on a controlled test set.
5. Discover recurring incident patterns via clustering.
6. Define and implement one justified supervised ML task (incident category classification), evaluated properly.
7. Provide explainability for both similarity results and classifier predictions.
8. Deliver an interactive Streamlit dashboard and a report generator.
9. Cover the system with automated tests (data, NLP, similarity, clustering, ML, app).
10. Maintain a clean, meaningful Git history from `git init` to a tagged `v1.0.0` release.

## Scope

**In scope:** text-based incident similarity retrieval, historical root-cause/resolution display, clustering-based pattern discovery, one supervised classification task, EDA, Streamlit dashboard, report generation, automated testing, documentation.

**Out of scope (unless reached as optional stages):** transformer/embedding-based semantic search, real-time ticketing-system integration, multi-user auth, cloud-hosted production deployment, paid-API-backed reasoning.

## Functional Requirements

- FR1: Load and validate incident data from the chosen dataset format(s).
- FR2: Clean and normalize incident text and metadata without mutating raw data.
- FR3: Preprocess incident text for NLP (tokenize, normalize, optionally stem/lemmatize — each choice justified experimentally).
- FR4: Represent incidents as TF-IDF vectors.
- FR5: Given a new incident description, return the top-N most similar historical incidents ranked by cosine similarity.
- FR6: Display root cause and resolution for each retrieved incident where available.
- FR7: Identify and display recurring incident clusters/patterns.
- FR8: Predict incident category via a trained classifier, with an evaluation report.
- FR9: Explain why incidents were considered similar (shared terms, category, cluster, etc.).
- FR10: Provide a dashboard for search, results, patterns, trends, and ML performance.
- FR11: Generate a structured analysis report per query.
- FR12: Validate and safely reject malformed/invalid/oversized input.

## Non-Functional Requirements

- Runs locally on a standard laptop (no GPU required).
- Deterministic retrieval results for a fixed dataset/configuration.
- No paid APIs or services required for core functionality.
- Reasonable performance on the primary dataset (measured, not assumed, from Stage 23 onward).
- Test coverage for each core module.
- Clear separation of raw vs. processed data.
- No execution of uploaded/incident-supplied code or shell content.

## Assumptions

- The primary dataset is text-rich enough (description, root cause, resolution fields) to support meaningful similarity retrieval.
- A single machine, single user, offline-capable environment is sufficient for the certification scope.
- Traditional NLP/ML (TF-IDF, cosine similarity, classical classifiers) is sufficient to meet the objectives without transformer models.

## Limitations

- TF-IDF captures lexical overlap, not deep semantic meaning — synonym-heavy paraphrases may rank lower than expected.
- Retrieval quality depends entirely on how representative and clean the historical dataset is.
- The classifier's usefulness is bounded by the label quality and class balance of the primary dataset (to be confirmed, not assumed, once inspected).
- The system supports, not replaces, human root-cause analysis.

## Methodology

Iterative, stage-gated development: data acquisition → cleaning → EDA → NLP preprocessing → TF-IDF representation → similarity engine → retrieval evaluation → clustering → trend analysis → feature engineering → supervised ML → model evaluation → integration into an "intelligence engine" → explainability → dashboard → reporting → comprehensive testing → performance measurement → security hardening → optional enhancements → documentation → final clean-environment verification.

## Technology Stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, NLTK and/or spaCy (as justified), Streamlit, pytest, Git/GitHub. No paid APIs.

## Architecture

Target structure (introduced incrementally, changed only when technically justified):

```
IncidentLens/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_nlp_preprocessing.ipynb
│   ├── 04_similarity_analysis.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_ml_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── nlp_processor.py
│   ├── vectorizer.py
│   ├── similarity_engine.py
│   ├── clustering.py
│   ├── feature_engineering.py
│   ├── classifier.py
│   ├── analyzer.py
│   ├── visualizer.py
│   └── report_generator.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_nlp_processor.py
│   ├── test_similarity.py
│   ├── test_clustering.py
│   └── test_classifier.py
│
├── reports/
├── docs/
│
├── app.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── .gitignore
└── LICENSE
```

## Expected Outputs

- A working Streamlit application.
- A documented, reproducible retrieval + clustering + classification pipeline.
- An evaluation report with real (post-execution) metrics.
- Automated test suite with a results table.
- Final documentation, certification report, presentation, and viva prep — all built from actual results.

## MVP (Minimum Viable Product)

A single developer can, from a clean environment:

1. Load the primary dataset.
2. Clean and NLP-preprocess incident text.
3. Represent incidents with TF-IDF.
4. Enter a new incident description and retrieve the top-N most similar historical incidents via cosine similarity, with their root causes and resolutions shown.
5. See this working through a minimal Streamlit interface.

Everything beyond this (clustering, classification, trend analysis, event-log analytics, explainability UI, reporting, optional log analysis, portfolio engineering) is a planned enhancement layered on top of a working MVP, not a prerequisite for it.

## Future Enhancements (post-v1.0.0, optional)

- Sentence-embedding-based semantic similarity (still free/local, e.g. sentence-transformers) as a comparison against TF-IDF.
- LogHub-based operational context linking.
- FastAPI service layer, containerization, CI, cloud deployment — each evaluated on merit, not added by default.
