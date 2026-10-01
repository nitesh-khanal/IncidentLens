# IncidentLens demonstration guide

## Before the audience arrives

1. Use Python 3.12 and install `requirements.txt` in a local virtual environment. Keep the environment downloaded on this device; iCloud placeholders can stall imports.
2. Prepare language resources once: `python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4`.
3. Run `python -m pytest tests/ -q`, then `streamlit run app.py` from the project directory.
4. Open the dashboard and run the Billing example once to warm the artifact cache. Confirm that Reports offers HTML and text downloads.
5. Keep the app open, increase browser zoom if needed, and close unrelated tabs. The demonstration uses committed data and models; no Kaggle download or retraining is needed.

## Five-minute walkthrough

**0:00–0:45 · Problem and dataset.** “Support teams can lose time searching old tickets. IncidentLens finds related historical descriptions and recorded resolutions.” Show incident count and resolution coverage. Say immediately that the dataset is synthetic: 100,000 tickets contain just 96 unique descriptions.

**0:45–2:30 · Analyze an incident.** Navigate to Analyze New Incident. Select Billing · duplicate charge, press Use selected example, then Analyze Incident. These are illustrative queries, not guaranteed predictions. Explain the suggested category, model comparison, and first expanded historical ticket. Point out missing resolutions when present.

**2:30–3:15 · Explain the evidence.** Show the contributing words in Why similar. “TF-IDF cosine similarity measures shared words. A high score does not prove a common cause.” Show Recurring Patterns and explain that clusters group language.

**3:15–4:00 · Evaluate honestly.** On ML Performance, contrast 100% row-split accuracy with 54.47% unseen-template accuracy. Show the grouped validation: preprocessing is fitted on training rows only, and Logistic Regression has the best mean macro F1. Repeated templates make the row split too easy; none of the models establishes production accuracy.

**4:00–4:30 · Show a boundary.** Load Unrecognized text · evidence boundary and analyze it. Explain the explicit insufficient-evidence outcome, absence of historical matches, and absence of a cluster. Weak category suggestions and their statistics are withheld. Optionally try Account access to show unfamiliar wording and possible model disagreement.

**4:30–5:00 · Export.** Analyze Billing again, visit Reports, and download the HTML report. It contains evidence labels, a similarity chart, and limitations, and can be opened without external assets.

## Likely questions

- **Is this semantic search?** No. It uses lexical TF-IDF matching; embedding retrieval is future work.
- **Does it diagnose root causes?** No. Historical resolutions and model suggestions support a human decision.
- **Why three models?** They provide a baseline comparison. Grouped validation selects Logistic Regression by mean macro F1; the models can still disagree on individual incidents.
- **Why does an arbitrary description still get a category?** Classifiers must select a known label. The app keeps raw candidates for diagnostics but withholds category suggestions when the evidence gate fails.
- **Are workflow charts from the same tickets?** No. They come from a separate event-log dataset, and the datasets are never joined.
- **What would improve real-world performance?** Evaluate on real, diverse tickets, expand account-access examples, compare embeddings, and measure generalization on unseen wording.

## Changes in this demonstration pass

Persistent saved results; guided examples; readable labels; a consistent light theme; a visible first historical match; clearer evaluation context; loading and empty-input feedback; safer HTML notes; corrected category-frequency wording; no arbitrary cluster for zero-overlap text.
