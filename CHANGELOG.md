# Changelog

All notable changes to IncidentLens, in the order they were made. Commit
hashes are from the real repository history.

## [Unreleased] (targeting v1.0.0; the tag is created in Stage 28)

### Project setup
- `96af7a6` first commit
- `3ab6e9d` Added .gitignore file
- `0cbd0df` specification commit (Stage 0, `docs/specification.md`)
- `9f37d63` venv setup (Stage 1; the environment is kept outside the repo)

### Data
- `c93a054` data: add and document primary dataset (Stage 2; 100,000 rows, CC BY 4.0)
- `5963172` fix: correct .gitignore to exclude raw data and venv
- `303b9d4` feat: add incident data loading (Stage 3)
- `df80a19` feat: add incident data preprocessing (Stage 4)
- `c0f03b3` data analysis (Stage 5; EDA notebook)

### Text pipeline and retrieval
- `9cf1792` feat: add incident NLP preprocessing (Stage 6)
- `802fc54` feat: implement TF-IDF incident representation (Stage 7)
- `dee144f` feat: implement incident similarity engine (Stage 8)
- `6597e6e` feat: add historical incident retrieval (Stage 9)
- `13d3d99` test: evaluate incident retrieval quality (Stage 10)

### Analytics
- `3805b6a` feat: add incident pattern clustering (Stage 11)
- `a829c5f` feat: add incident trend analysis (Stage 12)
- `06cb245` feat: add incident workflow analytics (Stage 13; secondary dataset)

### Machine learning
- `bba26f0` feat: add incident feature engineering (Stage 14)
- `736ea12` docs: define IncidentLens machine learning task (Stage 15)
- `4b28ab6` feat: implement incident classification models (Stage 16)
- `73b0048` test: evaluate incident classification models (Stage 17)
- `2ee9102` docs: document account_access generalization limitation and considered fix

### Application
- `2b5c503` feat: build IncidentLens intelligence engine (Stage 18)
- `3d2a61f` feat: add incident analysis explanations (Stage 19)
- `a74368c` feat: build IncidentLens Streamlit dashboard (Stage 20)
- `e6972a0` feat: add incident analysis reports (Stage 21)

### Quality
- `e5aa678` test: add comprehensive IncidentLens test suite (Stage 22; 105 tests)
- `58d2306` perf: measure IncidentLens processing performance (Stage 23)
- `189f174` fix: harden IncidentLens input handling (Stage 24; XSS fix in the report generator)

### Evaluated, not built
- `1d49f49` docs: evaluate optional LogHub integration (Stage 25; no defensible link to the primary dataset)
- Stage 26 (FastAPI, PostgreSQL, Docker, CI, cloud deployment) was skipped by decision.

### Documentation
- Stage 27: README, CHANGELOG, and a pinned `requirements.txt` (this release's documentation commit).

### Notable fixes made inside a stage, before its commit
- Stage 3: malformed-CSV test replaced after pandas turned out to tolerate extra trailing fields
- Stage 8: two test assumptions corrected (lemmatizer verb forms; zero-similarity results are excluded)
- Stage 12: pandas index-alignment bug that silently produced NaN trend correlations
- Stage 13: the secondary CSV is semicolon-delimited; negative stage durations fixed with groupby and shift
- Stage 16: PyArrow-backed labels broke scikit-learn indexing
- Stage 20: the 3-model comparison crashed when metadata was omitted (now reuses DEFAULT_METADATA)
- Stage 23: the same missing-metadata bug reappeared in the performance script and was fixed the same way
- Stage 24: stored XSS in the HTML report generator (fixed with html.escape, regression tests added)
