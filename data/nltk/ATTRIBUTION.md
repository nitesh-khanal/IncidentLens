# Bundled English NLTK resources

These are unchanged English resources distributed through the NLTK data repository:
https://github.com/nltk/nltk_data

- `corpora/wordnet.zip`: Princeton WordNet 3.0. Its permission and copyright notice is included inside the archive and reproduced in `WORDNET-LICENSE`.
- `corpora/stopwords/english`: the NLTK English stopword list. The upstream corpus README is preserved beside the file; it records Snowball/PostgreSQL provenance and NLTK additions.
- `tokenizers/punkt_tab/english`: English Punkt tokenizer parameters by Jan Strunk and Tibor Kiss, distributed by NLTK. The upstream README is preserved in the parent directory with the paper citation and training-source attribution.

These data files keep their upstream terms; they are not relicensed as IncidentLens source code. The NLTK registry does not declare a standard license for the stopword and Punkt packages; this file does not claim one. Upstream licensing overview: https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md

Only English runtime resources are bundled. The app does not download language resources, access a personal NLTK cache, require a Kaggle account, or request an API key. The included SHA-256 manifest records the exact shipped resource bytes.
