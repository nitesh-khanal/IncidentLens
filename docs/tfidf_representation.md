# TF-IDF Text Representation

## What it is
- **TF (Term Frequency):** how often a term appears within a single incident description.
- **IDF (Inverse Document Frequency):** how rare a term is across the whole corpus — common words get a low score, distinctive words get a high score.
- **TF-IDF:** TF x IDF — high only for terms that are both frequent in one document and rare overall.

## Why chosen
- Simple, fast, fully explainable (every score traces back to word frequency — no black box).
- Works well for short, keyword-driven text like this dataset's incident descriptions.
- Standard baseline before considering embeddings (Stage 26, optional).

## Advantages
- No training data or labels required.
- Deterministic and reproducible.
- Directly explainable: "these incidents are similar because they share these high-weight terms."

## Limitations (stated explicitly, not glossed over)
- **Purely lexical, not semantic.** "Cannot log in" and "unable to access account" share almost no words and would score as dissimilar under TF-IDF, despite meaning the same thing.
- Sensitive to vocabulary size — very short descriptions (this dataset: mean 56 characters) limit how much signal each vector carries.
- Does not capture word order or context (a "bag of words" model).

_Real vocabulary size, matrix shape, and density are filled in below after running scripts/build_tfidf.py — not estimated in advance._

## Real measured results (full 100,000-document corpus)

- **Vocabulary size:** 99 terms
- **Matrix shape:** (100000, 99)
- **Non-zero entries:** 525,723
- **Density:** 5.31% (94.69% sparse)

The vocabulary is small (99 terms) because descriptions are short (Stage 5 EDA: mean 56 characters) and heavily preprocessed (Stage 6: stopwords removed, lemmatized). This is a real constraint worth stating plainly in the certification report: with only 99 distinct terms across the whole corpus, TF-IDF similarity has limited room to distinguish between incidents that happen to share common vocabulary but describe different problems.

**Most distinctive terms (highest IDF):** crashing, keep, open, whenever, category, fit, sure, amount, compared, invoice — a mix of billing-specific and generic action words.

**Most common terms (lowest IDF):** need, login, data, account, export, seems, team, option, plan, section — these appear across many incidents and contribute little to distinguishing similarity.
