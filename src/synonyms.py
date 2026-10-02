"""Reviewed WordNet senses and support phrases; no unrestricted expansion.

Canonical targets already occur in the historical training descriptions.
WordNet sense IDs document why each approved variant was selected.
"""
import re

# Only these reviewed lemmas are accepted, not every lemma of every sense.
REVIEWED_WORDNET = {
    "sluggish": ("slow", "dull.s.05"),
    "wrong": ("incorrect", "incorrect.a.01"),
    "customise": ("customize", "customize.v.02"),
}
# Domain phrases require context; payment alone stays payment.
PHRASE_ALIASES = (
    (r"\b(?:billed|charged|paid)\s+(?:two\s+times|twice)\b", "charged twice"),
    (r"\b(?:duplicate|double)\s+(?:charge|charges|payment|payments)\b", "charged twice"),
    (r"\b(?:double[ -]charged|payment\s+taken\s+twice)\b", "charged twice"),
)


def normalize_support_synonyms(text):
    """Normalize reviewed equivalents without assigning an issue category."""
    for pattern, replacement in PHRASE_ALIASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for variant, (canonical, _) in REVIEWED_WORDNET.items():
        text = re.sub(r"\b" + re.escape(variant) + r"\b", canonical,
                      text, flags=re.IGNORECASE)
    return text
