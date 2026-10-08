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
    "slowly": ("slow", "slowly.r.01"),
    "exportation": ("export", "export.n.01"),
    "computer_code": ("code", "code.n.03"),
    "computer_error": ("error", "error.n.06"),
    "mouse_click": ("click", "click.n.04"),
}
# Domain phrases require context; payment alone stays payment.
PHRASE_ALIASES = (
    # Explicit technical names only: these do not infer a failure or category.
    (r"\btwo[\s-]+factor[\s-]+authentication(?=\s+code\b)", "2fa"),
    (r"\bapplication[\s-]+programming[\s-]+interface\b", "api"),
    (r"\bmobile[\s-]+application\b", "mobile app"),
    (r"\blogin[\s-]+authentication\b", "login auth"),
    (r"\b(?:billed|charged|paid)\s+(?:two\s+times|twice)\b", "charged twice"),
    (r"\b(?:duplicate|double)\s+(?:charge|charges|payment|payments)\b", "charged twice"),
    (r"\b(?:double[ -]charged|payment\s+taken\s+twice)\b", "charged twice"),
)


def normalize_support_synonyms(text):
    """Normalize reviewed equivalents without assigning an issue category."""
    for pattern, replacement in PHRASE_ALIASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for variant, (canonical, _) in REVIEWED_WORDNET.items():
        # WordNet stores multiword lemmas with underscores. Accept their
        # spaced/hyphenated forms while retaining whole-word boundaries.
        pattern = r"[\s_-]+".join(re.escape(part) for part in variant.split("_"))
        text = re.sub(r"\b" + pattern + r"\b", canonical,
                      text, flags=re.IGNORECASE)
    return text
