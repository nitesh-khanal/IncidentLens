# Security and Robustness Review (Stage 24)

## Risk model
IncidentLens is a local CLI/dashboard tool, run by its own user against
data they control. It is NOT a multi-tenant web service accepting input
from untrusted remote users. This shapes what's treated as a real risk
versus a lower-priority one below.

## Findings

### 1. Code execution safety — CHECKED, CLEAN
`grep` across `src/`, `app.py`, `scripts/` for `eval(`, `exec(`,
`os.system`, `subprocess`, `__import__` found zero matches. No user
input or dataset content is ever executed as code or shell commands.

### 2. XSS in HTML report generation — FOUND AND FIXED
**Real vulnerability**, not theoretical: `src/report_generator.py` embedded
the user's incident query and dataset text (resolution summaries, etc.)
into HTML via raw f-string interpolation with no escaping. A query
containing `<script>...</script>` was confirmed to appear unescaped in
the generated report. Since these reports are meant to be downloaded and
potentially shared with others, this was exploitable.

**Fix:** all user-supplied and dataset-derived text fields are now passed
through `html.escape()` before insertion into the HTML template. Verified
fixed with a live test, and locked in with two permanent regression tests
(`test_html_injection_in_query_is_escaped`,
`test_html_injection_in_resolution_summary_is_escaped`).

### 3. Oversized input — CHECKED, HANDLED GRACEFULLY
A 1.2MB (200,000-word) query was passed through the full retrieval
pipeline. Completed in ~2 seconds with no crash, no hang, no memory
issue. No explicit size limit is enforced, but real behavior at this
scale is acceptable for a local single-user tool.

### 4. Malformed/invalid data — CHECKED (Stage 3/4)
Already covered thoroughly in Stage 3 (missing file, wrong type, empty
file, malformed CSV, bad encoding, missing columns) and Stage 4 (missing
values, duplicates, inconsistent categories) — not repeated here.

### 5. Path handling — CHECKED, ACCEPTABLE FOR RISK MODEL
Path traversal attempts (`../../../../etc/passwd`, `/etc/passwd`) were
tested against `load_tickets()`. Both were rejected — one because the
path didn't exist relative to the working directory, the other because
`/etc/passwd`'s content doesn't match the required CSV schema (caught by
schema validation, not path sanitization specifically). Given the local,
single-user risk model, explicit path sanitization was judged unnecessary
beyond what schema validation already provides as a side effect — this is
a conscious, stated decision, not an oversight.

### 6. Secrets — CHECKED, CLEAN
- Searched all source files for hardcoded credentials/keys — no matches
  beyond false positives (the word "token" inside NLTK's `word_tokenize`
  function name, and a demo string using "password" as sample incident
  text).
- Searched full git history (`git log --all -p`) for the Kaggle API token
  pattern used during Stage 2/13 setup — confirmed NOT present anywhere
  in history. The token was typed directly into the terminal by the user,
  never written to a file by this project's tooling, and never committed.
- `~/.kaggle/access_token` lives outside the repository entirely and is
  not tracked.

### 7. Error message safety — CHECKED, CLEAN
Tested that a file-not-found error only echoes back the path the caller
provided — confirmed no internal system paths, stack details, or
environment information leak into user-facing error messages beyond
what the user themselves already supplied.

## Summary
One real, exploitable vulnerability was found (XSS) and fixed with a
verified test and permanent regression coverage. All other checks
(execution safety, oversized input, path handling, secrets, error
messages) came back clean, appropriate for this tool's actual risk model
as a local, single-user application — not treated as a checklist to
rubber-stamp, but as genuine tests that could have found more problems
and, in the one case where something real was there, did.
