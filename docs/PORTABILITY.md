# Running on another device

The public clone includes the dashboard corpus, vectorizer, matrix, feature
builder, clusterer, three classifiers, and English NLTK resources. Runtime code
reads resources relative to the repository, never from the author's home folder.
No login, secret, Kaggle token, or external service is required for the app.

Use Python 3.12 and `run.py` as described in the README. Initial setup needs
internet to install public Python packages, enough disk space, and a writable
project folder. Subsequent launches use the local `.venv` and bundled resources.
Downloading the optional raw datasets is a separate research workflow.

`python -m scripts.check_setup` checks SHA-256 file integrity, pinned package
versions, dataset/vector dimensions, and actual predictions from all classifiers
and the clusterer. `run.py --check` creates the environment and runs this check.
The GitHub Actions workflow repeats preflight and tests on three operating systems.

If setup reports a changed/missing model or data file, obtain a complete clone or
ZIP. When intentionally rebuilding the artifacts, run
`python -m scripts.generate_runtime_manifest` to refresh the manifest (the
validated rebuild script does this automatically). If an existing `.venv` uses
another Python version, rename or remove it and rerun with Python 3.12. Do not
transfer virtual environments between machines. A busy port can be changed using
`run.py --port 8502`. Installation failures can also mean your platform has no
compatible package wheel; the launcher reports the underlying installer error.

Local clean-environment verification: 132 tests passed on macOS. Offline
preflight and every dashboard page were also exercised with socket connections
and NLTK downloads disabled. See GitHub Actions for current platform results.
