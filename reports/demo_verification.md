# Demonstration and validation verification — 1 October 2026

The full test suite passed against the updated local project: **129 passed**, no failures.

The run used Python 3.12 and the pinned core runtime packages in a fresh local environment. Language resources and Python caches were kept outside the iCloud-backed environment. Warnings were limited to physical-core detection falling back to logical cores and the intentional uniform-data clustering test.

Coverage includes train-only vocabulary and category fitting, grouped template isolation, authentication phrase normalization, low-score and low-overlap abstention, hidden unsupported category statistics, standard Precision@K with missing slots counted as misses, saved analysis persistence, safe HTML notes, demonstration examples, and all six application pages.

Three grouped folds refit preprocessing from raw training rows. The corrected unseen-template holdout shares zero templates. Logistic Regression was selected by grouped mean macro F1. Results and counts are recorded in validated_evaluation.json.

The exploratory challenge set contains 48 assistant-authored category cases and 6 boundary inputs; it is excluded from fitting and is not an independent real-world benchmark. Category suggestion coverage is 21/48; 20 of those 21 suggestions match the intended labels. All six boundary inputs are withheld. Raw classifier accuracy is 35/48, compared with 36/48 before the changes. These results support a more conservative interface, not a claim that overall classification accuracy improved.

The controlled retrieval check improved from 5/7 to 6/7 labelled cases after normalization; the short "Login broken" case remains a known failure. Exploratory retrieval category Precision@1 remains 81.25%.

The prior test_report.md is retained as its historical run record. This verification concerns application behavior and evaluation correctness; it does not validate real-world incident diagnosis.
