# Formal Test Report

Generated: 2026-09-27T09:27:08.901132+00:00
Source: reports/junit_results.xml (real pytest run — every result below was actually executed)

## Summary

- **Total tests:** 105
- **Passed:** 105
- **Failed:** 0
- **Errors:** 0
- **Skipped:** 0
- **Total duration:** 8.79s

## Results by category

### Data (15/15 passed)

| Test ID | Description | Source (exact input/assertions) | Duration | Status |
|---|---|---|---|---|
| test_data_loader::test_missing_file | Missing file | `tests/test_data_loader.py` | 0.002s | PASS |
| test_data_loader::test_path_is_directory | Path is directory | `tests/test_data_loader.py` | 0.002s | PASS |
| test_data_loader::test_empty_file | Empty file | `tests/test_data_loader.py` | 0.001s | PASS |
| test_data_loader::test_header_only_no_rows | Header only no rows | `tests/test_data_loader.py` | 0.001s | PASS |
| test_data_loader::test_missing_required_columns | Missing required columns | `tests/test_data_loader.py` | 0.001s | PASS |
| test_data_loader::test_malformed_csv_unterminated_quote | Malformed csv unterminated quote | `tests/test_data_loader.py` | 0.001s | PASS |
| test_data_loader::test_bad_encoding | Bad encoding | `tests/test_data_loader.py` | 0.001s | PASS |
| test_data_loader::test_valid_load | Valid load | `tests/test_data_loader.py` | 0.001s | PASS |
| test_preprocessing::test_strips_whitespace | Strips whitespace | `tests/test_preprocessing.py` | 0.007s | PASS |
| test_preprocessing::test_normalizes_categories_lowercase | Normalizes categories lowercase | `tests/test_preprocessing.py` | 0.003s | PASS |
| test_preprocessing::test_parses_timestamps_and_flags_invalid | Parses timestamps and flags invalid | `tests/test_preprocessing.py` | 0.002s | PASS |
| test_preprocessing::test_missing_region_filled_unknown | Missing region filled unknown | `tests/test_preprocessing.py` | 0.003s | PASS |
| test_preprocessing::test_missing_resolution_not_fabricated | Missing resolution not fabricated | `tests/test_preprocessing.py` | 0.002s | PASS |
| test_preprocessing::test_empty_description_flagged | Empty description flagged | `tests/test_preprocessing.py` | 0.002s | PASS |
| test_preprocessing::test_duplicate_ticket_ids_removed | Duplicate ticket ids removed | `tests/test_preprocessing.py` | 0.002s | PASS |

### NLP (10/10 passed)

| Test ID | Description | Source (exact input/assertions) | Duration | Status |
|---|---|---|---|---|
| test_nlp_processor::test_normal_incident_text | Normal incident text | `tests/test_nlp_processor.py` | 0.023s | PASS |
| test_nlp_processor::test_punctuation_removed | Punctuation removed | `tests/test_nlp_processor.py` | 0.001s | PASS |
| test_nlp_processor::test_abbreviations_preserved_as_tokens | Abbreviations preserved as tokens | `tests/test_nlp_processor.py` | 0.000s | PASS |
| test_nlp_processor::test_numbers_preserved | Numbers preserved | `tests/test_nlp_processor.py` | 0.000s | PASS |
| test_nlp_processor::test_short_text | Short text | `tests/test_nlp_processor.py` | 0.000s | PASS |
| test_nlp_processor::test_empty_text_returns_empty_string | Empty text returns empty string | `tests/test_nlp_processor.py` | 0.000s | PASS |
| test_nlp_processor::test_repeated_words_all_kept | Repeated words all kept | `tests/test_nlp_processor.py` | 0.000s | PASS |
| test_nlp_processor::test_stopword_removal_removes_common_words | Stopword removal removes common words | `tests/test_nlp_processor.py` | 0.001s | PASS |
| test_nlp_processor::test_stemming_collapses_related_forms | Stemming collapses related forms | `tests/test_nlp_processor.py` | 0.000s | PASS |
| test_nlp_processor::test_stem_and_lemmatize_mutually_exclusive | Stem and lemmatize mutually exclusive | `tests/test_nlp_processor.py` | 0.000s | PASS |

### Similarity (30/30 passed)

| Test ID | Description | Source (exact input/assertions) | Duration | Status |
|---|---|---|---|---|
| test_evaluation::test_hit_at_k_true_when_present | Hit at k true when present | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_hit_at_k_false_when_absent | Hit at k false when absent | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_hit_at_k_respects_k_boundary | Hit at k respects k boundary | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_hit_at_k_none_when_no_expected_category | Hit at k none when no expected category | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_precision_at_k_all_relevant | Precision at k all relevant | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_precision_at_k_none_relevant | Precision at k none relevant | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_precision_at_k_partial | Precision at k partial | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_precision_at_k_empty_results | Precision at k empty results | `tests/test_evaluation.py` | 0.000s | PASS |
| test_evaluation::test_precision_at_k_none_when_no_expected_category | Precision at k none when no expected category | `tests/test_evaluation.py` | 0.000s | PASS |
| test_retrieval::test_empty_query_returns_empty_list | Empty query returns empty list | `tests/test_retrieval.py` | 0.046s | PASS |
| test_retrieval::test_very_short_query | Very short query | `tests/test_retrieval.py` | 0.047s | PASS |
| test_retrieval::test_unseen_words_returns_empty_no_crash | Unseen words returns empty no crash | `tests/test_retrieval.py` | 0.047s | PASS |
| test_retrieval::test_unrelated_query_matches_correct_incident | Unrelated query matches correct incident | `tests/test_retrieval.py` | 0.047s | PASS |
| test_retrieval::test_duplicate_incident_retrieves_itself_top | Duplicate incident retrieves itself top | `tests/test_retrieval.py` | 0.048s | PASS |
| test_retrieval::test_valid_incident_returns_enriched_metadata | Valid incident returns enriched metadata | `tests/test_retrieval.py` | 0.048s | PASS |
| test_retrieval::test_deterministic_results | Deterministic results | `tests/test_retrieval.py` | 0.049s | PASS |
| test_similarity::test_near_identical_ranks_highest | Near identical ranks highest | `tests/test_similarity.py` | 0.047s | PASS |
| test_similarity::test_related_incident_ranks_above_unrelated | Related incident ranks above unrelated | `tests/test_similarity.py` | 0.047s | PASS |
| test_similarity::test_unrelated_incident_scores_low_or_absent | Unrelated incident scores low or absent | `tests/test_similarity.py` | 0.047s | PASS |
| test_similarity::test_paraphrase_limitation_documented | Paraphrase limitation documented | `tests/test_similarity.py` | 0.047s | PASS |
| test_similarity::test_empty_query_returns_empty_list | Empty query returns empty list | `tests/test_similarity.py` | 0.046s | PASS |
| test_similarity::test_very_short_query | Very short query | `tests/test_similarity.py` | 0.047s | PASS |
| test_similarity::test_top_n_respected | Top n respected | `tests/test_similarity.py` | 0.047s | PASS |
| test_similarity::test_result_fields_present | Result fields present | `tests/test_similarity.py` | 0.047s | PASS |
| test_vectorizer::test_fit_transform_shape | Fit transform shape | `tests/test_vectorizer.py` | 0.001s | PASS |
| test_vectorizer::test_transform_before_fit_raises | Transform before fit raises | `tests/test_vectorizer.py` | 0.000s | PASS |
| test_vectorizer::test_vocabulary_contains_expected_terms | Vocabulary contains expected terms | `tests/test_vectorizer.py` | 0.001s | PASS |
| test_vectorizer::test_shared_terms_produce_higher_similarity | Shared terms produce higher similarity | `tests/test_vectorizer.py` | 0.001s | PASS |
| test_vectorizer::test_empty_corpus_raises | Empty corpus raises | `tests/test_vectorizer.py` | 0.000s | PASS |
| test_vectorizer::test_save_and_load_roundtrip | Save and load roundtrip | `tests/test_vectorizer.py` | 0.002s | PASS |

### Clustering (6/6 passed)

| Test ID | Description | Source (exact input/assertions) | Duration | Status |
|---|---|---|---|---|
| test_clustering::test_expected_grouping_recovered | Expected grouping recovered | `tests/test_clustering.py` | 0.009s | PASS |
| test_clustering::test_predict_before_fit_raises | Predict before fit raises | `tests/test_clustering.py` | 0.001s | PASS |
| test_clustering::test_top_terms_per_cluster_returns_real_terms | Top terms per cluster returns real terms | `tests/test_clustering.py` | 0.007s | PASS |
| test_clustering::test_inertia_before_fit_raises | Inertia before fit raises | `tests/test_clustering.py` | 0.000s | PASS |
| test_clustering::test_inertia_decreases_with_more_clusters | Inertia decreases with more clusters | `tests/test_clustering.py` | 0.017s | PASS |
| test_clustering::test_single_cluster_on_uniform_data_handles_gracefully | Single cluster on uniform data handles gracefully | `tests/test_clustering.py` | 0.009s | PASS |

### ML (18/18 passed)

| Test ID | Description | Source (exact input/assertions) | Duration | Status |
|---|---|---|---|---|
| test_classifier::test_train_test_split_reproducible | Train split reproducible | `tests/test_classifier.py` | 0.003s | PASS |
| test_classifier::test_split_is_stratified | Split is stratified | `tests/test_classifier.py` | 0.001s | PASS |
| test_classifier::test_train_model_unknown_name_raises | Train model unknown name raises | `tests/test_classifier.py` | 0.000s | PASS |
| test_classifier::test_train_all_models_returns_three_fitted_models | Train all models returns three fitted models | `tests/test_classifier.py` | 0.052s | PASS |
| test_classifier::test_each_model_beats_random_baseline_on_easy_synthetic_data | Each model beats random baseline on easy synthetic data | `tests/test_classifier.py` | 0.047s | PASS |
| test_explainability::test_similarity_explanation_shared_terms_identified | Similarity explanation shared terms identified | `tests/test_explainability.py` | 0.001s | PASS |
| test_explainability::test_similarity_explanation_contributions_sum_to_total | Similarity explanation contributions sum to total | `tests/test_explainability.py` | 0.001s | PASS |
| test_explainability::test_similarity_explanation_unrelated_has_no_contributing_terms | Similarity explanation unrelated has no contributing terms | `tests/test_explainability.py` | 0.001s | PASS |
| test_explainability::test_classification_explanation_logistic_regression_is_class_specific | Classification explanation logistic regression is class specific | `tests/test_explainability.py` | 0.002s | PASS |
| test_explainability::test_classification_explanation_tree_is_global_not_class_specific | Classification explanation tree is global not class specific | `tests/test_explainability.py` | 0.001s | PASS |
| test_explainability::test_shared_metadata_identifies_matches | Shared metadata identifies matches | `tests/test_explainability.py` | 0.001s | PASS |
| test_feature_engineering::test_temporal_features_extracted_correctly | Temporal features extracted correctly | `tests/test_feature_engineering.py` | 0.002s | PASS |
| test_feature_engineering::test_message_length_computed | Message length computed | `tests/test_feature_engineering.py` | 0.001s | PASS |
| test_feature_engineering::test_fit_transform_produces_correct_shape | Fit transform produces correct shape | `tests/test_feature_engineering.py` | 0.005s | PASS |
| test_feature_engineering::test_transform_before_fit_raises | Transform before fit raises | `tests/test_feature_engineering.py` | 0.000s | PASS |
| test_feature_engineering::test_no_leakage_fields_in_feature_names | No leakage fields in feature names | `tests/test_feature_engineering.py` | 0.004s | PASS |
| test_feature_engineering::test_unseen_category_handled_without_crash | Unseen category handled without crash | `tests/test_feature_engineering.py` | 0.007s | PASS |
| test_feature_engineering::test_feature_names_length_matches_matrix_columns | Feature names length matches matrix columns | `tests/test_feature_engineering.py` | 0.004s | PASS |

### Application (26/26 passed)

| Test ID | Description | Source (exact input/assertions) | Duration | Status |
|---|---|---|---|---|
| test_analyzer::test_analyze_returns_all_sections | Analyze returns all sections | `tests/test_analyzer.py` | 1.850s | PASS |
| test_analyzer::test_classification_flags_missing_metadata | Classification flags missing metadata | `tests/test_analyzer.py` | 0.098s | PASS |
| test_analyzer::test_classification_flags_provided_metadata | Classification flags provided metadata | `tests/test_analyzer.py` | 0.090s | PASS |
| test_analyzer::test_account_access_prediction_includes_limitation_note | Account access prediction includes limitation note | `tests/test_analyzer.py` | 0.090s | PASS |
| test_analyzer::test_empty_query_returns_none_sections_gracefully | Empty query returns none sections gracefully | `tests/test_analyzer.py` | 0.082s | PASS |
| test_analyzer::test_cluster_note_present_and_not_a_root_cause_claim | Cluster note present and not a root cause claim | `tests/test_analyzer.py` | 0.089s | PASS |
| test_analyzer::test_recurring_patterns_frequency_computed_correctly | Recurring patterns frequency computed correctly | `tests/test_analyzer.py` | 0.091s | PASS |
| test_analyzer::test_format_report_produces_readable_string | Format report produces readable string | `tests/test_analyzer.py` | 0.091s | PASS |
| test_app::test_app_starts_without_exception | App starts without exception | `tests/test_app.py` | 1.737s | PASS |
| test_app::test_dashboard_page_loads_with_metrics | Dashboard page loads with metrics | `tests/test_app.py` | 0.028s | PASS |
| test_app::test_navigate_to_analyze_page | Navigate to analyze page | `tests/test_app.py` | 0.014s | PASS |
| test_app::test_analyze_with_valid_query_produces_results | Analyze with valid query produces results | `tests/test_app.py` | 0.156s | PASS |
| test_app::test_analyze_with_empty_query_does_not_crash | Analyze with empty query does not crash | `tests/test_app.py` | 0.039s | PASS |
| test_app::test_analyze_with_whitespace_only_query_does_not_crash | Analyze with whitespace only query does not crash | `tests/test_app.py` | 0.038s | PASS |
| test_app::test_navigate_to_recurring_patterns_page | Navigate to recurring patterns page | `tests/test_app.py` | 0.085s | PASS |
| test_app::test_navigate_to_trends_page | Navigate to trends page | `tests/test_app.py` | 0.021s | PASS |
| test_app::test_navigate_to_ml_performance_page | Navigate to ml performance page | `tests/test_app.py` | 0.014s | PASS |
| test_app::test_reports_page_before_any_analysis_shows_info_not_crash | Reports page before any analysis shows info not crash | `tests/test_app.py` | 0.188s | PASS |
| test_app::test_reports_page_after_analysis_shows_download_buttons | Reports page after analysis shows download buttons | `tests/test_app.py` | 0.119s | PASS |
| test_report_generator::test_report_contains_query_text | Report contains query text | `tests/test_report_generator.py` | 0.028s | PASS |
| test_report_generator::test_report_contains_ticket_ids | Report contains ticket ids | `tests/test_report_generator.py` | 0.029s | PASS |
| test_report_generator::test_report_shows_resolution_and_no_resolution_correctly | Report shows resolution and no resolution correctly | `tests/test_report_generator.py` | 0.029s | PASS |
| test_report_generator::test_report_always_includes_limitations | Report always includes limitations | `tests/test_report_generator.py` | 0.027s | PASS |
| test_report_generator::test_report_includes_embedded_chart_when_results_exist | Report includes embedded chart when results exist | `tests/test_report_generator.py` | 0.027s | PASS |
| test_report_generator::test_empty_result_does_not_crash_and_still_has_limitations | Empty result does not crash and still has limitations | `tests/test_report_generator.py` | 0.000s | PASS |
| test_report_generator::test_no_chart_when_no_similar_incidents | No chart when no similar incidents | `tests/test_report_generator.py` | 0.000s | PASS |

## Note on Input / Expected Result columns

Per-test exact input values and expected-result assertions are not
paraphrased in this table — doing so risks misrepresenting what was
actually asserted. Each test's real input construction and assertion
logic is in its linked source file. This table instead reports what
can be stated with certainty from an actual run: which test executed,
against which real inputs (via source), how long it took, and its
real outcome.