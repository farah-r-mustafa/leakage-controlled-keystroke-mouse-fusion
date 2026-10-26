# 3_Results_and_Statistics — UEBA results

| Folder / file | Content |
|---|---|
| `UEBA_test_metrics/per_user_test_metrics.csv` | Per-user test metrics (EER, AUC, FAR, FRR, F1, MCC, …) for every cohort, model and protocol |
| `UEBA_test_metrics/macro_user_test_metrics.csv` | User-macro averages (Tables 6 and 9 of the article) |
| `UEBA_test_metrics/overall_test_metrics.csv` | Pooled (claim-level) metrics |
| `UEBA_test_metrics/model_resource_summary.csv` | Training time, model size, parameters / tree nodes and scoring time |
| `UEBA_test_metrics/integrity_checks.csv` | Integrity checks of the test-evaluation stage |
| `UEBA_statistics/paired_user_signflip_*.csv` | Exact paired-user sign-flip tests with Holm correction |
| `UEBA_statistics/mcnemar_rf_vs_deep_claim_paired.csv` | McNemar tests (descriptive only) |
| `UEBA_statistics/clustered_session_bootstrap_*.csv`, `paired_session_cluster_bootstrap_*.csv` | Session-cluster bootstrap intervals (5,000 replicates; supplementary, not used for reported comparisons) |
| `UEBA_statistics/deployment_candidate_scorecard.csv` | Input to `4_Matched_Window_Ablation/recompute_table9.py` (Table 9) |
| `UEBA_statistics/integrity_checks.csv` | Integrity checks of the statistics stage |

The protocol labelled `realistic_all_impostor` in these files is called "Enrolled Impostor-All" in the article. Local temporary paths in the integrity reports were replaced by `<temp>` or `<local>`; the original SHA-256 manifests (`output_manifest_sha256.csv`) refer to the files as produced by the pipeline.
