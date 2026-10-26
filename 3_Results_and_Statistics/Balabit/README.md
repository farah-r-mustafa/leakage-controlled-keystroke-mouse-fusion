# Balabit results (Stage 5.1)

| File | Content |
|---|---|
| `source_v3__06_metrics_by_user.csv`, `paper_tables/table_07_metrics_by_user.csv` | Per-user metrics for every model and protocol |
| `source_v3__05_metrics_summary.csv`, `paper_tables/table_01_performance_with_user_ci.csv` | User-macro metrics with user-level bootstrap intervals (Table 4 of the article) |
| `source_v3__03_thresholds_per_user.csv`, `source_v3__02_temperature_scaling.csv` | Per-user validation thresholds and temperature calibration |
| `10_exact_user_signflip_holm_full.csv`, `09_paired_performance_deltas_full.csv` | Exact user-level sign-flip tests with Holm correction and paired differences |
| `08_mcnemar_exact_holm_annotated.csv` | McNemar tests (descriptive only) |
| `07_bootstrap_95ci_full.csv` | Bootstrap intervals |
| `11_hybrid_cost_justification_full.csv`, `source_v3__10_cost_summary.csv` | Cost comparison of the hybrid and CNN |

The protocol labelled `realistic_all_impostors` is called "Enrolled Impostor-All" in the article. Local paths were replaced by `<local_path>`.
