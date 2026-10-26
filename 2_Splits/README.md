# 2_Splits — UEBA session-level split manifests

Frozen train / validation / test split of the Keystroke and Mouse Dynamics for UEBA dataset (Stage 2A). Sessions were split chronologically per user (60/20/20, at least one validation and one test session) before any windowing, so no window or idle segment crosses two parts.

| File | Content |
|---|---|
| `frozen_session_manifest.csv` | One row per session: user, session_id, split (train/validation/test), event counts and timestamps |
| `primary_segment_manifest_60s.csv` | Idle segments (60 s gap rule); each segment inherits the split of its session |
| `user_split_summary.csv` | Number of train / validation / test sessions per user |
| `split_protocol.json` | Split rule, ordering and leakage guards |
| `excluded_sessions.csv`, `auxiliary_sessions.csv` | Sessions not used for enrolment and the reason |
| `window_configuration_summary.csv` | Window definitions per cohort |
| `stage2a_leakage_control_report.md` | Leakage-control checks of this stage |
| `SHA256SUMS.txt` | SHA-256 hashes of the files in this folder |

Frozen manifest SHA-256 (as recorded by the pipeline): `e35422f0e48f96b6d1e4da99c9063379a6a488dd2bfa3e99d3e6e68857a7613d`.
