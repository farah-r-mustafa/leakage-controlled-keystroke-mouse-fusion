# Stage 2A Leakage-Control Report

## Binding split rule

The session-level split is frozen by the SHA-256 value `e35422f0e48f96b6d1e4da99c9063379a6a488dd2bfa3e99d3e6e68857a7613d`. All later segments, windows, feature rows, claims, scores, calibration records, and thresholds must retain the original `session_id` and its frozen split.

## Prohibited leakage paths

1. A segment or window from one original session may not appear in more than one split.
2. Overlapping windows may be generated only after splitting and only inside one idle segment.
3. `user`, `session_id`, `_id`, raw timestamp, user-agent, screen/window/document dimensions, cursor, file path, and source-row order are metadata and must not enter model features.
4. Duplicate filtering is frozen before window construction; duplicate-derived windows must not be generated.
5. Scalers, encoders, feature selection, imputation, calibration, and personal thresholds must be fitted without test data.
6. Genuine and impostor claims must be constructed independently inside Validation and Test after the window manifest is frozen.
7. `Alluserspreauth` is not a twelfth enrolled user and may not be mixed into genuine claims.
8. Test coverage must not be used to choose a more favorable split or to retune window parameters; parameter alternatives must be reported as a prespecified feasibility/sensitivity analysis.

## Stage status

No model, claim, calibration, or threshold was created in this stage.
