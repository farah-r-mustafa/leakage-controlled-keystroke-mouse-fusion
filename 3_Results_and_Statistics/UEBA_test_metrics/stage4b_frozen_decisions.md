# Keystroke–Mouse Stage 4B Frozen Decisions

- Version: 4.2.0
- Test evaluation opened only after Stage 4A was complete.
- Frozen calibrated scores, thresholds, and predictions were used without refitting.
- Balanced 1:1 and Realistic All-Impostor protocols were evaluated separately.
- Macro-user metrics give equal weight to each claimed identity.
- Adequate-session subset: claimed users with at least 5 Core multimodal sessions in Stage 2A.
- TTA definition: first accepted genuine window end minus first genuine window start within a session; reported on successfully authenticated sessions, with coverage reported separately.
- Inference time, model size, and parameter counts are inherited from frozen Stage 3B/3C resource audits.
- No McNemar, bootstrap, sign-flip, Holm correction, or model-selection retraining occurs here; those are reserved for Stage 5.
