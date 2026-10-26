# Reference Test Report

- Input: `keystroke_mouse_stage4b_review(1).zip`
- Input SHA-256: `01bb3d64d80717b4e4b8fd5b1093b3d0b8f7f5b3ec0cdc0d0a0011bad53646ff`
- Run mode: Pilot
- Bootstrap replicates: 100
- Seed: 5102026
- Status: PASS
- Elapsed in reference environment: 30.190 seconds
- Integrity checks: 400 PASS, 0 FAIL
- McNemar comparisons: 24
- Cluster-bootstrap CI rows: 192
- Paired cluster-bootstrap delta rows: 144
- User-level exact sign-flip rows: 528
- Adequacy/threshold subgroup test rows: 384
- Balabit comparison executed: No
- Retraining/recalibration/threshold changes: No
- Provisional deployment recommendation: `multimodal_any / random_forest`
- Claim-level McNemar significant after family Holm: 19/24
- RF-vs-deep user-level sign-flip significant after family Holm: 0/144

The pilot therefore preserves Random Forest as the strongest provisional deployment candidate, while correctly avoiding a claim that its superiority is statistically established across users.
