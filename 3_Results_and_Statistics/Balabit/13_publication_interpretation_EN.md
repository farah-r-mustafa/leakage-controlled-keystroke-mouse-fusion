# Stage 5.1 Full — Publication Interpretation

## Primary conclusion
CNN is selected as the primary deployment model. Hybrid achieved marginal numerical improvements under the realistic all-impostor protocol, but no primary Hybrid–CNN comparison remained significant after exact user-level sign-flip testing with Holm correction (no significant comparison). The final cost decision is **NOT_JUSTIFIED_OVER_CNN**.

## Realistic protocol
- CNN: EER 5.321%, MCC 0.7338, FAR 5.765%, FRR 5.893%.
- Hybrid: EER 5.045%, MCC 0.7481, FAR 5.096%, FRR 6.089%.
- Hybrid − CNN: ΔEER -0.276 percentage points; ΔMCC 0.0143; ΔFAR -0.669 percentage points; ΔFRR 0.195 percentage points.

## Balanced common protocol
CNN and Hybrid substantially outperform Random Forest and Linear SVM on the same 1:1 claim set. Hybrid and CNN remain statistically comparable on the primary cluster-aware comparison.

## Edge-cost interpretation
Hybrid uses 4.00× the parameters, 3.84× the checkpoint size, and 2.39× the single-window inference latency of CNN. Therefore, the limited numerical improvement does not justify replacing CNN as the resource-aware deployment choice.

## Permitted manuscript claim
“Both deep models markedly outperformed the classical baselines under the balanced common-claim protocol. Although the hybrid architecture produced marginal numerical gains in several metrics, cluster-aware user-level inference did not establish a robust overall advantage over CNN. CNN therefore offers the preferable performance–efficiency trade-off for continuous authentication in resource-constrained Zero Trust deployments.”

## Required limitations
1. The test split contains one genuine source session per user.
2. Claim-level McNemar is descriptive because windows and claims are dependent.
3. All-impostor baseline evaluation is unavailable because reusable Stage 3 baseline models were not saved.
4. Exact user-level inference is based on ten users, limiting statistical power.
