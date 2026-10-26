"""Recompute the deployment utility U (Table 9) from the released Stage 5.1 scorecard.

Usage: python recompute_table9.py deployment_candidate_scorecard.csv

U = 0.7 P + 0.2 C + 0.1 S  (Enrolled Impostor-All protocol, labelled realistic_all_impostor)
P: mean of the six rank scores (AUC, EER, FAR, FRR, F1, MCC) already stored in the scorecard.
C: mean of three min-max cost scores (higher = cheaper): total model size (log10 scale),
   scoring time per claim, total training time.
S: 0.5 for every candidate (no Holm-significant win or loss).
Also repeats the ranking for 90 weight settings (performance weight 0.40-0.95).
"""
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1])
d = d[d.protocol == 'realistic_all_impostor'].copy()
mm = lambda x: 1 - (x - x.min()) / (x.max() - x.min())
d['C'] = (mm(np.log10(d.total_model_size_bytes)) + mm(d.score_ms_per_claim_output) + mm(d.total_train_seconds)) / 3
d['S'] = 0.5
d['U'] = 0.7 * d.performance_score + 0.2 * d.C + 0.1 * d.S
cols = ['cohort', 'model_family', 'eer', 'auc', 'performance_score', 'C', 'U',
        'total_train_seconds', 'total_model_size_bytes', 'total_parameter_count']
print(d.sort_values('U', ascending=False)[cols].head(5).round(4).to_string(index=False))
res = []
for wp in np.round(np.arange(0.40, 0.951, 0.05), 2):
    for wc in np.round(np.arange(0.0, 1 - wp + 1e-9, 0.05), 2):
        ws = round(1 - wp - wc, 2)
        U = wp * d.performance_score + wc * d.C + ws * d.S
        top = d.loc[U.idxmax()]; top3 = d.loc[U.nlargest(3).index]
        res.append((wp, wc, ws, top.cohort, top.model_family, (top3.model_family == 'random_forest').all()))
r = pd.DataFrame(res, columns=['w_perf', 'w_cost', 'w_support', 'top_cohort', 'top_model', 'top3_all_rf'])
print(f"\nWeight settings: {len(r)}; Random Forest first in {(r.top_model=='random_forest').sum()}; "
      f"keyboard first in {(r.top_cohort=='keyboard').sum()}; "
      f"top three all RF when w_perf >= 0.5: {r[r.w_perf>=0.5].top3_all_rf.all()}")
