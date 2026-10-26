"""Matched-window feature ablation on UEBA multimodal windows (cohort = argv[3], default multimodal_any).
Same windows, same splits, same per-user RF protocol; only the input columns differ:
keyboard-only (k_* + keyboard masks), mouse-only (m_* + mouse masks), fused (all)."""
import sys, json, itertools, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
D=sys.argv[1]; OUT=sys.argv[2]; CO=sys.argv[3] if len(sys.argv)>3 else 'multimodal_any'; SEEDS=[42,43,44]
proc=json.load(open(f'{D}/processors/tabular_{CO}_processor.json'))
feat=proc['kept_features']
if len(feat)<np.load(f'{D}/tabular/{CO}_train.npz')['X'].shape[1]: feat=feat+proc.get('mask_features_included',[])
def load(s):
    z=np.load(f'{D}/tabular/{CO}_{s}.npz',allow_pickle=True); idx=pd.read_csv(f'{D}/indexes/{CO}_{s}_index.csv')
    assert len(idx)==len(z['X']); return z['X'],idx
Xtr,Itr=load('train'); Xva,Iva=load('validation'); Xte,Ite=load('test')
assert Xtr.shape[1]==len(feat), (Xtr.shape, len(feat))
feat=np.array(feat)
cols={'keyboard_only':[i for i,f in enumerate(feat) if f.startswith('k_') or f.startswith('keyboard_')],
      'mouse_only':[i for i,f in enumerate(feat) if f.startswith('m_') or f.startswith('mouse_')],
      'fused':list(range(len(feat)))}
print({k:len(v) for k,v in cols.items()}, len(feat))
users=sorted(Itr.user_id.unique())
def impostor_sample(I,u,n,rng):
    # deterministic user- and session-balanced impostor windows
    others=[v for v in users if v!=u]; per=int(np.ceil(n/len(others))); pick=[]
    for v in others:
        sub=I[I.user_id==v]; 
        if len(sub)==0: continue
        # round-robin over sessions
        groups=[g.index.values[rng.permutation(len(g))] for _,g in sub.groupby('session_id')]
        rr=[x for t in itertools.zip_longest(*groups) for x in t if x is not None]
        pick+=rr[:per]
    pick=np.array(pick); return pick[rng.permutation(len(pick))][:n]
def eer(y,s):
    f,t,_=roc_curve(y,s); fn=1-t; i=np.argmin(abs(fn-f)); return (f[i]+fn[i])/2*100
rows=[]
for seed in SEEDS:
  for var,c in cols.items():
    for u in users:
        rng=np.random.default_rng(seed+users.index(u))
        g=Itr.index[Itr.user_id==u].values; im=impostor_sample(Itr,u,len(g),rng)
        X=np.vstack([Xtr[g][:,c],Xtr[im][:,c]]); y=np.r_[np.ones(len(g)),np.zeros(len(im))]
        rf=RandomForestClassifier(n_estimators=100,max_depth=12,random_state=seed,n_jobs=-1).fit(X,y)
        s=rf.predict_proba(Xte[:,c])[:,1]; yt=(Ite.user_id==u).values.astype(int)
        # realistic / enrolled impostor-all
        rows.append(dict(seed=seed,variant=var,user=u,protocol='impostor_all',eer=eer(yt,s),auc=roc_auc_score(yt,s)*100,n_gen=yt.sum(),n_imp=(1-yt).sum()))
        # balanced 1:1 (deterministic impostor sample of equal size)
        rb=np.random.default_rng(1000+users.index(u)); gi=np.where(yt==1)[0]; ii=Ite.index[Ite.user_id!=u].values
        ii=impostor_sample(Ite,u,len(gi),rb); sel=np.r_[gi,ii]
        rows.append(dict(seed=seed,variant=var,user=u,protocol='balanced',eer=eer(yt[sel],s[sel]),auc=roc_auc_score(yt[sel],s[sel])*100,n_gen=len(gi),n_imp=len(ii)))
    print(seed,var,flush=True)
R=pd.DataFrame(rows); R.to_csv(OUT,index=False)
print(R.groupby(['protocol','variant','seed'])[['eer','auc']].mean().round(2))
