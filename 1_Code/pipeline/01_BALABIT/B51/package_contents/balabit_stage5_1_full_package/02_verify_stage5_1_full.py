#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, hashlib, json, sys, zipfile
from pathlib import Path
import pandas as pd

OUTPUT_DIR = "balabit_stage5_1_full_output"

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def find_root(start: Path, explicit: str|None) -> Path:
    if explicit: return Path(explicit).expanduser().resolve()
    for p in [start, start.parent, Path.cwd(), Path.cwd().parent]:
        if (p/OUTPUT_DIR).exists(): return p.resolve()
    for p in start.parent.glob(f"**/{OUTPUT_DIR}"):
        return p.parent.resolve()
    raise FileNotFoundError(OUTPUT_DIR)

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', default=None)
    a=ap.parse_args()
    root=find_root(Path(__file__).resolve().parent, a.project_root)
    out=root/OUTPUT_DIR
    manifest=json.loads((out/'00_full_manifest.json').read_text(encoding='utf-8-sig'))
    checks=[]
    checks.append(('manifest_status', manifest.get('status')=='FULL_COMPLETE', manifest.get('status')))
    for name in ['07_bootstrap_95ci_full.csv','09_paired_performance_deltas_full.csv','10_exact_user_signflip_holm_full.csv','11_hybrid_cost_justification_full.csv','13_publication_interpretation_EN.md','14_publication_interpretation_AR.md','balabit_stage5_1_full_review.zip']:
        checks.append((name, (out/name).exists(), 'exists' if (out/name).exists() else 'missing'))
    b=pd.read_csv(out/'07_bootstrap_95ci_full.csv')
    d=pd.read_csv(out/'09_paired_performance_deltas_full.csv')
    checks.append(('metric_ci_point_sanity', bool(b['ci_contains_point'].all()), f"outside={(~b['ci_contains_point']).sum()}"))
    checks.append(('delta_ci_point_sanity', bool(d['ci_contains_point'].all()), f"outside={(~d['ci_contains_point']).sum()}"))
    sess=b[b['cluster_level'].astype(str).str.startswith('stratified')]
    checks.append(('session_class_support', bool(sess['all_users_have_both_classes'].all()), str(bool(sess['all_users_have_both_classes'].all()))))
    z=out/'balabit_stage5_1_full_review.zip'
    if z.exists():
        with zipfile.ZipFile(z) as f:
            bad=f.testzip()
        checks.append(('review_zip_integrity', bad is None, str(bad)))
        hp=out/'balabit_stage5_1_full_review.zip.sha256.txt'
        if hp.exists():
            expected=hp.read_text(encoding='utf-8').split()[0]
            checks.append(('review_zip_sha256', sha256(z)==expected, f"expected={expected} actual={sha256(z)}"))
    ok=True
    print('Balabit Stage 5.1 Full verification')
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print('\nFINAL:', 'PASS' if ok else 'REVIEW_REQUIRED')
    return 0 if ok else 1

if __name__=='__main__': raise SystemExit(main())
