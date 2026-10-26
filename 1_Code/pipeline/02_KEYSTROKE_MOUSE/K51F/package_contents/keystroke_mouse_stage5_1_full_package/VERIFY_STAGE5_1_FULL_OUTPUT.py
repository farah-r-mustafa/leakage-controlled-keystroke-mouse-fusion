#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, pandas as pd, sys

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('output')
    args = ap.parse_args()
    root = Path(args.output).expanduser().resolve()
    required = ['STAGE5_1_FULL_COMPLETE.ok','stage5_1_full_summary.json','integrity_checks.csv','output_manifest_sha256.csv']
    missing = [x for x in required if not (root/x).is_file()]
    if missing:
        print('FAIL missing:', missing); return 2
    summary = json.loads((root/'stage5_1_full_summary.json').read_text(encoding='utf-8'))
    checks = pd.read_csv(root/'integrity_checks.csv')
    manifest = pd.read_csv(root/'output_manifest_sha256.csv')
    failures = []
    if summary.get('status') != 'PASS' or summary.get('mode') != 'full': failures.append('summary status/mode')
    if int((checks.status != 'PASS').sum()) != 0: failures.append('integrity checks')
    for r in manifest.itertuples(index=False):
        p = root / str(r.relative_path)
        if not p.is_file() or p.stat().st_size != int(r.bytes) or sha256_file(p).lower() != str(r.sha256).lower():
            failures.append(str(r.relative_path))
    if failures:
        print('FAIL', failures); return 2
    print(json.dumps({'status':'PASS','manifest_files':len(manifest),'bootstrap_reps':summary.get('bootstrap_reps'),'recommendation':summary.get('recommendation')}, indent=2))
    return 0
if __name__ == '__main__': raise SystemExit(main())
