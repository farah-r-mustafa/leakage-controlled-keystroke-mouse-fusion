#!/usr/bin/env python3
"""
Keystroke–Mouse Stage 3B Classical Pilot
Model training and raw claim-score generation for:
  Centroid, One-Class SVM, Linear SVM, Random Forest.

Safety / methodology:
- Training uses Stage 2C train arrays only.
- Validation and test are transformed already by frozen Stage 2C processors.
- Stage 3A claim files are preserved row-for-row; raw scores are appended.
- No calibration, thresholding, test metric calculation, or model selection.
- Deterministic user/session-aware negative sampling for discriminative models.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.svm import LinearSVC, OneClassSVM
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with: python -m pip install numpy scikit-learn joblib psutil\n"
        f"Original error: {exc}"
    )

try:
    import psutil  # optional but recommended
except ImportError:  # pragma: no cover
    psutil = None

VERSION = "3.1.2"
STAGE = "Keystroke–Mouse Stage 3B Classical Pilot"
BASE_SEED = 31082026
COHORTS = ["keyboard", "mouse", "multimodal_primary", "multimodal_any"]
MODELS = ["centroid", "oneclass_svm", "linear_svm", "random_forest"]
PROTOCOLS = ["balanced_1to1", "realistic_all_impostor"]
SCORE_SPLITS = ["validation", "test"]
USERS = [f"User{i}" for i in range(1, 12)]
LOW_GENUINE_WARNING = 10
OCSVM_MAX_GENUINE = 1500

CLAIM_REQUIRED = [
    "claim_id", "cohort", "protocol", "split", "source_row_index", "window_id",
    "session_id", "segment_id", "window_start_epoch", "window_end_epoch",
    "actual_user_id", "actual_user_index", "claimed_user_id", "claimed_user_index",
    "is_genuine", "sampling_role", "sampling_seed",
]
SCORE_APPEND = ["model_id", "model_family", "raw_score", "score_higher_is_genuine"]




def remove_tree_with_retries(path: Path, *, fatal: bool) -> bool:
    """Remove a directory reliably on Windows after releasing Python/NumPy handles."""
    gc.collect()
    last_error: Exception | None = None
    for delay in (0.0, 0.25, 0.75, 1.5, 3.0):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError) as exc:
            last_error = exc
            gc.collect()
    if fatal:
        raise RuntimeError(f"Could not remove existing directory after retries: {path}; {last_error}")
    return False


@contextmanager
def resilient_temp_workspace(prefix: str = "km_stage3b_"):
    """Create a temp workspace whose cleanup can never invalidate completed results."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        # A delayed Windows/antivirus lock must never cancel completed results.
        remove_tree_with_retries(path, fatal=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_rank(*parts: Any) -> int:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def stable_seed32(*parts: Any) -> int:
    return stable_rank(BASE_SEED, *parts) % (2**32 - 1)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_root(base: Path, marker: str) -> Path:
    if (base / marker).exists():
        return base
    matches = list(base.rglob(marker))
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely locate {marker} under {base}; matches={len(matches)}")
    return matches[0].parent


def materialize_input(path: Path, marker: str, temp_base: Path, label: str) -> tuple[Path, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return detect_root(path, marker), ""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{label} must be a directory or ZIP: {path}")
    dest = temp_base / label
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest)
    return detect_root(dest, marker), sha256_file(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_index_sessions(path: Path, expected_rows: int) -> list[str]:
    sessions: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "session_id" not in r.fieldnames:
            raise ValueError(f"Missing session_id in {path}")
        for row in r:
            sessions.append(row["session_id"])
    if len(sessions) != expected_rows:
        raise ValueError(f"Index/array mismatch: {path.name} rows={len(sessions)}, expected={expected_rows}")
    return sessions


def session_round_robin(indices: list[int], sessions: list[str], target: int, *rank_parts: Any) -> list[int]:
    if target >= len(indices):
        return sorted(indices)
    by_session: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_session[sessions[idx]].append(idx)
    session_order = sorted(by_session, key=lambda s: stable_rank(*rank_parts, "session", s))
    for s in session_order:
        by_session[s].sort(key=lambda i: stable_rank(*rank_parts, "window", s, i))
    pos = {s: 0 for s in session_order}
    selected: list[int] = []
    while len(selected) < target:
        progressed = False
        for s in session_order:
            p = pos[s]
            if p < len(by_session[s]):
                selected.append(by_session[s][p])
                pos[s] += 1
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            raise RuntimeError("Session round-robin exhausted unexpectedly")
    return selected


def allocate_quotas(target: int, capacities: dict[int, int], *rank_parts: Any) -> dict[int, int]:
    if target > sum(capacities.values()):
        raise ValueError(f"Insufficient negative capacity: target={target}, capacity={sum(capacities.values())}")
    order = sorted(capacities, key=lambda u: stable_rank(*rank_parts, "user", u))
    quotas = {u: 0 for u in capacities}
    remain = target
    while remain:
        progressed = False
        for u in order:
            if quotas[u] < capacities[u]:
                quotas[u] += 1
                remain -= 1
                progressed = True
                if remain == 0:
                    break
        if not progressed:
            raise RuntimeError("Quota allocation stalled")
    return quotas


def select_balanced_negatives(y: np.ndarray, sessions: list[str], claimed: int,
                              target: int, cohort: str, model: str) -> list[int]:
    by_user: dict[int, list[int]] = {}
    for u in sorted(int(v) for v in np.unique(y) if int(v) != claimed):
        by_user[u] = np.flatnonzero(y == u).astype(int).tolist()
    capacities = {u: len(v) for u, v in by_user.items()}
    quotas = allocate_quotas(target, capacities, cohort, model, claimed, "negative_quota")
    selected: list[int] = []
    for u in sorted(by_user):
        selected.extend(session_round_robin(
            by_user[u], sessions, quotas[u], cohort, model, claimed, u, "negative_select"
        ))
    selected.sort(key=lambda i: stable_rank(cohort, model, claimed, "negative_final", i))
    return selected


class PeakRSSMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_mb = float("nan")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if psutil is None:
            return self
        proc = psutil.Process(os.getpid())
        self.peak_mb = proc.memory_info().rss / (1024**2)

        def poll() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self.peak_mb = max(self.peak_mb, proc.memory_info().rss / (1024**2))
                except Exception:
                    return

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


@dataclass
class FittedModel:
    family: str
    claimed_user_index: int
    artifact_path: Path
    object: Any
    parameter_count: int
    parameter_definition: str


def fit_one_model(family: str, X: np.ndarray, y: np.ndarray, sessions: list[str],
                  claimed: int, cohort: str, artifact_dir: Path,
                  warnings_rows: list[dict[str, Any]]) -> tuple[FittedModel, dict[str, Any], dict[str, Any]]:
    user_id = USERS[claimed]
    pos_all = np.flatnonzero(y == claimed).astype(int).tolist()
    if not pos_all:
        raise ValueError(f"No positive training samples for {cohort}/{user_id}")
    if len(pos_all) < LOW_GENUINE_WARNING:
        warnings_rows.append({
            "severity": "HIGH", "code": "LOW_TRAIN_GENUINE", "cohort": cohort,
            "model_family": family, "user_id": user_id,
            "message": f"Only {len(pos_all)} genuine training windows; model estimates may be unstable.",
        })

    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed = stable_seed32(cohort, family, claimed)
    start = time.perf_counter()
    convergence_messages: list[str] = []

    if family == "centroid":
        pos_used = pos_all
        neg_used: list[int] = []
        center = X[pos_used].mean(axis=0, dtype=np.float64).astype(np.float32)
        model_obj = {"center": center}
        artifact = artifact_dir / f"{user_id}.npz"
        np.savez_compressed(artifact, center=center, claimed_user_index=np.int16(claimed))
        parameter_count = int(center.size)
        parameter_definition = "centroid coordinates"

    elif family == "oneclass_svm":
        target = min(len(pos_all), OCSVM_MAX_GENUINE)
        pos_used = session_round_robin(pos_all, sessions, target, cohort, family, claimed, "positive_select")
        neg_used = []
        if len(pos_all) > target:
            warnings_rows.append({
                "severity": "INFO", "code": "OCSVM_GENUINE_CAP", "cohort": cohort,
                "model_family": family, "user_id": user_id,
                "message": f"Used deterministic session-balanced cap {target}/{len(pos_all)} for CPU-safe OCSVM fitting.",
            })
        model_obj = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05, cache_size=512)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model_obj.fit(X[pos_used])
            convergence_messages = [str(w.message) for w in caught]
        artifact = artifact_dir / f"{user_id}.joblib"
        joblib.dump(model_obj, artifact, compress=3)
        parameter_count = int(model_obj.support_vectors_.size + model_obj.dual_coef_.size + model_obj.intercept_.size)
        parameter_definition = "support-vector coordinates + dual coefficients + intercept"

    elif family == "linear_svm":
        pos_used = pos_all
        neg_used = select_balanced_negatives(y, sessions, claimed, len(pos_used), cohort, family)
        train_idx = np.asarray(pos_used + neg_used, dtype=np.int64)
        target_y = np.concatenate([
            np.ones(len(pos_used), dtype=np.int8), np.zeros(len(neg_used), dtype=np.int8)
        ])
        order = np.argsort([stable_rank(cohort, family, claimed, "train_order", int(i)) for i in train_idx])
        train_idx, target_y = train_idx[order], target_y[order]
        model_obj = LinearSVC(C=1.0, class_weight=None, dual="auto", max_iter=20000, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model_obj.fit(X[train_idx], target_y)
            convergence_messages = [str(w.message) for w in caught]
        artifact = artifact_dir / f"{user_id}.joblib"
        joblib.dump(model_obj, artifact, compress=3)
        parameter_count = int(model_obj.coef_.size + model_obj.intercept_.size)
        parameter_definition = "linear coefficients + intercept"

    elif family == "random_forest":
        pos_used = pos_all
        neg_used = select_balanced_negatives(y, sessions, claimed, len(pos_used), cohort, family)
        train_idx = np.asarray(pos_used + neg_used, dtype=np.int64)
        target_y = np.concatenate([
            np.ones(len(pos_used), dtype=np.int8), np.zeros(len(neg_used), dtype=np.int8)
        ])
        order = np.argsort([stable_rank(cohort, family, claimed, "train_order", int(i)) for i in train_idx])
        train_idx, target_y = train_idx[order], target_y[order]
        model_obj = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=1, max_features="sqrt",
            bootstrap=True, class_weight=None, random_state=seed, n_jobs=1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model_obj.fit(X[train_idx], target_y)
            convergence_messages = [str(w.message) for w in caught]
        artifact = artifact_dir / f"{user_id}.joblib"
        joblib.dump(model_obj, artifact, compress=3)
        parameter_count = int(sum(est.tree_.node_count for est in model_obj.estimators_))
        parameter_definition = "total decision-tree nodes (structural count)"
    else:
        raise ValueError(family)

    elapsed = time.perf_counter() - start
    for msg in convergence_messages:
        warnings_rows.append({
            "severity": "MEDIUM", "code": "SKLEARN_FIT_WARNING", "cohort": cohort,
            "model_family": family, "user_id": user_id, "message": msg,
        })

    training_audit = {
        "cohort": cohort, "model_family": family, "user_id": user_id,
        "claimed_user_index": claimed, "feature_count": int(X.shape[1]),
        "genuine_available": len(pos_all), "genuine_used": len(pos_used),
        "impostor_used": len(neg_used), "training_rows": len(pos_used) + len(neg_used),
        "positive_sessions_available": len({sessions[i] for i in pos_all}),
        "positive_sessions_used": len({sessions[i] for i in pos_used}),
        "negative_users_used": len({int(y[i]) for i in neg_used}),
        "negative_sessions_used": len({sessions[i] for i in neg_used}),
        "random_seed": seed,
    }
    resource = {
        "cohort": cohort, "model_family": family, "user_id": user_id,
        "train_seconds": round(elapsed, 6), "model_size_bytes": artifact.stat().st_size,
        "parameter_count": parameter_count, "parameter_count_definition": parameter_definition,
    }
    fitted = FittedModel(family, claimed, artifact, model_obj, parameter_count, parameter_definition)
    return fitted, training_audit, resource


def score_model(fitted: FittedModel, X: np.ndarray) -> np.ndarray:
    if fitted.family == "centroid":
        center = fitted.object["center"]
        # Negative RMS Euclidean distance; higher (closer to zero) means more genuine.
        d = X.astype(np.float64) - center.astype(np.float64)
        return (-np.sqrt(np.mean(d * d, axis=1))).astype(np.float64)
    if fitted.family == "oneclass_svm":
        return np.asarray(fitted.object.decision_function(X), dtype=np.float64).reshape(-1)
    if fitted.family == "linear_svm":
        return np.asarray(fitted.object.decision_function(X), dtype=np.float64).reshape(-1)
    if fitted.family == "random_forest":
        classes = fitted.object.classes_.tolist()
        if 1 not in classes:
            raise ValueError("Random forest lacks positive class")
        col = classes.index(1)
        return np.asarray(fitted.object.predict_proba(X)[:, col], dtype=np.float64)
    raise ValueError(fitted.family)


def write_claim_scores(claim_path: Path, out_path: Path, score_matrix: np.ndarray,
                       family: str, cohort: str, protocol: str, split: str,
                       diagnostic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_ids = 0
    seen: set[str] = set()
    finite_count = 0
    total = 0
    y_global: list[int] = []
    s_global: list[float] = []
    by_user_y: dict[str, list[int]] = defaultdict(list)
    by_user_s: dict[str, list[float]] = defaultdict(list)

    with claim_path.open(newline="", encoding="utf-8-sig") as fi, \
         out_path.open("w", newline="", encoding="utf-8-sig") as fo:
        r = csv.DictReader(fi)
        if not r.fieldnames or any(c not in r.fieldnames for c in CLAIM_REQUIRED):
            raise ValueError(f"Claim schema mismatch: {claim_path}")
        fields = list(r.fieldnames) + SCORE_APPEND
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        for row in r:
            total += 1
            cid = row["claim_id"]
            if cid in seen:
                duplicate_ids += 1
            seen.add(cid)
            src = int(row["source_row_index"])
            claimed = int(row["claimed_user_index"])
            if src < 0 or src >= score_matrix.shape[0] or claimed < 0 or claimed >= score_matrix.shape[1]:
                raise IndexError(f"Claim index out of bounds in {claim_path}: src={src}, claimed={claimed}")
            score = float(score_matrix[src, claimed])
            if math.isfinite(score):
                finite_count += 1
            score_text = format(score, ".12g")
            score_for_output = float(score_text)
            row.update({
                "model_id": f"{family}__{cohort}__{row['claimed_user_id']}",
                "model_family": family,
                "raw_score": score_text,
                "score_higher_is_genuine": 1,
            })
            w.writerow(row)
            if split == "validation":
                yy = int(row["is_genuine"])
                uid = row["claimed_user_id"]
                # Diagnostics are computed from the exact persisted score precision.
                y_global.append(yy); s_global.append(score_for_output)
                by_user_y[uid].append(yy); by_user_s[uid].append(score_for_output)

    if split == "validation":
        user_aucs: list[float] = []
        for uid in USERS:
            yy, ss = by_user_y[uid], by_user_s[uid]
            auc = float("nan")
            if len(set(yy)) == 2:
                auc = float(roc_auc_score(yy, ss))
                user_aucs.append(auc)
            diagnostic_rows.append({
                "cohort": cohort, "model_family": family, "protocol": protocol,
                "scope": "user", "claimed_user_id": uid,
                "genuine_claims": sum(yy), "impostor_claims": len(yy) - sum(yy),
                "validation_auc_raw": "" if not math.isfinite(auc) else f"{auc:.12g}",
                "test_metrics_computed": 0,
            })
        global_auc = float(roc_auc_score(y_global, s_global)) if len(set(y_global)) == 2 else float("nan")
        diagnostic_rows.append({
            "cohort": cohort, "model_family": family, "protocol": protocol,
            "scope": "global_claim_weighted", "claimed_user_id": "__GLOBAL__",
            "genuine_claims": sum(y_global), "impostor_claims": len(y_global) - sum(y_global),
            "validation_auc_raw": "" if not math.isfinite(global_auc) else f"{global_auc:.12g}",
            "test_metrics_computed": 0,
        })
        macro_auc = float(np.mean(user_aucs)) if user_aucs else float("nan")
        diagnostic_rows.append({
            "cohort": cohort, "model_family": family, "protocol": protocol,
            "scope": "macro_user", "claimed_user_id": "__MACRO_USER__",
            "genuine_claims": sum(y_global), "impostor_claims": len(y_global) - sum(y_global),
            "validation_auc_raw": "" if not math.isfinite(macro_auc) else f"{macro_auc:.12g}",
            "test_metrics_computed": 0,
        })

    return {
        "cohort": cohort, "model_family": family, "protocol": protocol, "split": split,
        "relative_path": str(out_path), "claims": total, "finite_scores": finite_count,
        "duplicate_claim_ids": duplicate_ids, "sha256": sha256_file(out_path),
    }


def zip_directory(root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path(root.name) / p.relative_to(root)))


def make_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "output_manifest_sha256.csv":
            rows.append({
                "relative_path": p.relative_to(root).as_posix(),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    return rows


def verify_manifest(root: Path, rows: list[dict[str, Any]]) -> None:
    failures: list[str] = []
    for row in rows:
        p = root / str(row["relative_path"])
        if not p.exists():
            failures.append(f"missing:{row['relative_path']}")
            continue
        if p.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size:{row['relative_path']}")
            continue
        if sha256_file(p) != row["sha256"]:
            failures.append(f"sha256:{row['relative_path']}")
    if failures:
        raise RuntimeError(f"Final output manifest verification failed: {failures[:10]}")


def verify_zip(zip_path: Path, expected_files: int) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Final ZIP CRC verification failed at: {bad_member}")
        actual_files = sum(1 for n in zf.namelist() if not n.endswith("/"))
    if actual_files != expected_files:
        raise RuntimeError(
            f"Final ZIP file-count mismatch: expected={expected_files}, actual={actual_files}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=STAGE)
    ap.add_argument("--input-stage2c", required=True, help="Stage 2C review ZIP or directory")
    ap.add_argument("--input-stage3a", required=True, help="Stage 3A review ZIP or directory")
    ap.add_argument("--output", required=True, help="Output directory; ZIP is also created")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    started_at = now_iso()
    t0 = time.perf_counter()
    input2c = Path(args.input_stage2c).expanduser().resolve()
    input3a = Path(args.input_stage3a).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out_zip = out.with_suffix(".zip")

    if out.exists() or out_zip.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists. Use --overwrite: {out} or {out_zip}")
        if out.exists():
            remove_tree_with_retries(out, fatal=True)
        if out_zip.exists():
            out_zip.unlink()
    out.mkdir(parents=True)

    warning_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    score_index_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    execution_lines = [f"{STAGE} v{VERSION}", f"Started: {started_at}"]

    with resilient_temp_workspace(prefix="km_stage3b_") as temp:
        root2c, zip2c_sha = materialize_input(input2c, "stage2c_summary.json", temp, "stage2c")
        root3a, zip3a_sha = materialize_input(input3a, "stage3a_summary.json", temp, "stage3a")
        s2 = load_json(root2c / "stage2c_summary.json")
        s3 = load_json(root3a / "stage3a_summary.json")
        split2 = s2.get("input_stage2b", {}).get("stage2a_split_sha256", "")
        split3 = s3.get("input_stage2c", {}).get("stage2a_split_sha256", "")
        if not split2 or split2 != split3:
            raise ValueError(f"Frozen split fingerprint mismatch: Stage2C={split2}, Stage3A={split3}")
        if s2.get("failure_count", 0) or s3.get("failure_count", 0):
            raise ValueError("Input stage reports failures")

        execution_lines += [
            f"Stage2C root: {root2c}", f"Stage3A root: {root3a}",
            f"Frozen split SHA256: {split2}", f"Python: {sys.version.split()[0]}",
            f"NumPy: {np.__version__}", f"scikit-learn: {sklearn.__version__}",
        ]

        frozen_protocol = {
            "stage": STAGE, "version": VERSION, "base_seed": BASE_SEED,
            "inputs": {
                "stage2c_path": str(input2c), "stage2c_zip_sha256": zip2c_sha,
                "stage3a_path": str(input3a), "stage3a_zip_sha256": zip3a_sha,
                "frozen_stage2a_split_sha256": split2,
            },
            "cohorts": COHORTS, "model_families": MODELS,
            "training": {
                "centroid": "all genuine training windows; negative RMS Euclidean distance score",
                "oneclass_svm": {
                    "training": "genuine windows only", "kernel": "RBF", "gamma": "scale",
                    "nu": 0.05, "max_genuine_windows": OCSVM_MAX_GENUINE,
                    "cap_sampling": "deterministic session-aware round robin",
                },
                "linear_svm": {
                    "training": "all genuine + equal number unique impostor windows",
                    "C": 1.0, "dual": "auto", "max_iter": 20000,
                    "negative_sampling": "capacity-aware user-balanced and session-aware; training only",
                },
                "random_forest": {
                    "training": "all genuine + equal number unique impostor windows",
                    "n_estimators": 100, "max_depth": 12, "min_samples_leaf": 1,
                    "max_features": "sqrt", "bootstrap": True, "n_jobs": 1,
                    "negative_sampling": "capacity-aware user-balanced and session-aware; training only",
                },
            },
            "score_orientation": "Higher raw score means more genuine for every model family.",
            "score_outputs": "Validation and test claim-level raw scores for both frozen Stage 3A protocols.",
            "prohibitions": [
                "No calibration", "No threshold selection", "No test metric calculation",
                "No test-driven model or hyperparameter selection", "No claim resampling",
                "No refitting of Stage 2C preprocessors",
            ],
        }
        write_json(out / "frozen_model_protocol.json", frozen_protocol)

        all_score_seconds: dict[tuple[str, str], float] = defaultdict(float)
        all_score_claims: dict[tuple[str, str], int] = defaultdict(int)

        with PeakRSSMonitor() as peak_monitor:
            for cohort in COHORTS:
                with np.load(root2c / "tabular" / f"{cohort}_train.npz", allow_pickle=False) as train_npz:
                    X_train = np.array(train_npz["X"], dtype=np.float32, copy=True)
                    y_train = np.array(train_npz["y_user"], dtype=np.int16, copy=True)
                sessions = load_index_sessions(root2c / "indexes" / f"{cohort}_train_index.csv", len(y_train))
                if not np.isfinite(X_train).all():
                    raise ValueError(f"Non-finite training array: {cohort}")
                if sorted(np.unique(y_train).tolist()) != list(range(11)):
                    raise ValueError(f"Unexpected user labels in {cohort}")

                split_arrays = {}
                for split in SCORE_SPLITS:
                    with np.load(root2c / "tabular" / f"{cohort}_{split}.npz", allow_pickle=False) as z:
                        Xs = np.array(z["X"], dtype=np.float32, copy=True)
                    if not np.isfinite(Xs).all():
                        raise ValueError(f"Non-finite {cohort}/{split} array")
                    split_arrays[split] = Xs

                for family in MODELS:
                    fitted_models: dict[int, FittedModel] = {}
                    for claimed in range(11):
                        fitted, audit, resource = fit_one_model(
                            family, X_train, y_train, sessions, claimed, cohort,
                            out / "models" / cohort / family, warning_rows,
                        )
                        fitted_models[claimed] = fitted
                        training_rows.append(audit)
                        resource_rows.append(resource)

                    score_matrices: dict[str, np.ndarray] = {}
                    for split, Xs in split_arrays.items():
                        matrix = np.empty((Xs.shape[0], 11), dtype=np.float64)
                        st = time.perf_counter()
                        for claimed in range(11):
                            matrix[:, claimed] = score_model(fitted_models[claimed], Xs)
                        elapsed = time.perf_counter() - st
                        if not np.isfinite(matrix).all():
                            raise ValueError(f"Non-finite score matrix: {cohort}/{family}/{split}")
                        score_matrices[split] = matrix
                        resource_rows.append({
                            "cohort": cohort, "model_family": family, "user_id": "__ALL_USERS__",
                            "train_seconds": "", "model_size_bytes": "",
                            "parameter_count": "", "parameter_count_definition": "",
                            "score_split": split, "score_windows": Xs.shape[0],
                            "score_user_models": 11, "score_seconds": round(elapsed, 6),
                            "score_ms_per_window_user_pair": round(elapsed * 1000 / (Xs.shape[0] * 11), 9),
                        })

                    for protocol in PROTOCOLS:
                        for split in SCORE_SPLITS:
                            claim_path = root3a / "claims" / cohort / protocol / f"{split}_claims.csv"
                            out_path = out / "scores" / cohort / family / protocol / f"{split}_claim_scores.csv"
                            st = time.perf_counter()
                            idx_row = write_claim_scores(
                                claim_path, out_path, score_matrices[split], family, cohort,
                                protocol, split, diagnostic_rows,
                            )
                            elapsed = time.perf_counter() - st
                            idx_row["relative_path"] = out_path.relative_to(out).as_posix()
                            score_index_rows.append(idx_row)
                            all_score_seconds[(cohort, family)] += elapsed
                            all_score_claims[(cohort, family)] += int(idx_row["claims"])
                            integrity_rows.extend([
                                {"cohort": cohort, "model_family": family, "protocol": protocol,
                                 "split": split, "check": "row_count_positive", "value": idx_row["claims"],
                                 "status": "PASS" if idx_row["claims"] > 0 else "FAIL"},
                                {"cohort": cohort, "model_family": family, "protocol": protocol,
                                 "split": split, "check": "all_scores_finite", "value": idx_row["finite_scores"],
                                 "expected": idx_row["claims"],
                                 "status": "PASS" if idx_row["finite_scores"] == idx_row["claims"] else "FAIL"},
                                {"cohort": cohort, "model_family": family, "protocol": protocol,
                                 "split": split, "check": "duplicate_claim_ids", "value": idx_row["duplicate_claim_ids"],
                                 "expected": 0, "status": "PASS" if idx_row["duplicate_claim_ids"] == 0 else "FAIL"},
                            ])

                    # Release potentially large forest objects before next family/cohort.
                    fitted_models.clear()
                    score_matrices.clear()
                    gc.collect()

                # Explicitly release all cohort arrays before the temporary workspace is cleaned.
                split_arrays.clear()
                del X_train, y_train, sessions
                gc.collect()

            peak_rss_mb = peak_monitor.peak_mb

        # Aggregate resource totals.
        for (cohort, family), seconds in sorted(all_score_seconds.items()):
            claims = all_score_claims[(cohort, family)]
            resource_rows.append({
                "cohort": cohort, "model_family": family, "user_id": "__CLAIM_SCORE_OUTPUT__",
                "train_seconds": "", "model_size_bytes": "", "parameter_count": "",
                "parameter_count_definition": "", "score_split": "validation+test;both_protocols",
                "score_windows": "", "score_user_models": 11,
                "score_seconds": round(seconds, 6),
                "score_ms_per_claim_output": round(seconds * 1000 / claims, 9) if claims else "",
                "claims_written": claims,
            })

        # Integrity status.
        failed_checks = [r for r in integrity_rows if r.get("status") != "PASS"]
        for r in failed_checks:
            failure_rows.append({
                "severity": "CRITICAL", "code": "INTEGRITY_CHECK_FAILED",
                "cohort": r.get("cohort", ""), "model_family": r.get("model_family", ""),
                "user_id": "", "message": json.dumps(r, ensure_ascii=False),
            })

        write_csv(out / "training_sample_audit.csv", training_rows, [
            "cohort", "model_family", "user_id", "claimed_user_index", "feature_count",
            "genuine_available", "genuine_used", "impostor_used", "training_rows",
            "positive_sessions_available", "positive_sessions_used", "negative_users_used",
            "negative_sessions_used", "random_seed",
        ])
        resource_fields = [
            "cohort", "model_family", "user_id", "train_seconds", "model_size_bytes",
            "parameter_count", "parameter_count_definition", "score_split", "score_windows",
            "score_user_models", "score_seconds", "score_ms_per_window_user_pair",
            "score_ms_per_claim_output", "claims_written",
        ]
        write_csv(out / "model_resource_metrics.csv", resource_rows, resource_fields)
        write_csv(out / "score_file_index.csv", score_index_rows, [
            "cohort", "model_family", "protocol", "split", "relative_path", "claims",
            "finite_scores", "duplicate_claim_ids", "sha256",
        ])
        write_csv(out / "score_integrity_checks.csv", integrity_rows, [
            "cohort", "model_family", "protocol", "split", "check", "value", "expected", "status",
        ])
        write_csv(out / "validation_auc_diagnostics.csv", diagnostic_rows, [
            "cohort", "model_family", "protocol", "scope", "claimed_user_id",
            "genuine_claims", "impostor_claims", "validation_auc_raw", "test_metrics_computed",
        ])
        warning_fields = ["severity", "code", "cohort", "model_family", "user_id", "message"]
        write_csv(out / "warnings_and_failures.csv", warning_rows + failure_rows, warning_fields)

        decisions = f"""# Stage 3B Classical Pilot — Frozen Decisions\n\n- Version: {VERSION}\n- Frozen Stage 2A split SHA-256: `{split2}`\n- Base seed: `{BASE_SEED}`\n- Cohorts: {', '.join(COHORTS)}\n- Models: Centroid, One-Class SVM, Linear SVM, Random Forest.\n- Training data: Stage 2C Training only.\n- Scored data: Stage 3A Validation and Test claims only.\n- Higher raw score always means more genuine.\n- No calibration, threshold, test metrics, or test-driven model selection.\n- Validation AUC is diagnostic only and does not alter scores, models, or hyperparameters.\n- Balanced discriminative training uses all genuine windows plus an equal number of unique impostor windows, selected deterministically across users and sessions.\n- One-Class SVM uses genuine windows only, capped at {OCSVM_MAX_GENUINE} with deterministic session-aware selection for CPU safety.\n- Deep sequence models are deferred to Stage 3C.\n"""
        (out / "stage3b_frozen_decisions.md").write_text(decisions, encoding="utf-8")

        finished_at = now_iso()
        elapsed_total = time.perf_counter() - t0
        status = "FAIL" if failure_rows else ("PASS_WITH_WARNINGS" if warning_rows else "PASS")
        summary = {
            "stage": STAGE, "version": VERSION, "status": status,
            "started_at": started_at, "finished_at": finished_at,
            "elapsed_seconds": round(elapsed_total, 3),
            "inputs": {
                "stage2c": str(input2c), "stage2c_zip_sha256": zip2c_sha,
                "stage3a": str(input3a), "stage3a_zip_sha256": zip3a_sha,
                "frozen_stage2a_split_sha256": split2,
            },
            "environment": {
                "python": sys.version.split()[0], "numpy": np.__version__,
                "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
                "psutil": getattr(psutil, "__version__", "not_installed"),
            },
            "cohorts": COHORTS, "model_families": MODELS,
            "trained_user_models": len(COHORTS) * len(MODELS) * len(USERS),
            "score_files": len(score_index_rows),
            "total_claim_scores": sum(int(r["claims"]) for r in score_index_rows),
            "validation_diagnostics_only": True, "test_metrics_computed": False,
            "warning_count": len(warning_rows), "failure_count": len(failure_rows),
            "failed_integrity_checks": len(failed_checks),
            "peak_process_rss_mb": None if not math.isfinite(peak_rss_mb) else round(peak_rss_mb, 3),
        }
        write_json(out / "stage3b_summary.json", summary)
        execution_lines += [
            f"Finished: {finished_at}", f"Elapsed seconds: {elapsed_total:.3f}",
            f"Status: {status}", f"Warnings: {len(warning_rows)}", f"Failures: {len(failure_rows)}",
            f"Score files: {len(score_index_rows)}", f"Claim scores: {summary['total_claim_scores']}",
            f"Peak RSS MB: {summary['peak_process_rss_mb']}",
        ]
        (out / "execution_log.txt").write_text("\n".join(execution_lines) + "\n", encoding="utf-8")

    # Completion marker is written only after training/scoring and temp-workspace exit.
    (out / "STAGE3B_COMPLETE.ok").write_text(
        f"{STAGE} v{VERSION} completed successfully at {now_iso()}\n", encoding="utf-8"
    )
    manifest_rows = make_manifest(out)
    write_csv(out / "output_manifest_sha256.csv", manifest_rows, ["relative_path", "size_bytes", "sha256"])
    verify_manifest(out, manifest_rows)
    zip_directory(out, out_zip)
    verify_zip(out_zip, expected_files=len(manifest_rows) + 1)
    print(json.dumps({
        "status": load_json(out / "stage3b_summary.json")["status"],
        "output_directory": str(out), "output_zip": str(out_zip),
        "zip_size_bytes": out_zip.stat().st_size,
        "manifest_files_verified": len(manifest_rows),
        "zip_crc_verified": True,
    }, ensure_ascii=False, indent=2))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
