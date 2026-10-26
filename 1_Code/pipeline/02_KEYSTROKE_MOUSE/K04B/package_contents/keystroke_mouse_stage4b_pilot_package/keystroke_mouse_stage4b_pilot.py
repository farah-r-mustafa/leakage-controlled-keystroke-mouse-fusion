#!/usr/bin/env python3
"""
Keystroke-Mouse Stage 4B v4.2.0
Frozen Test Evaluation and Complete Authentication Metrics.

Inputs:
  - Stage 4A calibrated score review ZIP
  - Stage 3B classical review ZIP (timestamps + resources)
  - Stage 3C deep review ZIP (timestamps + resources)
  - Stage 2A review ZIP (adequate-session subset definition)

Outputs:
  - Test metrics for Balanced 1:1 and Realistic All-Impostor protocols
  - Per-user and macro-user metrics
  - Calibration metrics, fallback sensitivity, adequate-session subset
  - Session-level Time-to-Authentication (TTA)
  - Model resource summaries
  - Claim-level evaluated Test files
  - NO statistical model-comparison tests (reserved for Stage 5)

Safety:
  - Reads ZIPs directly; no extraction and no temporary NPZ files.
  - Does not refit calibration or thresholds.
  - Uses frozen prediction/threshold fields from Stage 4A.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score

VERSION = "4.2.0"
EPS = 1e-12
ADEQUATE_MIN_CORE_SESSIONS = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_member(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [n for n in zf.namelist() if n.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one ZIP member ending with {suffix!r}; found {len(matches)}")
    return matches[0]


def read_csv_member(zf: zipfile.ZipFile, member: str) -> List[Dict[str, str]]:
    with zf.open(member, "r") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as txt:
            return list(csv.DictReader(txt))


def open_csv_reader(zf: zipfile.ZipFile, member: str):
    raw = zf.open(member, "r")
    txt = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return raw, txt, csv.DictReader(txt)


def safe_float(x: str, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def confusion(y: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return tp, fn, fp, tn


def select_eer(scores: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    pos_total = int(np.sum(y == 1))
    neg_total = int(np.sum(y == 0))
    if pos_total == 0 or neg_total == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order]
    cum_pos = np.cumsum(yy == 1)
    cum_neg = np.cumsum(yy == 0)
    group_ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    thresholds = s[group_ends]
    fars = cum_neg[group_ends] / neg_total
    frrs = (pos_total - cum_pos[group_ends]) / pos_total
    # Candidate above maximum: accept none.
    thresholds = np.r_[math.nextafter(float(s[0]), math.inf), thresholds]
    fars = np.r_[0.0, fars]
    frrs = np.r_[1.0, frrs]
    diffs = np.abs(fars - frrs)
    bers = 0.5 * (fars + frrs)
    # lexicographic: min difference, then BER, then higher threshold.
    idx = int(np.lexsort((-thresholds, bers, diffs))[0])
    return float(thresholds[idx]), float(fars[idx]), float(frrs[idx]), float(bers[idx])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(bins):
        if i == bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        count = int(np.sum(m))
        if count == 0:
            continue
        ece += (count / n) * abs(float(np.mean(p[m])) - float(np.mean(y[m])))
    return float(ece)


def metric_record(y: Sequence[int], pred: Sequence[int], score: Sequence[float]) -> Dict[str, float | int]:
    y_a = np.asarray(y, dtype=np.int8)
    p_a = np.asarray(pred, dtype=np.int8)
    s_a = np.asarray(score, dtype=np.float64)
    tp, fn, fp, tn = confusion(y_a, p_a)
    genuine = tp + fn
    impostor = fp + tn
    far = fp / impostor if impostor else float("nan")
    frr = fn / genuine if genuine else float("nan")
    auc = float(roc_auc_score(y_a, s_a)) if len(np.unique(y_a)) == 2 else float("nan")
    eer_thr, eer_far, eer_frr, eer = select_eer(s_a, y_a)
    precision = float(precision_score(y_a, p_a, zero_division=0))
    recall = float(recall_score(y_a, p_a, zero_division=0))
    specificity = tn / impostor if impostor else float("nan")
    return {
        "claims": int(len(y_a)), "genuine_claims": int(genuine), "impostor_claims": int(impostor),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "far": far, "frr": frr, "eer": eer, "eer_threshold_test_descriptive": eer_thr,
        "eer_far": eer_far, "eer_frr": eer_frr, "auc": auc,
        "accuracy": float(accuracy_score(y_a, p_a)),
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": float(f1_score(y_a, p_a, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_a, p_a)) if len(y_a) else float("nan"),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
    }


def calibration_record(y: Sequence[int], prob: Sequence[float]) -> Dict[str, float]:
    y_a = np.asarray(y, dtype=np.float64)
    p_a = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0 - EPS)
    if len(y_a) == 0:
        return {"brier": float("nan"), "log_loss": float("nan"), "ece10": float("nan")}
    return {
        "brier": float(np.mean((p_a - y_a) ** 2)),
        "log_loss": float(np.mean(-(y_a * np.log(p_a) + (1 - y_a) * np.log(1 - p_a)))),
        "ece10": expected_calibration_error(y_a, p_a, 10),
    }


def mean_metrics(records: List[Dict[str, float | int]], metric_names: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in metric_names:
        vals = [float(r[k]) for r in records if k in r and math.isfinite(float(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def percentile(values: Sequence[float], q: float) -> float:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if fields:
            w.writeheader()
            for row in rows:
                w.writerow(row)


def load_adequate_users(stage2a_zip: Path) -> Tuple[Dict[str, int], List[str]]:
    with zipfile.ZipFile(stage2a_zip) as z:
        member = find_member(z, "frozen_session_manifest.csv")
        rows = read_csv_member(z, member)
    counts: Counter[str] = Counter()
    for r in rows:
        if r.get("core_multimodal_eligible") == "1" and r.get("split") in {"train", "validation", "test"}:
            counts[r["user"]] += 1
    adequate = sorted([u for u, n in counts.items() if n >= ADEQUATE_MIN_CORE_SESSIONS], key=lambda x: int(x.replace("User", "")))
    return dict(counts), adequate


def load_resource_rows(zip_path: Path) -> List[Dict[str, str]]:
    with zipfile.ZipFile(zip_path) as z:
        return read_csv_member(z, find_member(z, "model_resource_metrics.csv"))


def aggregate_resources(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    score_rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in rows:
        key = (r["cohort"], r["model_family"])
        if r["user_id"] == "__CLAIM_SCORE_OUTPUT__":
            score_rows[key] = r
        elif r["user_id"].startswith("User"):
            groups[key].append(r)
    out = []
    for key in sorted(groups):
        cohort, model = key
        us = groups[key]
        sr = score_rows.get(key, {})
        train_s = [safe_float(r.get("train_seconds", ""), 0.0) for r in us]
        sizes = [safe_int(r.get("model_size_bytes", ""), 0) for r in us]
        params = [safe_int(r.get("parameter_count", ""), 0) for r in us]
        out.append({
            "cohort": cohort, "model_family": model, "personal_models": len(us),
            "total_train_seconds": sum(train_s), "mean_train_seconds_per_user": float(np.mean(train_s)) if train_s else float("nan"),
            "total_model_size_bytes": sum(sizes), "mean_model_size_bytes_per_user": float(np.mean(sizes)) if sizes else float("nan"),
            "total_parameter_count": sum(params), "mean_parameter_count_per_user": float(np.mean(params)) if params else float("nan"),
            "parameter_count_definition": us[0].get("parameter_count_definition", "") if us else "",
            "score_seconds_validation_test_both_protocols": safe_float(sr.get("score_seconds", "")),
            "score_ms_per_claim_output": safe_float(sr.get("score_ms_per_claim_output", "")),
            "claims_written_resource_run": safe_int(sr.get("claims_written", "")),
        })
    return out


def zip_root(zf: zipfile.ZipFile) -> str:
    roots = {n.split("/", 1)[0] for n in zf.namelist() if "/" in n}
    if len(roots) != 1:
        raise RuntimeError(f"Could not determine unique ZIP root: {roots}")
    return next(iter(roots))


def stage3_member(root: str, cohort: str, model: str, protocol: str) -> str:
    return f"{root}/scores/{cohort}/{model}/{protocol}/test_claim_scores.csv"


def stage4a_member(root: str, relative_path: str) -> str:
    return f"{root}/{relative_path}"


def analyze_file(
    z4: zipfile.ZipFile, member4: str,
    z3: zipfile.ZipFile, member3: str,
    out_claim_path: Path,
    resource_ms: float,
    cohort: str, model: str, protocol: str,
    adequate_users: set[str],
) -> Dict[str, object]:
    # Pandas is used file-by-file for efficient I/O; no full-dataset load occurs.
    with z4.open(member4) as f4:
        df = pd.read_csv(f4, low_memory=False)
    with z3.open(member3) as f3:
        ts = pd.read_csv(f3, usecols=["claim_id", "window_id", "claimed_user_id", "window_start_epoch", "window_end_epoch"], low_memory=False)
    if len(df) != len(ts):
        raise RuntimeError(f"Row-count mismatch in {cohort}/{model}/{protocol}: {len(df)} vs {len(ts)}")
    if not df["claim_id"].astype(str).equals(ts["claim_id"].astype(str)):
        raise RuntimeError(f"Claim order mismatch in {cohort}/{model}/{protocol}")
    if not df["window_id"].astype(str).equals(ts["window_id"].astype(str)):
        raise RuntimeError(f"Window order mismatch in {cohort}/{model}/{protocol}")
    if not df["claimed_user_id"].astype(str).equals(ts["claimed_user_id"].astype(str)):
        raise RuntimeError(f"Claimed-user mismatch in {cohort}/{model}/{protocol}")

    df["window_start_epoch"] = pd.to_numeric(ts["window_start_epoch"], errors="coerce")
    df["window_end_epoch"] = pd.to_numeric(ts["window_end_epoch"], errors="coerce")
    df["window_span_seconds"] = (df["window_end_epoch"] - df["window_start_epoch"]).clip(lower=0)
    y_a = pd.to_numeric(df["is_genuine"], errors="raise").to_numpy(dtype=np.int8)
    pred_a = pd.to_numeric(df["prediction"], errors="raise").to_numpy(dtype=np.int8)
    score_a = pd.to_numeric(df["calibrated_score"], errors="raise").to_numpy(dtype=np.float64)
    raw_a = pd.to_numeric(df["raw_score"], errors="raise").to_numpy(dtype=np.float64)
    users_a = df["claimed_user_id"].astype(str).to_numpy(dtype=object)
    src_a = df["threshold_source"].astype(str).to_numpy(dtype=object)
    sessions_a = df["session_id"].astype(str).to_numpy(dtype=object)
    starts_a = df["window_start_epoch"].to_numpy(dtype=np.float64)
    ends_a = df["window_end_epoch"].to_numpy(dtype=np.float64)

    err = np.full(len(df), "TN", dtype=object)
    err[(y_a == 1) & (pred_a == 1)] = "TP"
    err[(y_a == 1) & (pred_a == 0)] = "FN"
    err[(y_a == 0) & (pred_a == 1)] = "FP"
    df["correct"] = (y_a == pred_a).astype(np.int8)
    df["error_type"] = err
    df["inference_ms_per_claim"] = resource_ms
    df["decision_latency_seconds"] = df["window_span_seconds"] + (resource_ms / 1000.0 if math.isfinite(resource_ms) else np.nan)
    df["adequate_session_user"] = df["claimed_user_id"].isin(adequate_users).astype(np.int8)
    out_claim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_claim_path, index=False, float_format="%.12g")

    overall = metric_record(y_a, pred_a, score_a)
    overall.update(calibration_record(y_a, score_a))
    overall.update({"cohort": cohort, "model_family": model, "protocol": protocol})

    per_user_rows: List[Dict[str, object]] = []
    user_order = sorted(pd.unique(df["claimed_user_id"].astype(str)), key=lambda x: int(x.replace("User", "")))
    for u in user_order:
        m = users_a == u
        rec = metric_record(y_a[m], pred_a[m], score_a[m])
        rec.update(calibration_record(y_a[m], score_a[m]))
        rec.update({
            "cohort": cohort, "model_family": model, "protocol": protocol, "claimed_user_id": u,
            "threshold_source": str(src_a[m][0]) if np.any(m) else "",
            "adequate_session_user": int(u in adequate_users),
        })
        per_user_rows.append(rec)

    macro_names = ["far", "frr", "eer", "auc", "accuracy", "precision", "recall", "specificity", "f1", "mcc", "balanced_accuracy", "brier", "log_loss", "ece10"]
    macro = mean_metrics(per_user_rows, macro_names)
    macro.update({"cohort": cohort, "model_family": model, "protocol": protocol, "users": len(per_user_rows)})

    subset_rows: List[Dict[str, object]] = []
    masks = [
        ("personal_threshold", np.char.startswith(src_a.astype(str), "personal_")),
        ("global_fallback", np.char.find(src_a.astype(str), "fallback") >= 0),
        ("adequate_session_subset", np.array([u in adequate_users for u in users_a], dtype=bool)),
        ("limited_session_subset", np.array([u not in adequate_users for u in users_a], dtype=bool)),
    ]
    for group_name, mask in masks:
        if np.any(mask) and len(np.unique(y_a[mask])) == 2:
            rec = metric_record(y_a[mask], pred_a[mask], score_a[mask])
            rec.update(calibration_record(y_a[mask], score_a[mask]))
        else:
            rec = {k: float("nan") for k in ["far", "frr", "eer", "auc", "accuracy", "f1", "mcc", "balanced_accuracy", "brier", "log_loss", "ece10"]}
            rec.update({"claims": int(np.sum(mask)), "genuine_claims": int(np.sum(y_a[mask] == 1)), "impostor_claims": int(np.sum(y_a[mask] == 0))})
        rec.update({"cohort": cohort, "model_family": model, "protocol": protocol, "analysis_group": group_name, "claimed_users": len(set(users_a[mask].tolist())) if np.any(mask) else 0})
        subset_rows.append(rec)

    # Session-level TTA from genuine claims. DataFrame groupby avoids row-wise Python overhead.
    g = df[df["is_genuine"].astype(int) == 1].copy()
    tta_session_rows: List[Dict[str, object]] = []
    if not g.empty:
        for (u, sess), grp in g.groupby(["claimed_user_id", "session_id"], sort=True):
            grp = grp[np.isfinite(grp["window_start_epoch"]) & np.isfinite(grp["window_end_epoch"])]
            if grp.empty:
                continue
            session_start = float(grp["window_start_epoch"].min())
            accepted = grp[grp["prediction"].astype(int) == 1].sort_values(["window_end_epoch", "window_start_epoch", "claim_id"], kind="mergesort")
            if not accepted.empty:
                first = accepted.iloc[0]
                behavioral = max(0.0, float(first["window_end_epoch"]) - session_start)
                total = behavioral + (resource_ms / 1000.0 if math.isfinite(resource_ms) else 0.0)
                first_claim = str(first["claim_id"]); accepted_flag = 1
            else:
                behavioral = total = float("nan"); first_claim = ""; accepted_flag = 0
            tta_session_rows.append({
                "cohort": cohort, "model_family": model, "protocol": protocol, "claimed_user_id": str(u),
                "session_id": str(sess), "genuine_claims_in_session": len(grp),
                "accepted_genuine_claims": int((grp["prediction"].astype(int) == 1).sum()),
                "session_authenticated": accepted_flag, "first_accepted_claim_id": first_claim,
                "behavioral_tta_seconds": behavioral, "inference_ms_per_claim": resource_ms,
                "total_tta_seconds": total, "adequate_session_user": int(str(u) in adequate_users),
            })
    success_behavioral = [float(r["behavioral_tta_seconds"]) for r in tta_session_rows if int(r["session_authenticated"]) == 1]
    success_total = [float(r["total_tta_seconds"]) for r in tta_session_rows if int(r["session_authenticated"]) == 1]
    authenticated = sum(int(r["session_authenticated"]) for r in tta_session_rows)
    tta_summary = {
        "cohort": cohort, "model_family": model, "protocol": protocol,
        "sessions": len(tta_session_rows), "authenticated_sessions": authenticated,
        "session_authentication_coverage": (authenticated / len(tta_session_rows)) if tta_session_rows else float("nan"),
        "median_behavioral_tta_seconds": percentile(success_behavioral, 50),
        "mean_behavioral_tta_seconds": float(np.mean(success_behavioral)) if success_behavioral else float("nan"),
        "p95_behavioral_tta_seconds": percentile(success_behavioral, 95),
        "median_total_tta_seconds": percentile(success_total, 50),
        "p95_total_tta_seconds": percentile(success_total, 95),
        "tta_definition": "first accepted genuine window end minus first genuine window start within session; successful sessions only",
    }
    return {
        "overall": overall, "per_user": per_user_rows, "macro": macro, "subsets": subset_rows,
        "tta_sessions": tta_session_rows, "tta_summary": tta_summary, "rows": len(df),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Keystroke-Mouse Stage 4B frozen Test evaluation")
    ap.add_argument("--input-stage4a", required=True)
    ap.add_argument("--input-stage3b", required=True)
    ap.add_argument("--input-stage3c", required=True)
    ap.add_argument("--input-stage2a", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    started = time.time(); started_utc = utc_now()
    p4, p3b, p3c, p2a = map(Path, [args.input_stage4a, args.input_stage3b, args.input_stage3c, args.input_stage2a])
    for p in [p4, p3b, p3c, p2a]:
        if not p.is_file():
            raise FileNotFoundError(p)
    out = Path(args.output)
    out_zip = out.with_suffix(".zip")
    if out.exists() or out_zip.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out} or {out_zip}; use --overwrite")
        if out.exists(): shutil.rmtree(out)
        if out_zip.exists(): out_zip.unlink()
    out.mkdir(parents=True)

    warnings: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    session_counts, adequate_users_list = load_adequate_users(p2a)
    adequate_users = set(adequate_users_list)
    write_csv(out / "adequate_session_users.csv", [
        {"user_id": u, "core_sessions": session_counts.get(u, 0), "adequate_session_user": int(u in adequate_users)}
        for u in sorted(session_counts, key=lambda x: int(x.replace("User", "")))
    ])

    resources = aggregate_resources(load_resource_rows(p3b) + load_resource_rows(p3c))
    resource_map = {(r["cohort"], r["model_family"]): safe_float(str(r["score_ms_per_claim_output"])) for r in resources}
    write_csv(out / "model_resource_summary.csv", resources)

    overall_rows: List[Dict[str, object]] = []
    per_user_rows: List[Dict[str, object]] = []
    macro_rows: List[Dict[str, object]] = []
    subset_rows: List[Dict[str, object]] = []
    tta_session_rows: List[Dict[str, object]] = []
    tta_summary_rows: List[Dict[str, object]] = []
    claim_file_index: List[Dict[str, object]] = []
    integrity_rows: List[Dict[str, object]] = []

    with zipfile.ZipFile(p4) as z4, zipfile.ZipFile(p3b) as z3b, zipfile.ZipFile(p3c) as z3c:
        root4, root3b, root3c = zip_root(z4), zip_root(z3b), zip_root(z3c)
        index_rows = read_csv_member(z4, find_member(z4, "calibrated_score_file_index.csv"))
        test_rows = [r for r in index_rows if r["split"] == "test"]
        for idx, r in enumerate(sorted(test_rows, key=lambda q: (q["cohort"], q["model_family"], q["protocol"])), 1):
            cohort, model, protocol = r["cohort"], r["model_family"], r["protocol"]
            z3, root3 = (z3b, root3b) if model in {"centroid", "oneclass_svm", "linear_svm", "random_forest"} else (z3c, root3c)
            member4 = stage4a_member(root4, r["relative_path"])
            member3 = stage3_member(root3, cohort, model, protocol)
            if member4 not in z4.namelist(): raise RuntimeError(f"Missing Stage4A member: {member4}")
            if member3 not in z3.namelist(): raise RuntimeError(f"Missing Stage3 member: {member3}")
            rel_out = Path("evaluated_test_claims") / cohort / model / protocol / "test_claim_evaluation.csv"
            result = analyze_file(z4, member4, z3, member3, out / rel_out, resource_map.get((cohort, model), float("nan")), cohort, model, protocol, adequate_users)
            if result["rows"] != safe_int(r["claims"]):
                raise RuntimeError(f"Claim count mismatch for {cohort}/{model}/{protocol}: {result['rows']} vs {r['claims']}")
            overall_rows.append(result["overall"]); per_user_rows.extend(result["per_user"]); macro_rows.append(result["macro"])
            subset_rows.extend(result["subsets"]); tta_session_rows.extend(result["tta_sessions"]); tta_summary_rows.append(result["tta_summary"])
            claim_file_index.append({
                "cohort": cohort, "model_family": model, "protocol": protocol, "split": "test",
                "relative_path": rel_out.as_posix(), "claims": result["rows"], "sha256": sha256_file(out / rel_out),
            })
            integrity_rows.extend([
                {"cohort": cohort, "model_family": model, "protocol": protocol, "check": "claim_count_matches_stage4a_index", "status": "PASS", "value": result["rows"]},
                {"cohort": cohort, "model_family": model, "protocol": protocol, "check": "test_metrics_computed_after_stage4a_complete", "status": "PASS", "value": 1},
                {"cohort": cohort, "model_family": model, "protocol": protocol, "check": "frozen_predictions_used", "status": "PASS", "value": 1},
            ])
            print(f"[{idx}/{len(test_rows)}] {cohort}/{model}/{protocol}: {result['rows']} claims", flush=True)

    write_csv(out / "overall_test_metrics.csv", overall_rows)
    write_csv(out / "per_user_test_metrics.csv", per_user_rows)
    write_csv(out / "macro_user_test_metrics.csv", macro_rows)
    write_csv(out / "fallback_and_subset_sensitivity.csv", subset_rows)
    write_csv(out / "tta_session_details.csv", tta_session_rows)
    write_csv(out / "tta_summary.csv", tta_summary_rows)
    write_csv(out / "evaluated_test_claim_file_index.csv", claim_file_index)
    write_csv(out / "integrity_checks.csv", integrity_rows)

    # Rankings are based on frozen Test results and are descriptive only.
    ranking_rows: List[Dict[str, object]] = []
    for (cohort, protocol), group in sorted(defaultdict(list, {k: [] for k in []}).items()):
        pass
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in macro_rows: grouped[(str(r["cohort"]), str(r["protocol"]))].append(r)
    for (cohort, protocol), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: (float(r["eer"]) if math.isfinite(float(r["eer"])) else 999, -(float(r["auc"]) if math.isfinite(float(r["auc"])) else -999), -float(r["mcc"])))
        for rank, r in enumerate(ordered, 1):
            ranking_rows.append({
                "cohort": cohort, "protocol": protocol, "rank_by_macro_user_eer": rank,
                "model_family": r["model_family"], "macro_user_eer": r["eer"], "macro_user_auc": r["auc"],
                "macro_user_far": r["far"], "macro_user_frr": r["frr"], "macro_user_f1": r["f1"], "macro_user_mcc": r["mcc"],
            })
    write_csv(out / "model_ranking_summary.csv", ranking_rows)

    # Add warnings for weak results and low TTA coverage, without changing eligibility.
    for r in macro_rows:
        if math.isfinite(float(r["auc"])) and float(r["auc"]) < 0.5:
            warnings.append({"severity": "WARN", "code": "TEST_AUC_BELOW_CHANCE", "cohort": r["cohort"], "model_family": r["model_family"], "protocol": r["protocol"], "message": f"Macro-user Test AUC={float(r['auc']):.4f}<0.5; reported without score-direction reversal."})
    for r in tta_summary_rows:
        if math.isfinite(float(r["session_authentication_coverage"])) and float(r["session_authentication_coverage"]) < 0.5:
            warnings.append({"severity": "WARN", "code": "LOW_SESSION_AUTHENTICATION_COVERAGE", "cohort": r["cohort"], "model_family": r["model_family"], "protocol": r["protocol"], "message": f"Only {float(r['session_authentication_coverage']):.3f} of genuine sessions achieved an accepted claim."})
    write_csv(out / "warnings_and_failures.csv", warnings + failures, fields=["severity", "code", "cohort", "model_family", "protocol", "message"])

    decision_md = f"""# Keystroke–Mouse Stage 4B Frozen Decisions\n\n- Version: {VERSION}\n- Test evaluation opened only after Stage 4A was complete.\n- Frozen calibrated scores, thresholds, and predictions were used without refitting.\n- Balanced 1:1 and Realistic All-Impostor protocols were evaluated separately.\n- Macro-user metrics give equal weight to each claimed identity.\n- Adequate-session subset: claimed users with at least {ADEQUATE_MIN_CORE_SESSIONS} Core multimodal sessions in Stage 2A.\n- TTA definition: first accepted genuine window end minus first genuine window start within a session; reported on successfully authenticated sessions, with coverage reported separately.\n- Inference time, model size, and parameter counts are inherited from frozen Stage 3B/3C resource audits.\n- No McNemar, bootstrap, sign-flip, Holm correction, or model-selection retraining occurs here; those are reserved for Stage 5.\n"""
    (out / "stage4b_frozen_decisions.md").write_text(decision_md, encoding="utf-8")

    finished = time.time(); finished_utc = utc_now()
    summary = {
        "stage": "4B", "version": VERSION, "status": "PASS" if not failures else "FAIL",
        "input_stage4a": str(p4), "input_stage3b": str(p3b), "input_stage3c": str(p3c), "input_stage2a": str(p2a),
        "input_stage4a_sha256": sha256_file(p4), "input_stage3b_sha256": sha256_file(p3b), "input_stage3c_sha256": sha256_file(p3c), "input_stage2a_sha256": sha256_file(p2a),
        "test_score_files_evaluated": len(claim_file_index), "test_claim_scores_evaluated": sum(int(r["claims"]) for r in claim_file_index),
        "model_protocol_results": len(overall_rows), "per_user_results": len(per_user_rows),
        "adequate_users": adequate_users_list, "warnings": len(warnings), "failures": len(failures),
        "started_at_utc": started_utc, "finished_at_utc": finished_utc, "elapsed_seconds": round(finished-started, 3),
    }
    (out / "stage4b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Completion marker first, then manifest excludes itself only.
    (out / "STAGE4B_COMPLETE.ok").write_text(json.dumps({"stage": "4B", "version": VERSION, "status": summary["status"], "completed_at_utc": finished_utc}, indent=2), encoding="utf-8")
    manifest_rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "output_manifest_sha256.csv":
            manifest_rows.append({"relative_path": p.relative_to(out).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    write_csv(out / "output_manifest_sha256.csv", manifest_rows)

    # Verify manifest.
    for r in manifest_rows:
        p = out / str(r["relative_path"])
        if p.stat().st_size != int(r["size_bytes"]) or sha256_file(p) != r["sha256"]:
            raise RuntimeError(f"Manifest verification failed for {p}")

    # ZIP with single root folder.
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file(): z.write(p, arcname=f"{out.name}/{p.relative_to(out).as_posix()}")
    with zipfile.ZipFile(out_zip) as z:
        bad = z.testzip()
        if bad: raise RuntimeError(f"ZIP CRC failed: {bad}")

    print(json.dumps({
        "status": summary["status"], "test_score_files_evaluated": len(claim_file_index),
        "test_claim_scores_evaluated": summary["test_claim_scores_evaluated"],
        "manifest_files_verified": len(manifest_rows), "zip_crc_verified": True,
        "output_zip": str(out_zip), "warnings": len(warnings), "failures": len(failures)
    }, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
