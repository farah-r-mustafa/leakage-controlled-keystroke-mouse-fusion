#!/usr/bin/env python3
"""
Keystroke–Mouse Stage 5.1 — Unified Evaluation and Statistical Validation
Safe pilot/full runner. It consumes frozen Stage 4B outputs only and never retrains,
recalibrates, or changes thresholds/predictions.

Python >= 3.10
Dependencies: numpy, pandas, scipy, scikit-learn
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

STAGE_VERSION = "5.1.0-pilot"
DATASET_ID = "keystroke_mouse_ueba"
SELECTED_MODELS = ["random_forest", "cnn", "bilstm", "cnn_bilstm"]
DEEP_MODELS = ["cnn", "bilstm", "cnn_bilstm"]
COHORTS = ["keyboard", "mouse", "multimodal_primary", "multimodal_any"]
PROTOCOLS = ["balanced_1to1", "realistic_all_impostor"]
METRICS = ["auc", "eer", "far", "frr", "f1", "mcc"]
HIGHER_IS_BETTER = {"auc": True, "eer": False, "far": False, "frr": False, "f1": True, "mcc": True}
CLAIM_COLUMNS = [
    "claim_id", "window_id", "session_id", "actual_user_id", "claimed_user_id",
    "is_genuine", "calibrated_score", "prediction", "correct", "threshold_source",
    "adequate_session_user", "split", "cohort", "protocol", "model_family"
]


@dataclass
class RunContext:
    mode: str
    bootstrap_reps: int
    alpha: float
    seed: int
    started_at: str
    input_path: str
    input_sha256: str | None
    stage4b_root: Path
    output_dir: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def safe_json_dump(obj: object, path: Path) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def locate_stage4b_root(base: Path) -> Path:
    candidates = []
    if (base / "stage4b_summary.json").exists():
        candidates.append(base)
    candidates.extend(p.parent for p in base.rglob("stage4b_summary.json"))
    unique = sorted(set(candidates), key=lambda p: len(p.parts))
    if len(unique) != 1:
        raise RuntimeError(f"Expected exactly one Stage 4B root; found {len(unique)}: {unique}")
    return unique[0]


def prepare_input(input_path: Path, work_dir: Path) -> tuple[Path, str | None]:
    if input_path.is_dir():
        return locate_stage4b_root(input_path), None
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.suffix.lower() != ".zip":
        raise ValueError("--input must be a Stage 4B directory or .zip archive")
    digest = sha256_file(input_path)
    extract_dir = work_dir / "stage4b_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe ZIP path: {member.filename}")
        zf.extractall(extract_dir)
    return locate_stage4b_root(extract_dir), digest


def append_check(checks: list[dict], check: str, passed: bool, details: str = "") -> None:
    checks.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})
    if not passed:
        raise RuntimeError(f"Integrity check failed: {check}: {details}")


def validate_stage4b(root: Path, checks: list[dict]) -> tuple[pd.DataFrame, dict]:
    required = [
        "STAGE4B_COMPLETE.ok", "stage4b_summary.json", "stage4b_frozen_decisions.md",
        "evaluated_test_claim_file_index.csv", "overall_test_metrics.csv",
        "per_user_test_metrics.csv", "macro_user_test_metrics.csv",
        "model_resource_summary.csv", "adequate_session_users.csv"
    ]
    for name in required:
        append_check(checks, f"required_file::{name}", (root / name).is_file(), str(root / name))

    summary = json.loads((root / "stage4b_summary.json").read_text(encoding="utf-8"))
    append_check(checks, "stage4b_status_pass", summary.get("status") == "PASS", str(summary.get("status")))
    append_check(checks, "stage4b_failures_zero", int(summary.get("failures", -1)) == 0, str(summary.get("failures")))
    append_check(checks, "stage4b_score_file_count", int(summary.get("test_score_files_evaluated", -1)) == 56,
                 str(summary.get("test_score_files_evaluated")))
    append_check(checks, "stage4b_claim_count", int(summary.get("test_claim_scores_evaluated", -1)) == 887068,
                 str(summary.get("test_claim_scores_evaluated")))

    idx = pd.read_csv(root / "evaluated_test_claim_file_index.csv")
    expected_cols = {"cohort", "model_family", "protocol", "split", "relative_path", "claims", "sha256"}
    append_check(checks, "index_schema", expected_cols.issubset(idx.columns), ",".join(idx.columns))
    append_check(checks, "index_rows_56", len(idx) == 56, str(len(idx)))
    append_check(checks, "index_test_only", set(idx["split"].astype(str)) == {"test"}, str(sorted(idx["split"].unique())))
    for cohort in COHORTS:
        for model in SELECTED_MODELS:
            for protocol in PROTOCOLS:
                rows = idx[(idx.cohort == cohort) & (idx.model_family == model) & (idx.protocol == protocol)]
                append_check(checks, f"selected_cell::{cohort}::{model}::{protocol}", len(rows) == 1, f"rows={len(rows)}")

    # Verify every indexed file against frozen hash and count. This is intentionally strict.
    for row in idx.itertuples(index=False):
        path = root / str(row.relative_path)
        append_check(checks, f"indexed_file_exists::{row.cohort}::{row.model_family}::{row.protocol}", path.is_file(), str(path))
        actual_sha = sha256_file(path)
        append_check(checks, f"indexed_sha256::{row.cohort}::{row.model_family}::{row.protocol}",
                     actual_sha.lower() == str(row.sha256).lower(), actual_sha)
        # Count lines without loading all columns.
        with path.open("rb") as f:
            n_rows = sum(1 for _ in f) - 1
        append_check(checks, f"indexed_claims::{row.cohort}::{row.model_family}::{row.protocol}",
                     n_rows == int(row.claims), f"expected={row.claims}; actual={n_rows}")
    return idx, summary


def claim_path(root: Path, idx: pd.DataFrame, cohort: str, model: str, protocol: str) -> Path:
    row = idx[(idx.cohort == cohort) & (idx.model_family == model) & (idx.protocol == protocol)]
    if len(row) != 1:
        raise RuntimeError(f"Missing/duplicate claim file: {cohort}/{model}/{protocol}")
    return root / str(row.iloc[0]["relative_path"])


def read_claims(root: Path, idx: pd.DataFrame, cohort: str, model: str, protocol: str) -> pd.DataFrame:
    path = claim_path(root, idx, cohort, model, protocol)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in CLAIM_COLUMNS if c not in header]
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")
    df = pd.read_csv(path, usecols=CLAIM_COLUMNS,
                     dtype={
                         "claim_id": "string", "window_id": "string", "session_id": "string",
                         "actual_user_id": "string", "claimed_user_id": "string",
                         "threshold_source": "string", "split": "string", "cohort": "string",
                         "protocol": "string", "model_family": "string"
                     })
    for c in ["is_genuine", "prediction", "correct", "adequate_session_user"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.int8)
    df["calibrated_score"] = pd.to_numeric(df["calibrated_score"], errors="raise").astype(float)
    return df


def semantic_claim_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["window_id"].astype(str) + "|" + df["session_id"].astype(str) + "|" +
        df["actual_user_id"].astype(str) + "|" + df["claimed_user_id"].astype(str) + "|" +
        df["is_genuine"].astype(str)
    )


def compute_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    far = fp / (fp + tn) if (fp + tn) else np.nan
    frr = fn / (fn + tp) if (fn + tp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mcc = float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else np.nan
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan
    eer = np.nan
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, score, pos_label=1)
        fnr = 1.0 - tpr
        i = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[i] + fnr[i]) / 2.0)
    return {
        "auc": auc, "eer": eer, "far": float(far), "frr": float(frr),
        "f1": float(f1), "mcc": mcc,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "claims": int(len(y))
    }


def build_stratified_session_clusters(df: pd.DataFrame) -> list[list[np.ndarray]]:
    # Nested list: one list of session index arrays per actual user.
    groups: list[list[np.ndarray]] = []
    for _, udf in df.groupby("actual_user_id", sort=True, observed=True):
        user_clusters = [g.index.to_numpy(dtype=np.int64) for _, g in udf.groupby("session_id", sort=True, observed=True)]
        if user_clusters:
            groups.append(user_clusters)
    if not groups:
        raise RuntimeError("No session clusters found")
    return groups


def sample_cluster_indices(groups: list[list[np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    chosen: list[np.ndarray] = []
    for user_clusters in groups:
        picks = rng.integers(0, len(user_clusters), size=len(user_clusters))
        chosen.extend(user_clusters[int(i)] for i in picks)
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.concatenate(chosen)


def percentile_ci(values: Sequence[float], alpha: float) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, 0
    lo, hi = np.quantile(arr, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0), int(len(arr))


def two_sided_bootstrap_p(favorable_deltas: Sequence[float]) -> float:
    arr = np.asarray(favorable_deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    p_lo = (np.sum(arr <= 0) + 1) / (len(arr) + 1)
    p_hi = (np.sum(arr >= 0) + 1) / (len(arr) + 1)
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


def exact_sign_flip(diffs: Sequence[float]) -> tuple[float, float, int, int]:
    x = np.asarray(diffs, dtype=float)
    x = x[np.isfinite(x)]
    x = x[np.abs(x) > 1e-15]
    n = len(x)
    if n == 0:
        return 1.0, 0.0, 0, 1
    observed = float(np.mean(x))
    if n <= 20:
        total = 1 << n
        extreme = 0
        abs_obs = abs(observed) - 1e-15
        for mask in range(total):
            signs = np.fromiter((1.0 if (mask >> i) & 1 else -1.0 for i in range(n)), dtype=float, count=n)
            stat = float(np.mean(signs * x))
            if abs(stat) >= abs_obs:
                extreme += 1
        p = extreme / total
        return float(p), observed, n, total
    # Deterministic Monte Carlo fallback, practically unused for this dataset.
    rng = np.random.default_rng(5102026)
    reps = 200000
    signs = rng.choice([-1.0, 1.0], size=(reps, n))
    stats = np.mean(signs * x, axis=1)
    p = (np.sum(np.abs(stats) >= abs(observed) - 1e-15) + 1) / (reps + 1)
    return float(p), observed, n, reps


def exact_label_permutation(values: Sequence[float], groups: Sequence[int]) -> tuple[float, float, int]:
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups, dtype=int)
    keep = np.isfinite(v) & np.isin(g, [0, 1])
    v, g = v[keep], g[keep]
    n1 = int(np.sum(g == 1))
    n = len(v)
    if n1 == 0 or n1 == n:
        return np.nan, np.nan, 0
    observed = float(np.mean(v[g == 1]) - np.mean(v[g == 0]))
    combos = list(itertools.combinations(range(n), n1))
    extreme = 0
    abs_obs = abs(observed) - 1e-15
    all_idx = set(range(n))
    for c in combos:
        s1 = np.array(c, dtype=int)
        s0 = np.array(sorted(all_idx.difference(c)), dtype=int)
        stat = float(np.mean(v[s1]) - np.mean(v[s0]))
        if abs(stat) >= abs_obs:
            extreme += 1
    return float(extreme / len(combos)), observed, len(combos)


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out


def apply_holm(df: pd.DataFrame, family_cols: list[str], p_col: str = "p_value") -> pd.DataFrame:
    out = df.copy()
    out["p_holm_family"] = np.nan
    for _, inds in out.groupby(family_cols, dropna=False, sort=False).groups.items():
        inds = list(inds)
        out.loc[inds, "p_holm_family"] = holm_adjust(out.loc[inds, p_col].to_numpy())
    out["p_holm_global"] = holm_adjust(out[p_col].to_numpy())
    out["significant_family_0_05"] = (out["p_holm_family"] < 0.05).astype(int)
    out["significant_global_0_05"] = (out["p_holm_global"] < 0.05).astype(int)
    return out


def run_claim_pair_and_bootstrap(ctx: RunContext, idx: pd.DataFrame, checks: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ci_rows: list[dict] = []
    delta_rows: list[dict] = []
    mcnemar_rows: list[dict] = []
    overlap_rows: list[dict] = []
    rng_master = np.random.default_rng(ctx.seed)

    for cohort in COHORTS:
        protocol_frames: dict[str, dict[str, pd.DataFrame]] = {}
        for protocol in PROTOCOLS:
            frames: dict[str, pd.DataFrame] = {}
            for model in SELECTED_MODELS:
                df = read_claims(ctx.stage4b_root, idx, cohort, model, protocol)
                append_check(checks, f"claims_split_test::{cohort}::{model}::{protocol}", set(df.split.astype(str)) == {"test"}, "")
                append_check(checks, f"claims_metadata::{cohort}::{model}::{protocol}",
                             set(df.cohort.astype(str)) == {cohort} and set(df.protocol.astype(str)) == {protocol} and set(df.model_family.astype(str)) == {model}, "")
                append_check(checks, f"claims_no_duplicate_id::{cohort}::{model}::{protocol}", not df.claim_id.duplicated().any(), "")
                append_check(checks, f"claims_prediction_binary::{cohort}::{model}::{protocol}", set(df.prediction.unique()).issubset({0, 1}), "")
                correct_recomputed = (df.prediction.to_numpy() == df.is_genuine.to_numpy()).astype(np.int8)
                append_check(checks, f"claims_correct_consistent::{cohort}::{model}::{protocol}",
                             np.array_equal(correct_recomputed, df.correct.to_numpy()), "")
                frames[model] = df.reset_index(drop=True)

            base = frames[SELECTED_MODELS[0]]
            for model in SELECTED_MODELS[1:]:
                same_ids = base.claim_id.equals(frames[model].claim_id)
                same_sem = semantic_claim_key(base).equals(semantic_claim_key(frames[model]))
                same_y = base.is_genuine.equals(frames[model].is_genuine)
                append_check(checks, f"paired_claims::{cohort}::{protocol}::{model}", same_ids and same_sem and same_y,
                             f"claim_id={same_ids}; semantic={same_sem}; y={same_y}")

            # McNemar RF vs deep models on exactly paired claims.
            a = frames["random_forest"]
            for bmodel in DEEP_MODELS:
                b = frames[bmodel]
                a_correct = a.correct.to_numpy(dtype=np.int8)
                b_correct = b.correct.to_numpy(dtype=np.int8)
                n_a_correct_b_wrong = int(np.sum((a_correct == 1) & (b_correct == 0)))
                n_a_wrong_b_correct = int(np.sum((a_correct == 0) & (b_correct == 1)))
                discordant = n_a_correct_b_wrong + n_a_wrong_b_correct
                exact_p = float(binomtest(n_a_correct_b_wrong, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
                cc_chi2 = ((abs(n_a_correct_b_wrong - n_a_wrong_b_correct) - 1.0) ** 2 / discordant) if discordant else 0.0
                odds = (n_a_correct_b_wrong + 0.5) / (n_a_wrong_b_correct + 0.5)
                mcnemar_rows.append({
                    "analysis_type": "claim_paired_mcnemar", "cohort": cohort, "protocol": protocol,
                    "model_a": "random_forest", "model_b": bmodel, "claims_paired": len(a),
                    "a_correct_b_wrong": n_a_correct_b_wrong, "a_wrong_b_correct": n_a_wrong_b_correct,
                    "discordant_claims": discordant, "delta_accuracy_a_minus_b": float(a_correct.mean() - b_correct.mean()),
                    "discordant_odds_a_over_b_haldane": float(odds), "mcnemar_exact_p": exact_p,
                    "mcnemar_continuity_corrected_chi2": float(cc_chi2),
                    "inference_note": "Claim-level paired test; clustered bootstrap and user sign-flip are primary for dependence-aware inference."
                })

            # Session-cluster bootstrap, stratified by actual user. Same resamples for all models.
            groups = build_stratified_session_clusters(base)
            point = {}
            boot: dict[str, dict[str, list[float]]] = {
                m: {metric: [] for metric in METRICS} for m in SELECTED_MODELS
            }
            for model, df in frames.items():
                point[model] = compute_metrics(df.is_genuine.to_numpy(), df.calibrated_score.to_numpy(), df.prediction.to_numpy())

            local_seed = int(rng_master.integers(0, np.iinfo(np.uint32).max))
            rng = np.random.default_rng(local_seed)
            for _ in range(ctx.bootstrap_reps):
                sample_idx = sample_cluster_indices(groups, rng)
                y = base.is_genuine.to_numpy()[sample_idx]
                for model, df in frames.items():
                    met = compute_metrics(y, df.calibrated_score.to_numpy()[sample_idx], df.prediction.to_numpy()[sample_idx])
                    for metric in METRICS:
                        boot[model][metric].append(met[metric])

            for model in SELECTED_MODELS:
                for metric in METRICS:
                    lo, hi, se, valid_n = percentile_ci(boot[model][metric], ctx.alpha)
                    ci_rows.append({
                        "dataset_id": DATASET_ID, "cohort": cohort, "model_family": model, "protocol": protocol,
                        "analysis_group": "all_test_claims", "metric": metric, "estimate": point[model][metric],
                        "ci_low": lo, "ci_high": hi, "bootstrap_se": se,
                        "confidence_level": 1.0 - ctx.alpha, "bootstrap_reps_requested": ctx.bootstrap_reps,
                        "bootstrap_reps_valid": valid_n, "resampling_unit": "session",
                        "bootstrap_design": "stratified_by_actual_user_then_sessions_with_replacement",
                        "claims": len(base), "sessions": int(base.session_id.nunique()),
                        "actual_users": int(base.actual_user_id.nunique()), "seed": local_seed
                    })

            for bmodel in DEEP_MODELS:
                for metric in METRICS:
                    raw = np.asarray(boot["random_forest"][metric]) - np.asarray(boot[bmodel][metric])
                    favorable = raw if HIGHER_IS_BETTER[metric] else -raw
                    lo, hi, se, valid_n = percentile_ci(raw, ctx.alpha)
                    flo, fhi, _, _ = percentile_ci(favorable, ctx.alpha)
                    delta_rows.append({
                        "analysis_type": "paired_session_cluster_bootstrap_model_difference",
                        "cohort": cohort, "protocol": protocol, "model_a": "random_forest", "model_b": bmodel,
                        "metric": metric, "estimate_a": point["random_forest"][metric], "estimate_b": point[bmodel][metric],
                        "delta_a_minus_b": point["random_forest"][metric] - point[bmodel][metric],
                        "favorable_delta_for_a": (point["random_forest"][metric] - point[bmodel][metric]) * (1 if HIGHER_IS_BETTER[metric] else -1),
                        "delta_ci_low": lo, "delta_ci_high": hi,
                        "favorable_delta_ci_low": flo, "favorable_delta_ci_high": fhi,
                        "bootstrap_se_raw_delta": se, "bootstrap_p_two_sided": two_sided_bootstrap_p(favorable),
                        "bootstrap_reps_valid": valid_n, "resampling_unit": "session",
                        "bootstrap_design": "paired_same_session_resamples_stratified_by_actual_user"
                    })
            protocol_frames[protocol] = frames

        # Verify balanced semantic claims are a subset of realistic for each model/cohort.
        for model in SELECTED_MODELS:
            bdf = protocol_frames["balanced_1to1"][model]
            rdf = protocol_frames["realistic_all_impostor"][model]
            bkeys = set(semantic_claim_key(bdf))
            rkeys = set(semantic_claim_key(rdf))
            overlap_rows.append({
                "cohort": cohort, "model_family": model,
                "balanced_claims": len(bkeys), "realistic_claims": len(rkeys),
                "semantic_overlap": len(bkeys & rkeys), "balanced_subset_of_realistic": int(bkeys.issubset(rkeys)),
                "note": "Protocol comparison uses paired user metrics, not independent-claim tests."
            })

    mcnemar = pd.DataFrame(mcnemar_rows)
    mcnemar["p_value"] = mcnemar["mcnemar_exact_p"]
    mcnemar = apply_holm(mcnemar, ["cohort", "protocol"], p_col="p_value")
    deltas = pd.DataFrame(delta_rows)
    deltas["p_value"] = deltas["bootstrap_p_two_sided"]
    deltas = apply_holm(deltas, ["cohort", "protocol"], p_col="p_value")
    return pd.DataFrame(ci_rows), deltas, mcnemar, pd.DataFrame(overlap_rows)


def paired_signflip_rows(per_user: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    pu = per_user[per_user.model_family.isin(SELECTED_MODELS) & per_user.cohort.isin(COHORTS) & per_user.protocol.isin(PROTOCOLS)].copy()

    # RF vs deep models, matched by claimed identity.
    for cohort in COHORTS:
        for protocol in PROTOCOLS:
            cell = pu[(pu.cohort == cohort) & (pu.protocol == protocol)]
            for bmodel in DEEP_MODELS:
                a = cell[cell.model_family == "random_forest"]
                b = cell[cell.model_family == bmodel]
                merged = a.merge(b, on="claimed_user_id", suffixes=("_a", "_b"), validate="one_to_one")
                for metric in METRICS:
                    raw = merged[f"{metric}_a"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
                    favorable = raw if HIGHER_IS_BETTER[metric] else -raw
                    p, mean_fav, n, perms = exact_sign_flip(favorable)
                    rows.append({
                        "analysis_type": "model_comparison", "family_id": f"model::{cohort}::{protocol}",
                        "cohort_a": cohort, "cohort_b": cohort, "protocol_a": protocol, "protocol_b": protocol,
                        "model_a": "random_forest", "model_b": bmodel, "metric": metric,
                        "paired_users": len(merged), "raw_delta_a_minus_b": float(np.nanmean(raw)),
                        "favorable_delta_for_a": float(mean_fav), "p_value": p, "nonzero_user_differences": n,
                        "signflip_permutations": perms, "unit": "claimed_user_id"
                    })

    # Modality/cohort comparisons within model/protocol. Includes primary and any multimodal definitions.
    cohort_pairs = list(itertools.combinations(COHORTS, 2))
    for model in SELECTED_MODELS:
        for protocol in PROTOCOLS:
            cell = pu[(pu.model_family == model) & (pu.protocol == protocol)]
            for ca, cb in cohort_pairs:
                a = cell[cell.cohort == ca]
                b = cell[cell.cohort == cb]
                merged = a.merge(b, on="claimed_user_id", suffixes=("_a", "_b"), validate="one_to_one")
                for metric in METRICS:
                    raw = merged[f"{metric}_a"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
                    favorable = raw if HIGHER_IS_BETTER[metric] else -raw
                    p, mean_fav, n, perms = exact_sign_flip(favorable)
                    rows.append({
                        "analysis_type": "cohort_comparison", "family_id": f"cohort::{model}::{protocol}",
                        "cohort_a": ca, "cohort_b": cb, "protocol_a": protocol, "protocol_b": protocol,
                        "model_a": model, "model_b": model, "metric": metric,
                        "paired_users": len(merged), "raw_delta_a_minus_b": float(np.nanmean(raw)),
                        "favorable_delta_for_a": float(mean_fav), "p_value": p, "nonzero_user_differences": n,
                        "signflip_permutations": perms, "unit": "claimed_user_id"
                    })

    # Balanced vs realistic within the same model/cohort.
    for model in SELECTED_MODELS:
        for cohort in COHORTS:
            cell = pu[(pu.model_family == model) & (pu.cohort == cohort)]
            a = cell[cell.protocol == "balanced_1to1"]
            b = cell[cell.protocol == "realistic_all_impostor"]
            merged = a.merge(b, on="claimed_user_id", suffixes=("_a", "_b"), validate="one_to_one")
            for metric in METRICS:
                raw = merged[f"{metric}_a"].to_numpy(float) - merged[f"{metric}_b"].to_numpy(float)
                favorable = raw if HIGHER_IS_BETTER[metric] else -raw
                p, mean_fav, n, perms = exact_sign_flip(favorable)
                rows.append({
                    "analysis_type": "protocol_comparison", "family_id": f"protocol::{model}::{cohort}",
                    "cohort_a": cohort, "cohort_b": cohort,
                    "protocol_a": "balanced_1to1", "protocol_b": "realistic_all_impostor",
                    "model_a": model, "model_b": model, "metric": metric,
                    "paired_users": len(merged), "raw_delta_a_minus_b": float(np.nanmean(raw)),
                    "favorable_delta_for_a": float(mean_fav), "p_value": p, "nonzero_user_differences": n,
                    "signflip_permutations": perms, "unit": "claimed_user_id"
                })

    out = pd.DataFrame(rows)
    return apply_holm(out, ["family_id"], p_col="p_value")


def subgroup_analysis(per_user: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pu = per_user[per_user.model_family.isin(SELECTED_MODELS) & per_user.cohort.isin(COHORTS) & per_user.protocol.isin(PROTOCOLS)].copy()
    desc_rows: list[dict] = []
    test_rows: list[dict] = []
    for cohort in COHORTS:
        for model in SELECTED_MODELS:
            for protocol in PROTOCOLS:
                cell = pu[(pu.cohort == cohort) & (pu.model_family == model) & (pu.protocol == protocol)].copy()
                analyses = [
                    ("session_adequacy", cell["adequate_session_user"].astype(int), {1: "adequate_session_subset", 0: "limited_session_subset"}),
                    ("threshold_source", cell["threshold_source"].astype(str).eq("personal_validation_eer").astype(int), {1: "personal_threshold", 0: "global_fallback"})
                ]
                for analysis_name, grouping, labels in analyses:
                    for group_value, label in labels.items():
                        part = cell[grouping.to_numpy() == group_value]
                        for metric in METRICS:
                            vals = part[metric].to_numpy(float)
                            desc_rows.append({
                                "analysis_type": analysis_name, "cohort": cohort, "model_family": model,
                                "protocol": protocol, "analysis_group": label, "metric": metric,
                                "users": len(part), "mean_user_metric": float(np.nanmean(vals)) if len(part) else np.nan,
                                "median_user_metric": float(np.nanmedian(vals)) if len(part) else np.nan,
                                "std_user_metric": float(np.nanstd(vals, ddof=1)) if len(part) > 1 else np.nan,
                                "min_user_metric": float(np.nanmin(vals)) if len(part) else np.nan,
                                "max_user_metric": float(np.nanmax(vals)) if len(part) else np.nan,
                                "interpretation": "Observational subgroup sensitivity; not a causal threshold/session effect."
                            })
                    for metric in METRICS:
                        vals = cell[metric].to_numpy(float)
                        p, raw_delta, perms = exact_label_permutation(vals, grouping.to_numpy())
                        favorable = raw_delta if HIGHER_IS_BETTER[metric] else -raw_delta
                        test_rows.append({
                            "analysis_type": analysis_name, "family_id": f"subgroup::{analysis_name}::{cohort}::{model}::{protocol}",
                            "cohort": cohort, "model_family": model, "protocol": protocol, "metric": metric,
                            "group_a": labels[1], "group_b": labels[0],
                            "users_a": int(np.sum(grouping == 1)), "users_b": int(np.sum(grouping == 0)),
                            "raw_delta_a_minus_b": raw_delta, "favorable_delta_for_a": favorable,
                            "p_value": p, "label_permutations": perms, "unit": "claimed_user_id",
                            "interpretation": "Exact user-level label permutation; observational sensitivity only."
                        })
    tests = apply_holm(pd.DataFrame(test_rows), ["family_id"], p_col="p_value")
    return pd.DataFrame(desc_rows), tests


def build_deployment_scorecard(root: Path, signflip: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
    macro = pd.read_csv(root / "macro_user_test_metrics.csv")
    resources = pd.read_csv(root / "model_resource_summary.csv")
    primary = macro[
        macro.model_family.isin(SELECTED_MODELS) & macro.cohort.isin(COHORTS) &
        (macro.protocol == "realistic_all_impostor")
    ].copy()
    resource_cols = ["cohort", "model_family", "total_train_seconds", "total_model_size_bytes", "score_ms_per_claim_output", "total_parameter_count"]
    primary = primary.merge(resources[resource_cols], on=["cohort", "model_family"], how="left", validate="one_to_one")

    # Rank-based performance score avoids mixing incompatible metric units.
    rank_parts = []
    for metric in METRICS:
        ascending = not HIGHER_IS_BETTER[metric]
        # rank 1 = best. Convert to [0,1], 1 = best.
        rank = primary[metric].rank(method="average", ascending=ascending)
        if len(primary) > 1:
            score = 1.0 - (rank - 1.0) / (len(primary) - 1.0)
        else:
            score = pd.Series(np.ones(len(primary)), index=primary.index)
        primary[f"rank_score_{metric}"] = score
        rank_parts.append(f"rank_score_{metric}")
    primary["performance_score"] = primary[rank_parts].mean(axis=1)

    # Cost score from log-scaled size, inference and training. 1 = lowest cost.
    cost_parts = []
    for col in ["total_model_size_bytes", "score_ms_per_claim_output", "total_train_seconds"]:
        x = np.log1p(pd.to_numeric(primary[col], errors="coerce"))
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        score = 1.0 - ((x - lo) / (hi - lo) if hi > lo else 0.0)
        cname = f"cost_score_{col}"
        primary[cname] = score
        cost_parts.append(cname)
    primary["computational_cost_score"] = primary[cost_parts].mean(axis=1)

    # Count Holm-significant user-level wins/losses for realistic model/cohort comparisons.
    relevant = signflip[(signflip.protocol_a == "realistic_all_impostor") & (signflip.protocol_b == "realistic_all_impostor") &
                        (signflip.analysis_type.isin(["model_comparison", "cohort_comparison"]))]
    wins, losses = [], []
    for row in primary.itertuples():
        w = l = 0
        for t in relevant.itertuples():
            if int(t.significant_family_0_05) != 1:
                continue
            candidate_is_a = (t.model_a == row.model_family and t.cohort_a == row.cohort)
            candidate_is_b = (t.model_b == row.model_family and t.cohort_b == row.cohort)
            if not (candidate_is_a or candidate_is_b):
                continue
            fav_a = float(t.favorable_delta_for_a)
            if candidate_is_a:
                w += int(fav_a > 0); l += int(fav_a < 0)
            else:
                w += int(fav_a < 0); l += int(fav_a > 0)
        wins.append(w); losses.append(l)
    primary["holm_significant_wins"] = wins
    primary["holm_significant_losses"] = losses
    max_tests = max(1, int(max(np.asarray(wins) + np.asarray(losses))))
    primary["statistical_support_score"] = 0.5 + 0.5 * (primary.holm_significant_wins - primary.holm_significant_losses) / max_tests
    primary["statistical_support_score"] = primary["statistical_support_score"].clip(0, 1)

    # Explicit documented pilot utility. Full mode uses same rule with stronger resampling.
    primary["deployment_utility"] = (
        0.70 * primary["performance_score"] +
        0.20 * primary["computational_cost_score"] +
        0.10 * primary["statistical_support_score"]
    )
    primary = primary.sort_values(
        ["deployment_utility", "performance_score", "computational_cost_score"], ascending=False
    ).reset_index(drop=True)
    primary["deployment_rank"] = np.arange(1, len(primary) + 1)
    top = primary.iloc[0]
    status = "PROVISIONAL_PILOT" if mode == "pilot" else "FINAL_FULL_STAGE5_1"
    recommendation = {
        "status": status,
        "dataset_id": DATASET_ID,
        "primary_protocol": "realistic_all_impostor",
        "recommended_cohort": str(top.cohort),
        "recommended_model_family": str(top.model_family),
        "deployment_utility": float(top.deployment_utility),
        "performance_score": float(top.performance_score),
        "computational_cost_score": float(top.computational_cost_score),
        "statistical_support_score": float(top.statistical_support_score),
        "holm_significant_wins": int(top.holm_significant_wins),
        "holm_significant_losses": int(top.holm_significant_losses),
        "selection_rule": "70% rank-based macro-user performance on Realistic protocol + 20% computational cost + 10% Holm-significant support.",
        "guardrail": "Pilot recommendation is provisional. Freeze only after full mode and scientific review; no retraining/calibration/threshold changes are performed."
    }
    return primary, recommendation


def build_cross_dataset_exports(ci: pd.DataFrame, signflip: pd.DataFrame, subgroup_desc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_export = ci.copy()
    metric_export.insert(1, "stage", "5.1")
    metric_export["threshold_policy"] = "frozen_stage4a_personal_with_global_fallback"
    metric_export["evaluation_split"] = "frozen_test"
    metric_export["comparison_ready_for"] = "Balabit Stage 5.1 (not executed here)"

    tests = signflip.copy()
    tests.insert(0, "dataset_id", DATASET_ID)
    tests.insert(1, "stage", "5.1")
    tests["comparison_ready_for"] = "Balabit Stage 5.1 (not executed here)"
    return metric_export, tests


def write_manifest(output_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.name != "output_manifest_sha256.csv":
            rows.append({
                "relative_path": p.relative_to(output_dir).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p)
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "output_manifest_sha256.csv", index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Keystroke–Mouse Stage 5.1 unified statistical validation")
    parser.add_argument("--input", required=True, help="Frozen Stage 4B .zip or extracted directory")
    parser.add_argument("--output", required=True, help="New Stage 5.1 output directory; must not be Stage 4B input")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--bootstrap-reps", type=int, default=None,
                        help="Pilot default 100; full default 2000. Minimum 20.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=5102026)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args()

    bootstrap_reps = args.bootstrap_reps if args.bootstrap_reps is not None else (100 if args.mode == "pilot" else 2000)
    if bootstrap_reps < 20:
        raise ValueError("--bootstrap-reps must be >= 20")
    if not (0 < args.alpha < 0.5):
        raise ValueError("--alpha must be between 0 and 0.5")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if input_path == output_dir or (input_path.is_dir() and output_dir.is_relative_to(input_path)):
        raise ValueError("Output must be outside the frozen Stage 4B input")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    started = time.time()
    checks: list[dict] = []
    warnings: list[dict] = []
    failures: list[dict] = []
    temp_root = Path(tempfile.mkdtemp(prefix="km_stage5_1_"))
    try:
        stage4b_root, input_sha = prepare_input(input_path, temp_root)
        ctx = RunContext(
            mode=args.mode, bootstrap_reps=bootstrap_reps, alpha=args.alpha, seed=args.seed,
            started_at=utc_now(), input_path=str(input_path), input_sha256=input_sha,
            stage4b_root=stage4b_root, output_dir=output_dir
        )
        idx, stage4b_summary = validate_stage4b(stage4b_root, checks)

        overall = pd.read_csv(stage4b_root / "overall_test_metrics.csv")
        per_user = pd.read_csv(stage4b_root / "per_user_test_metrics.csv")
        macro = pd.read_csv(stage4b_root / "macro_user_test_metrics.csv")
        resources = pd.read_csv(stage4b_root / "model_resource_summary.csv")
        selected_filter = lambda d: d[d.model_family.isin(SELECTED_MODELS) & d.cohort.isin(COHORTS) & d.protocol.isin(PROTOCOLS)].copy()
        selected_filter(overall).to_csv(output_dir / "frozen_stage4b_overall_metrics_selected.csv", index=False)
        selected_filter(per_user).to_csv(output_dir / "frozen_stage4b_per_user_metrics_selected.csv", index=False)
        selected_filter(macro).to_csv(output_dir / "frozen_stage4b_macro_user_metrics_selected.csv", index=False)
        resources[resources.model_family.isin(SELECTED_MODELS) & resources.cohort.isin(COHORTS)].to_csv(
            output_dir / "frozen_stage4b_resource_metrics_selected.csv", index=False)

        ci, paired_boot_delta, mcnemar, protocol_overlap = run_claim_pair_and_bootstrap(ctx, idx, checks)
        signflip = paired_signflip_rows(per_user)
        subgroup_desc, subgroup_tests = subgroup_analysis(per_user)
        scorecard, recommendation = build_deployment_scorecard(stage4b_root, signflip, args.mode)
        metric_export, test_export = build_cross_dataset_exports(ci, signflip, subgroup_desc)

        ci.to_csv(output_dir / "clustered_session_bootstrap_confidence_intervals.csv", index=False)
        paired_boot_delta.to_csv(output_dir / "paired_session_cluster_bootstrap_model_differences.csv", index=False)
        mcnemar.to_csv(output_dir / "mcnemar_rf_vs_deep_claim_paired.csv", index=False)
        signflip.to_csv(output_dir / "paired_user_signflip_all_comparisons_holm.csv", index=False)
        signflip[signflip.analysis_type == "model_comparison"].to_csv(output_dir / "paired_user_signflip_rf_vs_deep_holm.csv", index=False)
        signflip[signflip.analysis_type == "cohort_comparison"].to_csv(output_dir / "paired_user_signflip_cohort_comparisons_holm.csv", index=False)
        signflip[signflip.analysis_type == "protocol_comparison"].to_csv(output_dir / "paired_user_signflip_protocol_comparisons_holm.csv", index=False)
        subgroup_desc.to_csv(output_dir / "adequacy_and_threshold_subgroup_descriptives.csv", index=False)
        subgroup_tests.to_csv(output_dir / "adequacy_and_threshold_user_permutation_holm.csv", index=False)
        protocol_overlap.to_csv(output_dir / "balanced_realistic_semantic_overlap_audit.csv", index=False)
        scorecard.to_csv(output_dir / "deployment_candidate_scorecard.csv", index=False)
        safe_json_dump(recommendation, output_dir / "deployment_recommendation.json")
        metric_export.to_csv(output_dir / "balabit_comparison_ready_metric_export.csv", index=False)
        test_export.to_csv(output_dir / "balabit_comparison_ready_statistical_test_export.csv", index=False)

        checks_df = pd.DataFrame(checks)
        checks_df.to_csv(output_dir / "integrity_checks.csv", index=False)
        warnings.append({
            "severity": "INFO", "code": "PILOT_NOT_FINAL" if args.mode == "pilot" else "FULL_MODE",
            "message": "Pilot confidence intervals and recommendation are provisional; use full mode before freezing." if args.mode == "pilot" else "Full resampling mode completed; scientific review remains required before publication."
        })
        warnings.append({
            "severity": "INFO", "code": "DEPENDENCE_HANDLING",
            "message": "McNemar is reported at claim level, while primary dependence-aware inference uses session-cluster bootstrap and user-level exact tests."
        })
        warnings.append({
            "severity": "INFO", "code": "SUBGROUP_OBSERVATIONAL",
            "message": "Adequate/limited and personal/fallback analyses compare different user groups and are observational, not causal."
        })
        pd.DataFrame(warnings).to_csv(output_dir / "warnings_and_failures.csv", index=False)

        elapsed = time.time() - started
        summary = {
            "stage": "5.1", "version": STAGE_VERSION, "status": "PASS", "mode": args.mode,
            "dataset_id": DATASET_ID, "input_stage4b": str(input_path), "input_stage4b_sha256": input_sha,
            "stage4b_status": stage4b_summary.get("status"), "selected_models": SELECTED_MODELS,
            "cohorts": COHORTS, "protocols": PROTOCOLS, "metrics": METRICS,
            "bootstrap_reps": bootstrap_reps, "bootstrap_unit": "session",
            "bootstrap_design": "stratified_by_actual_user_then_sessions_with_replacement",
            "mcnemar_comparisons": int(len(mcnemar)), "cluster_bootstrap_ci_rows": int(len(ci)),
            "paired_cluster_delta_rows": int(len(paired_boot_delta)), "user_signflip_rows": int(len(signflip)),
            "subgroup_test_rows": int(len(subgroup_tests)), "integrity_checks": int(len(checks_df)),
            "integrity_failures": int((checks_df.status != "PASS").sum()),
            "recommendation": recommendation, "started_at_utc": ctx.started_at,
            "finished_at_utc": utc_now(), "elapsed_seconds": round(elapsed, 3),
            "python": sys.version, "platform": platform.platform(),
            "balabit_comparison_executed": False, "retraining_executed": False,
            "recalibration_executed": False, "threshold_changes_executed": False
        }
        safe_json_dump(summary, output_dir / "stage5_1_pilot_summary.json")
        atomic_write_text(output_dir / "STAGE5_1_PILOT_COMPLETE.ok",
                          f"PASS\nstage=5.1\nmode={args.mode}\nfinished_at_utc={summary['finished_at_utc']}\n")
        write_manifest(output_dir)
        print(json.dumps({
            "status": "PASS", "output": str(output_dir), "elapsed_seconds": summary["elapsed_seconds"],
            "recommended": f"{recommendation['recommended_cohort']} / {recommendation['recommended_model_family']}",
            "recommendation_status": recommendation["status"]
        }, indent=2))
        return 0
    except Exception as exc:
        failures.append({"severity": "FAIL", "code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        pd.DataFrame(failures).to_csv(output_dir / "warnings_and_failures.csv", index=False)
        safe_json_dump({
            "stage": "5.1", "version": STAGE_VERSION, "status": "FAIL", "mode": args.mode,
            "input": str(input_path), "error": str(exc), "finished_at_utc": utc_now()
        }, output_dir / "stage5_1_pilot_summary.json")
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
